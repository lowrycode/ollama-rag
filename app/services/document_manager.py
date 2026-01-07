from pathlib import Path
from collections import defaultdict
import hashlib
import json
import logging
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from app.config import DATA_DIR, CACHE_DIR, VECTOR_DB_DIR, EMBED_MODEL


logger = logging.getLogger(__name__)


class DocumentManager:

    def __init__(self):
        self.embedding = self._get_embedding(EMBED_MODEL)
        self.data_dir = DATA_DIR
        cache_dir = CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.hash_cache_file = cache_dir / "hash_cache.json"
        self.hash_cache = self._load_hash_cache()
        self.documents = []
        self.vector_db = self._load_or_create_vector_db(VECTOR_DB_DIR)
        self.db_sync = None

    # Private Methods
    def _load_hash_cache(self):
        if self.hash_cache_file.exists():
            try:
                with open(self.hash_cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.exception("Failed to load hash cache")
        return {}

    def _save_hash_cache(self):
        try:
            with open(self.hash_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.hash_cache, f, indent=2)
        except Exception:
            logger.exception("Failed to save hash cache")

    def _add_chunks_to_db(self, chunks):
        if not chunks:
            logger.debug("- No document chunks to add to DB")
            return

        self.vector_db.add_documents(chunks)
        logger.debug("Added %d new document chunks to DB", len(chunks))

    def _chunk_documents(self, documents):
        logger.debug("- Chunking documents...")
        if not documents:
            logger.debug("- No documents were found to split into chunks")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        chunks = splitter.split_documents(documents)
        logger.debug("- Documents have been split into chunks")
        return chunks

    def _get_embedding(self, emb_model):
        try:
            existing = ollama.list()
            if emb_model not in [m["model"] for m in existing["models"]]:
                ollama.pull(model=emb_model)
        except Exception:
            ollama.pull(model=emb_model)
        return OllamaEmbeddings(model=emb_model)

    def _hash_documents(self, docs):
        for d in docs:
            source = d.metadata.get("source", "")
            if not source:
                continue

            try:
                file_path = Path(source)
                mtime = file_path.stat().st_mtime  # modification time
                cached = self.hash_cache.get(source)

                if cached and cached.get("mtime") == mtime:
                    file_hash = cached["hash"]
                else:
                    with open(source, "rb") as f:
                        file_bytes = f.read()
                        file_hash = hashlib.sha256(file_bytes).hexdigest()
                    # Update cache
                    self.hash_cache[source] = {"hash": file_hash, "mtime": mtime}
                d.metadata["file_hash"] = file_hash
            except Exception:
                d.metadata["file_hash"] = "unknown"
                logger.exception("Error hashing file %s", source)
        # Save updated cache to disk
        self._save_hash_cache()
        return docs

    def _load_documents(self, pdf=True, txt=True, docx=True):
        """
        Load all documents of specified formats within directory
        (including sub-directories)
        """
        logger.debug("- Loading documents...")

        pdf_loader = (
            DirectoryLoader(
                self.data_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
            )
            if pdf
            else None
        )

        txt_loader = (
            DirectoryLoader(
                self.data_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
            )
            if txt
            else None
        )

        docx_loader = (
            DirectoryLoader(
                self.data_dir,
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
            )
            if docx
            else None
        )

        documents = []
        for ldr in [pdf_loader, txt_loader, docx_loader]:
            if ldr:
                try:
                    docs = ldr.load()
                except Exception:
                    logger.exception("Loader failed for %s", ldr)
                    continue
                docs = self._merge_document_pages(docs)  # because loader splits by page
                docs = self._hash_documents(docs)
                documents.extend(docs)

        document_count = len(documents)
        if document_count == 0:
            logger.debug("- No documents were found")
        else:
            logger.debug(
                "Finished loading %d documents/chunks",
                document_count,
            )

        self.documents = documents
        return documents

    def _merge_document_pages(self, docs):
        merged = defaultdict(list)
        metadata = {}

        for d in docs:
            src = d.metadata.get("source")
            merged[src].append(d.page_content)
            metadata.setdefault(src, d.metadata)

        return [
            Document(
                page_content="\n\n".join(pages),
                metadata=metadata[src]
            )
            for src, pages in merged.items()
        ]

    def _load_or_create_vector_db(self, vector_db_dir, collection_name="simple-rag"):
        vector_db_dir = Path(vector_db_dir)
        if vector_db_dir.exists() and any(vector_db_dir.iterdir()):
            logger.info("Loading existing vector db...")
            vector_db = Chroma(
                embedding_function=self.embedding,
                collection_name=collection_name,
                persist_directory=vector_db_dir,
            )
            logger.info("Vector database has been loaded")
        else:
            logger.info("Creating new vector db...")
            vector_db = Chroma(
                embedding_function=self.embedding,
                collection_name=collection_name,
                persist_directory=vector_db_dir,
            )
            logger.info("Vector database has been created")

        return vector_db

    def _display_db_sync_summary(self):
        logger.info("## SYNC SUMMARY ##")
        in_sync = self.db_sync.get("in_sync", False)
        if in_sync:
            logger.info("Vector DB is in sync with local files")
        else:
            added = self.db_sync.get("added", [])
            removed = self.db_sync.get("removed", [])
            renamed = self.db_sync.get("renamed", [])
            updated = self.db_sync.get("updated", [])

            logger.debug(
                "The following changes were made to local files "
                "and are not yet reflected in the DB:"
            )
            added_count = len(added)
            if added_count > 0:
                logger.debug("Added: %d", added_count)
                for s, _ in added:
                    logger.debug("- %s", s)

            removed_count = len(removed)
            if removed_count > 0:
                logger.debug("Removed: %d", removed_count)
                for s, _ in removed:
                    logger.debug("- %s", s)

            renamed_count = len(renamed)
            if renamed_count > 0:
                logger.debug("Renamed: %d", renamed_count)
                for new_s, _, s in renamed:
                    logger.debug("- %s renamed as %s", s, new_s)

            updated_count = len(updated)
            if updated_count > 0:
                logger.debug("Updated: %d", updated_count)
                for s, _, _ in updated:
                    logger.debug("- %s", s)

    # Public methods
    def check_db_sync(self, show_summary=True):
        logger.info("Checking sync status...")
        if not self.documents:
            self._load_documents()

        db_entries = self.vector_db.get()

        db_map = {
            meta.get("source", "unknown_source"): meta.get("file_hash", "unknown_hash")
            for meta in db_entries.get("metadatas", [])
            if "source" in meta and "file_hash" in meta
        }

        local_map = {
            d.metadata.get("source", "unknown_source"): d.metadata.get(
                "file_hash", "unknown_hash"
            )
            for d in self.documents
            if "source" in d.metadata and "file_hash" in d.metadata
        }

        # Build reverse maps (hash → source) for rename detection
        db_hash_to_source = {v: k for k, v in db_map.items()}
        local_hash_to_source = {v: k for k, v in local_map.items()}

        # --- Detect changes ---
        added = []
        removed = []
        updated = []
        renamed = []

        # Check for added / updated / renamed
        for local_src, local_hash in local_map.items():
            if local_src not in db_map:
                # Could be new or renamed
                if local_hash in db_hash_to_source:
                    # source, hash, db_source
                    renamed.append(
                        (local_src, local_hash, db_hash_to_source[local_hash])
                    )
                else:
                    added.append((local_src, local_hash))
            else:
                # Source already in DB — check if file changed
                if db_map[local_src] != local_hash:
                    # source, hash, db_hash
                    updated.append((local_src, local_hash, db_map[local_src]))

        # Check for removed
        for db_src, db_hash in db_map.items():
            if db_src not in local_map and db_hash not in local_hash_to_source:
                # db_source, db_hash
                removed.append((db_src, db_hash))

        # Update sync status
        db_sync = {
            "added": added,
            "removed": removed,
            "renamed": renamed,
            "updated": updated,
            "in_sync": all(len(x) == 0 for x in [added, removed, renamed, updated]),
        }
        self.db_sync = db_sync

        # Log in_sync
        in_sync = db_sync["in_sync"]
        if in_sync:
            logger.info("DB is in sync with local files")
        else:
            logger.info("DB is not in sync with local files")

        if show_summary:
            self._display_db_sync_summary()

        return in_sync

    def update_db_sync(self):
        logger.info("Updating DB Sync")
        if not self.db_sync:
            self.check_db_sync(show_summary=False)

        in_sync = self.db_sync.get("in_sync", False)
        if in_sync:
            self.db_sync = None
            logger.info("- DB is already in sync with local files")
            return

        added = self.db_sync.get("added", [])
        removed = self.db_sync.get("removed", [])
        renamed = self.db_sync.get("renamed", [])
        updated = self.db_sync.get("updated", [])

        local_doc_map = {
            d.metadata.get("file_hash", "unknown_hash"): d for d in self.documents
        }

        docs_to_add = []

        # Remove
        if removed:
            for src, db_hash in removed:
                self.vector_db.delete(where={"file_hash": db_hash})
                self.hash_cache.pop(src, None)  # remove from cache
            self._save_hash_cache()  # update cache file

        if renamed:
            for _, db_hash, _ in renamed:
                self.vector_db.delete(where={"file_hash": db_hash})
                local_hash = db_hash
                docs_to_add.append(local_doc_map[local_hash])

        if updated:
            for src, local_hash, db_hash in updated:
                self.vector_db.delete(where={"file_hash": db_hash})
                docs_to_add.append(local_doc_map[local_hash])

        if added:
            for _, local_hash in added:
                docs_to_add.append(local_doc_map[local_hash])

        # Add
        if docs_to_add:
            chunks = self._chunk_documents(docs_to_add)
            self._add_chunks_to_db(chunks)

        self.db_sync = None

        logger.info("DB is now in sync with local files")
