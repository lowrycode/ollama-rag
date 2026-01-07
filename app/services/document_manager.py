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
        self.local_sync_changes = None

    # ----- Private Methods -----

    # Hash cache for tracking local file changes
    def _load_hash_cache(self):
        logger.debug("Loading hash cache from %s", self.hash_cache_file)
        if self.hash_cache_file.exists():
            try:
                with open(self.hash_cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                logger.debug("Loaded hash cache with %d entries", len(cache))
                return cache
            except Exception:
                logger.exception("Failed to load hash cache")
        else:
            logger.debug("No existing hash cache file found")
        return {}

    def _save_hash_cache(self):
        try:
            with open(self.hash_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.hash_cache, f, indent=2)
            logger.debug("Hash cache saved with %d entries", len(self.hash_cache))
        except Exception:
            logger.exception("Failed to save hash cache")

    def _hash_documents(self, docs):
        logger.debug("Hashing %d documents", len(docs))
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
        logger.debug("Completed hashing documents")
        return docs

    # Document loading and chunking
    def _load_documents(self, pdf=True, txt=True, docx=True):
        """
        Load all documents of specified formats within directory
        (including sub-directories)
        """
        logger.debug("- Loading documents..")

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

                if not docs:
                    logger.debug(
                        "No documents found for loader with pattern %s", ldr.glob
                    )
                    continue

                docs = self._merge_document_pages(docs)
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
        logger.debug("Merging %d document pages", len(docs))

        # Optional: check for missing source metadata
        missing_source = [d for d in docs if not d.metadata.get("source")]
        if missing_source:
            logger.warning(
                "Found %d documents without source metadata", len(missing_source)
            )

        merged = defaultdict(list)
        metadata = {}

        for d in docs:
            src = d.metadata.get("source")
            merged[src].append(d.page_content)
            metadata.setdefault(src, d.metadata)

        return [
            Document(page_content="\n\n".join(pages), metadata=metadata[src])
            for src, pages in merged.items()
        ]

    def _chunk_documents(self, documents):
        logger.debug("- Chunking %d documents", len(documents))
        if not documents:
            logger.debug("- No documents were found to split into chunks")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        chunks = splitter.split_documents(documents)
        logger.debug("- Documents have been split into %d chunks", len(chunks))
        return chunks

    # Vector DB operations
    def _get_embedding(self, emb_model):
        try:
            existing = ollama.list()
            logger.debug(
                "Available ollama models: %s", [m["model"] for m in existing["models"]]
            )
            if emb_model not in [m["model"] for m in existing["models"]]:
                logger.debug("Model %s not found locally, pulling...", emb_model)
                ollama.pull(model=emb_model)
            else:
                logger.debug("Model %s found locally", emb_model)
        except Exception:
            logger.debug("Exception checking ollama models, pulling %s", emb_model)
            ollama.pull(model=emb_model)
        return OllamaEmbeddings(model=emb_model)

    def _load_or_create_vector_db(self, vector_db_dir, collection_name="simple-rag"):
        vector_db_dir = Path(vector_db_dir)
        if vector_db_dir.exists() and any(vector_db_dir.iterdir()):
            logger.info("Loading existing vector db..")
            vector_db = Chroma(
                embedding_function=self.embedding,
                collection_name=collection_name,
                persist_directory=vector_db_dir,
            )
            logger.info("- Vector database has been loaded")
        else:
            logger.info("Creating new vector db..")
            vector_db = Chroma(
                embedding_function=self.embedding,
                collection_name=collection_name,
                persist_directory=vector_db_dir,
            )
            logger.info("Vector database has been created")

        return vector_db

    def _add_chunks_to_db(self, chunks):
        if not chunks:
            logger.debug("- No document chunks to add to DB")
            return

        BATCH_SIZE = 50

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            logger.info(
                "Embedding chunks %d–%d of %d",
                i + 1,
                min(i + BATCH_SIZE, len(chunks)),
                len(chunks),
            )
            self.vector_db.add_documents(batch)
        logger.debug("Added %d new document chunks to DB", len(chunks))

    # Local sync changes
    def _compute_local_sync_changes(self, show_summary):
        logger.debug("Computing local sync changes")
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
        local_sync_changes = {
            "added": added,
            "removed": removed,
            "renamed": renamed,
            "updated": updated,
            "has_changes": any([added, removed, renamed, updated]),
        }
        self.local_sync_changes = local_sync_changes

        # Show summary
        if show_summary:
            self._display_local_sync_changes_summary()

        return local_sync_changes

    def _display_local_sync_changes_summary(self):
        logger.info("## SYNC SUMMARY ##")
        has_changes = self.local_sync_changes.get("has_changes", False)
        if not has_changes:
            logger.info("Vector DB is in sync with local files")
        else:
            added = self.local_sync_changes.get("added", [])
            removed = self.local_sync_changes.get("removed", [])
            renamed = self.local_sync_changes.get("renamed", [])
            updated = self.local_sync_changes.get("updated", [])

            logger.info(
                "The following changes were made to local files "
                "and are not yet reflected in the DB:"
            )
            added_count = len(added)
            if added_count > 0:
                logger.info("Added: %d", added_count)
                for s, _ in added:
                    logger.debug("- %s", s)

            removed_count = len(removed)
            if removed_count > 0:
                logger.info("Removed: %d", removed_count)
                for s, _ in removed:
                    logger.debug("- %s", s)

            renamed_count = len(renamed)
            if renamed_count > 0:
                logger.info("Renamed: %d", renamed_count)
                for new_s, _, s in renamed:
                    logger.debug("- %s renamed as %s", s, new_s)

            updated_count = len(updated)
            if updated_count > 0:
                logger.info("Updated: %d", updated_count)
                for s, _, _ in updated:
                    logger.debug("- %s", s)

    def _apply_local_sync_changes(self):
        logger.debug("Applying local sync changes")
        if not self.local_sync_changes:
            self._compute_local_sync_changes(show_summary=False)

        has_changes = self.local_sync_changes.get("has_changes", False)
        if not has_changes:
            self.local_sync_changes = None
            logger.debug("DB is already in sync with local files")
            return

        added = self.local_sync_changes.get("added", [])
        removed = self.local_sync_changes.get("removed", [])
        renamed = self.local_sync_changes.get("renamed", [])
        updated = self.local_sync_changes.get("updated", [])

        local_doc_map = {
            d.metadata.get("file_hash", "unknown_hash"): d for d in self.documents
        }

        docs_to_add = []

        # Remove
        if removed:
            for src, db_hash in removed:
                logger.debug("Removing file from DB: %s with hash %s", src, db_hash)
                self.vector_db.delete(where={"file_hash": db_hash})
                self.hash_cache.pop(src, None)  # remove from cache
            self._save_hash_cache()  # update cache file

        if renamed:
            for _, db_hash, _ in renamed:
                logger.debug("Removing renamed file hash from DB: %s", db_hash)
                self.vector_db.delete(where={"file_hash": db_hash})
                local_hash = db_hash
                docs_to_add.append(local_doc_map[local_hash])

        if updated:
            for src, local_hash, db_hash in updated:
                logger.debug(
                    "Updating file %s in DB, removing old hash %s", src, db_hash
                )
                self.vector_db.delete(where={"file_hash": db_hash})
                docs_to_add.append(local_doc_map[local_hash])

        if added:
            for _, local_hash in added:
                logger.debug("Adding new file hash to DB: %s", local_hash)
                docs_to_add.append(local_doc_map[local_hash])

        # Add
        if docs_to_add:
            chunks = self._chunk_documents(docs_to_add)
            self._add_chunks_to_db(chunks)
            self._save_hash_cache()  # update cache file

        self.local_sync_changes = None

        logger.debug("Local sync changes have been applied")

    # ----- Public methods -----
    def is_in_sync(self, show_summary=False):
        logger.info("Checking if local files are in sync with database..")
        local_sync_changes = self._compute_local_sync_changes(show_summary=show_summary)

        in_sync = not local_sync_changes["has_changes"]
        if in_sync:
            logger.info("- Database is in sync with local files")
        else:
            logger.info("- Changes to local files have been detected")
        return in_sync

    def sync(self, show_summary=False):
        logger.info("Syncing local files with database..")
        if not self.local_sync_changes:
            self._compute_local_sync_changes(show_summary=show_summary)

        if self.local_sync_changes["has_changes"]:
            self._apply_local_sync_changes()
            logger.info("Local changes have been synced with database")
        else:
            logger.info("Local files are already in-sync with database")
            self.local_sync_changes = None
