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
    """
    Manages local document ingestion, hashing, and synchronization with a vector
    database.

    Responsibilities:
    - Load documents from the local data directory
    - Merge multi-page documents
    - Hash files and maintain a persistent hash cache
    - Detect differences between local files and the vector DB
    - Apply incremental sync changes (add, remove, update, rename)
    - Chunk documents and embed them into a Chroma vector database

    This class is stateful and maintains:
    - self.hash_cache: persisted mapping of file paths to hashes and mtimes
    - self.vector_db: a persistent Chroma vector database
    """

    def __init__(self):
        self.embedding = self._get_embedding(EMBED_MODEL)
        self.data_dir = DATA_DIR
        cache_dir = CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.hash_cache_file = cache_dir / "hash_cache.json"
        self.hash_cache = self._load_hash_cache()
        self.vector_db = self._load_or_create_vector_db(VECTOR_DB_DIR)

    # ----- Private Methods -----

    # Hash cache for tracking local file changes
    def _load_hash_cache(self):
        """
        Load the persisted file hash cache from disk.

        The hash cache maps file paths to:
        - SHA-256 hash of file contents
        - Last modification time (mtime)

        Returns:
            dict: Cached hash metadata keyed by file path.
                Returns an empty dict if the cache does not exist or fails to load.
        """

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
        """
        Persist the in-memory hash cache to disk.

        This should be called whenever the hash cache is modified
        (e.g. after hashing new files or removing deleted files).
        """

        try:
            with open(self.hash_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.hash_cache, f, indent=2)
            logger.debug("Hash cache saved with %d entries", len(self.hash_cache))
        except Exception:
            logger.exception("Failed to save hash cache")

    def _hash_documents(self, docs):
        """
        Compute and attach file hashes to document metadata.

        For each document:
        - Uses the source file path from metadata
        - Reuses cached hashes if the file mtime is unchanged
        - Otherwise computes a new SHA-256 hash
        - Stores the hash in document.metadata["file_hash"]

        Side effects:
        - Updates self.hash_cache
        - Persists the hash cache to disk

        Args:
            docs (list[Document]): Documents whose source files should be hashed.

        Returns:
            list[Document]: The same documents with updated metadata.
        """

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
        Load documents from the data directory and prepare them for syncing.

        This method:
        - Recursively loads supported file types
        - Merges multi-page documents into single documents per source file
        - Computes and attaches file hashes

        Args:
            pdf (bool): Whether to load PDF files.
            txt (bool): Whether to load text files.
            docx (bool): Whether to load Word documents.

        Returns:
            list[Document]: Loaded and processed documents.
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

        return documents

    def _merge_document_pages(self, docs):
        """
        Merge multiple document pages into a single document per source file.

        Documents are grouped by their 'source' metadata field.
        Page contents are concatenated with double newlines.

        Args:
            docs (list[Document]): Page-level documents.

        Returns:
            list[Document]: One document per source file.
        """

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
        """
        Split documents into overlapping text chunks for embedding.

        Uses a RecursiveCharacterTextSplitter with fixed chunk size
        and overlap to preserve context.

        Args:
            documents (list[Document]): Documents to split.

        Returns:
            list[Document]: Chunked documents ready for embedding.
        """

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
        """
        Initialize and return an Ollama embedding model.

        Ensures the requested model is available locally,
        pulling it if necessary.

        Args:
            emb_model (str): Ollama embedding model name.

        Returns:
            OllamaEmbeddings: Embedding function for vector DB usage.
        """

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
        """
        Load an existing Chroma vector database or create a new one.

        Args:
            vector_db_dir (str | Path): Directory for persisted vector DB data.
            collection_name (str): Name of the Chroma collection.

        Returns:
            Chroma: Initialized vector database instance.
        """

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
        """
        Add document chunks to the vector database in batches.

        Chunks are embedded and inserted in fixed-size batches
        to improve performance and provide progress visibility.

        Args:
            chunks (list[Document]): Chunked documents to add.
        """

        if not chunks:
            logger.debug("- No document chunks to add to DB")
            return

        BATCH_SIZE = 50

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            logger.info(
                "Embedding chunks %d–%d of %d",
                i + 1,
                min(i + BATCH_SIZE, len(chunks)),
                len(chunks),
            )
            self.vector_db.add_documents(batch)
        logger.debug("Added %d new document chunks to DB", len(chunks))

    # Local sync changes
    def _compute_local_sync_changes(self, documents, show_summary):
        """
        Compare local documents with the vector database and detect differences.

        Detects:
        - Added files
        - Removed files
        - Updated files (same path, different hash)
        - Renamed files (same hash, different path)

        Args:
            documents (list[Document]): Current local docs to compare against the DB.
            show_summary (bool): Whether to log a human-readable summary.

        Returns:
            dict: A dictionary describing detected changes and sync status.
        """

        logger.debug("Computing local sync changes")

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
            for d in documents
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

        # Show summary
        if show_summary:
            self._display_local_sync_changes_summary(local_sync_changes)

        return local_sync_changes

    def _display_local_sync_changes_summary(self, local_sync_changes):
        """
        Log a human-readable summary of detected local sync changes.

        Intended for informational output during startup or manual sync checks.

        Args:
            local_sync_changes (dict): Output of `_compute_local_sync_changes`.
        """

        logger.info("## SYNC SUMMARY ##")
        has_changes = local_sync_changes.get("has_changes", False)
        if not has_changes:
            logger.info("Vector DB is in sync with local files")
        else:
            added = local_sync_changes.get("added", [])
            removed = local_sync_changes.get("removed", [])
            renamed = local_sync_changes.get("renamed", [])
            updated = local_sync_changes.get("updated", [])

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

    def _apply_local_sync_changes(self, documents, local_sync_changes):
        """
        Apply detected local sync changes to the vector database.

        This method:
        - Removes deleted or renamed documents
        - Re-adds updated or renamed documents
        - Adds newly discovered documents
        - Updates the hash cache accordingly

        Side effects:
        - Mutates the vector database
        - Updates and persists the hash cache

        Args:
            documents (list[Document]): Current local documents.
            local_sync_changes (dict): Precomputed sync changes to apply.
        """

        logger.debug("Applying local sync changes")
        has_changes = local_sync_changes.get("has_changes", False)
        if not has_changes:
            logger.debug("DB is already in sync with local files")
            return

        added = local_sync_changes.get("added", [])
        removed = local_sync_changes.get("removed", [])
        renamed = local_sync_changes.get("renamed", [])
        updated = local_sync_changes.get("updated", [])

        local_doc_map = {
            d.metadata.get("file_hash", "unknown_hash"): d for d in documents
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

        logger.debug("Local sync changes have been applied")

    # ----- Public methods -----
    def is_in_sync(self, show_summary=False):
        """
        Check whether local files are in sync with the vector database.

        Args:
            show_summary (bool): Whether to log a sync summary.

        Returns:
            bool: True if no differences are detected, False otherwise.
        """

        logger.info("Checking if local files are in sync with database..")
        documents = self._load_documents()
        local_sync_changes = self._compute_local_sync_changes(
            documents=documents, show_summary=show_summary
        )

        in_sync = not local_sync_changes["has_changes"]
        if in_sync:
            logger.info("- Database is in sync with local files")
        else:
            logger.info("- Changes to local files have been detected")
        return in_sync

    def sync(self, show_summary=False):
        """
        Synchronize local files with the vector database.

        If changes are detected, applies them incrementally.
        Otherwise, logs that the database is already up to date.

        Args:
            show_summary (bool): Whether to log a sync summary.
        """

        logger.info("Syncing local files with database..")
        documents = self._load_documents()
        local_sync_changes = self._compute_local_sync_changes(
            documents=documents, show_summary=show_summary
        )

        if local_sync_changes["has_changes"]:
            self._apply_local_sync_changes(
                documents=documents, local_sync_changes=local_sync_changes
            )
            logger.info("Local changes have been synced with database")
        else:
            logger.info("Local files are already in-sync with database")
