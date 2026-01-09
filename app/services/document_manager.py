from pathlib import Path
from collections import defaultdict
from typing import Any
from hashlib import sha1
from datetime import datetime, timezone
import json
import logging
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from app.config import DATA_DIR, CACHE_DIR, VECTOR_DB_DIR, EMBED_MODEL

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Manages local document ingestion, metadata caching, and synchronization with a
    vector database using file metadata (mtime and size) instead of hashing content.

    Responsibilities:
    - Load documents from the local data directory
    - Merge multi-page documents
    - Maintain a persistent metadata cache (mtime and size)
    - Detect differences between local files and the vector DB by metadata
    - Apply incremental sync changes (add, remove, update)
    - Chunk documents and embed them into a Chroma vector database

    This class is stateful and maintains:
    - self.file_index: persisted mapping of file paths to mtime and size
    - self.vector_db: a persistent Chroma vector database
    """

    def __init__(self):
        self.embedding = self._get_embedding(EMBED_MODEL)
        self.data_dir = DATA_DIR
        cache_dir = CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.file_index_path = cache_dir / "file_index.json"
        self.file_index = self._load_file_index_cache()
        self.vector_db = self._load_or_create_vector_db(VECTOR_DB_DIR)

    # ----- Private Methods -----

    # General helpers
    def _normalize_path(self, path: Path | str) -> str:
        """Normalize a file path to an absolute resolved string."""
        return str(Path(path).resolve())

    def _file_id(self, path: str) -> str:
        """
        Compute a unique file identifier by hashing the file path.

        Args:
            path: The normalized file path.

        Returns:
            A SHA-1 hex digest string representing the file ID.
        """
        return sha1(path.encode()).hexdigest()

    # Metadata cache for tracking local file changes
    def _load_file_index_cache(self) -> dict[str, Any]:
        """
        Load the persisted file metadata cache from a JSON file.

        The cache file stores a dictionary with the following structure:
        {
            "last_synced_at": str | None,  # ISO timestamp of the last sync, or None
            "files": {
                "<file_path>": {
                    "file_id": str,  # Unique file identifier (SHA-1 of file path)
                    "mtime": float,  # Last modification time (epoch timestamp)
                    "size": int,     # File size in bytes
                    # Additional metadata fields may be added in the future
                },
                ...
            }
        }

        Returns:
            A dictionary containing:
            - "last_synced_at": timestamp string or None
            - "files": mapping of file paths to their metadata dictionaries
        """
        logger.debug("Loading file_index cache from %s", self.file_index_path)
        if self.file_index_path.exists():
            try:
                with open(self.file_index_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                return {
                    "last_synced_at": cache.get("last_synced_at"),
                    "files": cache.get("files", {}),
                }
            except Exception:
                logger.exception("Failed to load file_index cache")
        else:
            logger.debug("No existing file_index cache file found")

        return {
            "last_synced_at": None,
            "files": {},
        }

    def _save_metadata_cache(self) -> None:
        """
        Persist the in-memory file metadata cache to disk as JSON.

        This method overwrites the existing cache file.
        Logs an error if saving fails.
        """
        try:
            with open(self.file_index_path, "w", encoding="utf-8") as f:
                json.dump(self.file_index, f, indent=2)
            logger.debug(
                "Metadata cache saved (%d files)",
                len(self.file_index["files"]),
            )
        except Exception:
            logger.exception("Failed to save metadata cache")

    def get_last_synced_at(self) -> str | None:
        """Return the ISO timestamp string of the last successful sync, or None."""
        return self.file_index.get("last_synced_at")

    # Loading and chunking documents
    def _scan_local_files(self) -> dict[str, dict[str, float]]:
        """
        Scan the data directory recursively for supported document files.

        Collects metadata (modification time and file size) for files with supported
        extensions (pdf, txt, docx).

        Returns:
            A dictionary mapping normalized file paths to their metadata dicts.
        """
        logger.debug("Scanning local files for metadata..")

        files = {}
        for path in self.data_dir.rglob("*"):
            if path.suffix.lower() not in {".pdf", ".txt", ".docx"}:
                continue

            try:
                stat = path.stat()
                files[self._normalize_path(path)] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }

            except Exception:
                logger.exception("Failed to stat file %s", path)

        logger.debug(".. Finished scanning %d local files", len(files))
        return files

    def _load_documents_filtered(self, paths: set[str]) -> list[Document]:
        """
        Load documents from a given set of file paths, based on file type.

        Supports PDF, TXT, and DOCX formats with respective loaders.

        Args:
            paths: Set of normalized file paths to load.

        Returns:
            A list of Document objects loaded and merged by source file.
        """
        logger.debug("Loading %d changed documents..", len(paths))

        docs = []
        for path in paths:
            suffix = Path(path).suffix.lower()

            if suffix == ".pdf":
                loader = PyPDFLoader(path)
            elif suffix == ".txt":
                loader = TextLoader(path, encoding="utf-8")
            elif suffix == ".docx":
                loader = Docx2txtLoader(path)
            else:
                continue

            loaded = loader.load()
            loaded = self._merge_document_pages(loaded)
            docs.extend(loaded)

        logger.debug(".. Finished loading %d documents..", len(docs))
        return docs

    def _merge_document_pages(self, docs: list[Document]) -> list[Document]:
        """
        Merge multi-page documents into single Document instances grouped by source
        file.

        Concatenates page contents with double newlines, preserving metadata.

        Args:
            docs: List of page-level Document objects.

        Returns:
            List of merged Document objects, one per source file.
        """
        logger.debug("Merging %d pages of document..", len(docs))

        missing_source = [d for d in docs if not d.metadata.get("source")]
        if missing_source:
            logger.warning(
                "Found %d documents without source metadata", len(missing_source)
            )

        merged = defaultdict(list)
        metadata = {}

        for d in docs:
            src = self._normalize_path(d.metadata.get("source"))
            merged[src].append(d.page_content)
            metadata.setdefault(src, d.metadata)

        logger.debug(".. Finished merging pages")
        return [
            Document(page_content="\n\n".join(pages), metadata=metadata[src])
            for src, pages in merged.items()
        ]

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks suitable for embedding.

        Uses RecursiveCharacterTextSplitter with defined chunk size and overlap.

        Args:
            documents: List of Document objects to chunk.

        Returns:
            A list of chunked Document objects, each with metadata including file_id,
            source, and chunk_index.
        """
        logger.debug("Chunking %d documents", len(documents))
        if not documents:
            logger.warning("No documents were found to split into chunks")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)

        chunks = []
        for doc in documents:
            split_chunks = splitter.split_text(doc.page_content)
            for i, chunk_text in enumerate(split_chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "file_id": self._file_id(doc.metadata["source"]),
                        "source": doc.metadata["source"],
                        "chunk_index": i,
                    },
                )
                chunks.append(chunk_doc)

        logger.debug(
            ".. Finished chunking documents into a total of %d chunks", len(chunks)
        )
        return chunks

    # Vector DB operations
    def _get_embedding(self, emb_model: str) -> OllamaEmbeddings:
        """
        Initialize and return an Ollama embedding model.

        Pulls the model if not found locally.

        Args:
            emb_model: Name of the Ollama embedding model.

        Returns:
            An OllamaEmbeddings instance for embedding documents.
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

    def _load_or_create_vector_db(
        self, vector_db_dir: Path | str, collection_name: str = "simple-rag"
    ) -> Chroma:
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
            logger.info(".. Vector database has been loaded")
        else:
            logger.info("Creating new vector db..")
            vector_db = Chroma(
                embedding_function=self.embedding,
                collection_name=collection_name,
                persist_directory=vector_db_dir,
            )
            logger.info(".. Vector database has been created")

        return vector_db

    def _add_chunks_to_db(self, chunks: list[Document]) -> None:
        """
        Add document chunks to the vector database in batches.

        Chunks are embedded and inserted in fixed-size batches
        to improve performance and provide progress visibility.

        Args:
            chunks (list[Document]): Chunked documents to add.
        """
        logger.debug("Adding %d chunks to database in batches..", len(chunks))

        if not chunks:
            logger.warning("No document chunks to add to DB")
            return

        BATCH_SIZE = 50

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            logger.info(
                "Adding chunks %d-%d of %d to database",
                i + 1,
                min(i + BATCH_SIZE, len(chunks)),
                len(chunks),
            )
            self.vector_db.add_documents(batch)
        logger.debug(".. Finished adding chunks to DB")

    # Local sync changes
    def _compute_local_sync_changes(self, show_summary: bool = False) -> dict[str, Any]:
        """
        Compare local file metadata with the cached index to detect added, updated, or
        removed files.

        Updated files will be removed and then re-added.

        Args:
            show_summary: If True, logs a human-readable summary of detected changes.

        Returns:
            A dictionary containing lists of 'added', 'updated', and 'removed' file
            paths, and a 'has_changes' boolean flag.
        """
        logger.debug(
            "Computing local sync changes - comparing file system "
            "with file_index record.."
        )

        local_files = self._scan_local_files()
        indexed_files = self.file_index["files"]

        added = []
        updated = []
        removed = []

        # New or modified files
        for path, meta in local_files.items():
            cached = indexed_files.get(path)
            if not cached:
                added.append(path)
            else:
                if cached["mtime"] != meta["mtime"] or cached["size"] != meta["size"]:
                    updated.append(path)

        # Deleted files
        for path in indexed_files:
            if path not in local_files:
                removed.append(path)

        # Normalize all paths
        added = [self._normalize_path(p) for p in added]
        updated = [self._normalize_path(p) for p in updated]
        removed = [self._normalize_path(p) for p in removed]

        changes = {
            "added": added,
            "updated": updated,
            "removed": removed,
            "has_changes": bool(added or updated or removed),
        }

        if show_summary:
            self._display_local_sync_changes_summary(changes)

        logger.debug(".. Finished computing local sync changes")
        return changes

    def _display_local_sync_changes_summary(
        self, local_sync_changes: dict[str, Any]
    ) -> None:
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
            updated = local_sync_changes.get("updated", [])

            logger.info(
                "The following changes were made to local files "
                "and are not yet reflected in the DB:"
            )

            if added:
                logger.info("Added: %d", len(added))
                for s in added:
                    logger.debug("- %s", s)

            if removed:
                logger.info("Removed: %d", len(removed))
                for s in removed:
                    logger.debug("- %s", s)

            if updated:
                logger.info("Updated: %d", len(updated))
                for s in updated:
                    logger.debug("- %s", s)

    def _apply_local_sync_changes(
        self, documents: list[Document], local_sync_changes: dict[str, Any]
    ) -> None:
        """
        Apply detected local sync changes to the vector database and update metadata
        cache.

        Handles deletion, updating, and addition of document chunks in the vector DB.

        Args:
            documents: List of loaded Document objects corresponding to added or
            updated files.
            local_sync_changes: Dictionary containing 'added', 'updated', and 'removed'
            file paths.

        Side Effects:
            - Deletes outdated documents from vector DB.
            - Inserts new/updated document chunks.
            - Updates the metadata cache file_index and saves to disk.
        """
        logger.debug("Applying local sync changes..")

        added = local_sync_changes["added"]
        updated = local_sync_changes["updated"]
        removed = local_sync_changes["removed"]

        local_doc_map = {
            self._normalize_path(d.metadata["source"]): d for d in documents
        }

        # 1. Handle deletions
        for path in removed:
            logger.debug("Deleting file from DB: %s", path)
            file_id = self._file_id(path)
            self.vector_db.delete(where={"file_id": file_id})
            self.file_index["files"].pop(path, None)

        # 2. Handle updates (delete first)
        for path in updated:
            logger.debug("Updating file in DB: %s", path)
            file_id = self._file_id(path)
            self.vector_db.delete(where={"file_id": file_id})

        # 3. Add new & updated docs
        docs_to_add = [local_doc_map[p] for p in added + updated]

        if docs_to_add:
            chunks = self._chunk_documents(docs_to_add)
            self._add_chunks_to_db(chunks)

            # 4. Update file_index ONLY after DB write success
            for path in added + updated:
                stat = Path(path).stat()
                self.file_index["files"][path] = {
                    "file_id": self._file_id(path),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }

        # 5. Update last_synced_at
        self.file_index["last_synced_at"] = (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        self._save_metadata_cache()
        logger.debug(".. Local sync changes applied successfully")

    # ----- Public methods -----

    def is_in_sync(self, show_summary: bool = False) -> bool:
        """
        Check whether the local files are in sync with the vector database.

        Compares current file metadata with cached metadata to detect changes rapidly.

        Args:
            show_summary: If True, logs a summary of detected differences.

        Returns:
            True if no changes are detected (in sync), False otherwise.
        """
        logger.debug("Starting is_in_sync procedure..")

        changes = self._compute_local_sync_changes(show_summary=show_summary)
        in_sync = not changes["has_changes"]

        logger.info("Result of is_in_sync: %s", in_sync)
        return in_sync

    def sync(self, show_summary: bool = False) -> None:
        """
        Perform a synchronization process between local files and the vector database.

        Detects added, updated, and removed files and applies corresponding changes.
        Updated files will be removed and then re-added

        Args:
            show_summary: If True, logs a summary of sync operations.

        Side Effects:
            Updates the vector database and metadata cache to reflect the current
            file state.
        """
        logger.info("Starting sync procedure..")

        # Compute changes
        changes = self._compute_local_sync_changes(show_summary=show_summary)

        # Exit early if already in-sync
        in_sync = not changes["has_changes"]
        if in_sync:
            logger.info("Database already in-sync")
            return

        # Load only affected files
        paths_to_load = set(changes["added"] + changes["updated"])
        documents = self._load_documents_filtered(paths_to_load)

        # Apply changes to db
        self._apply_local_sync_changes(
            documents=documents,
            local_sync_changes=changes,
        )

        logger.info("Database is now in-sync")
