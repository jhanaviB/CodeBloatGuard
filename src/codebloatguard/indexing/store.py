import hashlib

import chromadb

from codebloatguard.config import CHROMA_PATH
from codebloatguard.indexing.chunker import Chunk
from codebloatguard.indexing.embedder import embed, embed_one


def _collection_name(repo_root) -> str:
    """One collection per repo, keyed by path."""
    return "repo-" + hashlib.sha256(str(repo_root).encode()).hexdigest()[:16]


class RepoStore:
    def __init__(self, repo_root):
        self._db = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self._name = _collection_name(repo_root)
        self._col = self._open()

    def _open(self):
        return self._db.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}
        )

    def sync(self, chunks: list[Chunk]):
        """
        A repo-wide sync
        chunks -> Repo is the source
        For every path we sync the chunks in the repo and the store
        """
        live_paths = {c.path for c in chunks}
        for path in live_paths:
            self._sync_file(path, [c for c in chunks if c.path == path])
        self._drop_missing_files(live_paths)

    def _drop_missing_files(self, live_paths: set[str]):
        """
        Removed deleted files from the chunk store.
        """
        indexed = self._col.get(include=["metadatas"])
        orphans = [
            i for i, m in zip(indexed["ids"], indexed["metadatas"])
            if m["path"] not in live_paths
        ]
        if orphans:
            self._col.delete(ids=orphans)

    @staticmethod
    def _meta(c: Chunk) -> dict:
        return {"path": c.path, "name": c.name, "start_line": c.start_line, "end_line": c.end_line}

    def _sync_file(self, path: str, chunks: list[Chunk]):
        chunks = list({c.id: c for c in chunks}.values())
        existing = set(self._col.get(where={"path": path})["ids"])
        current = {c.id for c in chunks}

        stale = existing - current
        if stale:
            self._col.delete(ids=list(stale))

        #  Refresh line numbers for unchanged code
        stayed = [c for c in chunks if c.id in existing]
        if stayed:
            self._col.update(
                ids=[c.id for c in stayed],
                metadatas=[self._meta(c) for c in stayed],
            )

        fresh = [c for c in chunks if c.id not in existing]
        if not fresh:
            return

        vectors = embed([c.for_embedding() for c in fresh])
        self._col.upsert(
            ids=[c.id for c in fresh],
            embeddings=vectors,
            documents=[c.code for c in fresh],
            metadatas=[self._meta(c) for c in fresh],
        )

    def search(self, code: str, k: int = 5, exclude_paths: set[str] | None = None):
        return self.search_vec(embed_one(code), k=k, exclude_paths=exclude_paths)

    def search_vec(self, vector: list[float], k: int = 5, exclude_paths: set[str] | None = None):
        """
        Queries the collection to find top k similar embeddings, skipping any
        file that are currently under review (since the file is changing).
        """
        hits = self._col.query(
            query_embeddings=[vector],
            n_results=k,
            where={"path": {"$nin": sorted(exclude_paths)}} if exclude_paths else None,
        )
        return [
            {"id": i, "code": d, "meta": m, "distance": dist}
            for i, d, m, dist in zip(
                hits["ids"][0], hits["documents"][0], hits["metadatas"][0], hits["distances"][0]
            )
        ]

    def count(self) -> int:
        return self._col.count()

    def reset(self):
        self._db.delete_collection(self._name)
        self._col = self._open()
