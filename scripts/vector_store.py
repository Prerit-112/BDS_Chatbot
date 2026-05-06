"""ChromaDB-based RAG store: persistent vector storage with reranking support."""
from __future__ import annotations

import chromadb
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class ChromaRagIndex:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "bds_abacus_docs",
        model_name: str = "all-mpnet-base-v2",
        rerank_model_name: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        # Using cosine similarity for the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(rerank_model_name) if rerank_model_name else None

    def clear(self) -> None:
        """Delete all data in the collection."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name="bds_abacus_docs",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass

    def count(self) -> int:
        """Return number of chunks in the collection."""
        return self.collection.count()

    def upsert_batch(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add or update documents in the collection with self-healing for stale collections."""
        if not ids:
            return
            
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False, 
            batch_size=32
        ).tolist()
        
        try:
            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception:
            # If collection reference is stale (e.g. after a deletion), re-fetch and try once more
            name = getattr(self.collection, "name", "bds_abacus_docs")
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

    def query(self, text: str, k: int = 10) -> list[dict[str, Any]]:
        """Perform vector search and optional reranking."""
        query_embedding = self.model.encode(
            [text], 
            convert_to_numpy=True, 
            show_progress_bar=False
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Chroma returns a list of results for each query embedding. 
        # Since we only have one query, we look at index 0.
        out: list[dict[str, Any]] = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                # Distance is cosine distance (1 - similarity)
                dist = results["distances"][0][i]
                score = 1.0 - dist
                
                out.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": float(score)
                })

        # Optional Reranking
        if self.reranker and out:
            pairs = [[text, res["document"]] for res in out]
            rerank_scores = self.reranker.predict(pairs)
            for res, score in zip(out, rerank_scores):
                res["rerank_score"] = float(score)
            out = sorted(out, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        return out

    def close(self) -> None:
        """No explicit close needed for Chroma PersistentClient, but we'll null it out."""
        self.client = None

# For backward compatibility with existing code during transition
SqliteRagIndex = ChromaRagIndex
