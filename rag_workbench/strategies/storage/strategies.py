from typing import List, Optional, Any
from rag_workbench.core.interfaces import VectorStore, Document

class ChromaDBVectorStore(VectorStore):
    def __init__(self, collection_name: str = "rag_workbench", persist_directory: str = "./chroma_db"):
        try:
            import chromadb
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(path=persist_directory)
        except ImportError:
            raise ImportError("ChromaDB library is not installed. Please install it with `pip install chromadb`.")
        
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        ids = [doc.id for doc in documents]
        # Chroma expects metadatas to be flat dicts of str, int, float, bool
        # We might need to sanitize metadata here if it's complex
        metadatas = [doc.metadata for doc in documents]
        documents_text = [doc.content for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )

    def search(self, query_embedding: List[float], k: int = 4) -> List[Document]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Parse results back to Document objects
        documents = []
        if results['ids']:
            # Chroma returns lists of lists for batch queries
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            documents_text = results['documents'][0]
            
            for i in range(len(ids)):
                doc = Document(
                    id=ids[i],
                    content=documents_text[i],
                    metadata=metadatas[i] if metadatas else {}
                )
                documents.append(doc)
                
        return documents

class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing/prototyping without dependencies."""
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], k: int = 4) -> List[Document]:
        import math
        
        def cosine_similarity(v1, v2):
            dot_product = sum(a*b for a, b in zip(v1, v2))
            magnitude1 = math.sqrt(sum(a*a for a in v1))
            magnitude2 = math.sqrt(sum(b*b for b in v2))
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            return dot_product / (magnitude1 * magnitude2)

        scores = []
        for i, emb in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, emb)
            scores.append((score, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k_indices = [idx for _, idx in scores[:k]]
        
        return [self.documents[i] for i in top_k_indices]
