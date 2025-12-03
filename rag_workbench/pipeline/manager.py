from typing import List, Optional
from rag_workbench.core.interfaces import (
    IngestionStrategy,
    ChunkingStrategy,
    EmbeddingModel,
    VectorStore,
    RetrievalStrategy,
    GenerationModel,
    Document
)

class RAGPipeline:
    def __init__(
        self,
        chunking_strategy: ChunkingStrategy,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        retrieval_strategy: Optional[RetrievalStrategy] = None,
        generation_model: Optional[GenerationModel] = None,
    ):
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy
        self.generation_model = generation_model

    def ingest(self, documents: List[Document]):
        """Full ingestion flow: Chunk -> Embed -> Store"""
        print(f"Ingesting {len(documents)} documents...")
        
        # 1. Chunk
        chunks = self.chunking_strategy.chunk(documents)
        print(f"Created {len(chunks)} chunks.")
        
        # 2. Embed
        # Extract text content for embedding
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        print(f"Generated {len(embeddings)} embeddings.")
        
        # 3. Store
        self.vector_store.add_documents(chunks, embeddings)
        print("Stored documents in vector store.")

    def query(self, query_text: str, k: int = 4) -> List[Document]:
        """Retrieval flow: Embed Query -> Search Store"""
        # 1. Embed Query
        query_embedding = self.embedding_model.embed_query(query_text)
        
        # 2. Search
        # If a specific retrieval strategy is defined (e.g. for re-ranking), use it
        # Otherwise, default to vector store search
        if self.retrieval_strategy:
            # Note: This is a simplification. A real retrieval strategy might need access to the vector store
            # or might be a wrapper around it. For now, let's assume direct vector store search 
            # is the default "strategy" if none is provided.
            return self.retrieval_strategy.retrieve(query_text, k=k)
        else:
            return self.vector_store.search(query_embedding, k=k)

    def generate(self, query_text: str) -> str:
        """Full RAG flow: Retrieve -> Generate"""
        if not self.generation_model:
            raise ValueError("No generation model configured for this pipeline.")
            
        context_docs = self.query(query_text)
        return self.generation_model.generate(query_text, context_docs)
