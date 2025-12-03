from rag_workbench.config.settings import (
    PipelineConfig,
    ChunkingStrategyType,
    EmbeddingModelType,
    VectorStoreType
)
from rag_workbench.core.interfaces import (
    ChunkingStrategy,
    EmbeddingModel,
    VectorStore
)
from rag_workbench.strategies.chunking import FixedSizeChunker, RecursiveCharacterChunker
from rag_workbench.strategies.embedding import MockEmbeddingModel, OpenAIEmbeddingModel
from rag_workbench.strategies.storage import ChromaDBVectorStore, InMemoryVectorStore
from rag_workbench.pipeline.manager import RAGPipeline

class PipelineBuilder:
    @staticmethod
    def build(config: PipelineConfig) -> RAGPipeline:
        # 1. Build Chunking Strategy
        chunker = PipelineBuilder._build_chunker(config.chunking)
        
        # 2. Build Embedding Model
        embedder = PipelineBuilder._build_embedder(config.embedding)
        
        # 3. Build Vector Store
        store = PipelineBuilder._build_vector_store(config.vector_store)
        
        # 4. Build Retrieval & Generation (TODO)
        
        return RAGPipeline(
            chunking_strategy=chunker,
            embedding_model=embedder,
            vector_store=store
        )

    @staticmethod
    def _build_chunker(config) -> ChunkingStrategy:
        if config.strategy == ChunkingStrategyType.FIXED:
            return FixedSizeChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        elif config.strategy == ChunkingStrategyType.RECURSIVE:
            return RecursiveCharacterChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=config.separators
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {config.strategy}")

    @staticmethod
    def _build_embedder(config) -> EmbeddingModel:
        if config.model_type == EmbeddingModelType.MOCK:
            return MockEmbeddingModel()
        elif config.model_type == EmbeddingModelType.OPENAI:
            if not config.api_key:
                raise ValueError("API key required for OpenAI embedding model")
            return OpenAIEmbeddingModel(
                api_key=config.api_key,
                model_name=config.model_name
            )
        else:
            raise ValueError(f"Unknown embedding model type: {config.model_type}")

    @staticmethod
    def _build_vector_store(config) -> VectorStore:
        if config.store_type == VectorStoreType.CHROMA:
            return ChromaDBVectorStore(
                collection_name=config.collection_name,
                persist_directory=config.persist_directory
            )
        elif config.store_type == VectorStoreType.MEMORY:
            return InMemoryVectorStore()
        else:
            raise ValueError(f"Unknown vector store type: {config.store_type}")
