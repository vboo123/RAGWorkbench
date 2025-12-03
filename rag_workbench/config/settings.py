from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ChunkingStrategyType(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    # Add more as implemented

class EmbeddingModelType(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    MOCK = "mock"

class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"
    MEMORY = "memory"

class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategyType = ChunkingStrategyType.FIXED
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: Optional[list[str]] = None

class EmbeddingConfig(BaseModel):
    model_type: EmbeddingModelType = EmbeddingModelType.MOCK
    model_name: str = "text-embedding-3-small" # Default for OpenAI
    api_key: Optional[str] = None

class VectorStoreConfig(BaseModel):
    store_type: VectorStoreType = VectorStoreType.CHROMA
    collection_name: str = "rag_workbench"
    persist_directory: str = "./chroma_db"

class RetrievalConfig(BaseModel):
    k: int = 4

class GenerationConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1000

class PipelineConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
