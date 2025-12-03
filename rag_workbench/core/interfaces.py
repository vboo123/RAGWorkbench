from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None

class IngestionStrategy(ABC):
    @abstractmethod
    def load(self, source: Any) -> List[Document]:
        """Load data from a source and return a list of Documents."""
        pass

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        pass

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 4) -> List[Document]:
        """Search for similar documents using a query embedding."""
        pass

class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a query."""
        pass

class GenerationModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: List[Document]) -> str:
        """Generate a response based on the prompt and context."""
        pass
