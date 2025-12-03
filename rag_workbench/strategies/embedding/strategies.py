import random
from typing import List, Optional
from rag_workbench.core.interfaces import EmbeddingModel

class MockEmbeddingModel(EmbeddingModel):
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[random.random() for _ in range(self.dimension)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [random.random() for _ in range(self.dimension)]

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI library is not installed. Please install it with `pip install openai`.")
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenAI batch limit is usually 2048, might need batching logic for large lists
        # For now, simple implementation
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding
