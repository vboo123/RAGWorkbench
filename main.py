import argparse
import sys
from rag_workbench.config.settings import (
    PipelineConfig, 
    ChunkingConfig, 
    EmbeddingConfig, 
    VectorStoreConfig,
    ChunkingStrategyType,
    EmbeddingModelType,
    VectorStoreType
)
from rag_workbench.pipeline.builder import PipelineBuilder
from rag_workbench.core.interfaces import Document

def run_vector_demo():
    print("=== Running Vector RAG Demo ===")
    
    # 1. Define Configuration (Declarative)
    config = PipelineConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategyType.RECURSIVE,
            chunk_size=50,
            chunk_overlap=10
        ),
        embedding=EmbeddingConfig(
            model_type=EmbeddingModelType.MOCK, # Use Mock for demo without keys
            model_name="mock-model"
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStoreType.MEMORY # Use Memory for demo
        )
    )
    
    # 2. Build Pipeline
    print("Building pipeline...")
    pipeline = PipelineBuilder.build(config)
    
    # 3. Ingest Data
    sample_text = """
    RAG Workbench is a modular system for building Retrieval Augmented Generation pipelines.
    It allows you to swap out components like chunkers, embedders, and vector stores.
    This is a demo of the vector RAG capability.
    Graph RAG is another type of RAG that uses knowledge graphs.
    SQL RAG allows querying relational databases.
    """
    
    docs = [Document(content=sample_text, metadata={"source": "demo_text"})]
    pipeline.ingest(docs)
    
    # 4. Query
    query = "What is RAG Workbench?"
    print(f"\nQuerying: '{query}'")
    results = pipeline.query(query, k=2)
    
    print("\nResults:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.content} (Score: N/A for mock)")

def run_graph_placeholder():
    print("=== Graph RAG Placeholder ===")
    print("This demonstrates where a Graph RAG pipeline would be implemented.")
    print("You would implement a `GraphStore` and `GraphRetrievalStrategy` in `rag_workbench/strategies`.")
    print("Then update `PipelineBuilder` to support building a graph-based pipeline.")

def main():
    parser = argparse.ArgumentParser(description="RAG Workbench CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Vector Demo
    subparsers.add_parser("vector-demo", help="Run a demo of the Vector RAG pipeline")
    
    # Graph Demo
    subparsers.add_parser("graph-demo", help="Run a placeholder for Graph RAG")
    
    args = parser.parse_args()
    
    if args.command == "vector-demo":
        run_vector_demo()
    elif args.command == "graph-demo":
        run_graph_placeholder()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
