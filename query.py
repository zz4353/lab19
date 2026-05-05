"""Simple query interface for testing GraphRAG and Flat RAG."""

import sys
import logging
from openai import OpenAI

from src.config import Config, validate_config
from src.data_loader import DataLoader
from src.graph_builder import GraphBuilder
from src.query_engine import GraphRAGQueryEngine, FlatRAGQueryEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main query interface."""
    if len(sys.argv) < 2:
        print("Usage: python query.py \"Your question here\"")
        print("Example: python query.py \"Who founded OpenAI?\"")
        sys.exit(1)
    
    question = sys.argv[1]
    
    try:
        # Load configuration
        try:
            config = Config.from_yaml('config.yaml')
        except FileNotFoundError:
            config = Config.from_env()
        
        validate_config(config)
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=config.openai_api_key)
        
        # Initialize GraphRAG
        print("\n" + "=" * 60)
        print("Initializing GraphRAG Query Engine...")
        print("=" * 60)
        
        graph_builder = GraphBuilder(
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_user,
            neo4j_password=config.neo4j_password
        )
        
        graphrag_engine = GraphRAGQueryEngine(
            graph_builder=graph_builder,
            llm_client=openai_client,
            model=config.openai_model
        )
        
        # Initialize Flat RAG
        print("\n" + "=" * 60)
        print("Initializing Flat RAG Query Engine...")
        print("=" * 60)
        
        flatrag_engine = FlatRAGQueryEngine(
            llm_client=openai_client,
            model=config.openai_model,
            embedding_model=config.embedding_model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            top_k=config.top_k
        )
        
        # Index documents for Flat RAG
        print("Loading and indexing documents...")
        data_loader = DataLoader(data_dir=config.data_dir)
        documents = data_loader.load_all_markdown_files()
        flatrag_engine.index_documents(documents)
        
        # Execute queries
        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)
        
        # GraphRAG
        print("\n[GraphRAG]")
        print("-" * 60)
        graphrag_result = graphrag_engine.query(question)
        print(f"Answer: {graphrag_result.answer}")
        print(f"Latency: {graphrag_result.latency_ms:.2f}ms")
        print(f"Tokens: {graphrag_result.token_usage}")
        
        # Flat RAG
        print("\n[Flat RAG]")
        print("-" * 60)
        flatrag_result = flatrag_engine.query(question)
        print(f"Answer: {flatrag_result.answer}")
        print(f"Latency: {flatrag_result.latency_ms:.2f}ms")
        print(f"Tokens: {flatrag_result.token_usage}")
        
        print("\n" + "=" * 60)
        print("Query completed!")
        print("=" * 60)
        
        # Close connections
        graph_builder.close()
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
