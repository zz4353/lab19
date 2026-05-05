"""Configuration management for GraphRAG system."""

import os
import yaml
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when system configuration is invalid."""
    pass


@dataclass
class Config:
    """System configuration."""
    
    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 2000
    
    # Data
    data_dir: str = "data/"
    
    # Flat RAG
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    vector_store_type: str = "chromadb"
    embedding_model: str = "text-embedding-3-small"
    
    # GraphRAG
    subgraph_hops: int = 2
    
    # Evaluation
    benchmark_questions_path: str = "benchmark_questions.json"
    output_dir: str = "output/"
    
    # Logging
    log_level: str = "INFO"
    debug_mode: bool = False
    log_file: str = "graphrag.log"
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance
            
        Raises:
            ConfigurationError: If config file not found or invalid
        """
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse YAML config: {e}")
        
        # Load environment variables (they override YAML)
        load_dotenv()
        
        return cls(
            # Neo4j
            neo4j_uri=os.getenv('NEO4J_URI', config_data.get('neo4j', {}).get('uri', '')),
            neo4j_user=os.getenv('NEO4J_USER', config_data.get('neo4j', {}).get('user', '')),
            neo4j_password=os.getenv('NEO4J_PASSWORD', config_data.get('neo4j', {}).get('password', '')),
            
            # OpenAI
            openai_api_key=os.getenv('OPENAI_API_KEY', config_data.get('openai', {}).get('api_key', '')),
            openai_model=os.getenv('OPENAI_MODEL', config_data.get('openai', {}).get('model', 'gpt-4o-mini')),
            openai_temperature=float(config_data.get('openai', {}).get('temperature', 0.0)),
            openai_max_tokens=int(config_data.get('openai', {}).get('max_tokens', 2000)),
            
            # Data
            data_dir=os.getenv('DATA_DIR', config_data.get('data', {}).get('data_dir', 'data/')),
            
            # Flat RAG
            chunk_size=int(os.getenv('CHUNK_SIZE', config_data.get('flatrag', {}).get('chunk_size', 512))),
            chunk_overlap=int(config_data.get('flatrag', {}).get('chunk_overlap', 50)),
            top_k=int(os.getenv('TOP_K', config_data.get('flatrag', {}).get('top_k', 5))),
            vector_store_type=config_data.get('flatrag', {}).get('vector_store_type', 'chromadb'),
            embedding_model=config_data.get('flatrag', {}).get('embedding_model', 'text-embedding-3-small'),
            
            # GraphRAG
            subgraph_hops=int(config_data.get('graphrag', {}).get('subgraph_hops', 2)),
            
            # Evaluation
            benchmark_questions_path=config_data.get('evaluation', {}).get('benchmark_questions_path', 'benchmark_questions.json'),
            output_dir=config_data.get('evaluation', {}).get('output_dir', 'output/'),
            
            # Logging
            log_level=os.getenv('LOG_LEVEL', config_data.get('logging', {}).get('log_level', 'INFO')),
            debug_mode=config_data.get('logging', {}).get('debug_mode', False),
            log_file=config_data.get('logging', {}).get('log_file', 'graphrag.log'),
        )
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables only.
        
        Returns:
            Config instance
            
        Raises:
            ConfigurationError: If required environment variables are missing
        """
        load_dotenv()
        
        return cls(
            # Neo4j
            neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
            neo4j_password=os.getenv('NEO4J_PASSWORD', ''),
            
            # OpenAI
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            openai_model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            
            # Data
            data_dir=os.getenv('DATA_DIR', 'data/'),
            
            # Flat RAG
            chunk_size=int(os.getenv('CHUNK_SIZE', '512')),
            top_k=int(os.getenv('TOP_K', '5')),
            
            # Logging
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
        )


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Config instance to validate
        
    Raises:
        ConfigurationError: If any required parameter is missing or invalid
    """
    # Validate required parameters
    if not config.openai_api_key:
        raise ConfigurationError(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable or add to config.yaml."
        )
    
    if not config.neo4j_uri:
        raise ConfigurationError(
            "Neo4j URI is required. Set NEO4J_URI environment variable or add to config.yaml."
        )
    
    if not config.neo4j_user:
        raise ConfigurationError(
            "Neo4j user is required. Set NEO4J_USER environment variable or add to config.yaml."
        )
    
    if not config.neo4j_password:
        raise ConfigurationError(
            "Neo4j password is required. Set NEO4J_PASSWORD environment variable or add to config.yaml."
        )
    
    # Validate data directory exists
    if not os.path.exists(config.data_dir):
        raise ConfigurationError(
            f"Data directory does not exist: {config.data_dir}"
        )
    
    # Validate numeric parameters
    if config.chunk_size <= 0:
        raise ConfigurationError(f"chunk_size must be positive, got {config.chunk_size}")
    
    if config.chunk_overlap < 0:
        raise ConfigurationError(f"chunk_overlap must be non-negative, got {config.chunk_overlap}")
    
    if config.chunk_overlap >= config.chunk_size:
        raise ConfigurationError(
            f"chunk_overlap ({config.chunk_overlap}) must be less than chunk_size ({config.chunk_size})"
        )
    
    if config.top_k <= 0:
        raise ConfigurationError(f"top_k must be positive, got {config.top_k}")
    
    if config.subgraph_hops <= 0:
        raise ConfigurationError(f"subgraph_hops must be positive, got {config.subgraph_hops}")
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
