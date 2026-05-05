"""Script to build Knowledge Graph from markdown files."""

import logging
import sys
from openai import OpenAI

from src.config import Config, validate_config, ConfigurationError
from src.data_loader import DataLoader
from src.entity_extractor import EntityExtractor
from src.graph_builder import GraphBuilder, Neo4jConnectionError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graphrag.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to build Knowledge Graph."""
    try:
        # Load configuration
        logger.info("Loading configuration...")
        try:
            config = Config.from_yaml('config.yaml')
        except FileNotFoundError:
            logger.warning("config.yaml not found, loading from environment variables")
            config = Config.from_env()
        
        # Validate configuration
        validate_config(config)
        logger.info("Configuration validated successfully")
        
        # Initialize OpenAI client
        logger.info(f"Initializing OpenAI client with model: {config.openai_model}")
        openai_client = OpenAI(api_key=config.openai_api_key)
        
        # Initialize components
        data_loader = DataLoader(data_dir=config.data_dir)
        entity_extractor = EntityExtractor(
            llm_client=openai_client,
            model=config.openai_model
        )
        graph_builder = GraphBuilder(
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_user,
            neo4j_password=config.neo4j_password
        )
        
        # Step 1: Load all markdown files
        logger.info("=" * 60)
        logger.info("STEP 1: Loading markdown files")
        logger.info("=" * 60)
        documents = data_loader.load_all_markdown_files()
        
        if not documents:
            logger.error("No markdown files found in data directory")
            return
        
        logger.info(f"Loaded {len(documents)} markdown files")
        
        # Step 2: Extract triples from all documents
        logger.info("=" * 60)
        logger.info("STEP 2: Extracting entities and relationships")
        logger.info("=" * 60)
        
        # TEMPORARY: Only process first document for testing
        # logger.info("*** TESTING MODE: Processing only first document ***")
        # documents = documents[:1]
        
        all_triples = []
        for filename, content in documents:
            logger.info(f"Processing {filename}...")
            try:
                triples = entity_extractor.extract_triples(content, filename)
                all_triples.extend(triples)
                logger.info(f"  -> Extracted {len(triples)} triples from {filename}")
            except Exception as e:
                logger.error(f"  -> Failed to extract from {filename}: {e}")
                continue
        
        logger.info(f"Total triples extracted: {len(all_triples)}")
        
        if not all_triples:
            logger.error("No triples extracted from documents")
            return
        
        # Step 3: Build Knowledge Graph
        logger.info("=" * 60)
        logger.info("STEP 3: Building Knowledge Graph in Neo4j")
        logger.info("=" * 60)
        
        graph_builder.build_graph(all_triples)
        
        # Step 4: Display statistics
        logger.info("=" * 60)
        logger.info("STEP 4: Graph Statistics")
        logger.info("=" * 60)
        
        stats = graph_builder.get_graph_statistics()
        logger.info(f"Total Nodes (Entities): {stats['nodes']}")
        logger.info(f"Total Edges (Relationships): {stats['edges']}")
        
        # Verify success criteria
        if stats['nodes'] >= 100 and stats['edges'] >= 200:
            logger.info("✓ Success criteria met: >= 100 entities and >= 200 relationships")
        else:
            logger.warning(f"⚠ Success criteria not met: Need >= 100 entities (got {stats['nodes']}) and >= 200 relationships (got {stats['edges']})")
        
        logger.info("=" * 60)
        logger.info("Knowledge Graph construction completed successfully!")
        logger.info("=" * 60)
        logger.info(f"You can now visualize the graph at: http://localhost:7474")
        logger.info(f"Login with username: {config.neo4j_user}")
        
        # Close connection
        graph_builder.close()
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Neo4jConnectionError as e:
        logger.error(f"Neo4j connection error: {e}")
        logger.error("Make sure Neo4j is running and credentials are correct")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
