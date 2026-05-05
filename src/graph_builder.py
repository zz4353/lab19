"""Graph construction module for Neo4j Knowledge Graph."""

import logging
from typing import List, Dict, Any
from datetime import datetime
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.models import Triple

logger = logging.getLogger(__name__)


class Neo4jConnectionError(Exception):
    """Raised when cannot connect to Neo4j."""
    pass


class GraphBuilder:
    """Build and manage Knowledge Graph in Neo4j."""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize Neo4j connection.
        
        Args:
            neo4j_uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            neo4j_user: Username
            neo4j_password: Password
            
        Raises:
            Neo4jConnectionError: If cannot connect to Neo4j
        """
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            # Test connection
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {neo4j_uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise Neo4jConnectionError(f"Cannot connect to Neo4j at {neo4j_uri}: {e}")
    
    def build_graph(self, triples: List[Triple]) -> None:
        """
        Build Knowledge Graph from triples.
        
        Args:
            triples: List of Triple objects
            
        Process:
            1. Create nodes for entities (with deduplication)
            2. Create edges for relationships
            3. Add metadata to nodes and edges
            4. Create indexes
        """
        logger.info(f"Building graph from {len(triples)} triples")
        
        # Create indexes first for performance
        self.create_indexes()
        
        # Batch insert nodes and edges
        batch_size = 100
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            self._insert_batch(batch)
            logger.info(f"Processed {min(i + batch_size, len(triples))}/{len(triples)} triples")
        
        # Log statistics
        stats = self.get_graph_statistics()
        logger.info(f"Graph construction complete: {stats['nodes']} nodes, {stats['edges']} edges")
    
    def _insert_batch(self, triples: List[Triple]) -> None:
        """
        Insert a batch of triples into the graph.
        
        Args:
            triples: List of Triple objects
        """
        with self.driver.session() as session:
            for triple in triples:
                # Create nodes with metadata
                self.create_node(
                    entity=triple.entity1,
                    entity_type=triple.entity1_type,
                    metadata={
                        'source_file': triple.source_file,
                        'timestamp': triple.timestamp.isoformat()
                    }
                )
                
                self.create_node(
                    entity=triple.entity2,
                    entity_type=triple.entity2_type,
                    metadata={
                        'source_file': triple.source_file,
                        'timestamp': triple.timestamp.isoformat()
                    }
                )
                
                # Create edge
                self.create_edge(
                    entity1=triple.entity1,
                    relation=triple.relation,
                    entity2=triple.entity2,
                    metadata={
                        'source_file': triple.source_file,
                        'timestamp': triple.timestamp.isoformat()
                    }
                )
    
    def create_node(self, entity: str, entity_type: str, metadata: dict) -> None:
        """
        Create or merge a node in the graph.
        
        Args:
            entity: Entity name
            entity_type: Type of entity (Company, Person, etc.)
            metadata: Additional metadata (source_file, timestamp)
        """
        with self.driver.session() as session:
            # Use MERGE for deduplication (case-insensitive)
            query = """
            MERGE (e:Entity {name_lower: toLower($name)})
            ON CREATE SET 
                e.name = $name,
                e.type = $type,
                e.source_files = [$source_file],
                e.created_at = datetime($timestamp),
                e.updated_at = datetime($timestamp)
            ON MATCH SET
                e.source_files = CASE 
                    WHEN NOT $source_file IN e.source_files 
                    THEN e.source_files + $source_file 
                    ELSE e.source_files 
                END,
                e.updated_at = datetime($timestamp)
            RETURN e
            """
            
            session.run(
                query,
                name=entity,
                type=entity_type,
                source_file=metadata['source_file'],
                timestamp=metadata['timestamp']
            )
    
    def create_edge(self, entity1: str, relation: str, entity2: str, metadata: dict) -> None:
        """
        Create an edge between two nodes.
        
        Args:
            entity1: Source entity name
            relation: Relationship type
            entity2: Target entity name
            metadata: Additional metadata (source_file, timestamp)
        """
        with self.driver.session() as session:
            # Create relationship with metadata
            query = f"""
            MATCH (a:Entity {{name_lower: toLower($entity1)}})
            MATCH (b:Entity {{name_lower: toLower($entity2)}})
            MERGE (a)-[r:{relation}]->(b)
            ON CREATE SET
                r.source_file = $source_file,
                r.created_at = datetime($timestamp)
            RETURN r
            """
            
            session.run(
                query,
                entity1=entity1,
                entity2=entity2,
                source_file=metadata['source_file'],
                timestamp=metadata['timestamp']
            )
    
    def create_indexes(self) -> None:
        """Create indexes on node properties for query optimization."""
        with self.driver.session() as session:
            # Index on name_lower for case-insensitive lookups
            try:
                session.run("CREATE INDEX entity_name_lower IF NOT EXISTS FOR (e:Entity) ON (e.name_lower)")
                logger.info("Created index on entity name_lower")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
            
            # Index on type
            try:
                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                logger.info("Created index on entity type")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
    
    def get_subgraph(self, entity: str, hops: int = 2) -> Dict[str, Any]:
        """
        Extract subgraph around an entity.
        
        Args:
            entity: Entity name
            hops: Number of hops to traverse (default: 2)
            
        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        with self.driver.session() as session:
            # Query for n-hop subgraph
            query = f"""
            MATCH path = (start:Entity {{name_lower: toLower($entity)}})-[*1..{hops}]-(connected:Entity)
            WITH start, connected, relationships(path) as rels
            RETURN 
                collect(DISTINCT {{name: start.name, type: start.type}}) as start_nodes,
                collect(DISTINCT {{name: connected.name, type: connected.type}}) as connected_nodes,
                [r in rels | {{
                    source: startNode(r).name, 
                    target: endNode(r).name, 
                    type: type(r)
                }}] as edges
            """
            
            result = session.run(query, entity=entity)
            record = result.single()
            
            if not record:
                logger.warning(f"No subgraph found for entity: {entity}")
                return {'nodes': [], 'edges': []}
            
            # Combine start and connected nodes
            start_nodes = record['start_nodes'] if record['start_nodes'] else []
            connected_nodes = record['connected_nodes'] if record['connected_nodes'] else []
            nodes = start_nodes + connected_nodes
            
            # Deduplicate nodes by name
            unique_nodes = {}
            for node in nodes:
                if isinstance(node, dict) and 'name' in node:
                    unique_nodes[node['name']] = node
            
            # Flatten and deduplicate edges
            edges = []
            seen_edges = set()
            raw_edges = record['edges'] if record['edges'] else []
            
            for edge_list in raw_edges:
                if isinstance(edge_list, list):
                    for edge in edge_list:
                        if isinstance(edge, dict) and 'source' in edge and 'type' in edge and 'target' in edge:
                            edge_key = (edge['source'], edge['type'], edge['target'])
                            if edge_key not in seen_edges:
                                edges.append(edge)
                                seen_edges.add(edge_key)
                elif isinstance(edge_list, dict) and 'source' in edge_list:
                    edge_key = (edge_list['source'], edge_list['type'], edge_list['target'])
                    if edge_key not in seen_edges:
                        edges.append(edge_list)
                        seen_edges.add(edge_key)
            
            return {
                'nodes': list(unique_nodes.values()),
                'edges': edges
            }
    
    def get_graph_statistics(self) -> Dict[str, int]:
        """
        Get graph statistics.
        
        Returns:
            Dict with node count and edge count
        """
        with self.driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            node_count = node_result.single()['count']
            
            # Count edges
            edge_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            edge_count = edge_result.single()['count']
            
            return {
                'nodes': node_count,
                'edges': edge_count
            }
    
    def clear_graph(self) -> None:
        """Clear all nodes and edges from the graph. Use with caution!"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Graph cleared - all nodes and edges deleted")
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
