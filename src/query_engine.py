"""Query engine modules for GraphRAG and Flat RAG."""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
import chromadb
from chromadb.config import Settings

from src.models import QueryResult
from src.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# LLM Prompts
QUESTION_ENTITY_EXTRACTION_PROMPT = """Extract the key entities from this question that should be looked up in a knowledge graph.

Question: {question}

Return only the entity names as a JSON list:
["entity1", "entity2", ...]

Return valid JSON only, no additional text."""


GRAPHRAG_ANSWER_PROMPT = """Answer the following question based on the provided knowledge graph context.

Question: {question}

Knowledge Graph Context:
{textualized_subgraph}

Provide a clear, accurate answer based only on the information in the context.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."
"""


FLATRAG_ANSWER_PROMPT = """Answer the following question based on the provided context.

Question: {question}

Context:
{retrieved_chunks}

Provide a clear, accurate answer based only on the information in the context.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."
"""


class QueryEngine(ABC):
    """Abstract base class for query engines."""
    
    @abstractmethod
    def query(self, question: str) -> QueryResult:
        """
        Process a question and return an answer.
        
        Args:
            question: User question
            
        Returns:
            QueryResult with answer and metadata
        """
        pass


class GraphRAGQueryEngine(QueryEngine):
    """GraphRAG query engine with multi-hop reasoning."""
    
    def __init__(self, graph_builder: GraphBuilder, llm_client: OpenAI, model: str = "gpt-4o-mini"):
        """
        Initialize GraphRAG query engine.
        
        Args:
            graph_builder: GraphBuilder instance
            llm_client: OpenAI client
            model: Model name
        """
        self.graph_builder = graph_builder
        self.llm_client = llm_client
        self.model = model
        logger.info(f"GraphRAGQueryEngine initialized with model: {model}")
    
    def query(self, question: str) -> QueryResult:
        """
        Process question using Knowledge Graph.
        
        Process:
            1. Extract entities from question
            2. Find matching nodes in graph
            3. Extract 2-hop subgraph
            4. Textualize subgraph
            5. Generate answer with LLM
        """
        start_time = time.time()
        total_tokens = 0
        
        logger.info(f"GraphRAG Query: {question}")
        
        # Step 1: Extract entities from question
        entities = self._extract_entities_from_question(question)
        logger.info(f"Extracted entities: {entities}")
        
        if not entities:
            latency_ms = (time.time() - start_time) * 1000
            return QueryResult(
                answer="Không thể xác định thực thể từ câu hỏi.",
                latency_ms=latency_ms,
                token_usage=total_tokens,
                supporting_context=None
            )
        
        # Step 2 & 3: Find nodes and extract subgraph
        subgraph = None
        for entity in entities:
            subgraph = self.graph_builder.get_subgraph(entity, hops=2)
            if subgraph and subgraph['nodes']:
                logger.info(f"Found subgraph for entity '{entity}': {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
                break
        
        if not subgraph or not subgraph['nodes']:
            latency_ms = (time.time() - start_time) * 1000
            return QueryResult(
                answer="Không tìm thấy thông tin liên quan trong cơ sở tri thức.",
                latency_ms=latency_ms,
                token_usage=total_tokens,
                supporting_context=None
            )
        
        # Step 4: Textualize subgraph
        textualized = self._textualize_subgraph(subgraph)
        logger.debug(f"Textualized subgraph: {textualized[:200]}...")
        
        # Step 5: Generate answer
        prompt = GRAPHRAG_ANSWER_PROMPT.format(
            question=question,
            textualized_subgraph=textualized
        )
        
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on knowledge graph information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"GraphRAG Answer generated in {latency_ms:.2f}ms using {total_tokens} tokens")
        
        return QueryResult(
            answer=answer,
            latency_ms=latency_ms,
            token_usage=total_tokens,
            supporting_context=subgraph
        )
    
    def _extract_entities_from_question(self, question: str) -> List[str]:
        """Extract key entities from question using LLM."""
        prompt = QUESTION_ENTITY_EXTRACTION_PROMPT.format(question=question)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting entities from questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if not content.startswith('['):
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    content = content[start_idx:end_idx]
            
            entities = json.loads(content)
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities from question: {e}")
            return []
    
    def _textualize_subgraph(self, subgraph: dict) -> str:
        """Convert subgraph to natural language description."""
        lines = []
        
        # Describe entities
        lines.append("Entities:")
        nodes = subgraph.get('nodes', [])
        if nodes:
            for node in nodes:
                if isinstance(node, dict):
                    name = node.get('name', 'Unknown')
                    node_type = node.get('type', 'Unknown')
                    lines.append(f"- {name} (Type: {node_type})")
        else:
            lines.append("- No entities found")
        
        lines.append("\nRelationships:")
        # Describe relationships
        edges = subgraph.get('edges', [])
        if edges:
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source', 'Unknown')
                    edge_type = edge.get('type', 'Unknown')
                    target = edge.get('target', 'Unknown')
                    lines.append(f"- {source} {edge_type} {target}")
        else:
            lines.append("- No relationships found")
        
        return "\n".join(lines)


class FlatRAGQueryEngine(QueryEngine):
    """Flat RAG baseline using vector similarity search."""
    
    def __init__(self, 
                 llm_client: OpenAI, 
                 model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 top_k: int = 5):
        """
        Initialize Flat RAG query engine.
        
        Args:
            llm_client: OpenAI client
            model: Model name for generation
            embedding_model: Model name for embeddings
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
        """
        self.llm_client = llm_client
        self.model = model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.chroma_client.create_collection(
            name="tech_company_corpus",
            get_or_create=True
        )
        
        logger.info(f"FlatRAGQueryEngine initialized with model: {model}, embedding: {embedding_model}")
    
    def index_documents(self, documents: List[tuple]) -> None:
        """
        Chunk and embed documents into vector store.
        
        Args:
            documents: List of (filename, content) tuples
        """
        logger.info(f"Indexing {len(documents)} documents into ChromaDB")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        chunk_id = 0
        for filename, content in documents:
            chunks = self._chunk_text(content)
            logger.info(f"  {filename}: {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    'source_file': filename,
                    'chunk_index': i,
                    'chunk_size': len(chunk.split())
                })
                all_ids.append(f"chunk_{chunk_id:04d}")
                chunk_id += 1
        
        # Generate embeddings using OpenAI
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self._generate_embeddings(all_chunks)
        
        # Add to ChromaDB
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
            embeddings=embeddings
        )
        
        logger.info(f"Indexed {len(all_chunks)} chunks into ChromaDB")
    
    def query(self, question: str) -> QueryResult:
        """
        Process question using vector similarity search.
        
        Process:
            1. Embed question
            2. Retrieve top-k similar chunks
            3. Generate answer with LLM
        """
        start_time = time.time()
        
        logger.info(f"FlatRAG Query: {question}")
        
        # Step 1: Embed question
        question_embedding = self._generate_embeddings([question])[0]
        
        # Step 2: Retrieve top-k chunks
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=self.top_k
        )
        
        retrieved_chunks = results['documents'][0]
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 3: Generate answer
        context = "\n\n".join([f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
        
        prompt = FLATRAG_ANSWER_PROMPT.format(
            question=question,
            retrieved_chunks=context
        )
        
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"FlatRAG Answer generated in {latency_ms:.2f}ms using {total_tokens} tokens")
        
        return QueryResult(
            answer=answer,
            latency_ms=latency_ms,
            token_usage=total_tokens,
            supporting_context=retrieved_chunks
        )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
            i += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        response = self.llm_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        
        return [item.embedding for item in response.data]
