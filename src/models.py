"""Data models for GraphRAG system."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Any, Optional


class EntityType(Enum):
    """Types of entities that can be extracted."""
    COMPANY = "Company"
    PERSON = "Person"
    PRODUCT = "Product"
    LOCATION = "Location"
    DATE = "Date"
    EVENT = "Event"


class RelationType(Enum):
    """Types of relationships between entities."""
    FOUNDED_BY = "FOUNDED_BY"
    FOUNDED_IN = "FOUNDED_IN"
    DEVELOPED = "DEVELOPED"
    RELEASED = "RELEASED"
    ACQUIRED = "ACQUIRED"
    PARTNERED_WITH = "PARTNERED_WITH"
    LOCATED_IN = "LOCATED_IN"
    CEO_OF = "CEO_OF"


@dataclass
class Triple:
    """Represents an entity-relationship-entity triple."""
    entity1: str
    relation: str
    entity2: str
    entity1_type: str
    entity2_type: str
    source_file: str
    timestamp: datetime = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class QueryResult:
    """Result from a query engine."""
    answer: str
    latency_ms: float
    token_usage: int
    supporting_context: Any  # Subgraph for GraphRAG, chunks for FlatRAG


@dataclass
class BenchmarkQuestion:
    """A benchmark question with ground truth."""
    id: int
    question: str
    ground_truth: str
    question_type: str  # simple, multi-hop, comparison, aggregation


@dataclass
class QuestionResult:
    """Result for a single benchmark question."""
    question: BenchmarkQuestion
    graphrag_answer: str
    flatrag_answer: str
    graphrag_correct: bool
    flatrag_correct: bool
    graphrag_latency_ms: float
    flatrag_latency_ms: float
    graphrag_tokens: int
    flatrag_tokens: int


@dataclass
class BenchmarkReport:
    """Complete benchmark evaluation report."""
    results: List[QuestionResult]
    graphrag_accuracy: float
    flatrag_accuracy: float
    graphrag_avg_latency: float
    flatrag_avg_latency: float
    graphrag_total_tokens: int
    flatrag_total_tokens: int
    accuracy_improvement: float  # Should be >= 20%
