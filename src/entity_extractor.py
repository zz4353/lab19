"""Entity extraction module using LLM."""

import json
import logging
from typing import List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models import Triple, EntityType, RelationType

logger = logging.getLogger(__name__)


# LLM Prompt for entity extraction
ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting structured information from text.

Extract all entities and relationships from the following text about technology companies.

IMPORTANT RULES:
1. Use FULL ENTITY NAMES, not pronouns (he, she, it, they, this, that)
2. Replace pronouns with the actual entity name from context
3. Only extract information explicitly stated in the text
4. Use specific names, not generic terms

Entity Types: Company, Person, Product, Location, Date, Event
Relationship Types: FOUNDED_BY, FOUNDED_IN, DEVELOPED, RELEASED, ACQUIRED, PARTNERED_WITH, LOCATED_IN, CEO_OF

EXAMPLES:

Input: "OpenAI was founded by Sam Altman in 2015. He previously worked at Y Combinator."
CORRECT Output:
{{
    "triples": [
        {{"entity1": "OpenAI", "entity1_type": "Company", "relation": "FOUNDED_BY", "entity2": "Sam Altman", "entity2_type": "Person"}},
        {{"entity1": "OpenAI", "entity1_type": "Company", "relation": "FOUNDED_IN", "entity2": "2015", "entity2_type": "Date"}},
        {{"entity1": "Sam Altman", "entity1_type": "Person", "relation": "PARTNERED_WITH", "entity2": "Y Combinator", "entity2_type": "Company"}}
    ]
}}

WRONG Output (DO NOT DO THIS):
{{
    "triples": [
        {{"entity1": "He", "entity1_type": "Person", "relation": "PARTNERED_WITH", "entity2": "Y Combinator", "entity2_type": "Company"}}
    ]
}}

Input: "ChatGPT is a product developed by OpenAI. It was released in November 2022."
CORRECT Output:
{{
    "triples": [
        {{"entity1": "ChatGPT", "entity1_type": "Product", "relation": "DEVELOPED", "entity2": "OpenAI", "entity2_type": "Company"}},
        {{"entity1": "ChatGPT", "entity1_type": "Product", "relation": "RELEASED", "entity2": "November 2022", "entity2_type": "Date"}}
    ]
}}

WRONG Output (DO NOT DO THIS):
{{
    "triples": [
        {{"entity1": "It", "entity1_type": "Product", "relation": "RELEASED", "entity2": "November 2022", "entity2_type": "Date"}}
    ]
}}

Now extract from this text:

Text:
{text}

Return the results in JSON format:
{{
    "triples": [
        {{
            "entity1": "entity name",
            "entity1_type": "entity type",
            "relation": "relationship type",
            "entity2": "entity name",
            "entity2_type": "entity type"
        }}
    ]
}}

Return valid JSON only, no additional text."""


class ExtractionError(Exception):
    """Raised when entity extraction fails."""
    pass


class EntityExtractor:
    """Extract entities and relationships from text using LLM."""
    
    def __init__(self, llm_client: OpenAI, model: str = "gpt-4o-mini"):
        """
        Initialize with LLM client.
        
        Args:
            llm_client: OpenAI client instance
            model: Model name (default: gpt-4o-mini)
        """
        self.llm_client = llm_client
        self.model = model
        logger.info(f"EntityExtractor initialized with model: {model}")
    
    def extract_triples(self, text: str, source_file: str) -> List[Triple]:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Markdown content
            source_file: Source filename for metadata
            
        Returns:
            List of Triple objects (entity1, relation, entity2)
            
        Raises:
            ExtractionError: If extraction fails after max retries
        """
        logger.info(f"Extracting entities from {source_file} ({len(text)} characters)")
        
        # Split text into chunks if too long
        chunks = self._chunk_text(text, max_chars=3500)
        logger.info(f"Split into {len(chunks)} chunks for processing")
        
        all_triples = []
        
        # Extract from each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
            
            # Try extraction with retry logic
            for attempt in range(1, 4):  # Max 3 attempts
                try:
                    response = self._call_llm(chunk)
                    triples = self._parse_llm_response(response, source_file)
                    all_triples.extend(triples)
                    logger.info(f"  -> Extracted {len(triples)} triples from chunk {i+1}")
                    break
                    
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Attempt {attempt}/3 failed for chunk {i+1}: {e}")
                    if attempt == 3:
                        logger.error(f"Failed to extract from chunk {i+1} after 3 attempts")
                        # Continue with next chunk instead of failing completely
                    continue
        
        # Deduplicate triples
        unique_triples = self._deduplicate_triples(all_triples)
        logger.info(f"Extracted {len(all_triples)} triples, {len(unique_triples)} unique triples from {source_file}")
        
        return unique_triples
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception)
    )
    def _call_llm(self, text: str) -> str:
        """
        Call LLM with retry logic for rate limits.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            LLM response as string
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured information from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _parse_llm_response(self, response: str, source_file: str) -> List[Triple]:
        """
        Parse LLM response into structured triples.
        
        Args:
            response: LLM response string
            source_file: Source filename for metadata
            
        Returns:
            List of Triple objects
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
            KeyError: If required fields are missing
        """
        # Try to extract JSON from response (in case LLM adds extra text)
        response = response.strip()
        if not response.startswith('{'):
            # Find JSON block
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                response = response[start_idx:end_idx]
        
        data = json.loads(response)
        triples = []
        
        for triple_data in data.get('triples', []):
            # Normalize entity names
            entity1 = self._normalize_entity_name(triple_data['entity1'])
            entity2 = self._normalize_entity_name(triple_data['entity2'])
            
            triple = Triple(
                entity1=entity1,
                relation=triple_data['relation'],
                entity2=entity2,
                entity1_type=triple_data['entity1_type'],
                entity2_type=triple_data['entity2_type'],
                source_file=source_file
            )
            triples.append(triple)
        
        return triples
    
    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name for deduplication.
        
        Args:
            name: Entity name to normalize
            
        Returns:
            Normalized entity name
        """
        # Strip whitespace
        name = name.strip()
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Consistent casing (title case for proper nouns)
        # Keep original casing for acronyms (all caps)
        if not name.isupper():
            name = name.title()
        
        return name
    
    def _chunk_text(self, text: str, max_chars: int = 3500) -> List[str]:
        """
        Split text into chunks with overlap to preserve context.
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first (preserve semantic boundaries)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If single paragraph is too long, split it
            if para_length > max_chars:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentences = para.split('. ')
                temp_chunk = []
                temp_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_length = len(sentence) + 2  # +2 for '. '
                    
                    if temp_length + sentence_length > max_chars and temp_chunk:
                        chunks.append('. '.join(temp_chunk) + '.')
                        # Keep last sentence for overlap
                        temp_chunk = [temp_chunk[-1]] if temp_chunk else []
                        temp_length = len(temp_chunk[0]) + 2 if temp_chunk else 0
                    
                    temp_chunk.append(sentence)
                    temp_length += sentence_length
                
                if temp_chunk:
                    chunks.append('. '.join(temp_chunk) + '.')
                
            elif current_length + para_length > max_chars:
                # Current chunk is full, start new chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep last paragraph for overlap
                    current_chunk = [current_chunk[-1]] if current_chunk else []
                    current_length = len(current_chunk[0]) if current_chunk else 0
                
                current_chunk.append(para)
                current_length += para_length
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_length += para_length + 2  # +2 for '\n\n'
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def _deduplicate_triples(self, triples: List[Triple]) -> List[Triple]:
        """
        Remove duplicate triples and merge similar entities.
        
        Args:
            triples: List of Triple objects
            
        Returns:
            Deduplicated list of Triple objects
        """
        # Use set to track unique triples
        seen = set()
        unique_triples = []
        
        for triple in triples:
            # Create a normalized key for deduplication
            # Use lowercase for case-insensitive comparison
            key = (
                triple.entity1.lower(),
                triple.relation.upper(),
                triple.entity2.lower()
            )
            
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
        
        return unique_triples
