"""
Knowledge graph builder for extracting entities and relationships from policy documents.
Specifically designed for insurance and healthcare policy documents.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone
import asyncio
import re

from graphiti_core import Graphiti
from dotenv import load_dotenv

from chunker import DocumentChunk

# Import graph utilities
try:
    from graph_utils import GraphitiClient
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph_utils import GraphitiClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class PolicyGraphBuilder:
    """Builds knowledge graph from policy document chunks."""
    
    def __init__(self):
        """Initialize policy graph builder."""
        self.graph_client = GraphitiClient()
        self._initialized = False
    
    async def initialize(self):
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True
    
    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False
    
    async def add_policy_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        policy_title: str,
        policy_uin: str,
        policy_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 3
    ) -> Dict[str, Any]:
        """
        Add policy document chunks to the knowledge graph.
        
        Args:
            chunks: List of document chunks
            policy_title: Title of the policy document
            policy_uin: Unique Identification Number of the policy
            policy_metadata: Additional policy metadata (insurer, effective date, etc.)
            batch_size: Number of chunks to process in each batch
        
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"episodes_created": 0, "errors": []}
        
        logger.info(f"Adding {len(chunks)} chunks to knowledge graph for policy: {policy_title} (UIN: {policy_uin})")
        logger.info("⚠️ Large policy sections will be truncated to avoid Graphiti token limits.")
        
        # Check for oversized chunks and warn
        oversized_chunks = [i for i, chunk in enumerate(chunks) if len(chunk.content) > 6000]
        if oversized_chunks:
            logger.warning(f"Found {len(oversized_chunks)} policy sections over 6000 chars that will be truncated: {oversized_chunks}")
        
        episodes_created = 0
        errors = []
        
        # Process chunks one by one to avoid overwhelming Graphiti
        for i, chunk in enumerate(chunks):
            try:
                # Create episode ID for policy section
                episode_id = f"{policy_uin}_{chunk.index}_{datetime.now().timestamp()}"
                
                # Prepare episode content with policy context
                episode_content = self._prepare_policy_episode_content(
                    chunk,
                    policy_title,
                    policy_uin,
                    policy_metadata
                )
                
                # Create source description for policy section
                source_description = f"Policy: {policy_title} (UIN: {policy_uin}, Section: {chunk.index})"
                
                # Add episode to graph
                await self.graph_client.add_episode(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_description,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "policy_title": policy_title,
                        "policy_uin": policy_uin,
                        "section_index": chunk.index,
                        "original_length": len(chunk.content),
                        "processed_length": len(episode_content),
                        "document_type": "insurance_policy"
                    }
                )
                
                episodes_created += 1
                logger.info(f"✓ Added policy section {episode_id} to knowledge graph ({episodes_created}/{len(chunks)})")
                
                # Small delay between each episode to reduce API pressure
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                error_msg = f"Failed to add policy section {chunk.index} to graph: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                # Continue processing other chunks even if one fails
                continue
        
        result = {
            "episodes_created": episodes_created,
            "total_sections": len(chunks),
            "errors": errors
        }
        
        logger.info(f"Policy graph building complete: {episodes_created} episodes created, {len(errors)} errors")
        return result
    
    def _prepare_policy_episode_content(
        self,
        chunk: DocumentChunk,
        policy_title: str,
        policy_uin: str,
        policy_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare policy episode content with minimal context to avoid token limits.
        
        Args:
            chunk: Document chunk
            policy_title: Title of the policy document
            policy_uin: Policy UIN
            policy_metadata: Additional policy metadata
        
        Returns:
            Formatted episode content (optimized for Graphiti)
        """
        # Limit chunk content to avoid Graphiti's 8192 token limit
        max_content_length = 6000
        
        content = chunk.content
        if len(content) > max_content_length:
            # Truncate content but try to end at a sentence boundary
            truncated = content[:max_content_length]
            last_sentence_end = max(
                truncated.rfind('. '),
                truncated.rfind('! '),
                truncated.rfind('? ')
            )
            
            if last_sentence_end > max_content_length * 0.7:
                content = truncated[:last_sentence_end + 1] + " [TRUNCATED]"
            else:
                content = truncated + "... [TRUNCATED]"
            
            logger.warning(f"Truncated policy section {chunk.index} from {len(chunk.content)} to {len(content)} chars for Graphiti")
        
        # Add policy context
        if policy_title and len(content) < max_content_length - 150:
            episode_content = f"[Policy: {policy_title[:50]} | UIN: {policy_uin}]\n\n{content}"
        else:
            episode_content = content
        
        return episode_content
    
    async def extract_policy_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        extract_coverage_terms: bool = True,
        extract_medical_conditions: bool = True,
        extract_financial_terms: bool = True,
        extract_regulatory_terms: bool = True
    ) -> List[DocumentChunk]:
        """
        Extract policy-specific entities from chunks and add to metadata.
        
        Args:
            chunks: List of document chunks
            extract_coverage_terms: Whether to extract coverage-related terms
            extract_medical_conditions: Whether to extract medical conditions
            extract_financial_terms: Whether to extract financial terms
            extract_regulatory_terms: Whether to extract regulatory terms
        
        Returns:
            Chunks with policy entity metadata added
        """
        logger.info(f"Extracting policy entities from {len(chunks)} chunks")
        
        enriched_chunks = []
        
        for chunk in chunks:
            entities = {
                "coverage_terms": [],
                "medical_conditions": [],
                "financial_terms": [],
                "regulatory_terms": [],
                "exclusions": [],
                "benefits": [],
                "procedures": []
            }
            
            content = chunk.content
            
            # Extract coverage terms
            if extract_coverage_terms:
                entities["coverage_terms"] = self._extract_coverage_terms(content)
                entities["exclusions"] = self._extract_exclusions(content)
                entities["benefits"] = self._extract_benefits(content)
            
            # Extract medical conditions and procedures
            if extract_medical_conditions:
                entities["medical_conditions"] = self._extract_medical_conditions(content)
                entities["procedures"] = self._extract_medical_procedures(content)
            
            # Extract financial terms
            if extract_financial_terms:
                entities["financial_terms"] = self._extract_financial_terms(content)
            
            # Extract regulatory terms
            if extract_regulatory_terms:
                entities["regulatory_terms"] = self._extract_regulatory_terms(content)
            
            # Create enriched chunk
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "policy_entities": entities,
                    "entity_extraction_date": datetime.now().isoformat()
                },
                token_count=chunk.token_count
            )
            
            # Preserve embedding if it exists
            if hasattr(chunk, 'embedding'):
                enriched_chunk.embedding = chunk.embedding
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info("Policy entity extraction complete")
        return enriched_chunks
    
    def _extract_coverage_terms(self, text: str) -> List[str]:
        """Extract coverage-related terms from policy text."""
        coverage_terms = {
            "sum insured", "deductible", "co-payment", "copay", "coinsurance",
            "premium", "coverage", "benefit", "limit", "sub-limit",
            "waiting period", "pre-existing condition", "renewal",
            "claim", "reimbursement", "cashless", "network hospital",
            "domiciliary treatment", "hospitalization", "daycare",
            "pre-hospitalization", "post-hospitalization", "ambulance",
            "room rent", "intensive care", "ICU", "surgery", "treatment"
        }
        
        found_terms = set()
        text_lower = text.lower()
        
        for term in coverage_terms:
            if term.lower() in text_lower:
                found_terms.add(term)
        
        return list(found_terms)
    
    def _extract_exclusions(self, text: str) -> List[str]:
        """Extract exclusion-related terms."""
        exclusion_indicators = [
            "excluded", "not covered", "exclusion", "shall not cover",
            "does not include", "not eligible", "not payable"
        ]
        
        found_exclusions = []
        text_lower = text.lower()
        
        for indicator in exclusion_indicators:
            if indicator in text_lower:
                found_exclusions.append(indicator)
        
        return found_exclusions
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract benefit-related terms."""
        benefit_terms = {
            "maternity", "dental", "optical", "wellness", "preventive care",
            "health check-up", "vaccination", "physiotherapy", "mental health",
            "organ transplant", "cancer treatment", "cardiac care",
            "emergency care", "telemedicine", "home nursing"
        }
        
        found_benefits = set()
        text_lower = text.lower()
        
        for benefit in benefit_terms:
            if benefit.lower() in text_lower:
                found_benefits.add(benefit)
        
        return list(found_benefits)
    
    def _extract_medical_conditions(self, text: str) -> List[str]:
        """Extract medical condition terms."""
        medical_conditions = {
            "diabetes", "hypertension", "cancer", "heart disease", "stroke",
            "kidney disease", "liver disease", "asthma", "COPD",
            "arthritis", "osteoporosis", "thyroid", "epilepsy",
            "mental illness", "depression", "anxiety", "HIV", "AIDS"
        }
        
        found_conditions = set()
        text_lower = text.lower()
        
        for condition in medical_conditions:
            if condition.lower() in text_lower:
                found_conditions.add(condition)
        
        return list(found_conditions)
    
    def _extract_medical_procedures(self, text: str) -> List[str]:
        """Extract medical procedure terms."""
        procedures = {
            "surgery", "operation", "transplant", "dialysis", "chemotherapy",
            "radiotherapy", "angioplasty", "bypass", "endoscopy",
            "biopsy", "CT scan", "MRI", "ultrasound", "X-ray",
            "blood test", "ECG", "EEG", "consultation"
        }
        
        found_procedures = set()
        text_lower = text.lower()
        
        for procedure in procedures:
            if procedure.lower() in text_lower:
                found_procedures.add(procedure)
        
        return list(found_procedures)
    
    def _extract_financial_terms(self, text: str) -> List[str]:
        """Extract financial and monetary terms."""
        financial_terms = {
            "rupees", "INR", "lakh", "crore", "premium", "deductible",
            "co-payment", "percentage", "annual limit", "per incident",
            "per policy year", "maximum payable", "sum insured",
            "claim amount", "reimbursement", "settlement"
        }
        
        found_terms = set()
        text_lower = text.lower()
        
        for term in financial_terms:
            if term.lower() in text_lower:
                found_terms.add(term)
        
        # Also look for currency amounts
        currency_pattern = r'(?:₹|Rs\.?|rupees?)\s*[\d,]+(?:\.\d{2})?'
        currency_matches = re.findall(currency_pattern, text, re.IGNORECASE)
        if currency_matches:
            found_terms.add("currency_amount")
        
        return list(found_terms)
    
    def _extract_regulatory_terms(self, text: str) -> List[str]:
        """Extract regulatory and compliance terms."""
        regulatory_terms = {
            "IRDAI", "Insurance Regulatory and Development Authority",
            "UIN", "Unique Identification Number", "policy wording",
            "terms and conditions", "grievance", "ombudsman",
            "complaint", "regulator", "compliance", "statutory",
            "mandatory", "as per regulations", "guidelines"
        }
        
        found_terms = set()
        text_lower = text.lower()
        
        for term in regulatory_terms:
            if term.lower() in text_lower:
                found_terms.add(term)
        
        return list(found_terms)
    
    async def clear_policy_graph(self):
        """Clear all policy data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        
        logger.warning("Clearing policy knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Policy knowledge graph cleared")


class SimplePolicyEntityExtractor:
    """Simple rule-based entity extractor for policy documents."""
    
    def __init__(self):
        """Initialize policy entity extractor."""
        self.coverage_patterns = [
            r'\b(?:sum insured|deductible|co-payment|premium|coverage)\b',
            r'\b(?:hospitalization|treatment|surgery|claim)\b'
        ]
        
        self.medical_patterns = [
            r'\b(?:diabetes|hypertension|cancer|heart disease|stroke)\b',
            r'\b(?:surgery|treatment|procedure|diagnosis)\b'
        ]
        
        self.financial_patterns = [
            r'₹\s*[\d,]+(?:\.\d{2})?',
            r'\b\d+\s*(?:lakh|crore|rupees?)\b',
            r'\b\d+%\b'
        ]
    
    def extract_policy_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract policy entities using patterns."""
        entities = {
            "coverage_terms": [],
            "medical_terms": [],
            "financial_terms": []
        }
        
        # Extract coverage terms
        for pattern in self.coverage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["coverage_terms"].extend(matches)
        
        # Extract medical terms
        for pattern in self.medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["medical_terms"].extend(matches)
        
        # Extract financial terms
        for pattern in self.financial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["financial_terms"].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities


# Factory function
def create_policy_graph_builder() -> PolicyGraphBuilder:
    """Create policy graph builder instance."""
    return PolicyGraphBuilder()


# Example usage for policy document
async def main():
    """Example usage of the policy graph builder with example.md."""
    from chunker import ChunkingConfig, create_chunker
    
    # Create chunker and policy graph builder
    config = ChunkingConfig(chunk_size=500, use_semantic_splitting=False)
    chunker = create_chunker(config)
    policy_graph_builder = create_policy_graph_builder()
    
    # Read the example.md file (your policy document)
    try:
        with open("example.md", "r", encoding="utf-8") as file:
            policy_content = file.read()
        
        print(f"Loaded policy document: {len(policy_content)} characters")
        
        # Chunk the policy document
        chunks = chunker.chunk_document(
            content=policy_content,
            title="Global Health Care Policy Wordings",
            source="example.md"
        )
        
        print(f"Created {len(chunks)} policy sections")
        
        # Extract policy-specific entities
        enriched_chunks = await policy_graph_builder.extract_policy_entities_from_chunks(chunks)
        
        print("\nExtracted Policy Entities:")
        for i, chunk in enumerate(enriched_chunks[:3]):  # Show first 3 chunks
            entities = chunk.metadata.get('policy_entities', {})
            print(f"\nSection {i}:")
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"  {entity_type}: {entity_list}")
        
        # Add to knowledge graph
        try:
            result = await policy_graph_builder.add_policy_document_to_graph(
                chunks=enriched_chunks,
                policy_title="Global Health Care Policy Wordings",
                policy_uin="BAJHLIP23020V012223",
                policy_metadata={
                    "document_type": "health_insurance_policy",
                    "insurer": "Bajaj Allianz",
                    "version": "012223"
                }
            )
            
            print(f"\nPolicy graph building result: {result}")
            
        except Exception as e:
            print(f"Policy graph building failed: {e}")
        
    except FileNotFoundError:
        print("Error: example.md file not found. Please ensure the policy document exists.")
    except Exception as e:
        print(f"Error reading policy document: {e}")
    
    finally:
        await policy_graph_builder.close()


if __name__ == "__main__":
    asyncio.run(main())