# processing/text_processor.py
import re
from typing import List, Dict, Any, NamedTuple
import tiktoken
from settings import settings


class DocumentSection(NamedTuple):
    content: str
    category: str
    importance_score: float
    metadata: Dict[str, Any]


class TextProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        return text.strip()

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap"""
        if not text.strip():
            return []

        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = self._get_token_count(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_data = {
                    'content': current_chunk.strip(),
                    'metadata': {
                        **(metadata or {}),
                        'token_count': current_length,
                        'chunk_type': 'text'
                    }
                }
                chunks.append(chunk_data)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_length = self._get_token_count(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length

        # Add the last chunk
        if current_chunk.strip():
            chunk_data = {
                'content': current_chunk.strip(),
                'metadata': {
                    **(metadata or {}),
                    'token_count': current_length,
                    'chunk_type': 'text'
                }
            }
            chunks.append(chunk_data)

        return chunks

    def categorize_and_partition_document(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentSection]:
        """Categorize and partition document into sections with importance scores"""
        if not text.strip():
            return []

        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Split into paragraphs for better categorization
        paragraphs = [p.strip() for p in processed_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [processed_text]

        sections = []
        
        for i, paragraph in enumerate(paragraphs):
            # Categorize paragraph
            category = self._categorize_content(paragraph)
            
            # Calculate importance score
            importance_score = self._calculate_importance_score(paragraph, category)
            
            # Create chunks from paragraph if it's too long
            if self._get_token_count(paragraph) > self.chunk_size:
                chunks = self.chunk_text(paragraph, metadata)
                for j, chunk in enumerate(chunks):
                    section_metadata = {
                        **(metadata or {}),
                        'paragraph_index': i,
                        'chunk_index': j,
                        'total_chunks': len(chunks)
                    }
                    sections.append(DocumentSection(
                        content=chunk['content'],
                        category=category,
                        importance_score=importance_score,
                        metadata=section_metadata
                    ))
            else:
                section_metadata = {
                    **(metadata or {}),
                    'paragraph_index': i,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
                sections.append(DocumentSection(
                    content=paragraph,
                    category=category,
                    importance_score=importance_score,
                    metadata=section_metadata
                ))

        return sections

    def _categorize_content(self, text: str) -> str:
        """Categorize content based on keywords and patterns"""
        text_lower = text.lower()
        
        # Define category keywords
        categories = {
            'introduction': ['introduction', 'abstract', 'summary', 'overview', 'begin'],
            'methodology': ['method', 'approach', 'technique', 'procedure', 'process', 'algorithm'],
            'results': ['result', 'finding', 'outcome', 'data', 'analysis', 'experiment'],
            'conclusion': ['conclusion', 'summary', 'end', 'final', 'closing', 'wrap'],
            'reference': ['reference', 'citation', 'bibliography', 'source'],
            'technical': ['implementation', 'code', 'system', 'architecture', 'design'],
            'discussion': ['discussion', 'analysis', 'interpretation', 'implication']
        }
        
        # Count keyword matches
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'general' if no matches
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'

    def _calculate_importance_score(self, text: str, category: str) -> float:
        """Calculate importance score based on content and category"""
        base_score = 0.5
        
        # Category-based scoring
        category_scores = {
            'introduction': 0.8,
            'conclusion': 0.8,
            'results': 0.9,
            'methodology': 0.7,
            'discussion': 0.6,
            'technical': 0.5,
            'reference': 0.3,
            'general': 0.5
        }
        
        score = category_scores.get(category, base_score)
        
        # Length-based adjustment (longer content might be more important)
        length_factor = min(len(text) / 1000, 1.0)  # Cap at 1.0
        score += length_factor * 0.1
        
        # Keyword-based importance boost
        important_keywords = [
            'important', 'significant', 'key', 'main', 'primary', 'crucial',
            'essential', 'critical', 'major', 'fundamental', 'core'
        ]
        
        text_lower = text.lower()
        keyword_boost = sum(0.05 for keyword in important_keywords if keyword in text_lower)
        score += min(keyword_boost, 0.2)  # Cap boost at 0.2
        
        # Ensure score is between 0 and 1
        return min(max(score, 0.0), 1.0)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with more sophisticated methods
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_token_count(self, text: str) -> int:
        """Get approximate token count"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate 4 characters per token
            return len(text) // 4

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        overlap_words = words[-self.chunk_overlap // 10:]  # Approximate word overlap
        return " ".join(overlap_words)