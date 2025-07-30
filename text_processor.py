# processing/enhanced_text_processor.py
import re
from typing import List, Dict, Any, Tuple
import tiktoken
from settings import settings
from dataclasses import dataclass


@dataclass
class DocumentSection:
    category: str
    content: str
    importance_score: float
    metadata: Dict[str, Any]


class EnhancedTextProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE * 2  # Increased chunk size for large documents
        self.chunk_overlap = settings.CHUNK_OVERLAP
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace but preserve paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        # Keep more punctuation for better section detection
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"\/\n\r]+', '', text)
        return text.strip()

    def categorize_and_partition_document(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentSection]:
        """Intelligently categorize and partition document into relevant sections"""

        # Step 1: Identify document structure and sections
        sections = self._identify_document_sections(text)

        # Step 2: Categorize each section and create DocumentSection objects
        categorized_sections = []
        for section in sections:
            category = self._categorize_section(section['content'])
            importance = self._calculate_importance_score(section['content'], category)

            doc_section = DocumentSection(
                category=category,
                content=section['content'],
                importance_score=importance,
                metadata={
                    **(metadata or {}),
                    'section_type': section['type'],
                    'position': section['position'],
                    'token_count': self._get_token_count(section['content'])
                }
            )
            categorized_sections.append(doc_section)

        # Step 3: Create chunks within each category
        final_chunks = []
        for section in categorized_sections:
            chunks = self._create_category_chunks(section)
            final_chunks.extend(chunks)

        return final_chunks

    def _identify_document_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify different sections in the document"""
        sections = []

        # Split by common section indicators
        section_patterns = [
            r'(?i)(?:^|\n)(?:abstract|summary|introduction|background|methodology|methods|results|discussion|conclusion|references|appendix|acknowledgments)(?:\s*:|\s*\n)',
            r'(?i)(?:^|\n)(?:\d+\.?\s+[A-Z][^.\n]{5,50})',  # Numbered sections
            r'(?i)(?:^|\n)(?:[A-Z\s]{3,30}:)',  # All caps headings
            r'\n\n+',  # Paragraph breaks
        ]

        # Try to split by headings first
        for pattern in section_patterns[:3]:
            matches = list(re.finditer(pattern, text))
            if len(matches) >= 2:  # Found meaningful sections
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    section_text = text[start:end].strip()

                    if len(section_text) > 100:  # Minimum section length
                        sections.append({
                            'content': section_text,
                            'type': 'structured_section',
                            'position': i
                        })
                break

        # Fallback: Split by paragraphs if no clear structure
        if not sections:
            paragraphs = re.split(r'\n\n+', text)
            current_section = ""
            section_count = 0

            for para in paragraphs:
                if len(current_section) + len(para) > self.chunk_size:
                    if current_section:
                        sections.append({
                            'content': current_section.strip(),
                            'type': 'paragraph_group',
                            'position': section_count
                        })
                        section_count += 1
                    current_section = para
                else:
                    current_section += "\n\n" + para if current_section else para

            if current_section:
                sections.append({
                    'content': current_section.strip(),
                    'type': 'paragraph_group',
                    'position': section_count
                })

        return sections

    def _categorize_section(self, content: str) -> str:
        """Categorize section content using pattern matching and keywords"""
        content_lower = content.lower()

        # Category keywords
        categories = {
            'abstract': ['abstract', 'summary', 'overview'],
            'introduction': ['introduction', 'background', 'motivation', 'problem statement'],
            'methodology': ['method', 'approach', 'technique', 'algorithm', 'procedure'],
            'results': ['result', 'finding', 'outcome', 'analysis', 'evaluation'],
            'discussion': ['discussion', 'interpretation', 'implication', 'significance'],
            'conclusion': ['conclusion', 'summary', 'future work', 'recommendation'],
            'technical': ['formula', 'equation', 'algorithm', 'code', 'implementation'],
            'data': ['table', 'figure', 'chart', 'graph', 'data', 'statistics'],
            'references': ['reference', 'bibliography', 'citation', 'source'],
            'general': []  # Default category
        }

        # Score each category
        category_scores = {}
        for category, keywords in categories.items():
            score = 0
            for keyword in keywords:
                score += content_lower.count(keyword)
            category_scores[category] = score

        # Find best category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        return best_category[0] if best_category[1] > 0 else 'general'

    def _calculate_importance_score(self, content: str, category: str) -> float:
        """Calculate importance score for content prioritization"""
        base_scores = {
            'abstract': 0.9,
            'introduction': 0.8,
            'methodology': 0.7,
            'results': 0.8,
            'discussion': 0.7,
            'conclusion': 0.9,
            'technical': 0.6,
            'data': 0.5,
            'references': 0.3,
            'general': 0.5
        }

        base_score = base_scores.get(category, 0.5)

        # Adjust based on content characteristics
        content_lower = content.lower()

        # Boost for key terms
        key_terms = ['important', 'significant', 'key', 'main', 'primary', 'crucial', 'essential']
        key_term_boost = sum(content_lower.count(term) for term in key_terms) * 0.1

        # Boost for questions and statements
        question_boost = content.count('?') * 0.05

        # Length penalty for very short or very long sections
        length_factor = min(1.0, len(content) / 500) * min(1.0, 2000 / len(content))

        final_score = min(1.0, base_score + key_term_boost + question_boost) * length_factor
        return final_score

    def _create_category_chunks(self, section: DocumentSection) -> List[DocumentSection]:
        """Create chunks within a category - returns DocumentSection objects"""
        chunks = []
        content = section.content

        if len(content) <= self.chunk_size:
            # Single chunk - return the original section
            chunks.append(section)
        else:
            # Multiple chunks with overlap
            sentences = self._split_into_sentences(content)
            current_chunk = ""
            current_length = 0
            chunk_index = 0

            for sentence in sentences:
                sentence_length = self._get_token_count(sentence)

                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Create chunk as DocumentSection
                    chunk_metadata = section.metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': chunk_index,
                        'total_chunks': 'unknown'  # Will be updated later
                    })

                    chunks.append(DocumentSection(
                        category=section.category,
                        content=current_chunk.strip(),
                        importance_score=section.importance_score,
                        metadata=chunk_metadata
                    ))
                    chunk_index += 1

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                    current_length = self._get_token_count(current_chunk)
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_length += sentence_length

            # Add the last chunk
            if current_chunk.strip():
                chunk_metadata = section.metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'total_chunks': chunk_index + 1
                })

                chunks.append(DocumentSection(
                    category=section.category,
                    content=current_chunk.strip(),
                    importance_score=section.importance_score,
                    metadata=chunk_metadata
                ))

            # Update total_chunks for all chunks
            total_chunks = len(chunks)
            for chunk in chunks:
                chunk.metadata['total_chunks'] = total_chunks

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling"""
        # Handle academic text with citations
        text = re.sub(r'\([^)]*\)', '', text)  # Remove citations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _get_token_count(self, text: str) -> int:
        """Get token count"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        overlap_size = min(self.chunk_overlap // 10, len(words) // 4)
        overlap_words = words[-overlap_size:] if overlap_size > 0 else []
        return " ".join(overlap_words)