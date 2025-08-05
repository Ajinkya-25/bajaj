# processing/text_processor.py
# Fixed version with proper chunking that handles large documents

import re
from typing import List, Dict, Any
from settings import settings
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def preprocess_text(self, text: str) -> str:
        """Cleans text by removing extra whitespace and unifying line breaks."""
        text = re.sub(r' \n', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Splits text into semantic chunks with proper size management.
        This version ensures chunks stay within byte limits while maintaining semantic coherence.
        """
        if not text.strip():
            return []

        # Maximum chunk size in characters (roughly 1 byte per char for English text)
        # Set to be safe under the 36KB byte limit
        MAX_CHUNK_CHARS = 8000  # This should be well under 36KB when encoded as UTF-8
        
        chunks = []
        
        # First, try to split by paragraphs (double newlines)
        sections = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If adding this section would make the chunk too large
            if len(current_chunk) + len(section) + 2 > MAX_CHUNK_CHARS:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {**(metadata or {}), 'chunk_type': 'semantic_chunk'}
                    })
                
                # If the section itself is too large, split it further
                if len(section) > MAX_CHUNK_CHARS:
                    sub_chunks = self._split_large_section(section, MAX_CHUNK_CHARS)
                    for sub_chunk in sub_chunks:
                        chunks.append({
                            'content': sub_chunk,
                            'metadata': {**(metadata or {}), 'chunk_type': 'split_section'}
                        })
                    current_chunk = ""
                else:
                    current_chunk = section
            else:
                # Add section to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {**(metadata or {}), 'chunk_type': 'final_chunk'}
            })
        
        # Verify all chunks are under byte limit
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            byte_size = len(chunk['content'].encode('utf-8'))
            if byte_size <= 36000:  # 36KB limit
                valid_chunks.append(chunk)
                logger.debug(f"Chunk {i}: {len(chunk['content'])} chars, {byte_size} bytes ✓")
            else:
                logger.warning(f"Chunk {i} still too large ({byte_size} bytes), attempting further split")
                # Further split oversized chunks
                sub_chunks = self._force_split_by_size(chunk['content'], metadata)
                valid_chunks.extend(sub_chunks)
        
        logger.info(f"Successfully generated {len(valid_chunks)} valid chunks from {len(text)} characters")
        
        return valid_chunks

    def _split_large_section(self, section: str, max_chars: int) -> List[str]:
        """Split a large section by sentences, then by words if necessary."""
        # Try splitting by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single sentence is too long, split by words
                if len(sentence) > max_chars:
                    word_chunks = self._split_by_words(sentence, max_chars)
                    chunks.extend(word_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _split_by_words(self, text: str, max_chars: int) -> List[str]:
        """Split text by words when sentences are too long."""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _force_split_by_size(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Force split text into chunks that fit within byte limit."""
        # Use a conservative character limit to ensure we stay under 36KB
        MAX_SAFE_CHARS = 7000
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + MAX_SAFE_CHARS
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:].strip()
            else:
                # Try to break at a word boundary
                chunk_text = text[start:end]
                last_space = chunk_text.rfind(' ')
                if last_space > len(chunk_text) * 0.8:  # If we found a space in the last 20%
                    end = start + last_space
                    chunk_text = text[start:end].strip()
                else:
                    chunk_text = chunk_text.strip()
            
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'metadata': {**(metadata or {}), 'chunk_type': 'force_split'}
                })
            
            start = end + 1
        
        return chunks