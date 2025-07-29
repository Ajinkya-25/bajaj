# processing/text_processor.py
import re
from typing import List, Dict, Any
import tiktoken
from settings import settings


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
