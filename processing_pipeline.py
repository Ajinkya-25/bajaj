# processing/processing_pipeline.py
import logging
from typing import Dict, Any, List
from document_processor import DocumentProcessorFactory
from text_processor import TextProcessor
from embedding_manager import EmbeddingManager
from vectordb import FAISSManager
from postgres_manager import PostgresManager
import re

# Configure module-level logger
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Strip out non-printables, PDF control bytes and weird symbols."""
    # remove control chars
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # keep only sensible punctuation + unicode word chars
    text = re.sub(r"[^\w\s\.,;:'\"()\[\]\{\}\?!%₹\$@&#\*\-]", "", text)
    return text.strip()

class ProcessingPipeline:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_db = FAISSManager(self.embedding_manager.get_dimension())
        self.postgres_db = PostgresManager()
        logger.info("Initialized ProcessingPipeline")

    def process_document(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        logger.info(f"Starting process_document for '{filename}' (size={len(file_content)} bytes)")
        try:
            # 1) Extract raw text
            processor = DocumentProcessorFactory.get_processor(filename)
            logger.debug(f"Using processor: {processor.__class__.__name__}")
            raw = processor.extract_text(file_content)
            logger.debug(f"Raw extract length: {len(raw)} chars")

            if not raw.strip():
                logger.warning(f"No text content extracted from '{filename}'")
                return {"error": "No text content extracted from document"}

            # 2) Clean out PDF plumbing / control chars
            cleaned = clean_text(raw)
            logger.debug(f"Cleaned text length: {len(cleaned)} chars")

            # 🔍 Preview the first 500 chars of cleaned text
            print("📄 Cleaned text preview:")
            print(cleaned[:500])

            # 3) Store metadata
            document_id = self.postgres_db.store_document(
                filename=filename,
                file_type=processor.get_file_type(),
                file_size=len(file_content),
                metadata={"original_length": len(raw), "cleaned_length": len(cleaned)}
            )
            logger.info(f"Stored document metadata (id={document_id})")

            # 4) Further preprocessing (normalize punctuation, etc.)
            processed_text = self.text_processor.preprocess_text(cleaned)
            logger.info(f"Processed text length: {len(processed_text)} chars")

            # 5) Chunking with improved logic
            chunks = self.text_processor.chunk_text(
                processed_text,
                metadata={"document_id": document_id, "filename": filename}
            )
            logger.info(f"Generated {len(chunks)} text chunks")
            
            if not chunks:
                logger.warning(f"No chunks generated for document id={document_id}")
                return {"error": "No chunks generated from document"}

            # 6) Verify chunks and prepare for embedding
            MAX_BYTES = 36000
            valid_texts, valid_metadata, valid_chunks = [], [], []
            
            for i, chunk in enumerate(chunks):
                content = chunk["content"]
                byte_size = len(content.encode("utf-8"))
                
                if byte_size <= MAX_BYTES:
                    valid_texts.append(content)
                    valid_metadata.append({
                        "document_id": document_id,
                        "chunk_index": i,
                        "filename": filename,
                        "byte_size": byte_size,
                        "char_count": len(content)
                    })
                    valid_chunks.append(chunk)
                    logger.debug(f"✅ Chunk {i}: {len(content)} chars, {byte_size} bytes")
                else:
                    logger.error(f"❌ Chunk {i} still exceeds limit: {byte_size} bytes")

            if not valid_texts:
                logger.error("No valid chunks after size filtering")
                return {"error": "All chunks exceed the size limit. Document may be too dense or contain unusual formatting."}

            logger.info(f"Processing {len(valid_texts)} valid chunks for embedding")

            # 7) Generate embeddings
            try:
                embeddings = self.embedding_manager.generate_embeddings(valid_texts)
                logger.info(f"Generated {len(embeddings)} embeddings (dim={self.embedding_manager.get_dimension()})")
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return {"error": f"Embedding generation failed: {str(e)}"}

            # 8) Store in vector database
            try:
                vector_ids = self.vector_db.add_vectors(embeddings, valid_metadata)
                logger.info(f"Stored embeddings in FAISS, got {len(vector_ids)} vector IDs")
            except Exception as e:
                logger.error(f"Failed to store vectors: {e}")
                return {"error": f"Vector storage failed: {str(e)}"}

            # 9) Attach vector_ids back to chunks and store in PostgreSQL
            try:
                for meta, vid in zip(valid_metadata, vector_ids):
                    chunk_idx = meta["chunk_index"]
                    if chunk_idx < len(chunks):
                        chunks[chunk_idx]["vector_id"] = vid

                self.postgres_db.store_chunks(document_id, valid_chunks)
                self.postgres_db.mark_document_processed(document_id)
                logger.info(f"Document id={document_id} marked as processed")
            except Exception as e:
                logger.error(f"Failed to store chunks in PostgreSQL: {e}")
                return {"error": f"Chunk storage failed: {str(e)}"}

            # 10) Success response with detailed statistics
            total_chars = sum(len(chunk["content"]) for chunk in valid_chunks)
            avg_chunk_size = total_chars // len(valid_chunks) if valid_chunks else 0
            
            return {
                "document_id": document_id,
                "chunks_created": len(valid_chunks),
                "total_characters": total_chars,
                "average_chunk_size": avg_chunk_size,
                "original_text_length": len(raw),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Processing failed for '{filename}': {e}", exc_info=True)
            return {"error": f"Processing failed: {str(e)}"}