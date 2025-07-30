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
            logger.debug(f"Processed text length: {len(processed_text)} chars")

            # 5) Chunking
            chunks = self.text_processor.chunk_text(
                processed_text,
                metadata={"document_id": document_id, "filename": filename}
            )
            logger.info(f"Generated {len(chunks)} text chunks")
            if not chunks:
                logger.warning(f"No chunks generated for document id={document_id}")
                return {"error": "No chunks generated from document"}

            # 6) Filter by byte‑size, embed, and store
            MAX_BYTES = 36000
            valid_texts, valid_metadata, valid_chunks = [], [], []
            for i, chunk in enumerate(chunks):
                b = chunk["content"].encode("utf-8")
                if len(b) <= MAX_BYTES:
                    valid_texts.append(chunk["content"])
                    valid_metadata.append({
                        "document_id": document_id,
                        "chunk_index": i,
                        "filename": filename
                    })
                    valid_chunks.append(chunk)
                else:
                    logger.warning(f"⚠️ Skipping over‑sized chunk {i} ({len(b)} bytes)")

            if not valid_texts:
                raise RuntimeError("No valid chunks under byte limit")

            embeddings = self.embedding_manager.generate_embeddings(valid_texts)
            logger.info(f"Generated {len(embeddings)} embeddings (dim={self.embedding_manager.get_dimension()})")

            vector_ids = self.vector_db.add_vectors(embeddings, valid_metadata)
            logger.info(f"Stored embeddings in FAISS, got {len(vector_ids)} vector IDs")

            # attach vector_ids back to chunks
            for meta, vid in zip(valid_metadata, vector_ids):
                chunks[meta["chunk_index"]]["vector_id"] = vid

            self.postgres_db.store_chunks(document_id, chunks)
            self.postgres_db.mark_document_processed(document_id)
            logger.info(f"Document id={document_id} marked as processed")

            return {"document_id": document_id, "chunks_created": len(chunks), "status": "success"}

        except Exception as e:
            logger.error(f"Processing failed for '{filename}': {e}", exc_info=True)
            return {"error": f"Processing failed: {str(e)}"}
