import logging
from typing import Dict, Any, List
from document_processor import DocumentProcessorFactory
from text_processor import TextProcessor
from embedding_manager import EmbeddingManager
from vectordb import FAISSManager
from postgres_manager import PostgresManager

# Configure module-level logger
logger = logging.getLogger(__name__)

class ProcessingPipeline:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_db = FAISSManager(self.embedding_manager.get_dimension())
        self.postgres_db = PostgresManager()
        logger.info("Initialized ProcessingPipeline")

    def process_document(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Process a document through the entire pipeline"""
        logger.info(f"Starting process_document for '{filename}' (size={len(file_content)} bytes)")
        try:
            # Step 1: Document Type Check & Text Extraction
            processor = DocumentProcessorFactory.get_processor(filename)
            logger.debug(f"Using processor: {processor.__class__.__name__}")
            text = processor.extract_text(file_content)
            logger.debug(f"Extracted text length: {len(text)} characters")

            if not text.strip():
                logger.warning(f"No text content extracted from '{filename}'")
                return {"error": "No text content extracted from document"}

            # Step 2: Store document metadata
            document_id = self.postgres_db.store_document(
                filename=filename,
                file_type=processor.get_file_type(),
                file_size=len(file_content),
                metadata={"original_length": len(text)}
            )
            logger.info(f"Stored document metadata (id={document_id})")

            # Step 3: Text Preprocessing
            processed_text = self.text_processor.preprocess_text(text)
            logger.debug(f"Processed text length: {len(processed_text)} characters")

            # Step 4: Text Chunking
            chunks = self.text_processor.chunk_text(
                processed_text,
                metadata={"document_id": document_id, "filename": filename}
            )
            logger.info(f"Generated {len(chunks)} text chunks")

            if not chunks:
                logger.warning(f"No chunks generated for document id={document_id}")
                return {"error": "No chunks generated from document"}

            # Step 5: Generate Embeddings
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_manager.generate_embeddings(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings (dim={self.embedding_manager.get_dimension()})")

            # Step 6: Store in Vector DB
            vector_metadata = [
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "filename": filename
                }
                for i in range(len(chunks))
            ]
            vector_ids = self.vector_db.add_vectors(embeddings, vector_metadata)
            logger.info(f"Stored embeddings in FAISS, received {len(vector_ids)} vector IDs")

            # Step 7: Link Vector & SQL Records
            for i, chunk in enumerate(chunks):
                chunk['vector_id'] = vector_ids[i]
            self.postgres_db.store_chunks(document_id, chunks)
            self.postgres_db.mark_document_processed(document_id)
            logger.info(f"Document id={document_id} marked as processed")

            logger.info(f"Completed processing '{filename}' successfully")
            return {
                "document_id": document_id,
                "chunks_created": len(chunks),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Processing failed for '{filename}': {e}", exc_info=True)
            return {"error": f"Processing failed: {str(e)}"}
