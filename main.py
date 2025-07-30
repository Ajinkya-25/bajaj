# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List
import uvicorn
import logging
import time
import requests
import os
from urllib.parse import urlparse

from document_processor import DocumentProcessorFactory
from text_processor import EnhancedTextProcessor
from embedding_manager import EmbeddingManager
from vectordb import FAISSManager
from postgres_manager import PostgresManager
import google.generativeai as genai
from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG System API", version="1.0.0")

# Initialize components
text_processor = EnhancedTextProcessor()
embedding_manager = EmbeddingManager()
vector_db = FAISSManager(embedding_manager.get_dimension())
postgres_db = PostgresManager()
genai.configure(api_key=settings.GEMINI_API_KEY)


class ProcessRequest(BaseModel):
    document_url: HttpUrl
    questions: List[str]


@app.post("/process")
async def process_document_and_answer(request: ProcessRequest):
    """Process document and answer questions"""
    try:
        start_time = time.time()
        logger.info(f"Processing document: {request.document_url}")

        # Clear existing data
        postgres_db.clear_all_data()
        vector_db.clear_database()

        # Download document
        response = requests.get(str(request.document_url), timeout=30)
        response.raise_for_status()

        parsed_url = urlparse(str(request.document_url))
        filename = os.path.basename(parsed_url.path)
        if not filename or '.' not in filename:
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type:
                filename = 'document.pdf'
            elif 'word' in content_type or 'docx' in content_type:
                filename = 'document.docx'
            else:
                filename = 'document.txt'

        # Extract text
        processor = DocumentProcessorFactory.get_processor(filename)
        raw_text = processor.extract_text(response.content)

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text content extracted")

        # Process text with categorization
        processed_text = text_processor.preprocess_text(raw_text)
        document_sections = text_processor.categorize_and_partition_document(
            processed_text,
            metadata={"document_url": str(request.document_url), "filename": filename}
        )

        if not document_sections:
            raise HTTPException(status_code=400, detail="No chunks generated")

        # Store document
        categories = list(set(section.category for section in document_sections))
        processing_time = time.time() - start_time
        document_id = postgres_db.store_document_with_url(
            filename=filename,
            document_url=str(request.document_url),
            file_type=processor.get_file_type(),
            file_size=len(response.content),
            categories=categories,
            processing_time=processing_time
        )

        # Generate embeddings
        chunk_texts = [section.content for section in document_sections]
        embeddings = embedding_manager.generate_embeddings(chunk_texts)

        # Prepare metadata for vector storage
        vector_metadata = []
        for i, section in enumerate(document_sections):
            metadata = section.metadata.copy()
            metadata.update({
                'document_id': document_id,
                'chunk_index': i,
                'category': section.category,
                'importance_score': section.importance_score,
                'filename': filename,
                'document_url': str(request.document_url)
            })
            vector_metadata.append(metadata)

        # Store in vector database
        vector_ids = vector_db.add_vectors_with_categories(embeddings, vector_metadata)

        # Link vector IDs to chunks and store in PostgreSQL
        chunks_for_postgres = []
        for i, section in enumerate(document_sections):
            chunks_for_postgres.append({
                'content': section.content,
                'category': section.category,
                'importance_score': section.importance_score,
                'metadata': section.metadata,
                'vector_id': vector_ids[i]
            })

        postgres_db.store_enhanced_chunks(document_id, chunks_for_postgres)
        postgres_db.mark_document_processed(document_id)

        # Answer questions
        all_contexts = []
        for question in request.questions:
            # Generate query embedding
            query_embedding = embedding_manager.generate_query_embedding(question)

            # Search for relevant chunks
            scores, vector_indices, categories = vector_db.search_by_category(
                query_embedding, k=5
            )

            # Get chunks from postgres
            if vector_indices:
                chunks = postgres_db.get_chunks_by_vector_ids_with_categories(vector_indices)
                context = "\n\n".join([chunk['content'] for chunk in chunks])
                all_contexts.append(f"Question: {question}\nContext: {context}")

        # Combine all contexts for single API call
        combined_prompt = f"""Based on the provided contexts, answer each question accurately and concisely. Return ONLY the answers in a JSON array format.

{chr(10).join(all_contexts)}

Return the answers as a JSON array: ["answer1", "answer2", ...]"""

        # Single Gemini API call
        model = genai.GenerativeModel(model_name=settings.LLM_MODEL)
        response = model.generate_content(combined_prompt)

        if not response.text:
            raise HTTPException(status_code=500, detail="No response from LLM")

        # Try to parse JSON response
        try:
            import json
            answers = json.loads(response.text.strip())
            if not isinstance(answers, list):
                # Fallback: split by lines if not proper JSON
                answers = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        except:
            # Fallback: split response into answers
            answers = [line.strip() for line in response.text.strip().split('\n') if line.strip()]

        # Ensure we have the right number of answers
        while len(answers) < len(request.questions):
            answers.append("Could not generate answer")

        answers = answers[:len(request.questions)]

        total_time = time.time() - start_time
        logger.info(f"Completed processing in {total_time:.2f} seconds")

        return JSONResponse(content={"answers": answers})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)