# main.py
import os
import uvicorn
import logging
import time
from urllib.parse import urlparse
import requests
import json

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List

import google.generativeai as genai
from google.api_core.exceptions import DeadlineExceeded
from document_processor import DocumentProcessorFactory
from text_processor import TextProcessor
from embedding_manager import EmbeddingManager
from vectordb import FAISSManager
from postgres_manager import PostgresManager
from settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# Authentication dependency
def verify_api_key(authorization: str = Header(...)):
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token

# Initialize core components
text_processor = TextProcessor()
embedding_manager = EmbeddingManager()
# Use the actual embedding dimension
vector_db = FAISSManager(dimension=embedding_manager.dimension)
postgres_db = PostgresManager()
genai.configure(api_key=settings.GEMINI_API_KEY)

# Request model
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(request: HackRxRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    try:
        doc_url = str(request.documents)
        logger.info(f"Processing document: {doc_url}")

        # Clear data
        postgres_db.clear_all_data()
        vector_db.clear_database()

        # Download document
        resp = requests.get(doc_url, timeout=settings.REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()

        # Determine filename
        parsed = urlparse(doc_url)
        filename = os.path.basename(parsed.path) or ''
        if '.' not in filename:
            ctype = resp.headers.get('content-type', '').lower()
            filename = 'document.pdf' if 'pdf' in ctype else 'document.docx' if 'word' in ctype else 'document.txt'

        # Extract text
        processor = DocumentProcessorFactory.get_processor(filename)
        raw_text = processor.extract_text(resp.content)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text content extracted")

        # Preprocess and chunk
        sections = text_processor.categorize_and_partition_document(
            raw_text,
            metadata={"url": doc_url, "filename": filename}
        )
        if not sections:
            raise HTTPException(status_code=400, detail="No chunks generated")

        # Store metadata
        doc_id = postgres_db.store_document_with_url(
            filename=filename,
            document_url=doc_url,
            file_type=processor.get_file_type(),
            file_size=len(resp.content),
            categories=list({s.category for s in sections}),
            processing_time=time.time() - start_time
        )

        # Embeddings & vectors
        texts = [s.content for s in sections]
        embs = embedding_manager.generate_embeddings(texts)
        meta_list = []
        for i, s in enumerate(sections):
            m = s.metadata.copy()
            m.update({
                'document_id': doc_id,
                'chunk_index': i,
                'category': s.category,
                'importance_score': s.importance_score
            })
            meta_list.append(m)
        vec_ids = vector_db.add_vectors_with_categories(embs, meta_list)

        postgres_db.store_enhanced_chunks(
            doc_id,
            [
                {
                    'content': s.content,
                    'category': s.category,
                    'importance_score': s.importance_score,
                    'metadata': s.metadata,
                    'vector_id': vec_ids[i]
                }
                for i, s in enumerate(sections)
            ]
        )
        postgres_db.mark_document_processed(doc_id)

        # Retrieve contexts
        contexts = []
        for q in request.questions:
            q_emb = embedding_manager.generate_query_embedding(q)
            scores, idxs, cats = vector_db.search_by_category(q_emb, k=5)
            if idxs:
                chunks = postgres_db.get_chunks_by_vector_ids_with_categories(idxs)
                contexts.append(
                    f"Question: {q}\nContext:\n" + "\n\n".join(c['content'] for c in chunks)
                )

        # Build prompt
        prompt = (
            "Based on the provided contexts, answer each question accurately and concisely with explanation. "
            "Return ONLY the answers in a JSON array format.\n\n" +
            "\n\n".join(contexts) +
            "\n\nReturn the answers as a JSON array: ['answer1', 'answer2', ...]"
        )

        # LLM call with timeout handling
        try:
            model = genai.GenerativeModel(model_name=settings.LLM_MODEL)
            gen_response = model.generate_content(prompt)
        except DeadlineExceeded as de:
            logger.error(f"LLM timeout: {de}")
            raise HTTPException(status_code=504, detail="LLM deadline exceeded, please try again later")

        if not gen_response.text:
            raise HTTPException(status_code=500, detail="No response from LLM")

        # Clean and parse answers
        raw = gen_response.text.strip()
        # Strip markdown fences if present
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            # remove starting fence
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            # remove ending fence
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

        try:
            answers = json.loads(cleaned)
            if not isinstance(answers, list):
                raise ValueError()
        except Exception:
            # Fallback: split by lines
            answers = [line.strip() for line in cleaned.splitlines() if line.strip()]

        # Ensure count
        while len(answers) < len(request.questions):
            answers.append("Could not generate answer")
        answers = answers[:len(request.questions)]

        logger.info(f"Completed in {time.time() - start_time:.2f}s")
        return JSONResponse(content={"answers": answers})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing failed")
        detail = str(e) or "An unexpected error occurred (see server logs for details)."
        raise HTTPException(status_code=500, detail=detail)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)