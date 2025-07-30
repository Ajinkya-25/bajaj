import uvicorn
import traceback
import logging
import time
from typing import List, Optional, Dict

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Load settings from .env
class Settings(BaseSettings):
    hackrx_api_key: str

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipelines
try:
    from processing_pipeline import ProcessingPipeline
    from query_pipeline import QueryPipeline
    logger.info("Successfully imported pipeline modules")
except Exception as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error(traceback.format_exc())
    raise

app = FastAPI(title="RAG System API", version="1.0.0")

# Initialize pipelines
try:
    logger.info("Initializing processing pipeline...")
    processing_pipeline = ProcessingPipeline()
    logger.info("Processing pipeline initialized successfully")

    logger.info("Initializing query pipeline...")
    query_pipeline = QueryPipeline()
    logger.info("Query pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipelines: {e}")
    logger.error(traceback.format_exc())
    processing_pipeline = None
    query_pipeline = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class MultipleQueriesRequest(BaseModel):
    queries: List[str]
    top_k: Optional[int] = 5

# Utility: Validate Bearer token
def _validate_bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != settings.hackrx_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token

# Upload Document Endpoint
def _upload_document_sync(filename: str, content: bytes) -> Dict:
    if processing_pipeline is None:
        return {"error": "Processing pipeline not initialized"}
    return processing_pipeline.process_document(filename, content)

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload: {file.filename}")
        content = await file.read()
        result = _upload_document_sync(file.filename, content)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Single Query Endpoint
@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        if query_pipeline is None:
            raise HTTPException(status_code=503, detail="Query pipeline not initialized")

        result = query_pipeline.process_query(request.query, request.top_k)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Multiple Queries Endpoint
@app.post("/multiple-queries")
async def process_multiple_queries(request: MultipleQueriesRequest):
    try:
        if query_pipeline is None:
            raise HTTPException(status_code=503, detail="Query pipeline not initialized")

        results = []
        for q in request.queries:
            raw = query_pipeline.process_query(q, request.top_k)
            results.append(raw)
        return JSONResponse(content={"results": results})
    except Exception as e:
        logger.error(f"Multiple queries failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if processing_pipeline and query_pipeline else "degraded",
        "message": "RAG System is running",
        "processing_pipeline": "ready" if processing_pipeline else "failed",
        "query_pipeline": "ready" if query_pipeline else "failed"
    }

# Debug Info Endpoint
@app.get("/debug")
async def debug_info():
    try:
        from settings import settings as app_settings
        return {
            "faiss_path": getattr(app_settings, 'FAISS_INDEX_PATH', None),
            "postgres_host": getattr(app_settings, 'POSTGRES_HOST', None),
            "gemini_configured": bool(getattr(app_settings, 'GEMINI_API_KEY', None)),
            "processing_pipeline": "ready" if processing_pipeline else "failed",
            "query_pipeline": "ready" if query_pipeline else "failed"
        }
    except Exception as e:
        return {"error": str(e)}

# HackRx Run Endpoint with Bearer Auth and PDF fetching
@app.post("/hackrx/run")
async def hackrx_run(
    payload: Dict = Body(...),
    authorization: str = Header(None)
):
    _validate_bearer(authorization)

    url = payload.get("documents")
    questions: List[str] = payload.get("questions", [])
    if not url or not questions:
        raise HTTPException(status_code=400, detail="`documents` and `questions` are required")

    # 1) Download the PDF from the provided URL
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content
    except Exception as e:
        logger.error(f"Failed to fetch document from URL {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not fetch document: {e}")

    # 2) Ingest the PDF bytes
    try:
        processing_pipeline.process_document("datasheet.pdf", pdf_bytes)
    except Exception as e:
        logger.error(f"Failed to ingest PDF: {e}")
        # proceed anyway—pipeline might log internal errors

    # 3) Run each question
    answers: List[str] = []
    timings_ms: List[float] = []
    for q in questions:
        start = time.time()
        try:
            raw = query_pipeline.process_query(q, top_k=5)
            if isinstance(raw.get("answers"), list) and raw["answers"]:
                answer = raw["answers"][0]
            elif isinstance(raw.get("results"), list) and raw["results"]:
                answer = raw["results"][0].get("answer", "No answer returned.")
            else:
                answer = "No answer returned."
        except Exception as e:
            logger.error(f"Error during query '{q}': {e}")
            answer = f"Error generating answer: {e}"
        end = time.time()

        answers.append(answer)
        timings_ms.append((end - start) * 1000)

    return JSONResponse(content={
        "answers": answers,
        "timings_ms": timings_ms
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
