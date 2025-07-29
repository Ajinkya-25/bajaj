# main.py - Final HackRx Submission Version
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import requests
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from processing_pipeline import ProcessingPipeline
    from query_pipeline import QueryPipeline
    from settings import settings
    logger.info("Successfully imported pipeline modules")
except Exception as e:
    logger.error(f"Failed to import modules: {str(e)}")
    logger.error(traceback.format_exc())
    raise

app = FastAPI(title="RAG System API", version="1.0.0")

# Initialize pipelines
try:
    processing_pipeline = ProcessingPipeline()
    query_pipeline = QueryPipeline()
except Exception as e:
    logger.error(f"Pipeline initialization failed: {e}")
    processing_pipeline = None
    query_pipeline = None

# Request models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class MultipleQueriesRequest(BaseModel):
    queries: List[str]
    top_k: Optional[int] = 5

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# --- HackRx Required Endpoint ---
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: HackRxRequest, authorization: str = Header(...)):
    """HackRx submission endpoint: accept document URL and questions, return answers."""
    try:
        # Validate Bearer token
        VALID_TOKEN = "bc321bbad77ee026212bb14fed80fdb8a8a0e5972eef4d4b200574c4039756a8"
        if authorization.split()[-1] != VALID_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")

        logger.info(f"Downloading document from {request.documents}")
        response = requests.get(request.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to download document from URL")

        file_content = response.content
        filename = request.documents.split("/")[-1]

        logger.info("Processing downloaded document")
        doc_result = processing_pipeline.process_document(filename, file_content)

        if "error" in doc_result:
            raise HTTPException(status_code=500, detail=doc_result["error"])

        logger.info("Running semantic query on questions")
        answers = []
        for q in request.questions:
            result = query_pipeline.process_query(q)
            answers.append(result.get("answer", "No answer found."))

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        logger.error(f"Error in /hackrx/run: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Existing Endpoints ---
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = processing_pipeline.process_document(file.filename, content)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        result = query_pipeline.process_query(request.query, request.top_k)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/multiple-queries")
async def process_multiple_queries(request: MultipleQueriesRequest):
    try:
        results = []
        for q in request.queries:
            results.append(query_pipeline.process_query(q, request.top_k))
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multiple queries failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if processing_pipeline and query_pipeline else "degraded",
        "processing_pipeline": "ready" if processing_pipeline else "failed",
        "query_pipeline": "ready" if query_pipeline else "failed"
    }

@app.get("/debug")
async def debug_info():
    return {
        "faiss_path": settings.FAISS_INDEX_PATH,
        "postgres_host": settings.POSTGRES_HOST,
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "processing_pipeline": "ready" if processing_pipeline else "failed",
        "query_pipeline": "ready" if query_pipeline else "failed"
    }

# --- Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
