# api/main.py - Debug Version
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from processing_pipeline import ProcessingPipeline
    from query_pipeline import QueryPipeline

    logger.info("Successfully imported pipeline modules")
except Exception as e:
    logger.error(f"Failed to import modules: {str(e)}")
    logger.error(traceback.format_exc())
    raise

app = FastAPI(title="RAG System API", version="1.0.0")

# Initialize pipelines with error handling
try:
    logger.info("Initializing processing pipeline...")
    processing_pipeline = ProcessingPipeline()
    logger.info("Processing pipeline initialized successfully")

    logger.info("Initializing query pipeline...")
    query_pipeline = QueryPipeline()
    logger.info("Query pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipelines: {str(e)}")
    logger.error(traceback.format_exc())
    # Create dummy pipelines to prevent startup failure
    processing_pipeline = None
    query_pipeline = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class MultipleQueriesRequest(BaseModel):
    queries: List[str]
    top_k: Optional[int] = 5


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        logger.info(f"Received file upload: {file.filename}")

        if processing_pipeline is None:
            raise HTTPException(status_code=503, detail="Processing pipeline not initialized")

        # Read file content
        logger.info("Reading file content...")
        content = await file.read()
        logger.info(f"File content read: {len(content)} bytes")

        # Process document
        logger.info("Starting document processing...")
        result = processing_pipeline.process_document(file.filename, content)
        logger.info(f"Document processing result: {result}")

        if "error" in result:
            logger.error(f"Processing error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a single query"""
    try:
        logger.info(f"Received query: {request.query}")

        if query_pipeline is None:
            raise HTTPException(status_code=503, detail="Query pipeline not initialized")

        result = query_pipeline.process_query(request.query, request.top_k)
        logger.info(f"Query result: {result}")

        if "error" in result:
            logger.error(f"Query error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Query failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/multiple-queries")
async def process_multiple_queries(request: MultipleQueriesRequest):
    """Process multiple queries"""
    try:
        logger.info(f"Received multiple queries: {len(request.queries)}")

        if query_pipeline is None:
            raise HTTPException(status_code=503, detail="Query pipeline not initialized")

        results = []
        for query in request.queries:
            result = query_pipeline.process_query(query, request.top_k)
            results.append(result)

        return JSONResponse(content={"results": results})

    except Exception as e:
        error_msg = f"Multiple queries failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if processing_pipeline and query_pipeline else "degraded",
        "message": "RAG System is running",
        "processing_pipeline": "ready" if processing_pipeline else "failed",
        "query_pipeline": "ready" if query_pipeline else "failed"
    }
    return status


@app.get("/debug")
async def debug_info():
    """Debug information endpoint"""
    try:
        from settings import settings
        return {
            "faiss_path": settings.FAISS_INDEX_PATH,
            "postgres_host": settings.POSTGRES_HOST,
            "gemini_configured": bool(settings.GEMINI_API_KEY),
            "processing_pipeline": "ready" if processing_pipeline else "failed",
            "query_pipeline": "ready" if query_pipeline else "failed"
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)