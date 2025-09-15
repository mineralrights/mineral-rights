from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile
import os
import time
import json
import threading
import asyncio
import uuid
from contextlib import redirect_stdout
from io import StringIO
import psutil
import gc

# Import your pipeline
from src.mineral_rights.document_classifier import DocumentProcessor

# Initialize FastAPI app
app = FastAPI(title="Mineral-Rights API - Simple SSE Version")

# CORS configuration - Explicit and robust
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "Cache-Control",
        "Pragma",
        "Expires",
        "User-Agent",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["*"],
    max_age=3600  # Cache preflight for 1 hour
)

# Add cache-busting middleware
@app.middleware("http")
async def add_cache_busting_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Global processor
processor = None

def initialize_processor():
    global processor
    try:
        print("🔧 Initializing DocumentProcessor...")
        
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found in environment")
            return False
            
        # Get Document AI endpoint
        document_ai_endpoint = os.getenv("DOCUMENT_AI_ENDPOINT")
        if not document_ai_endpoint:
            print("❌ DOCUMENT_AI_ENDPOINT not found in environment")
            return False
            
        print(f"API Key present: {'Yes' if api_key else 'No'}")
        print(f"Document AI Endpoint: {document_ai_endpoint}")
        
        # Handle Google credentials
        credentials_path = None
        google_credentials_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        
        if google_credentials_base64:
            try:
                import base64
                import tempfile
                # Decode base64 credentials
                credentials_json = base64.b64decode(google_credentials_base64).decode('utf-8')
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                temp_file.write(credentials_json)
                temp_file.close()
                credentials_path = temp_file.name
                print(f"✅ Created temporary credentials file from base64")
            except Exception as e:
                print(f"⚠️ Failed to decode base64 credentials: {e}")
        
        # Initialize processor with explicit parameters including credentials
        processor = DocumentProcessor(
            api_key=api_key,
            document_ai_endpoint=document_ai_endpoint,
            document_ai_credentials=credentials_path
        )
        
        print("✅ DocumentProcessor initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize DocumentProcessor: {e}")
        import traceback
        traceback.print_exc()
        return False

# Processor will be initialized on first request, not during startup

# Simple in-memory job registry for SSE streaming
jobs: dict[str, asyncio.Queue[str]] = {}
job_metadata: dict[str, dict] = {}
job_start_times: dict[str, float] = {}
job_results: dict[str, dict] = {}

class QueueWriter:
    def __init__(self, queue):
        self.queue = queue
    
    def write(self, text):
        if text.strip():
            try:
                self.queue.put_nowait(text.strip())
            except asyncio.QueueFull:
                pass  # Ignore if queue is full
    
    def flush(self):
        pass

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    processing_mode: str = Form("single_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """Process document directly and return results"""
    print(f"🔍 Processing document - Mode: {processing_mode}, File: {file.filename}")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Save file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            if processor is None:
                try:
                    if not initialize_processor():
                        raise HTTPException(status_code=500, detail="Model not initialized")
                except Exception as e:
                    print(f"❌ Processor initialization failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
            
            print("🤖 Initializing document processor...")
            print(f"🔧 Processing with mode: {processing_mode}")
            
            # Process the document based on mode
            if processing_mode == "single_deed":
                result = processor.process_document(tmp_file_path)
                
                # Convert result to expected format
                return {
                    "has_reservation": result.get("classification", 0) == 1,
                    "confidence": result.get("confidence", 0.0),
                    "reasoning": result.get("detailed_samples", [{}])[0].get("reasoning", "No reasoning provided") if result.get("detailed_samples") else "No reasoning provided",
                    "processing_mode": processing_mode,
                    "filename": file.filename
                }
                
            elif processing_mode == "multi_deed":
                # Use the proper multi-deed processing with Document AI splitting
                result = processor.process_multi_deed_document(tmp_file_path, strategy=splitting_strategy)
                
                # Convert to expected format
                deed_results = []
                for deed_result in result:
                    deed_results.append({
                        "deed_number": deed_result.get("deed_number", 0),
                        "has_reservations": deed_result.get("has_reservation", False),
                        "confidence": deed_result.get("confidence", 0.0),
                        "reasoning": deed_result.get("reasoning", "No reasoning provided"),
                        "pages": deed_result.get("pages", [])
                    })
                
                return {
                    "deed_results": deed_results,
                    "total_deeds": len(deed_results),
                    "processing_mode": processing_mode,
                    "filename": file.filename
                }
                
            elif processing_mode == "page_by_page":
                result = processor.process_document(tmp_file_path, page_strategy="all_pages")
                
                # Convert to page-by-page format
                page_results = []
                if "page_results" in result:
                    for i, page_result in enumerate(result["page_results"]):
                        page_results.append({
                            "page_number": i + 1,
                            "has_reservations": page_result.get("classification", 0) == 1,
                            "confidence": page_result.get("confidence", 0.0),
                            "explanation": page_result.get("detailed_samples", [{}])[0].get("reasoning", "No reasoning provided") if page_result.get("detailed_samples") else "No reasoning provided"
                        })
                
                return {
                    "page_results": page_results,
                    "total_pages": len(page_results),
                    "processing_mode": processing_mode,
                    "filename": file.filename
                }
                
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except Exception as e:
        print(f"❌ Failed to process document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/stream/{job_id}")
async def stream(job_id: str):
    """Stream processing logs and results via Server-Sent Events"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    queue = jobs[job_id]

    async def event_generator():
        last_heartbeat = time.time()
        heartbeat_interval = 5  # Send heartbeat every 5 seconds
        session_start = time.time()
        
        while True:
            try:
                # Wait for message with timeout for heartbeats
                line = await asyncio.wait_for(queue.get(), timeout=2.0)
                
                # Send the message
                yield f"data: {line}\n\n"
                
                # Check if we're done
                if line == "__END__":
                    break
                    
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                current_time = time.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    session_duration = current_time - session_start
                    yield f"data: __HEARTBEAT__{current_time}|{session_duration}\n\n"
                    last_heartbeat = current_time

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "processor_initialized": processor is not None,
            "timestamp": time.time(),
            "api_key_present": bool(os.getenv("ANTHROPIC_API_KEY")),
            "document_ai_endpoint_present": bool(os.getenv("DOCUMENT_AI_ENDPOINT")),
            "google_credentials_present": bool(os.getenv("GOOGLE_CREDENTIALS_BASE64"))
        }
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/")
async def root():
    """Root endpoint for basic connectivity test"""
    return {"message": "Mineral Rights API is running", "timestamp": time.time()}

@app.get("/heartbeat")
async def heartbeat():
    """Simple heartbeat endpoint for Railway health checks"""
    return {"status": "alive"}

@app.get("/test")
async def test():
    """Simple test endpoint that doesn't require processor initialization"""
    return {"message": "Test endpoint working", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
