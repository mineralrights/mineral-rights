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

# CORS configuration - Simple and permissive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
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
    """Upload PDF and start processing - returns job_id for SSE streaming"""
    print(f"🔍 Processing mode: {processing_mode}")
    print(f"🔍 Filename: {file.filename}")
    
    if processor is None:
        if not initialize_processor():
            raise HTTPException(status_code=500, detail="Model not initialized")

    # Memory monitoring
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"💾 Memory usage: {initial_memory:.1f} MB")

    # Save file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Create job
    job_id = str(uuid.uuid4())
    log_q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
    jobs[job_id] = log_q
    job_start_times[job_id] = time.time()
    job_metadata[job_id] = {
        "filename": file.filename,
        "processing_mode": processing_mode,
        "splitting_strategy": splitting_strategy,
        "status": "processing"
    }
    
    def run():
        try:
            print(f"🚀 Starting processing for job {job_id}")
            log_q.put_nowait("🚀 Processing started")
            
            with redirect_stdout(QueueWriter(log_q)):
                print(f"📁 File: {file.filename}")
                print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"💾 Memory: {initial_memory:.1f} MB")
                
                if processing_mode == "single_deed":
                    print("📄 Single deed processing")
                    result = processor.process_document(tmp_path)
                elif processing_mode == "multi_deed":
                    print("📄 Multi-deed processing")
                    result = processor.process_multi_deed_document(tmp_path, strategy=splitting_strategy)
                elif processing_mode == "page_by_page":
                    print("📄 Page-by-page processing")
                    result = processor.process_document_page_by_page(tmp_path, max_samples=6, high_recall_mode=True)
                else:
                    raise ValueError(f"Unknown processing mode: {processing_mode}")
                
                job_results[job_id] = result
                log_q.put_nowait(f"__RESULT__{json.dumps(result)}")
                print("✅ Processing completed successfully")
                log_q.put_nowait("__END__")
                    
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            log_q.put_nowait(f"❌ Error: {str(e)}")
            log_q.put_nowait("__END__")
        finally:
            # Clean up
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    print(f"🧹 Cleaned up temp file")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error: {cleanup_error}")

    # Start background thread
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    
        return {"job_id": job_id}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
