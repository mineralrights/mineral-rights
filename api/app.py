from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, traceback
from typing import List, Optional
import asyncio, threading, uuid, json, time
from fastapi.responses import StreamingResponse
import io, sys
from contextlib import redirect_stdout
import gc  # For garbage collection
import psutil  # For memory monitoring
from enum import Enum
from dataclasses import dataclass, asdict

#  import your pipeline ----------------------------------------------
from src.mineral_rights.document_classifier import DocumentProcessor

# Long-running job support built into main app
JOB_ENDPOINTS_AVAILABLE = True
print("✅ Long-running job support enabled - SIMPLE VERSION")

# --------------------------------------------------------------------------
# Simple Job System (In-Memory)
# --------------------------------------------------------------------------
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time
import uuid

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobInfo:
    id: str
    filename: str
    processing_mode: str
    splitting_strategy: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: list = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []

class SimpleJobManager:
    """Simple in-memory job manager - works reliably for Railway"""
    
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}
        print("✅ Simple job manager initialized (in-memory)")
    
    def create_job(self, filename: str, processing_mode: str, splitting_strategy: str) -> str:
        """Create a new job"""
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        job = JobInfo(
            id=job_id,
            filename=filename,
            processing_mode=processing_mode,
            splitting_strategy=splitting_strategy,
            status=JobStatus.PENDING,
            created_at=time.time(),
            logs=[]
        )
        
        self.jobs[job_id] = job
        print(f"✅ Created job: {job_id}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         result: Optional[Dict[str, Any]] = None, 
                         error: Optional[str] = None):
        """Update job status"""
        job = self.jobs.get(job_id)
        if not job:
            print(f"⚠️ Job {job_id} not found for status update")
            return
        
        job.status = status
        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = time.time()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            job.completed_at = time.time()
        
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        
        print(f"✅ Updated job {job_id} status to {status.value}")
    
    def list_jobs(self) -> list:
        """List all jobs"""
        return [asdict(job) for job in self.jobs.values()]

# Global job manager
job_manager = SimpleJobManager()

# ---------------------------------------------------------------------------


## TESTING NEW DEPLOYMENT 
app = FastAPI(title="Mineral-Rights API")

# Job system is now built into the main app
print("✅ Long-running job system integrated")

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3000", 
        "https://mineral-rights-4o8gr9h79-lauragomezjurados-projects.vercel.app",
        "https://mineral-rights-3n3mc6fj6-lauragomezjurados-projects.vercel.app",
        "https://mineral-rights-lvgux6lc5-lauragomezjurados-projects.vercel.app",  # Current Vercel domain
        "https://*.vercel.app",  # Allow all Vercel domains
        "*"  # Allow all origins for debugging
    ],
    allow_credentials=False,  # Set to False when using wildcard origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Add cache-busting middleware to prevent SSL/cache issues
@app.middleware("http")
async def add_cache_busting_headers(request: Request, call_next):
    response = await call_next(request)
    # Add headers to prevent caching and SSL issues
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# 2️⃣  --- initialise once at startup ----------------------------------------
API_KEY = os.getenv("ANTHROPIC_API_KEY")  # or whatever key the processor needs
DOCUMENT_AI_ENDPOINT = os.getenv("DOCUMENT_AI_ENDPOINT", "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process")
DOCUMENT_AI_CREDENTIALS = os.getenv("DOCUMENT_AI_CREDENTIALS_PATH")  # Path to service account JSON
GOOGLE_CREDENTIALS_BASE64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")  # Base64 encoded credentials

# Debug environment variables - FORCE DEPLOY
print("=" * 80)
print("🚀🚀🚀 FORCE DEPLOY - Debug environment variables 🚀🚀🚀")
print("=" * 80)
print(f"🔍 DEBUG: ANTHROPIC_API_KEY present: {'Yes' if API_KEY else 'No'}")
print(f"🔍 DEBUG: DOCUMENT_AI_ENDPOINT: {DOCUMENT_AI_ENDPOINT}")
print(f"🔍 DEBUG: GOOGLE_CREDENTIALS_BASE64 present: {'Yes' if GOOGLE_CREDENTIALS_BASE64 else 'No'}")
print(f"🔍 DEBUG: All env vars: {list(os.environ.keys())}")
print("=" * 80)
print("🚀🚀🚀 END FORCE DEPLOY DEBUG 🚀🚀🚀")
print("=" * 80)
processor = None

def initialize_processor():
    global processor
    try:
        print("🔧 Initializing DocumentProcessor...")
        print(f"API Key present: {'Yes' if API_KEY else 'No'}")
        print(f"Document AI Endpoint: {DOCUMENT_AI_ENDPOINT}")
        print(f"Document AI Credentials: {'Yes' if DOCUMENT_AI_CREDENTIALS else 'No'}")
        
        # Test imports first
        try:
            import fitz
            print("✅ PyMuPDF (fitz) imported successfully")
        except ImportError as e:
            print(f"❌ PyMuPDF import failed: {e}")
            return False
            
        try:
            import psutil
            print("✅ psutil imported successfully")
        except ImportError as e:
            print(f"❌ psutil import failed: {e}")
            return False
            
        try:
            import anthropic
            print("✅ anthropic imported successfully")
        except ImportError as e:
            print(f"❌ anthropic import failed: {e}")
            return False
        
        # Handle credentials - try multiple approaches
        credentials_path = None

        # Method 1: Try base64 credentials (if available and valid)
        if GOOGLE_CREDENTIALS_BASE64:
            try:
                import base64
                import tempfile
                # Test if base64 string is valid
                if len(GOOGLE_CREDENTIALS_BASE64) > 100:  # Basic length check
                    credentials_json = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                    temp_file.write(credentials_json)
                    temp_file.close()
                    credentials_path = temp_file.name
                    print(f"✅ Created temporary credentials file from base64: {credentials_path}")
                else:
                    print("⚠️ Base64 string too short, skipping...")
            except Exception as e:
                print(f"⚠️ Failed to decode base64 credentials: {e}")
                print("🔄 Will try other methods...")

        # Method 2: Try file path credentials
        if not credentials_path and DOCUMENT_AI_CREDENTIALS:
            if os.path.exists(DOCUMENT_AI_CREDENTIALS):
                credentials_path = DOCUMENT_AI_CREDENTIALS
                print(f"✅ Using credentials file: {credentials_path}")
            else:
                print(f"⚠️ Credentials file not found: {DOCUMENT_AI_CREDENTIALS}")

        # Method 3: Try Application Default Credentials
        if not credentials_path:
            try:
                from google.auth import default
                credentials, project = default()
                print("✅ Using Application Default Credentials")
                # Don't set credentials_path - let Google Auth handle it
            except Exception as e:
                print(f"⚠️ Application Default Credentials not available: {e}")
                print("🔄 Will use fallback authentication")
        
        # Initialize processor with Document AI support
        processor = DocumentProcessor(
            api_key=API_KEY,
            document_ai_endpoint=DOCUMENT_AI_ENDPOINT,
            document_ai_credentials=credentials_path
        )
        print("✅ DocumentProcessor initialized successfully")
        
        # Check if Document AI service is available
        if processor.document_ai_service:
            print("✅ Document AI service is available")
        else:
            print("⚠️ Document AI service is not available - check credentials")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start DocumentProcessor: {e}")
        import traceback
        traceback.print_exc()
        processor = None
        return False

# Initialize on startup
initialize_processor()
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------------
# In-memory job registry with session persistence for long-running processes
# --------------------------------------------------------------------------
jobs: dict[str, asyncio.Queue[str]] = {}        # log lines per job-id
job_metadata: dict[str, dict] = {}              # job metadata for persistence
job_start_times: dict[str, float] = {}          # track job start times
job_results: dict[str, dict] = {}               # store completed results for retrieval


# --------------------------------------------------------------------------
# POST /predict  – upload PDF, return {"job_id": "..."}
# --------------------------------------------------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    processing_mode: str = Form("single_deed"),  # FIXED: Use Form() for FormData
    splitting_strategy: str = Form("document_ai")  # Using Document AI as default
):
    # Try to initialize processor if it's None
    if processor is None:
        print("⚠️ Processor not initialized, attempting to initialize...")
        if not initialize_processor():
            raise HTTPException(
                status_code=500, 
                detail="Model not initialised. Check server logs for details."
            )

    # Memory monitoring at start
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"💾 Starting job - Memory usage: {initial_memory:.1f} MB")

    # save upload to a temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    log_q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)  # Reduced queue size to prevent memory buildup
    jobs[job_id] = log_q
    
    # Initialize job metadata
    job_metadata[job_id] = {
        "filename": file.filename,
        "processing_mode": processing_mode,
        "splitting_strategy": splitting_strategy,
        "start_time": time.time(),
        "status": "processing",
        "pages_processed": 0,
        "total_pages": 0,
        "initial_memory_mb": initial_memory
    }
    job_start_times[job_id] = time.time()

    def logger(msg: str):
        try:
            log_q.put_nowait(msg)
        except asyncio.QueueFull:
            # If queue is full, remove oldest message and add new one
            try:
                log_q.get_nowait()
                log_q.put_nowait(msg)
            except:
                pass  # Ignore if still can't add
    
    # run pipeline in a background thread so we don't block the event loop
    def run():
        try:
            print(f"🚀 Background thread started for job {job_id}")
            log_q.put_nowait("🚀 Background thread started")
            
            with redirect_stdout(QueueWriter(log_q)):
                print(f"🎯 Processing mode: '{processing_mode}'")
                print(f"📁 File: {file.filename}")
                print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"💾 Initial memory: {initial_memory:.1f} MB")
                
                if processing_mode == "single_deed":
                    print("📄 Using single deed processing")
                    # Single deed processing doesn't use splitting strategy
                    result = processor.process_document(tmp_path)
                    # Store result for potential retrieval
                    job_results[job_id] = result
                    log_q.put_nowait(f"__RESULT__{json.dumps(result)}")
                    
                elif processing_mode == "multi_deed":
                    print(f"📑 Using multi-deed processing with strategy: '{splitting_strategy}'")
                    log_q.put_nowait("🚀 Starting Document AI Smart Chunking...")
                    try:
                        # Check if processor is still valid
                        if processor is None:
                            raise Exception("Processor became None during processing")
                        
                        log_q.put_nowait("🔧 Initializing Document AI processing...")
                        deed_results = processor.process_multi_deed_document(
                            tmp_path, 
                            strategy=splitting_strategy
                        )
                        
                        # Validate results
                        if not isinstance(deed_results, list):
                            raise Exception(f"Expected list of results, got {type(deed_results)}")
                        
                        # Wrap results in expected structure
                        response = {
                            "deed_results": deed_results,
                            "total_deeds": len(deed_results),
                            "summary": {
                                "reservations_found": sum(1 for deed in deed_results if deed.get('classification') == 1)
                            }
                        }
                        # Store result for potential retrieval
                        job_results[job_id] = response
                        log_q.put_nowait(f"__RESULT__{json.dumps(response)}")
                        print(f"✅ Multi-deed processing completed successfully: {len(deed_results)} deeds")
                        
                    except Exception as e:
                        print(f"❌ Multi-deed processing error: {e}")
                        traceback.print_exc()
                        error_msg = f"Multi-deed processing failed: {str(e)}"
                        log_q.put_nowait(f"__ERROR__{error_msg}")
                        # Don't re-raise, let the finally block handle cleanup
                        
                elif processing_mode == "page_by_page":
                    print("📄 Using page-by-page processing (treating each page as a deed)")
                    log_q.put_nowait("🚀 Starting page-by-page classification...")
                    try:
                        # Check if processor is still valid
                        if processor is None:
                            raise Exception("Processor became None during processing")
                        
                        log_q.put_nowait("🔧 Processing each page individually...")
                        result = processor.process_document_page_by_page(
                            tmp_path,
                            max_samples=6,  # Fewer samples for speed
                            high_recall_mode=True
                        )
                        
                        # Store result for potential retrieval
                        job_results[job_id] = result
                        log_q.put_nowait(f"__RESULT__{json.dumps(result)}")
                        print(f"✅ Page-by-page processing completed successfully: {result['total_pages']} pages processed")
                        print(f"🎯 Pages with reservations: {result['pages_with_reservations']}")
                        
                    except Exception as e:
                        print(f"❌ Page-by-page processing error: {e}")
                        traceback.print_exc()
                        error_msg = f"Page-by-page processing failed: {str(e)}"
                        log_q.put_nowait(f"__ERROR__{error_msg}")
                        # Don't re-raise, let the finally block handle cleanup
                        
                else:
                    raise ValueError(f"Unknown processing_mode: '{processing_mode}'")
                    
        except Exception as e:
            print(f"❌ Processing error: {e}")  # This will show in Render logs
            traceback.print_exc()  # Print full stack trace
            log_q.put_nowait(f"__ERROR__{str(e)}")
        finally:
            # Update job metadata
            if job_id in job_metadata:
                job_metadata[job_id]["status"] = "completed"
                job_metadata[job_id]["end_time"] = time.time()
                job_metadata[job_id]["duration"] = time.time() - job_start_times.get(job_id, time.time())
                
                # Final memory check
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_change = final_memory - initial_memory
                job_metadata[job_id]["final_memory_mb"] = final_memory
                job_metadata[job_id]["memory_change_mb"] = memory_change
                
                print(f"💾 Final memory: {final_memory:.1f} MB (change: {memory_change:+.1f} MB)")
            
            log_q.put_nowait("__END__")  # ALWAYS send end signal
            
            # Force garbage collection to free memory
            print("🧹 Running final garbage collection...")
            gc.collect()
            
            try:
                os.remove(tmp_path)
            except (FileNotFoundError, OSError):
                pass  # Ignore file deletion errors

    try:
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        print(f"✅ Background thread started successfully for job {job_id}")
        return {"job_id": job_id}
    except Exception as e:
        print(f"❌ Failed to start background thread: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


# --------------------------------------------------------------------------
# GET /stream/<job_id>  – SSE stream of log lines + final JSON result
# --------------------------------------------------------------------------
@app.get("/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    queue = jobs[job_id]

    async def event_generator():
        last_heartbeat = time.time()
        heartbeat_interval = 5  # Send heartbeat every 5 seconds for better reliability
        session_start = time.time()
        last_memory_check = time.time()
        last_progress_update = time.time()
        
        while True:
            try:
                # Wait for message with shorter timeout for more frequent heartbeats
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
                    
                    # Progress updates every 2 minutes
                    if current_time - last_progress_update > 120:  # 2 minutes
                        try:
                            # Send progress update with job metadata
                            if job_id in job_metadata:
                                metadata = job_metadata[job_id]
                                progress_info = {
                                    "session_duration": session_duration,
                                    "status": metadata.get("status", "processing"),
                                    "pages_processed": metadata.get("pages_processed", 0),
                                    "total_pages": metadata.get("total_pages", 0)
                                }
                                yield f"data: __PROGRESS__{json.dumps(progress_info)}\n\n"
                            last_progress_update = current_time
                        except Exception as e:
                            print(f"Progress update error: {e}")
                    
                    # Memory monitoring every 3 minutes during heartbeats
                    if current_time - last_memory_check > 180:  # 3 minutes
                        try:
                            process = psutil.Process(os.getpid())
                            current_memory = process.memory_info().rss / 1024 / 1024
                            yield f"data: __MEMORY__{current_memory}\n\n"
                            last_memory_check = current_time
                            
                            # Force garbage collection if memory is high
                            if current_memory > 400:  # Lowered threshold to 400MB
                                print("🧹 Running memory-triggered garbage collection...")
                                gc.collect()
                                after_gc_memory = process.memory_info().rss / 1024 / 1024
                                memory_freed = current_memory - after_gc_memory
                                print(f"🧹 Memory-triggered GC freed: {memory_freed:.1f} MB")
                                yield f"data: __MEMORY_GC__{after_gc_memory}|{memory_freed}\n\n"
                        except Exception as e:
                            print(f"Memory monitoring error: {e}")
                continue
                
            except Exception as e:
                # Send error and end
                yield f"data: __ERROR__Stream error: {str(e)}\n\n"
                break
        
        # Delay cleanup to allow client to receive final messages
        await asyncio.sleep(2)
        
        # cleanup (but preserve results for potential retrieval)
        if job_id in jobs:
            del jobs[job_id]
        # Keep job_metadata and job_results for potential retrieval
        # They will be cleaned up by a background task or on server restart

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "X-Accel-Buffering": "no"  # Disable nginx buffering for real-time streaming
        }
    )


# --------------------------------------------------------------------------
# GET /job-status/<job_id>  – Get job status and metadata
# --------------------------------------------------------------------------
@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_metadata:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    
    metadata = job_metadata[job_id].copy()
    if "start_time" in metadata:
        metadata["elapsed_time"] = time.time() - metadata["start_time"]
    
    return metadata


# --------------------------------------------------------------------------
# GET /job-result/<job_id>  – Get completed job result
# --------------------------------------------------------------------------
@app.get("/job-result/{job_id}")
async def get_job_result(job_id: str):
    """Get the result of a completed job"""
    if job_id not in job_results:
        # Check if job is still processing
        if job_id in job_metadata:
            status = job_metadata[job_id].get("status", "unknown")
            if status == "processing":
                raise HTTPException(status_code=202, detail="Job still processing")
            else:
                raise HTTPException(status_code=404, detail="Job result not found")
        else:
            raise HTTPException(status_code=404, detail="Unknown job_id")
    
    return job_results[job_id]


# --------------------------------------------------------------------------
# GET /health  – Health check endpoint
# --------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is working"""
    try:
        return {
            "status": "healthy",
            "processor_initialized": processor is not None,
            "api_key_present": bool(API_KEY),
            "job_endpoints_available": JOB_ENDPOINTS_AVAILABLE,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# --------------------------------------------------------------------------
# GET /debug  – Debug information endpoint
# --------------------------------------------------------------------------
@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system status"""
    try:
        import sys
        import platform
        
        # Check imports
        imports_status = {}
        for module_name in ['fitz', 'psutil', 'anthropic', 'PIL', 'sklearn', 'numpy']:
            try:
                __import__(module_name)
                imports_status[module_name] = "✅ OK"
            except ImportError as e:
                imports_status[module_name] = f"❌ {str(e)}"
        
        # System info
        system_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor_initialized": processor is not None,
            "api_key_present": bool(API_KEY),
            "active_jobs": len(jobs),
            "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 if 'psutil' in imports_status else "N/A"
        }
        
        return {
            "imports": imports_status,
            "system": system_info,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}

# --------------------------------------------------------------------------
# GET /memory-status  – Get current memory usage
# --------------------------------------------------------------------------
@app.get("/memory-status")
async def get_memory_status():
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
    except Exception as e:
        return {"error": str(e)}


# ——————————————————————————————————————————————
# helper: stream any print() output into our asyncio queue
# ——————————————————————————————————————————————
class QueueWriter(io.TextIOBase):
    def __init__(self, q: asyncio.Queue[str]):
        self.q = q
    def write(self, s: str):
        # split so multi-line prints are handled
        for line in s.rstrip().splitlines():
            if line:
                try:
                    self.q.put_nowait(line)
                except asyncio.QueueFull:
                    # If queue is full, remove oldest message and add new one
                    try:
                        self.q.get_nowait()
                        self.q.put_nowait(line)
                    except:
                        pass  # Ignore if still can't add
    def flush(self):            # required by TextIOBase
        pass


# --------------------------------------------------------------------------
# GET /test-jobs  – Test job system endpoint
# --------------------------------------------------------------------------
@app.get("/test-jobs")
async def test_job_system():
    """Test endpoint to verify job system is working"""
    try:
        # Test built-in job manager
        jobs = job_manager.list_jobs()
        
        return {
            "status": "success",
            "message": "Built-in job system is working correctly",
            "job_endpoints_available": True,
            "total_jobs": len(jobs),
            "active_jobs": len([j for j in jobs if j['status'] == 'running'])
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Job system test failed: {str(e)}",
            "job_endpoints_available": False
        }

# --------------------------------------------------------------------------
# Job Endpoints (Built-in)
# --------------------------------------------------------------------------

@app.post("/jobs/create")
async def create_long_running_job(
    file: UploadFile = File(...),
    processing_mode: str = Form("multi_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """
    Create a long-running processing job that can run for 8+ hours
    
    This endpoint creates a job instead of processing immediately,
    avoiding the 22-minute timeout limit of web services.
    """
    # Debug logging to see what parameters are received
    print(f"🔍 DEBUG: processing_mode = '{processing_mode}'")
    print(f"🔍 DEBUG: splitting_strategy = '{splitting_strategy}'")
    print(f"🔍 DEBUG: filename = '{file.filename}'")
    print(f"🔍 DEBUG: content_type = '{file.content_type}'")
    print(f"🔍 DEBUG: file size = {file.size if hasattr(file, 'size') else 'unknown'}")
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Save uploaded file temporarily
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        print(f"📁 Received file: {file.filename} (size: {len(contents)} bytes)")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        print(f"💾 Saved to temp file: {tmp_path}")
        
        # Create job
        print(f"🔄 Creating job for {file.filename}...")
        try:
            job_id = job_manager.create_job(
                file.filename,
                processing_mode,
                splitting_strategy
            )
            print(f"🎯 Created job: {job_id}")
        except Exception as job_error:
            print(f"❌ Job creation failed: {job_error}")
            import traceback
            traceback.print_exc()
            raise
        
        # Start processing in background
        def process_job():
            try:
                print(f"🚀 Starting job {job_id} processing...")
                job_manager.update_job_status(job_id, JobStatus.RUNNING)
                
                # Check if file exists and has content
                if not os.path.exists(tmp_path):
                    raise ValueError(f"Temporary file not found: {tmp_path}")
                
                file_size = os.path.getsize(tmp_path)
                if file_size == 0:
                    raise ValueError(f"File is empty: {tmp_path}")
                
                print(f"📁 Processing file: {tmp_path} (size: {file_size} bytes)")
                
                # Process the document
                if processing_mode == "single_deed":
                    # Single deed processing doesn't use splitting strategy
                    result = processor.process_document(tmp_path)
                elif processing_mode == "multi_deed":
                    result = processor.process_multi_deed_document(
                        tmp_path, 
                        strategy=splitting_strategy
                    )
                elif processing_mode == "page_by_page":
                    result = processor.process_document_page_by_page(
                        tmp_path,
                        max_samples=6,  # Fewer samples for speed
                        high_recall_mode=True
                    )
                else:
                    raise ValueError(f"Unknown processing_mode: {processing_mode}")
                
                print(f"✅ Job {job_id} completed successfully")
                job_manager.update_job_status(job_id, JobStatus.COMPLETED, result=result)
                
            except Exception as e:
                print(f"❌ Job {job_id} failed: {e}")
                import traceback
                traceback.print_exc()
                job_manager.update_job_status(job_id, JobStatus.FAILED, error=str(e))
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        print(f"🧹 Cleaned up temp file: {tmp_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Failed to cleanup temp file: {cleanup_error}")
        
        # Start background thread with a small delay to ensure response is sent first
        def delayed_start():
            time.sleep(1)  # Small delay to ensure HTTP response is sent
            process_job()
        
        thread = threading.Thread(target=delayed_start, daemon=True)
        thread.start()
        
        return {
            "job_id": job_id,
            "status": "created",
            "message": "Long-running job created successfully. Use /jobs/{job_id}/status to monitor progress.",
            "estimated_duration": "Up to 8 hours for large documents"
        }
        
    except Exception as e:
        print(f"❌ Job creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the current status of a processing job"""
    from fastapi.responses import JSONResponse
    
    job = job_manager.get_job(job_id)
    if not job:
        return JSONResponse({"detail": "Job not found"}, status_code=404, headers={"Access-Control-Allow-Origin": "*"})

    # Ensure Enum is serialized to its value
    data = asdict(job)
    if isinstance(job.status, JobStatus):
        data["status"] = job.status.value
    
    return JSONResponse(data, headers={"Access-Control-Allow-Origin": "*"})

@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the result of a completed job"""
    from fastapi.responses import JSONResponse
    
    job = job_manager.get_job(job_id)
    if not job:
        return JSONResponse({"detail": "Job not found"}, status_code=404, headers={"Access-Control-Allow-Origin": "*"})

    if job.status != JobStatus.COMPLETED:
        status_value = job.status.value if isinstance(job.status, JobStatus) else str(job.status)
        return JSONResponse({"detail": f"Job not completed yet. Current status: {status_value}"}, status_code=202, headers={"Access-Control-Allow-Origin": "*"})

    result_data = job.result if isinstance(job.result, dict) else {"result": job.result}
    return JSONResponse(result_data, headers={"Access-Control-Allow-Origin": "*"})

@app.get("/jobs/")
async def list_jobs():
    """List all jobs"""
    jobs = job_manager.list_jobs()
    return {"jobs": jobs, "total": len(jobs)}

@app.get("/jobs/health")
async def jobs_health_check():
    """Health check for the job system"""
    return {
        "status": "healthy",
        "job_manager_initialized": job_manager is not None,
        "active_jobs": len([j for j in job_manager.jobs.values() if j.status == JobStatus.RUNNING]),
        "total_jobs": len(job_manager.jobs)
    }

# --------------------------------------------------------------------------
# Deed Tracking Endpoints
# --------------------------------------------------------------------------

@app.get("/deed-tracking/sessions")
async def list_deed_tracking_sessions():
    """List all deed tracking sessions"""
    try:
        if processor is None or not hasattr(processor, 'deed_tracker'):
            raise HTTPException(status_code=500, detail="Deed tracker not available")
        
        sessions = processor.deed_tracker.list_sessions()
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.get("/deed-tracking/sessions/{session_id}")
async def get_deed_tracking_session(session_id: str):
    """Get details for a specific deed tracking session"""
    try:
        if processor is None or not hasattr(processor, 'deed_tracker'):
            raise HTTPException(status_code=500, detail="Deed tracker not available")
        
        summary = processor.deed_tracker.get_session_summary(session_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.get("/deed-tracking/sessions/{session_id}/boundaries")
async def get_deed_boundaries(session_id: str):
    """Get deed boundaries for a specific session"""
    try:
        if processor is None or not hasattr(processor, 'deed_tracker'):
            raise HTTPException(status_code=500, detail="Deed tracker not available")
        
        # Load session to get boundaries
        session = processor.deed_tracker._load_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "deed_boundaries": [
                {
                    "deed_number": b.deed_number,
                    "pages": b.pages,
                    "confidence": b.confidence,
                    "page_range": b.page_range,
                    "detected_at": b.detected_at,
                    "splitting_strategy": b.splitting_strategy,
                    "document_ai_used": b.document_ai_used
                }
                for b in session.deed_boundaries
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get boundaries: {str(e)}")

@app.get("/deed-tracking/sessions/{session_id}/results")
async def get_deed_classification_results(session_id: str):
    """Get classification results for a specific session"""
    try:
        if processor is None or not hasattr(processor, 'deed_tracker'):
            raise HTTPException(status_code=500, detail="Deed tracker not available")
        
        # Load session to get results
        session = processor.deed_tracker._load_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "classification_results": [
                {
                    "deed_number": r.deed_number,
                    "classification": r.classification,
                    "confidence": r.confidence,
                    "pages_in_deed": r.pages_in_deed,
                    "processing_time": r.processing_time,
                    "error": r.error,
                    "deed_boundary_info": r.deed_boundary_info
                }
                for r in session.classification_results
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")

@app.post("/jobs/test")
async def test_job_creation():
    """Test endpoint to verify job system works without file upload"""
    try:
        # Create a test job without file processing
        job_id = job_manager.create_job(
            "test.pdf",
            "multi_deed",
            "smart_detection"
        )
        
        # Simulate job completion
        job_manager.update_job_status(job_id, JobStatus.COMPLETED, result={
            "classification": 1,
            "confidence": 0.95,
            "message": "Test job completed successfully"
        })
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": "Test job created and completed successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Test job failed: {str(e)}"
        }