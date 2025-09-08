from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, traceback
from typing import List
import asyncio, threading, uuid, json, time
from fastapi.responses import StreamingResponse
import io, sys
from contextlib import redirect_stdout
import gc  # For garbage collection
import psutil  # For memory monitoring

#  import your pipeline ----------------------------------------------
from src.mineral_rights.document_classifier import DocumentProcessor
# ---------------------------------------------------------------------------


## TESTING NEW DEPLOYMENT 
app = FastAPI(title="Mineral-Rights API")

# More permissive CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3000", 
        "https://mineral-rights-4o8gr9h79-lauragomezjurados-projects.vercel.app",
        # Allow all Vercel domains
        "*"  # Temporarily allow all origins for debugging
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2️⃣  --- initialise once at startup ----------------------------------------
API_KEY = os.getenv("ANTHROPIC_API_KEY")  # or whatever key the processor needs
processor = None

def initialize_processor():
    global processor
    try:
        print("🔧 Initializing DocumentProcessor...")
        print(f"API Key present: {'Yes' if API_KEY else 'No'}")
        
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
        
        # Initialize processor
        processor = DocumentProcessor(API_KEY)
        print("✅ DocumentProcessor initialized successfully")
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


# --------------------------------------------------------------------------
# POST /predict  – upload PDF, return {"job_id": "..."}
# --------------------------------------------------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    processing_mode: str = Form("single_deed"),  # FIXED: Use Form() for FormData
    splitting_strategy: str = Form("smart_detection")  # FIXED: Use Form() for FormData
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
            with redirect_stdout(QueueWriter(log_q)):
                print(f"🎯 Processing mode: '{processing_mode}'")
                print(f"📁 File: {file.filename}")
                print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"💾 Initial memory: {initial_memory:.1f} MB")
                
                if processing_mode == "single_deed":
                    print("📄 Using single deed processing")
                    result = processor.process_document(tmp_path)
                    log_q.put_nowait(f"__RESULT__{json.dumps(result)}")
                
                elif processing_mode == "multi_deed":
                    print(f"📑 Using multi-deed processing with strategy: '{splitting_strategy}'")
                    try:
                        # Check if processor is still valid
                        if processor is None:
                            raise Exception("Processor became None during processing")
                        
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
                        log_q.put_nowait(f"__RESULT__{json.dumps(response)}")
                        print(f"✅ Multi-deed processing completed successfully: {len(deed_results)} deeds")
                        
                    except Exception as e:
                        print(f"❌ Multi-deed processing error: {e}")
                        traceback.print_exc()
                        error_msg = f"Multi-deed processing failed: {str(e)}"
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

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


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
        heartbeat_interval = 10  # Send heartbeat every 10 seconds for very long sessions
        session_start = time.time()
        last_memory_check = time.time()
        
        while True:
            try:
                # Wait for message with timeout
                line = await asyncio.wait_for(queue.get(), timeout=3.0)  # Increased timeout
                
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
                    
                    # Memory monitoring every 5 minutes during heartbeats
                    if current_time - last_memory_check > 300:  # 5 minutes
                        try:
                            process = psutil.Process(os.getpid())
                            current_memory = process.memory_info().rss / 1024 / 1024
                            yield f"data: __MEMORY__{current_memory}\n\n"
                            last_memory_check = current_time
                            
                            # Force garbage collection if memory is high
                            if current_memory > 500:  # More than 500MB
                                print("🧹 Running memory-triggered garbage collection...")
                                gc.collect()
                                after_gc_memory = process.memory_info().rss / 1024 / 1024
                                memory_freed = current_memory - after_gc_memory
                                print(f"🧹 Memory-triggered GC freed: {memory_freed:.1f} MB")
                        except Exception as e:
                            print(f"Memory monitoring error: {e}")
                continue
                
            except Exception as e:
                # Send error and end
                yield f"data: __ERROR__Stream error: {str(e)}\n\n"
                break
        
        # cleanup
        if job_id in jobs:
            del jobs[job_id]
        if job_id in job_metadata:
            del job_metadata[job_id]
        if job_id in job_start_times:
            del job_start_times[job_id]

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