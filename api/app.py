from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, traceback
from typing import List
import asyncio, threading, uuid, json, time
from fastapi.responses import StreamingResponse
import io, sys
from contextlib import redirect_stdout
import gc  # For garbage collection

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
try:
    processor = DocumentProcessor(API_KEY)
except Exception as e:
    print("❌ Failed to start DocumentProcessor:", e)
    processor = None
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
    if processor is None:
        raise HTTPException(status_code=500, detail="Model not initialised")

    # save upload to a temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    log_q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)  # Increased queue size for long sessions
    jobs[job_id] = log_q
    
    # Initialize job metadata
    job_metadata[job_id] = {
        "filename": file.filename,
        "processing_mode": processing_mode,
        "splitting_strategy": splitting_strategy,
        "start_time": time.time(),
        "status": "processing",
        "pages_processed": 0,
        "total_pages": 0
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
                
                if processing_mode == "single_deed":
                    print("📄 Using single deed processing")
                    result = processor.process_document(tmp_path)
                    log_q.put_nowait(f"__RESULT__{json.dumps(result)}")
                
                elif processing_mode == "multi_deed":
                    print(f"📑 Using multi-deed processing with strategy: '{splitting_strategy}'")
                    deed_results = processor.process_multi_deed_document(
                        tmp_path, 
                        strategy=splitting_strategy
                    )
                    
                    # Wrap results in expected structure
                    response = {
                        "deed_results": deed_results,
                        "total_deeds": len(deed_results),
                        "summary": {
                            "reservations_found": sum(1 for deed in deed_results if deed.get('classification') == 1)
                        }
                    }
                    log_q.put_nowait(f"__RESULT__{json.dumps(response)}")
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
            
            log_q.put_nowait("__END__")  # ALWAYS send end signal
            
            # Force garbage collection to free memory
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
                    
                    # Force garbage collection periodically
                    if int(session_duration) % 300 == 0:  # Every 5 minutes
                        gc.collect()
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