from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, traceback
from typing import List
import asyncio, threading, uuid, json
from fastapi.responses import StreamingResponse
import io, sys
from contextlib import redirect_stdout

#  import your pipeline ----------------------------------------------
from src.mineral_rights.document_classifier import DocumentProcessor
# ---------------------------------------------------------------------------


## TESTING NEW DEPLOYMENT 
app = FastAPI(title="Mineral-Rights API")

def is_cors_allowed(origin: str) -> bool:
    allowed = [
        "http://localhost:3000",
        "https://localhost:3000",
    ]
    if origin and (origin.endswith(".vercel.app") or origin in allowed):
        return True
    return False

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Regex pattern for all Vercel
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
# In-memory job registry  (works fine for a single container)
# --------------------------------------------------------------------------
jobs: dict[str, asyncio.Queue[str]] = {}        # log lines per job-id


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

    # DEBUG: Log received parameters
    print(f"🔍 DEBUG - Received parameters:")
    print(f"  - processing_mode: '{processing_mode}'")
    print(f"  - splitting_strategy: '{splitting_strategy}'")
    print(f"  - file: {file.filename}")

    # save upload to a temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    log_q: asyncio.Queue[str] = asyncio.Queue()
    jobs[job_id] = log_q

    def logger(msg: str):
        log_q.put_nowait(msg)
    
    # run pipeline in a background thread so we don't block the event loop
    def run():
        try:
            with redirect_stdout(QueueWriter(log_q)):
                print(f"🎯 Processing mode: '{processing_mode}'")
                
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
                    log_q.put_nowait(f"__RESULT__{json.dumps(deed_results)}")
                else:
                    raise ValueError(f"Unknown processing_mode: '{processing_mode}'")
                    
        except Exception as e:
            print(f"❌ Processing error: {e}")  # This will show in Render logs
            traceback.print_exc()  # Print full stack trace
            log_q.put_nowait(f"__ERROR__{str(e)}")
        finally:
            log_q.put_nowait("__END__")  # ALWAYS send end signal
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
        while True:
            line = await queue.get()
            # standard SSE format  (reconnects handled automatically by client)
            yield f"data: {line}\n\n"
            if line == "__END__":
                break
        # cleanup
        del jobs[job_id]

    return StreamingResponse(event_generator(),
                             media_type="text/event-stream")


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
                self.q.put_nowait(line)
    def flush(self):            # required by TextIOBase
        pass