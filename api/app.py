from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, traceback
from typing import List
import asyncio, threading, uuid, json
from fastapi.responses import StreamingResponse
import io, sys
from contextlib import redirect_stdout

# 1️⃣  --- import your pipeline ----------------------------------------------
from src.mineral_rights.document_classifier import DocumentProcessor
# ---------------------------------------------------------------------------

app = FastAPI(title="Mineral-Rights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://<your-frontend>.vercel.app"
    ],
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
async def predict(file: UploadFile = File(...)):
    if processor is None:
        raise HTTPException(status_code=500, detail="Model not initialised")

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
            # anything printed during processing goes to the queue
            with redirect_stdout(QueueWriter(log_q)):
                result = processor.process_document(tmp_path)
            log_q.put_nowait(f"__RESULT__{json.dumps(result)}")
        except Exception as e:
            log_q.put_nowait(f"__ERROR__{str(e)}")
        finally:
            log_q.put_nowait("__END__")
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

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