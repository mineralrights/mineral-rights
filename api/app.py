from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, traceback

# 1️⃣  --- import your pipeline ----------------------------------------------
from src.mineral_rights.document_classifier import DocumentProcessor
# ---------------------------------------------------------------------------

app = FastAPI(title="Mineral-Rights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receive one PDF, run the ML pipeline, return JSON"""
    if processor is None:
        raise HTTPException(status_code=500, detail="Model not initialised")

    # Save upload to a temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 3️⃣  --- run the pipeline ------------------------------------------
        result = processor.process_document(tmp_path)
        # Map numeric class → text label
        label = "has_reservation" if result["classification"] == 1 else "no_reservation"

        return {
            "prediction": label,
            "explanation": f"Confidence {result['confidence']:.2f}",
            "confidence": result["confidence"],
            # include anything else you want in the CSV later
        }
        # -------------------------------------------------------------------
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)