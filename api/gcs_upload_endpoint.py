"""
GCS Upload Endpoint for Large PDFs
Handles files up to 5TB using Google Cloud Storage
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import uuid
from google.cloud import storage
from google.oauth2 import service_account
import json

# Initialize FastAPI app
app = FastAPI(title="GCS Upload API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# GCS Configuration
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "mineral-rights-pdfs-1759435410")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def get_gcs_client():
    """Initialize GCS client with credentials"""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        return storage.Client(credentials=credentials)
    else:
        # Use default credentials (ADC)
        return storage.Client()

@app.post("/upload-large")
async def upload_large_file(
    file: UploadFile = File(...),
    processing_mode: str = Form("multi_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """
    Upload large files to Google Cloud Storage
    Handles files up to 5TB (Google's limit)
    """
    print(f"🔍 Uploading large file: {file.filename}")
    
    try:
        # Initialize GCS client
        client = get_gcs_client()
        bucket = client.bucket(BUCKET_NAME)
        
        # Generate unique blob name
        file_id = str(uuid.uuid4())
        blob_name = f"uploads/{file_id}/{file.filename}"
        blob = bucket.blob(blob_name)
        
        # Upload file content to GCS
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        print(f"📁 File size: {file_size_mb:.1f}MB")
        
        # Upload with resumable upload for large files
        blob.upload_from_string(file_content, content_type=file.content_type)
        
        # Make it publicly accessible
        blob.make_public()
        
        # Get public URL
        public_url = blob.public_url
        
        print(f"✅ Upload successful: {public_url}")
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "file_size_mb": file_size_mb,
            "gcs_url": public_url,
            "blob_name": blob_name,
            "processing_mode": processing_mode,
            "splitting_strategy": splitting_strategy,
            "message": f"File uploaded successfully. Size: {file_size_mb:.1f}MB"
        }
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process-from-gcs")
async def process_from_gcs(
    gcs_url: str = Form(...),
    processing_mode: str = Form("multi_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """
    Process a file from Google Cloud Storage URL
    Downloads the file and processes it
    """
    print(f"🔍 Processing file from GCS: {gcs_url}")
    
    try:
        # Download file from GCS
        client = get_gcs_client()
        
        # Extract blob name from URL
        # URL format: https://storage.googleapis.com/bucket-name/path/to/file
        url_parts = gcs_url.split('/')
        bucket_name = url_parts[3]
        blob_name = '/'.join(url_parts[4:])
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            blob.download_to_filename(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        print(f"📁 Downloaded file to: {tmp_file_path}")
        
        # TODO: Process the file using your existing DocumentProcessor
        # This is where you'd call your multi-deed processing logic
        
        return {
            "success": True,
            "gcs_url": gcs_url,
            "local_path": tmp_file_path,
            "processing_mode": processing_mode,
            "splitting_strategy": splitting_strategy,
            "message": "File downloaded and ready for processing"
        }
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "bucket_name": BUCKET_NAME,
        "credentials_available": bool(CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH))
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
