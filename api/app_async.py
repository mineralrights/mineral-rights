from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import time
import json
import uuid
import asyncio
from typing import Dict, Optional
import psutil
from google.cloud import storage, tasks_v2
from google.cloud import firestore
import base64

# Import your pipeline and job queue
from src.mineral_rights.document_classifier import DocumentProcessor
from api.job_queue import JobQueue

# Initialize FastAPI app
app = FastAPI(title="Mineral-Rights API - Async Job Pattern")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mineral-rights-perc6q1ij-lauragomezjurados-projects.vercel.app",
        "http://localhost:3000",
        "https://*.vercel.app"  # Allow all Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# Global processor
processor = None

# Initialize Google Cloud clients
storage_client = None
tasks_client = None
firestore_client = None
job_queue = None

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
                # Decode base64 credentials
                credentials_json = base64.b64decode(google_credentials_base64).decode('utf-8')
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                temp_file.write(credentials_json)
                temp_file.close()
                credentials_path = temp_file.name
                print(f"✅ Created temporary credentials file from base64")
            except Exception as e:
                print(f"⚠️ Failed to decode base64 credentials: {e}")
        
        # Initialize processor
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

def initialize_cloud_clients():
    global storage_client, tasks_client, firestore_client, job_queue
    try:
        storage_client = storage.Client()
        tasks_client = tasks_v2.CloudTasksClient()
        firestore_client = firestore.Client()
        job_queue = JobQueue()
        print("✅ Google Cloud clients initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Google Cloud clients: {e}")
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Mineral Rights API...")
    initialize_cloud_clients()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/heartbeat")
async def heartbeat():
    """Simple heartbeat endpoint"""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mineral Rights API - Async Job Pattern",
        "version": "2.0",
        "endpoints": {
            "health": "/health",
            "heartbeat": "/heartbeat",
            "create_job": "POST /jobs",
            "get_job": "GET /jobs/{job_id}",
            "list_jobs": "GET /jobs"
        }
    }

@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    processing_mode: str = Form("single_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """Create a new processing job and return job ID immediately"""
    print(f"🔍 Creating job - Mode: {processing_mode}, File: {file.filename}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Upload file to Cloud Storage
        bucket_name = os.getenv("GCS_BUCKET_NAME", "mineral-rights-storage")
        file_path = f"uploads/{job_id}/{file.filename}"
        
        # Read file contents
        contents = await file.read()
        
        # Upload to GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        blob.upload_from_string(contents, content_type=file.content_type)
        
        print(f"✅ File uploaded to GCS: gs://{bucket_name}/{file_path}")
        
        # Create job record in Firestore
        job_data = {
            "job_id": job_id,
            "filename": file.filename,
            "file_path": file_path,
            "processing_mode": processing_mode,
            "splitting_strategy": splitting_strategy,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "progress": 0,
            "logs": []
        }
        
        # Store in Firestore
        doc_ref = firestore_client.collection("jobs").document(job_id)
        doc_ref.set(job_data)
        
        # Queue the job for processing
        await job_queue.queue_job(job_id, file_path, processing_mode, splitting_strategy)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Job created successfully. Use the job_id to check status."
        }
        
    except Exception as e:
        print(f"❌ Failed to create job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and results"""
    try:
        doc_ref = firestore_client.collection("jobs").document(job_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = doc.to_dict()
        
        # Remove sensitive data
        if "file_path" in job_data:
            del job_data["file_path"]
        
        return job_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to get job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")

@app.get("/jobs")
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """List recent jobs"""
    try:
        query = firestore_client.collection("jobs").order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
        
        if status:
            query = query.where("status", "==", status)
        
        docs = query.stream()
        jobs = []
        
        for doc in docs:
            job_data = doc.to_dict()
            # Remove sensitive data
            if "file_path" in job_data:
                del job_data["file_path"]
            jobs.append(job_data)
        
        return {"jobs": jobs, "count": len(jobs)}
        
    except Exception as e:
        print(f"❌ Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
