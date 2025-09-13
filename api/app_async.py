#!/usr/bin/env python3
"""
Async Mineral Rights API
Uses Cloud Run Jobs for long-running processing tasks
"""

import os
import time
import uuid
import asyncio
from typing import Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import firestore
from google.cloud import storage

# Initialize FastAPI app
app = FastAPI(title="Mineral Rights API - Async", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Initialize Google Cloud clients
db = firestore.Client()
storage_client = storage.Client()

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'mineral-rights-app')
LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'mineral-rights-documents')
QUEUE_NAME = os.getenv('TASK_QUEUE_NAME', 'mineral-rights-queue')

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mineral Rights API - Async Version",
        "version": "2.0",
        "endpoints": {
            "health": "/health",
            "create_job": "POST /predict",
            "get_job": "GET /jobs/{job_id}",
            "list_jobs": "GET /jobs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "api_key_present": bool(os.getenv("ANTHROPIC_API_KEY")),
        "document_ai_endpoint_present": bool(os.getenv("DOCUMENT_AI_ENDPOINT")),
        "google_credentials_present": bool(os.getenv("GOOGLE_CREDENTIALS_BASE64")),
        "gcs_bucket_present": bool(os.getenv("GCS_BUCKET_NAME"))
    }

@app.post("/predict")
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
        # Create job data
        job_data = {
            "job_id": job_id,
            "filename": file.filename,
            "processing_mode": processing_mode,
            "splitting_strategy": splitting_strategy,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "progress": 0,
            "logs": [f"📋 Job created at {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        }
        
        # Save job to Firestore
        job_ref = db.collection('jobs').document(job_id)
        job_ref.set(job_data)
        
        # Upload file to Cloud Storage
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"uploads/{job_id}/{file.filename}"
        blob = bucket.blob(blob_name)
        
        # Read file content
        file_content = await file.read()
        blob.upload_from_string(file_content, content_type=file.content_type)
        
        print(f"📁 File uploaded to Cloud Storage: {blob_name}")
        
        # Update job with file info
        job_ref.update({
            "file_path": blob_name,
            "file_size": len(file_content),
            "logs": firestore.ArrayUnion([f"📁 File uploaded to Cloud Storage"])
        })
        
        # Mark job as ready for processing
        job_ref.update({
            "status": "queued",
            "logs": firestore.ArrayUnion([f"📤 Job queued for processing"])
        })
        
        print(f"✅ Job {job_id} created and queued")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Job created successfully. Use the job_id to check status."
        }
        
    except Exception as e:
        print(f"❌ Failed to create job: {e}")
        # Clean up job if it was created
        try:
            db.collection('jobs').document(job_id).delete()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and results"""
    try:
        job_ref = db.collection('jobs').document(job_id)
        job_doc = job_ref.get()
        
        if not job_doc.exists:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = job_doc.to_dict()
        
        # Remove sensitive data
        if 'file_path' in job_data:
            del job_data['file_path']
        
        return job_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to get job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")

@app.get("/jobs")
async def list_jobs(limit: int = 10):
    """List recent jobs"""
    try:
        jobs_ref = db.collection('jobs')
        jobs = jobs_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit).stream()
        
        job_list = []
        for job in jobs:
            job_data = job.to_dict()
            # Remove sensitive data
            if 'file_path' in job_data:
                del job_data['file_path']
            job_list.append(job_data)
        
        return {
            "jobs": job_list,
            "count": len(job_list)
        }
        
    except Exception as e:
        print(f"❌ Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@app.get("/heartbeat")
async def heartbeat():
    """Simple heartbeat endpoint"""
    return {"status": "alive", "timestamp": time.time()}

@app.get("/test")
async def test():
    """Test endpoint to verify async API is working"""
    return {
        "message": "Async API is working",
        "timestamp": time.time(),
        "environment": {
            "gcs_bucket": os.getenv("GCS_BUCKET_NAME"),
            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
            "location": os.getenv("GOOGLE_CLOUD_LOCATION")
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)