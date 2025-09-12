from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import tempfile
import os
import time
import json
import uuid
import asyncio
from typing import Dict, Optional
import psutil

# Initialize FastAPI app
app = FastAPI(title="Mineral-Rights API - Simple Async Version")

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

# Simple in-memory job storage (for demo purposes)
jobs: Dict[str, Dict] = {}

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
        "message": "Mineral Rights API - Simple Async Version",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "heartbeat": "/heartbeat",
            "create_job": "POST /jobs",
            "get_job": "GET /jobs/{job_id}",
            "list_jobs": "GET /jobs"
        }
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    processing_mode: str = Form("single_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """Create a new processing job and return job ID immediately"""
    print(f"🔍 Creating job - Mode: {processing_mode}, File: {file.filename}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    try:
        # For now, just simulate the job creation
        # In a real implementation, this would upload to Cloud Storage
        # and queue the job for processing
        
        job_data = {
            "job_id": job_id,
            "filename": file.filename,
            "processing_mode": processing_mode,
            "splitting_strategy": splitting_strategy,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "progress": 0,
            "logs": []
        }
        
        # Store in memory (in production, this would be Firestore)
        jobs[job_id] = job_data
        
        # Simulate queuing the job
        print(f"✅ Job {job_id} created and queued")
        
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
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = jobs[job_id]
        
        # Simulate job progress
        if job_data["status"] == "queued":
            # Simulate processing
            job_data["status"] = "processing"
            job_data["progress"] = 50
            job_data["updated_at"] = time.time()
        elif job_data["status"] == "processing" and job_data["progress"] < 100:
            # Simulate completion
            job_data["status"] = "completed"
            job_data["progress"] = 100
            job_data["updated_at"] = time.time()
            job_data["result"] = {
                "message": "Processing completed successfully",
                "deeds_found": 1,
                "mineral_rights_detected": True
            }
        
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
        job_list = list(jobs.values())
        
        if status:
            job_list = [job for job in job_list if job["status"] == status]
        
        # Sort by creation time (newest first)
        job_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"jobs": job_list[:limit], "count": len(job_list[:limit])}
        
    except Exception as e:
        print(f"❌ Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@app.get("/stream/{job_id}")
async def stream_job(job_id: str):
    """Stream job progress via Server-Sent Events"""
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = jobs[job_id]
        
        async def generate_events():
            # Send initial status
            yield f"data: Job {job_id} started\n\n"
            await asyncio.sleep(1)
            
            # Simulate processing steps
            yield f"data: 📄 Processing file: {job_data['filename']}\n\n"
            await asyncio.sleep(2)
            
            yield f"data: 🔍 Analyzing document structure\n\n"
            await asyncio.sleep(2)
            
            yield f"data: 🤖 Running AI classification\n\n"
            await asyncio.sleep(3)
            
            # Simulate result
            result = {
                "classification": 1,  # 1 = has reservation, 0 = no reservation
                "confidence": 0.85,
                "detailed_samples": [{
                    "reasoning": "Document contains mineral rights reservation clause in paragraph 3."
                }]
            }
            
            yield f"data: __RESULT__{json.dumps(result)}\n\n"
            await asyncio.sleep(1)
            
            yield f"data: __END__\n\n"
        
        return StreamingResponse(
            generate_events(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to stream job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stream job: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)