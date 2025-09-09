"""
Job API Endpoints for Render Jobs Integration
============================================

This module provides FastAPI endpoints to:
1. Create long-running processing jobs
2. Monitor job status and progress
3. Retrieve job results
4. Manage job lifecycle

Add these endpoints to your existing api/app.py
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import tempfile
import os
import time
import asyncio
from typing import Optional, List
import json

# Import the job manager
from job_manager import job_manager, JobStatus

# Create router for job endpoints
job_router = APIRouter(prefix="/jobs", tags=["jobs"])

@job_router.post("/create")
async def create_long_running_job(
    file: UploadFile = File(...),
    processing_mode: str = Form("multi_deed"),
    splitting_strategy: str = Form("smart_detection")
):
    """
    Create a long-running processing job that can run for 8+ hours
    
    This endpoint creates a Render Job instead of processing immediately,
    avoiding the 22-minute timeout limit of web services.
    """
    try:
        # Save uploaded file temporarily
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Create job
        job_id = job_manager.create_job(
            tmp_path,
            processing_mode,
            splitting_strategy
        )
        
        return {
            "job_id": job_id,
            "status": "created",
            "message": "Long-running job created successfully. Use /jobs/{job_id}/status to monitor progress.",
            "estimated_duration": "Up to 8 hours for large documents"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@job_router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Get the current status of a processing job
    
    Returns job status, progress, and metadata
    """
    status = job_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status

@job_router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the result of a completed job
    
    Returns the processing results if the job has completed successfully
    """
    result = job_manager.get_job_result(job_id)
    if not result:
        # Check if job exists but isn't completed
        status = job_manager.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        elif status['status'] != 'completed':
            raise HTTPException(
                status_code=202, 
                detail=f"Job not completed yet. Current status: {status['status']}"
            )
        else:
            raise HTTPException(status_code=404, detail="Job result not found")
    
    return result

@job_router.get("/{job_id}/logs")
async def get_job_logs(job_id: str):
    """
    Get logs for a job
    
    Returns the processing logs for monitoring progress
    """
    logs = job_manager.get_job_logs(job_id)
    if logs is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"job_id": job_id, "logs": logs}

@job_router.get("/{job_id}/logs/stream")
async def stream_job_logs(job_id: str):
    """
    Stream job logs in real-time using Server-Sent Events
    
    This provides real-time updates on job progress
    """
    async def event_generator():
        last_log_count = 0
        
        while True:
            try:
                logs = job_manager.get_job_logs(job_id)
                if logs is None:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break
                
                # Send new logs
                if len(logs) > last_log_count:
                    for i in range(last_log_count, len(logs)):
                        yield f"data: {json.dumps({'log': logs[i], 'timestamp': time.time()})}\n\n"
                    last_log_count = len(logs)
                
                # Check if job is completed
                status = job_manager.get_job_status(job_id)
                if status and status['status'] in ['completed', 'failed', 'cancelled']:
                    yield f"data: {json.dumps({'status': status['status'], 'final': True})}\n\n"
                    break
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@job_router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running job
    
    Attempts to cancel a job that is currently running
    """
    success = job_manager.cancel_job(job_id)
    if not success:
        status = job_manager.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel job in status: {status['status']}"
            )
    
    return {"message": "Job cancelled successfully", "job_id": job_id}

@job_router.get("/")
async def list_jobs():
    """
    List all jobs
    
    Returns a list of all jobs with their current status
    """
    jobs = job_manager.list_jobs()
    return {"jobs": jobs, "total": len(jobs)}

@job_router.get("/{job_id}/progress")
async def get_job_progress(job_id: str):
    """
    Get detailed progress information for a job
    
    Returns progress percentage, estimated time remaining, etc.
    """
    status = job_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Calculate progress based on job status and timing
    progress_info = {
        "job_id": job_id,
        "status": status['status'],
        "created_at": status['created_at'],
        "started_at": status.get('started_at'),
        "completed_at": status.get('completed_at'),
    }
    
    if status['status'] == JobStatus.RUNNING.value and status.get('started_at'):
        elapsed = time.time() - status['started_at']
        progress_info.update({
            "elapsed_time": elapsed,
            "estimated_remaining": "Unknown (job can run up to 8 hours)",
            "progress_percentage": min(10, (elapsed / 60) * 2)  # Rough estimate
        })
    elif status['status'] == JobStatus.COMPLETED.value:
        total_time = status['completed_at'] - status['started_at']
        progress_info.update({
            "total_time": total_time,
            "progress_percentage": 100
        })
    
    return progress_info

# Health check endpoint for jobs
@job_router.get("/health")
async def jobs_health_check():
    """
    Health check for the job system
    """
    return {
        "status": "healthy",
        "job_manager_initialized": job_manager is not None,
        "active_jobs": len([j for j in job_manager.jobs.values() if j.status == JobStatus.RUNNING]),
        "total_jobs": len(job_manager.jobs)
    }
