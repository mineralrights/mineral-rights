#!/usr/bin/env python3
"""
Render Jobs Manager
==================

This module provides functionality to:
1. Trigger Render Jobs for long-running processing
2. Monitor job status and progress
3. Retrieve job results
4. Manage job lifecycle

Usage:
    from job_manager import RenderJobManager
    manager = RenderJobManager()
    job_id = manager.create_job("document.pdf", "multi_deed", "smart_detection")
    status = manager.get_job_status(job_id)
"""

import os
import sys
import json
import time
import requests
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobInfo:
    id: str
    filename: str
    processing_mode: str
    splitting_strategy: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = None

class RenderJobManager:
    """
    Manager for Render Jobs - handles job creation, monitoring, and result retrieval
    """
    
    def __init__(self, render_api_key: Optional[str] = None):
        self.render_api_key = render_api_key or os.getenv("RENDER_API_KEY")
        self.render_service_id = os.getenv("RENDER_SERVICE_ID")  # Your job service ID
        self.base_url = "https://api.render.com/v1"
        self.jobs: Dict[str, JobInfo] = {}
        
    def create_job(self, pdf_file_path: str, processing_mode: str = "multi_deed", 
                   splitting_strategy: str = "smart_detection") -> str:
        """
        Create a new Render Job for long-running processing
        
        Args:
            pdf_file_path: Path to the PDF file to process
            processing_mode: "single_deed" or "multi_deed"
            splitting_strategy: "smart_detection" or "ai_assisted"
            
        Returns:
            Job ID for tracking
        """
        job_id = f"job_{int(time.time())}_{hash(pdf_file_path) % 10000}"
        
        # Store job info locally
        job_info = JobInfo(
            id=job_id,
            filename=os.path.basename(pdf_file_path),
            processing_mode=processing_mode,
            splitting_strategy=splitting_strategy,
            status=JobStatus.PENDING,
            created_at=time.time(),
            logs=[]
        )
        self.jobs[job_id] = job_info
        
        # In a real implementation, you would:
        # 1. Upload the PDF file to a storage service (S3, etc.)
        # 2. Trigger the Render Job via API
        # 3. Set environment variables for the job
        
        print(f"🚀 Created job {job_id} for {pdf_file_path}")
        print(f"   Mode: {processing_mode}, Strategy: {splitting_strategy}")
        
        # Simulate job start (in real implementation, this would be async)
        self._start_job(job_id, pdf_file_path)
        
        return job_id
    
    def _start_job(self, job_id: str, pdf_file_path: str):
        """Start the actual Render Job (simulated for now)"""
        job_info = self.jobs.get(job_id)
        if not job_info:
            return
        
        job_info.status = JobStatus.RUNNING
        job_info.started_at = time.time()
        job_info.logs.append(f"🚀 Job {job_id} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # In real implementation, this would trigger the Render Job API
        print(f"📋 Job {job_id} would be started on Render with:")
        print(f"   INPUT_PDF_PATH={pdf_file_path}")
        print(f"   PROCESSING_MODE={job_info.processing_mode}")
        print(f"   SPLITTING_STRATEGY={job_info.splitting_strategy}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a job
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Job status information or None if not found
        """
        job_info = self.jobs.get(job_id)
        if not job_info:
            return None
        
        # In real implementation, you would query Render API for actual status
        # For now, we'll simulate some progress
        if job_info.status == JobStatus.RUNNING:
            elapsed = time.time() - job_info.started_at
            if elapsed > 30:  # Simulate completion after 30 seconds
                self._complete_job(job_id)
        
        return asdict(job_info)
    
    def _complete_job(self, job_id: str):
        """Simulate job completion (in real implementation, this would be called by webhook)"""
        job_info = self.jobs.get(job_id)
        if not job_info:
            return
        
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = time.time()
        job_info.logs.append(f"✅ Job {job_id} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Simulate a result
        job_info.result = {
            "classification": 1,
            "confidence": 0.95,
            "processing_time": job_info.completed_at - job_info.started_at,
            "pages_processed": 25,
            "reservations_found": True
        }
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a completed job
        
        Args:
            job_id: The job ID
            
        Returns:
            Job result or None if not completed or not found
        """
        job_info = self.jobs.get(job_id)
        if job_info and job_info.status == JobStatus.COMPLETED:
            return job_info.result
        return None
    
    def get_job_logs(self, job_id: str) -> List[str]:
        """
        Get logs for a job
        
        Args:
            job_id: The job ID
            
        Returns:
            List of log messages
        """
        job_info = self.jobs.get(job_id)
        if job_info:
            return job_info.logs
        return []
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job
        
        Args:
            job_id: The job ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        job_info = self.jobs.get(job_id)
        if job_info and job_info.status == JobStatus.RUNNING:
            job_info.status = JobStatus.CANCELLED
            job_info.logs.append(f"❌ Job {job_id} cancelled at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        return False
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all jobs
        
        Returns:
            List of job information
        """
        return [asdict(job) for job in self.jobs.values()]

# Global job manager instance
job_manager = RenderJobManager()

# Convenience functions
def create_processing_job(pdf_file_path: str, processing_mode: str = "multi_deed", 
                         splitting_strategy: str = "smart_detection") -> str:
    """Create a new processing job"""
    return job_manager.create_job(pdf_file_path, processing_mode, splitting_strategy)

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status"""
    return job_manager.get_job_status(job_id)

def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job result"""
    return job_manager.get_job_result(job_id)

def get_job_logs(job_id: str) -> List[str]:
    """Get job logs"""
    return job_manager.get_job_logs(job_id)

# Example usage
if __name__ == "__main__":
    # Test the job manager
    manager = RenderJobManager()
    
    # Create a test job
    job_id = manager.create_job(
        "/path/to/test.pdf",
        "multi_deed",
        "smart_detection"
    )
    
    print(f"Created job: {job_id}")
    
    # Monitor job status
    for i in range(10):
        status = manager.get_job_status(job_id)
        if status:
            print(f"Job status: {status['status']}")
            if status['status'] == 'completed':
                result = manager.get_job_result(job_id)
                print(f"Job result: {result}")
                break
        time.sleep(5)
