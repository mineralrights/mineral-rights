"""
Simple job queue implementation for Cloud Run Jobs
This handles the queuing and execution of processing jobs
"""

import os
import time
import json
import subprocess
from typing import Dict, Any
from google.cloud import firestore

class JobQueue:
    def __init__(self):
        self.firestore_client = firestore.Client()
    
    async def queue_job(self, job_id: str, file_path: str, processing_mode: str, splitting_strategy: str):
        """Queue a job for processing by creating a Cloud Run Job execution"""
        try:
            # Get project and region from environment
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
            job_name = "mineral-rights-worker"
            
            # Prepare environment variables for the job
            env_vars = {
                "JOB_ID": job_id,
                "FILE_PATH": file_path,
                "PROCESSING_MODE": processing_mode,
                "SPLITTING_STRATEGY": splitting_strategy,
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                "DOCUMENT_AI_ENDPOINT": os.getenv("DOCUMENT_AI_ENDPOINT"),
                "GOOGLE_CREDENTIALS_BASE64": os.getenv("GOOGLE_CREDENTIALS_BASE64"),
                "GCS_BUCKET_NAME": os.getenv("GCS_BUCKET_NAME")
            }
            
            # Build the gcloud command to execute the job
            cmd = [
                "gcloud", "run", "jobs", "execute", job_name,
                "--region", region,
                "--project", project_id,
                "--wait"
            ]
            
            # Add environment variables
            for key, value in env_vars.items():
                if value:
                    cmd.extend(["--set-env-vars", f"{key}={value}"])
            
            print(f"🚀 Executing Cloud Run Job: {' '.join(cmd)}")
            
            # Execute the job
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Job {job_id} executed successfully")
                return True
            else:
                print(f"❌ Job {job_id} failed: {result.stderr}")
                # Update job status to failed
                await self.update_job_status(job_id, "failed", error=result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Failed to queue job: {e}")
            await self.update_job_status(job_id, "failed", error=str(e))
            return False
    
    async def update_job_status(self, job_id: str, status: str, progress: int = None, error: str = None, result_path: str = None):
        """Update job status in Firestore"""
        try:
            doc_ref = self.firestore_client.collection("jobs").document(job_id)
            
            update_data = {
                "status": status,
                "updated_at": time.time()
            }
            
            if progress is not None:
                update_data["progress"] = progress
            
            if error:
                update_data["error"] = error
            
            if result_path:
                update_data["result_path"] = result_path
            
            doc_ref.update(update_data)
            print(f"✅ Updated job {job_id} status to {status}")
            return True
        except Exception as e:
            print(f"❌ Failed to update job status: {e}")
            return False
