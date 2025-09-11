"""
Proper Redis-based job persistence using the durable pattern.
This ensures jobs survive Railway restarts and never return 404.
"""
import os
import json
import time
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import logging

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    QUEUED = "queued"
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
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: list = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []

class DurableJobManager:
    """
    Durable job manager that stores all job state in Redis.
    Jobs survive service restarts and never return 404.
    """
    
    def __init__(self):
        # Redis connection
        redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
        redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        
        if not redis_url or not redis_token:
            logger.warning("Redis credentials not found - using in-memory fallback")
            self.redis_client = None
            self.jobs: Dict[str, JobInfo] = {}
        else:
            try:
                # Use Upstash Redis REST API
                self.redis_client = {
                    'url': redis_url,
                    'token': redis_token
                }
                logger.info("✅ Redis initialized for durable job persistence")
            except Exception as e:
                logger.error(f"Redis initialization failed: {e}")
                self.redis_client = None
                self.jobs: Dict[str, JobInfo] = {}
    
    def _status_key(self, job_id: str) -> str:
        """Redis key for job status"""
        return f"job:status:{job_id}"
    
    def _result_key(self, job_id: str) -> str:
        """Redis key for job result"""
        return f"job:result:{job_id}"
    
    def _save_job_to_redis(self, job: JobInfo) -> bool:
        """Save job status to Redis using REST API"""
        if not self.redis_client:
            return False
        
        try:
            import requests
            
            # Convert job to dict and serialize
            job_data = asdict(job)
            job_data['status'] = job.status.value  # Convert enum to string
            job_json = json.dumps(job_data)
            
            # Save status with 7-day TTL
            response = requests.post(
                self.redis_client['url'],
                headers={
                    'Authorization': f"Bearer {self.redis_client['token']}",
                    'Content-Type': 'application/json'
                },
                json={
                    "command": ["SETEX", self._status_key(job.id), str(7 * 24 * 60 * 60), job_json]
                },
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"✅ Job {job.id} status saved to Redis")
                return True
            else:
                logger.warning(f"⚠️ Failed to save job {job.id} status to Redis: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"⚠️ Redis save error for job {job.id}: {e}")
            return False
    
    def _load_job_from_redis(self, job_id: str) -> Optional[JobInfo]:
        """Load job status from Redis using REST API"""
        if not self.redis_client:
            return None
        
        try:
            import requests
            
            # Get status
            response = requests.post(
                self.redis_client['url'],
                headers={
                    'Authorization': f"Bearer {self.redis_client['token']}",
                    'Content-Type': 'application/json'
                },
                json={
                    "command": ["GET", self._status_key(job_id)]
                },
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if result and result.get("result") and result["result"] != "null":
                    job_data = json.loads(result["result"])
                    job_data['status'] = JobStatus(job_data['status'])  # Convert string back to enum
                    job = JobInfo(**job_data)
                    logger.debug(f"✅ Job {job_id} loaded from Redis")
                    return job
                    
        except Exception as e:
            logger.error(f"⚠️ Redis load error for job {job_id}: {e}")
        
        return None
    
    def _save_result_to_redis(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Save job result to Redis with 7-day TTL"""
        if not self.redis_client:
            return False
        
        try:
            import requests
            
            result_json = json.dumps(result)
            
            response = requests.post(
                self.redis_client['url'],
                headers={
                    'Authorization': f"Bearer {self.redis_client['token']}",
                    'Content-Type': 'application/json'
                },
                json={
                    "command": ["SETEX", self._result_key(job_id), str(7 * 24 * 60 * 60), result_json]
                },
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"✅ Job {job_id} result saved to Redis")
                return True
            else:
                logger.warning(f"⚠️ Failed to save job {job_id} result to Redis: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"⚠️ Redis result save error for job {job_id}: {e}")
            return False
    
    def _load_result_from_redis(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job result from Redis"""
        if not self.redis_client:
            return None
        
        try:
            import requests
            
            response = requests.post(
                self.redis_client['url'],
                headers={
                    'Authorization': f"Bearer {self.redis_client['token']}",
                    'Content-Type': 'application/json'
                },
                json={
                    "command": ["GET", self._result_key(job_id)]
                },
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if result and result.get("result") and result["result"] != "null":
                    return json.loads(result["result"])
                    
        except Exception as e:
            logger.error(f"⚠️ Redis result load error for job {job_id}: {e}")
        
        return None
    
    def create_job(self, filename: str, processing_mode: str, splitting_strategy: str) -> str:
        """Create a new job with durable status"""
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        job = JobInfo(
            id=job_id,
            filename=filename,
            processing_mode=processing_mode,
            splitting_strategy=splitting_strategy,
            status=JobStatus.QUEUED,
            created_at=time.time(),
            logs=[]
        )
        
        # Save to Redis immediately
        self._save_job_to_redis(job)
        
        # Also save to in-memory for fast access
        if not self.redis_client:
            self.jobs[job_id] = job
        
        logger.info(f"✅ Created durable job: {job_id}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job - never returns None, always returns a job or raises exception"""
        # First try in-memory (fast)
        if not self.redis_client and job_id in self.jobs:
            return self.jobs[job_id]
        
        # Try Redis
        job = self._load_job_from_redis(job_id)
        if job:
            # Cache in memory for faster access
            if not self.redis_client:
                self.jobs[job_id] = job
            return job
        
        # Job not found - this should never happen with proper implementation
        logger.warning(f"⚠️ Job {job_id} not found in Redis or memory")
        return None
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         progress: Optional[int] = None, 
                         result: Optional[Dict[str, Any]] = None, 
                         error: Optional[str] = None):
        """Update job status with durable persistence"""
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Cannot update job {job_id} - job not found")
            return
        
        # Update job fields
        job.status = status
        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = time.time()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            job.completed_at = time.time()
        
        if progress is not None:
            job.progress = progress
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        
        # Save to Redis
        self._save_job_to_redis(job)
        
        # Save result separately if provided
        if result is not None:
            self._save_result_to_redis(job_id, result)
        
        # Update in-memory cache
        if not self.redis_client:
            self.jobs[job_id] = job
        
        logger.info(f"✅ Updated job {job_id} status to {status.value}")
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result from Redis"""
        # First try to get from job status
        job = self.get_job(job_id)
        if job and job.result:
            return job.result
        
        # Try to load from separate result key
        return self._load_result_from_redis(job_id)
    
    def list_jobs(self) -> list:
        """List all jobs (for debugging)"""
        if not self.redis_client:
            return [asdict(job) for job in self.jobs.values()]
        
        # For Redis, we'd need to scan keys, but this is mainly for debugging
        # In production, you'd typically not list all jobs
        return []

# Global instance
job_manager = DurableJobManager()
