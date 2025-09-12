#!/usr/bin/env python3
"""
Cloud Run Job Worker for Mineral Rights Processing
This runs as a separate Cloud Run Job to handle long-running processing tasks
"""

import os
import sys
import time
import json
import tempfile
import base64
from typing import Dict, Any
from google.cloud import storage, firestore
from google.cloud import tasks_v2

# Add the src directory to the path
sys.path.append('/app/src')

# Import your pipeline
from src.mineral_rights.document_classifier import DocumentProcessor

def initialize_processor():
    """Initialize the DocumentProcessor with environment variables"""
    try:
        print("🔧 Initializing DocumentProcessor...")
        
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found in environment")
            return None
            
        # Get Document AI endpoint
        document_ai_endpoint = os.getenv("DOCUMENT_AI_ENDPOINT")
        if not document_ai_endpoint:
            print("❌ DOCUMENT_AI_ENDPOINT not found in environment")
            return None
            
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
        return processor
        
    except Exception as e:
        print(f"❌ Failed to initialize DocumentProcessor: {e}")
        import traceback
        traceback.print_exc()
        return None

def download_file_from_gcs(bucket_name: str, file_path: str, local_path: str):
    """Download a file from Google Cloud Storage"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        blob.download_to_filename(local_path)
        print(f"✅ Downloaded file from GCS: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download file from GCS: {e}")
        return False

def upload_result_to_gcs(bucket_name: str, job_id: str, result: Dict[Any, Any]):
    """Upload processing result to Google Cloud Storage"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Upload result as JSON
        result_path = f"results/{job_id}/result.json"
        blob = bucket.blob(result_path)
        blob.upload_from_string(json.dumps(result, indent=2), content_type="application/json")
        
        print(f"✅ Uploaded result to GCS: {result_path}")
        return result_path
    except Exception as e:
        print(f"❌ Failed to upload result to GCS: {e}")
        return None

def update_job_status(job_id: str, status: str, progress: int = None, error: str = None, result_path: str = None):
    """Update job status in Firestore"""
    try:
        firestore_client = firestore.Client()
        doc_ref = firestore_client.collection("jobs").document(job_id)
        
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

def process_job(job_id: str, file_path: str, processing_mode: str, splitting_strategy: str):
    """Process a single job"""
    print(f"🚀 Starting processing for job {job_id}")
    print(f"📁 File: {file_path}")
    print(f"🔧 Mode: {processing_mode}")
    print(f"🔧 Strategy: {splitting_strategy}")
    
    # Initialize processor
    processor = initialize_processor()
    if not processor:
        update_job_status(job_id, "failed", error="Failed to initialize processor")
        return False
    
    # Get bucket name
    bucket_name = os.getenv("GCS_BUCKET_NAME", "mineral-rights-storage")
    
    # Download file from GCS
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        local_path = tmp_file.name
    
    try:
        if not download_file_from_gcs(bucket_name, file_path, local_path):
            update_job_status(job_id, "failed", error="Failed to download file from GCS")
            return False
        
        # Update status to processing
        update_job_status(job_id, "processing", progress=10)
        
        # Process the document
        print(f"📄 Processing document...")
        start_time = time.time()
        
        if processing_mode == "single_deed":
            print("📄 Single deed processing")
            result = processor.process_document(local_path)
        elif processing_mode == "multi_deed":
            print("📄 Multi-deed processing")
            result = processor.process_multi_deed_document(local_path, strategy=splitting_strategy)
        elif processing_mode == "page_by_page":
            print("📄 Page-by-page processing")
            result = processor.process_document_page_by_page(local_path, max_samples=6, high_recall_mode=True)
        else:
            raise ValueError(f"Unknown processing mode: {processing_mode}")
        
        processing_time = time.time() - start_time
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        
        # Update progress
        update_job_status(job_id, "processing", progress=80)
        
        # Upload result to GCS
        result_path = upload_result_to_gcs(bucket_name, job_id, result)
        if not result_path:
            update_job_status(job_id, "failed", error="Failed to upload result to GCS")
            return False
        
        # Update final status
        update_job_status(job_id, "completed", progress=100, result_path=result_path)
        
        print(f"✅ Job {job_id} completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        update_job_status(job_id, "failed", error=str(e))
        return False
    
    finally:
        # Clean up local file
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                print(f"🧹 Cleaned up local file")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")

def main():
    """Main entry point for the Cloud Run Job"""
    print("🚀 Starting Mineral Rights Job Worker...")
    
    # Get job parameters from environment variables
    job_id = os.getenv("JOB_ID")
    file_path = os.getenv("FILE_PATH")
    processing_mode = os.getenv("PROCESSING_MODE", "single_deed")
    splitting_strategy = os.getenv("SPLITTING_STRATEGY", "document_ai")
    
    if not job_id or not file_path:
        print("❌ Missing required environment variables: JOB_ID, FILE_PATH")
        sys.exit(1)
    
    print(f"📋 Job Parameters:")
    print(f"  Job ID: {job_id}")
    print(f"  File Path: {file_path}")
    print(f"  Processing Mode: {processing_mode}")
    print(f"  Splitting Strategy: {splitting_strategy}")
    
    # Process the job
    success = process_job(job_id, file_path, processing_mode, splitting_strategy)
    
    if success:
        print("✅ Job completed successfully")
        sys.exit(0)
    else:
        print("❌ Job failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
