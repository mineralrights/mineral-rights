#!/usr/bin/env python3
"""
Job Worker for Mineral Rights Processing
Runs as a Cloud Run Job to handle long-running document processing tasks
"""

import os
import sys
import time
import json
import tempfile
import traceback
from typing import Dict, Any
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from google.cloud import firestore
from google.cloud import storage
from src.mineral_rights.document_classifier import DocumentProcessor

# Initialize Google Cloud clients
db = firestore.Client()
storage_client = storage.Client()

def process_job(job_id: str) -> Dict[str, Any]:
    """Process a single job from Firestore"""
    print(f"🚀 Starting job processing for {job_id}")
    
    try:
        # Get job from Firestore
        job_ref = db.collection('jobs').document(job_id)
        job_doc = job_ref.get()
        
        if not job_doc.exists:
            raise Exception(f"Job {job_id} not found in Firestore")
        
        job_data = job_doc.to_dict()
        print(f"📋 Job data: {job_data}")
        
        # Update status to processing
        job_ref.update({
            'status': 'processing',
            'updated_at': time.time(),
            'logs': firestore.ArrayUnion([f"🚀 Job worker started processing at {time.strftime('%Y-%m-%d %H:%M:%S')}"])
        })
        
        # Download file from Cloud Storage
        bucket_name = os.getenv('GCS_BUCKET_NAME', 'mineral-rights-documents')
        bucket = storage_client.bucket(bucket_name)
        blob_name = f"uploads/{job_id}/{job_data['filename']}"
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            raise Exception(f"File {blob_name} not found in Cloud Storage")
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            blob.download_to_filename(tmp_file.name)
            tmp_path = tmp_file.name
        
        print(f"📁 Downloaded file to {tmp_path}")
        
        # Initialize DocumentProcessor
        processor = DocumentProcessor(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            document_ai_endpoint=os.getenv('DOCUMENT_AI_ENDPOINT'),
            document_ai_credentials=os.getenv('GOOGLE_CREDENTIALS_BASE64')
        )
        
        print("✅ DocumentProcessor initialized")
        
        # Process the document based on mode
        processing_mode = job_data.get('processing_mode', 'single_deed')
        splitting_strategy = job_data.get('splitting_strategy', 'document_ai')
        
        print(f"🔧 Processing mode: {processing_mode}, Strategy: {splitting_strategy}")
        
        # Update progress
        job_ref.update({
            'logs': firestore.ArrayUnion([f"🔧 Processing document with {processing_mode} mode"]),
            'progress': 25
        })
        
        # Process the document
        if processing_mode == "single_deed":
            result = processor.process_document(tmp_path)
        elif processing_mode == "multi_deed":
            result = processor.process_multi_deed_document(tmp_path, strategy=splitting_strategy)
        elif processing_mode == "page_by_page":
            result = processor.process_document_page_by_page(tmp_path, max_samples=6, high_recall_mode=True)
        else:
            raise ValueError(f"Unknown processing mode: {processing_mode}")
        
        print("✅ Document processing completed")
        
        # Update progress
        job_ref.update({
            'logs': firestore.ArrayUnion([f"✅ Processing completed successfully"]),
            'progress': 90
        })
        
        # Save results to Firestore
        job_ref.update({
            'status': 'completed',
            'result': result,
            'progress': 100,
            'completed_at': time.time(),
            'updated_at': time.time(),
            'logs': firestore.ArrayUnion([f"🎉 Job completed at {time.strftime('%Y-%m-%d %H:%M:%S')}"])
        })
        
        # Clean up temporary file
        os.unlink(tmp_path)
        print(f"🧹 Cleaned up temporary file")
        
        return {
            'status': 'success',
            'job_id': job_id,
            'result': result
        }
        
    except Exception as e:
        print(f"❌ Job processing failed: {e}")
        traceback.print_exc()
        
        # Update job status to failed
        try:
            job_ref.update({
                'status': 'failed',
                'error': str(e),
                'updated_at': time.time(),
                'logs': firestore.ArrayUnion([f"❌ Job failed: {str(e)}"])
            })
        except Exception as update_error:
            print(f"⚠️ Failed to update job status: {update_error}")
        
        # Clean up temporary file if it exists
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        
        return {
            'status': 'error',
            'job_id': job_id,
            'error': str(e)
        }

def main():
    """Main entry point for the job worker"""
    if len(sys.argv) != 2:
        print("Usage: python job_worker.py <job_id>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    print(f"🚀 Starting job worker for job: {job_id}")
    
    result = process_job(job_id)
    
    if result['status'] == 'success':
        print(f"✅ Job {job_id} completed successfully")
        sys.exit(0)
    else:
        print(f"❌ Job {job_id} failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()