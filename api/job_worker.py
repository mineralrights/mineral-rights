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
import base64
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
        
        # Handle base64-encoded credentials
        credentials_path = None
        if os.getenv('GOOGLE_CREDENTIALS_BASE64'):
            # Decode base64 credentials and create temporary file
            credentials_b64 = os.getenv('GOOGLE_CREDENTIALS_BASE64')
            credentials_json = base64.b64decode(credentials_b64).decode('utf-8')
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as cred_file:
                cred_file.write(credentials_json)
                credentials_path = cred_file.name
            
            print(f"✅ Created temporary credentials file: {credentials_path}")
        
        # Initialize DocumentProcessor
        processor = DocumentProcessor(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            document_ai_endpoint=os.getenv('DOCUMENT_AI_ENDPOINT'),
            document_ai_credentials=credentials_path
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
        
        # Process the document with detailed progress tracking
        if processing_mode == "single_deed":
            result = processor.process_document(tmp_path)
            job_ref.update({
                'logs': firestore.ArrayUnion([f"🤖 Running LLM analysis on single document"]),
                'progress': 50
            })
        elif processing_mode == "multi_deed":
            # Enhanced multi-deed processing with detailed logging
            job_ref.update({
                'logs': firestore.ArrayUnion([f"📄 Starting multi-deed document analysis"]),
                'progress': 30
            })
            
            # Check if we're using Document AI for segmentation
            if splitting_strategy == "document_ai":
                job_ref.update({
                    'logs': firestore.ArrayUnion([f"🔍 Calling Document AI for deed segmentation..."]),
                    'progress': 35
                })
            
            result = processor.process_multi_deed_document(tmp_path, strategy=splitting_strategy)
            
            # Log deed segmentation results
            if hasattr(result, 'deed_results') and result.deed_results:
                deed_count = len(result.deed_results)
                job_ref.update({
                    'logs': firestore.ArrayUnion([f"📋 Document AI identified {deed_count} potential deeds"]),
                    'progress': 60
                })
                
                # Log deed ranges if available
                for i, deed in enumerate(result.deed_results[:3]):  # Show first 3 deeds
                    if hasattr(deed, 'pages_in_deed'):
                        job_ref.update({
                            'logs': firestore.ArrayUnion([f"📄 Deed {i+1}: {deed.pages_in_deed} pages identified"])
                        })
                
                if deed_count > 3:
                    job_ref.update({
                        'logs': firestore.ArrayUnion([f"📄 ... and {deed_count - 3} more deeds"])
                    })
            
            job_ref.update({
                'logs': firestore.ArrayUnion([f"🤖 Running LLM analysis on {len(result.deed_results) if hasattr(result, 'deed_results') else 'unknown'} deeds"]),
                'progress': 70
            })
            
        elif processing_mode == "page_by_page":
            job_ref.update({
                'logs': firestore.ArrayUnion([f"📄 Starting page-by-page analysis"]),
                'progress': 30
            })
            result = processor.process_document_page_by_page(tmp_path, max_samples=6, high_recall_mode=True)
            
            if hasattr(result, 'page_results') and result.page_results:
                page_count = len(result.page_results)
                job_ref.update({
                    'logs': firestore.ArrayUnion([f"🤖 Running LLM analysis on {page_count} pages"]),
                    'progress': 60
                })
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
        
        # Clean up temporary files
        os.unlink(tmp_path)
        print(f"🧹 Cleaned up temporary file")
        
        if credentials_path and os.path.exists(credentials_path):
            os.unlink(credentials_path)
            print(f"🧹 Cleaned up temporary credentials file")
        
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
        
        # Clean up temporary files if they exist
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if 'credentials_path' in locals() and credentials_path and os.path.exists(credentials_path):
                os.unlink(credentials_path)
        except Exception:
            pass
        
        return {
            'status': 'error',
            'job_id': job_id,
            'error': str(e)
        }

def main():
    """Main entry point for the job worker"""
    if len(sys.argv) == 2:
        # Single job mode - process specific job
        job_id = sys.argv[1]
        print(f"🚀 Starting job worker for job: {job_id}")
        
        result = process_job(job_id)
        
        if result['status'] == 'success':
            print(f"✅ Job {job_id} completed successfully")
            sys.exit(0)
        else:
            print(f"❌ Job {job_id} failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        # Continuous mode - poll for queued jobs
        print("🚀 Starting job worker in continuous mode - polling for queued jobs")
        
        while True:
            try:
                # Get all queued jobs
                jobs_ref = db.collection('jobs')
                queued_jobs = jobs_ref.where('status', '==', 'queued').limit(1).get()
                
                if queued_jobs:
                    for job_doc in queued_jobs:
                        job_id = job_doc.id
                        job_data = job_doc.to_dict()
                        
                        print(f"🔄 Found queued job: {job_id}")
                        
                        # Process the job
                        result = process_job(job_id)
                        
                        if result['status'] == 'success':
                            print(f"✅ Job {job_id} completed successfully")
                        else:
                            print(f"❌ Job {job_id} failed: {result.get('error', 'Unknown error')}")
                else:
                    print("⏳ No queued jobs found, waiting...")
                    time.sleep(10)  # Wait 10 seconds before checking again
                    
            except Exception as e:
                print(f"❌ Error in job polling loop: {e}")
                time.sleep(30)  # Wait 30 seconds before retrying

if __name__ == "__main__":
    main()