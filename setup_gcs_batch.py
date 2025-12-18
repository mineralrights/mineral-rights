#!/usr/bin/env python3
"""
Setup Google Cloud Storage for Batch Processing

This script sets up GCS and implements true asynchronous batch processing.
"""

import os
import sys
from pathlib import Path
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tempfile

@dataclass
class BatchJobResult:
    """Result from batch processing job"""
    job_id: str
    status: str
    input_uri: str
    output_uri: str
    created_at: float

class GCSBatchProcessingService:
    """Service for asynchronous batch processing with GCS"""
    
    def __init__(self, project_id: str, location: str, bucket_name: str):
        """
        Initialize GCS batch processing service
        
        Args:
            project_id: Google Cloud project ID
            location: Document AI location (e.g., 'us')
            bucket_name: GCS bucket name for input/output
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients"""
        try:
            from google.cloud import documentai
            from google.cloud import storage
            
            self.documentai_client = documentai.DocumentProcessorServiceClient()
            self.storage_client = storage.Client()
            
            print("✅ Google Cloud clients initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize clients: {e}")
            raise
    
    def create_bucket_if_not_exists(self):
        """Create GCS bucket if it doesn't exist"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            
            if not bucket.exists():
                print(f"📦 Creating GCS bucket: {self.bucket_name}")
                bucket = self.storage_client.create_bucket(self.bucket_name, location=self.location)
                print(f"✅ Bucket created: {self.bucket_name}")
            else:
                print(f"✅ Bucket already exists: {self.bucket_name}")
            
            return bucket
            
        except Exception as e:
            print(f"❌ Error creating bucket: {e}")
            raise
    
    def upload_pdf_to_gcs(self, pdf_path: str) -> str:
        """Upload PDF to GCS and return the GCS URI"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            
            # Generate unique filename
            timestamp = int(time.time())
            filename = f"input/{timestamp}_{os.path.basename(pdf_path)}"
            
            print(f"📤 Uploading PDF to GCS: {filename}")
            
            # Upload file
            blob = bucket.blob(filename)
            blob.upload_from_filename(pdf_path)
            
            gcs_uri = f"gs://{self.bucket_name}/{filename}"
            print(f"✅ PDF uploaded: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            print(f"❌ Error uploading PDF: {e}")
            raise
    
    def start_batch_processing_job(self, input_gcs_uri: str) -> BatchJobResult:
        """Start asynchronous batch processing job"""
        try:
            from google.cloud import documentai
            
            # Create batch processing request with correct structure
            request = documentai.BatchProcessRequest(
                name=self.processor_version,
                input_documents=documentai.BatchDocumentsInputConfig(
                    gcs_prefix=documentai.GcsPrefix(
                        gcs_uri_prefix=input_gcs_uri.replace(os.path.basename(input_gcs_uri), "")
                    )
                ),
                document_output_config=documentai.DocumentOutputConfig(
                    gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                        gcs_uri=f"gs://{self.bucket_name}/output/"
                    )
                )
            )
            
            print("🚀 Starting batch processing job...")
            
            # Start the batch processing operation
            operation = self.documentai_client.batch_process_documents(request=request)
            
            print(f"✅ Batch processing job started")
            print(f"   - Input: {input_gcs_uri}")
            print(f"   - Output: gs://{self.bucket_name}/output/")
            
            # Get operation name from the operation object
            # The operation name is in the operation.operation attribute
            if hasattr(operation, 'operation') and hasattr(operation.operation, 'name'):
                operation_name = operation.operation.name
            elif hasattr(operation, 'name'):
                operation_name = operation.name
            else:
                # Extract from string representation
                operation_str = str(operation)
                if 'name: "' in operation_str:
                    operation_name = operation_str.split('name: "')[1].split('"')[0]
                else:
                    operation_name = operation_str
            
            return BatchJobResult(
                job_id=operation_name,
                status="RUNNING",
                input_uri=input_gcs_uri,
                output_uri=f"gs://{self.bucket_name}/output/",
                created_at=time.time()
            )
            
        except Exception as e:
            print(f"❌ Error starting batch job: {e}")
            raise
    
    def check_job_status(self, job_id: str) -> str:
        """Check the status of a batch processing job"""
        try:
            from google.cloud import documentai
            from google.api_core import operations_v1
            from google.api_core import grpc_helpers
            
            # Create operations client with proper channel
            channel = grpc_helpers.create_channel(
                f"{self.location}-documentai.googleapis.com:443"
            )
            operations_client = operations_v1.OperationsClient(channel=channel)
            
            # Get operation status
            operation = operations_client.get_operation(name=job_id)
            
            if operation.done:
                if operation.error and operation.error.code != 0:
                    return f"FAILED: {operation.error}"
                else:
                    return "COMPLETED"
            else:
                return "RUNNING"
                
        except Exception as e:
            print(f"❌ Error checking job status: {e}")
            return "ERROR"
    
    def download_results(self, output_uri: str) -> List[Dict[str, Any]]:
        """Download and parse batch processing results"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            
            # List files in output directory
            output_prefix = output_uri.replace(f"gs://{self.bucket_name}/", "")
            blobs = bucket.list_blobs(prefix=output_prefix)
            
            results = []
            
            for blob in blobs:
                if blob.name.endswith('.json'):
                    print(f"📥 Downloading result: {blob.name}")
                    
                    # Download and parse JSON
                    content = blob.download_as_text()
                    result_data = json.loads(content)
                    
                    # Extract deed information
                    if 'entities' in result_data:
                        for entity in result_data['entities']:
                            if entity.get('type') in ['DEED', 'COVER']:
                                results.append({
                                    'type': entity.get('type'),
                                    'confidence': entity.get('confidence', 0.0),
                                    'text': entity.get('mentionText', ''),
                                    'page_refs': entity.get('pageAnchor', {}).get('pageRefs', [])
                                })
            
            print(f"✅ Downloaded {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Error downloading results: {e}")
            return []

def test_gcs_batch_processing():
    """Test GCS batch processing"""
    print("🧪 Testing GCS Batch Processing")
    print("=" * 50)
    
    try:
        # Configuration
        project_id = "381937358877"  # Your project ID
        location = "us"
        bucket_name = "mineral-rights-batch-1757454568"  # Use the bucket we just created
        
        print(f"📊 Configuration:")
        print(f"   - Project ID: {project_id}")
        print(f"   - Location: {location}")
        print(f"   - Bucket: {bucket_name}")
        
        # Create service
        batch_service = GCSBatchProcessingService(project_id, location, bucket_name)
        
        # Create bucket
        batch_service.create_bucket_if_not_exists()
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        # Upload PDF
        input_uri = batch_service.upload_pdf_to_gcs(test_pdf_path)
        
        # Start batch job
        job_result = batch_service.start_batch_processing_job(input_uri)
        
        print(f"\n📊 Batch Job Started:")
        print(f"   - Job ID: {job_result.job_id}")
        print(f"   - Status: {job_result.status}")
        print(f"   - Input: {job_result.input_uri}")
        print(f"   - Output: {job_result.output_uri}")
        
        print(f"\n⏳ Batch processing is running asynchronously...")
        print(f"   - This may take several minutes")
        print(f"   - Check status with: batch_service.check_job_status('{job_result.job_id}')")
        print(f"   - Download results when completed")
        
        # Save job info
        job_info = {
            'job_id': job_result.job_id,
            'status': job_result.status,
            'input_uri': job_result.input_uri,
            'output_uri': job_result.output_uri,
            'created_at': job_result.created_at,
            'bucket_name': bucket_name
        }
        
        with open('batch_job_info.json', 'w') as f:
            json.dump(job_info, f, indent=2)
        
        print(f"\n💾 Job info saved to: batch_job_info.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_existing_job():
    """Check status of existing batch job"""
    try:
        if not os.path.exists('batch_job_info.json'):
            print("❌ No existing job found. Run test_gcs_batch_processing() first.")
            return
        
        with open('batch_job_info.json', 'r') as f:
            job_info = json.load(f)
        
        print(f"📊 Checking job status...")
        print(f"   - Job ID: {job_info['job_id']}")
        
        # Create service
        batch_service = GCSBatchProcessingService(
            project_id="381937358877",
            location="us",
            bucket_name=job_info['bucket_name']
        )
        
        # Check status
        status = batch_service.check_job_status(job_info['job_id'])
        print(f"   - Status: {status}")
        
        if status == "COMPLETED":
            print(f"\n🎉 Job completed! Downloading results...")
            results = batch_service.download_results(job_info['output_uri'])
            
            print(f"\n📊 Results:")
            print(f"   - Total entities: {len(results)}")
            
            for i, result in enumerate(results[:10]):  # Show first 10
                print(f"   - {i+1}. {result['type']}: confidence {result['confidence']:.3f}")
            
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more")
        
        return status == "COMPLETED"
        
    except Exception as e:
        print(f"❌ Error checking job: {e}")
        return False

def main():
    """Main function"""
    print("🚀 GCS Batch Processing Setup")
    print("=" * 40)
    
    # Check if we have an existing job
    if os.path.exists('batch_job_info.json'):
        print("📋 Found existing job, checking status...")
        success = check_existing_job()
    else:
        print("🆕 Starting new batch processing job...")
        success = test_gcs_batch_processing()
    
    if success:
        print("\n✅ GCS batch processing setup completed!")
    else:
        print("\n❌ Setup failed")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
