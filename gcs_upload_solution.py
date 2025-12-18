#!/usr/bin/env python3
"""
Google Cloud Storage Upload Solution for Large PDFs
This handles files up to 5TB (Google's limit)
"""

import os
import json
from google.cloud import storage
from google.oauth2 import service_account
import tempfile
from typing import Optional

class GCSFileHandler:
    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        self.bucket_name = bucket_name
        
        # Initialize GCS client
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = storage.Client(credentials=credentials)
        else:
            # Use default credentials (ADC)
            self.client = storage.Client()
        
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_file(self, file_path: str, blob_name: str) -> str:
        """Upload a file to GCS and return the public URL"""
        blob = self.bucket.blob(blob_name)
        
        # Upload with resumable upload for large files
        blob.upload_from_filename(file_path)
        
        # Make it publicly accessible
        blob.make_public()
        
        return blob.public_url
    
    def upload_file_content(self, file_content: bytes, blob_name: str) -> str:
        """Upload file content directly to GCS"""
        blob = self.bucket.blob(blob_name)
        
        # Upload with resumable upload for large files
        blob.upload_from_string(file_content)
        
        # Make it publicly accessible
        blob.make_public()
        
        return blob.public_url
    
    def get_signed_url(self, blob_name: str, expiration_minutes: int = 60) -> str:
        """Get a signed URL for private access"""
        blob = self.bucket.blob(blob_name)
        return blob.generate_signed_url(expiration=expiration_minutes * 60)

# Example usage
if __name__ == "__main__":
    # Initialize with your bucket
    handler = GCSFileHandler("mineral-rights-pdfs-1759435410")
    
    # Test with a local file
    test_file = "data/multi-deed/pdfs/ROBERT.pdf"
    if os.path.exists(test_file):
        print(f"📁 Uploading {test_file} to GCS...")
        url = handler.upload_file(test_file, "test/ROBERT.pdf")
        print(f"✅ Upload successful: {url}")
    else:
        print("❌ Test file not found")
