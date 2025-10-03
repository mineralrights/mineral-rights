#!/usr/bin/env python3
"""
Cloud Job processor for large PDF processing
Handles unlimited memory PDF processing
"""

import os
import sys
import tempfile
import base64
import json
from google.oauth2 import service_account
from src.mineral_rights.document_classifier import DocumentProcessor

def main():
    """Main job processor function"""
    if len(sys.argv) != 4:
        print("Usage: python process_job.py <gcs_url> <processing_mode> <splitting_strategy>")
        sys.exit(1)
    
    gcs_url = sys.argv[1]
    processing_mode = sys.argv[2]
    splitting_strategy = sys.argv[3]
    
    print(f"🚀 Starting Cloud Job processing...")
    print(f"📄 GCS URL: {gcs_url}")
    print(f"🔧 Processing mode: {processing_mode}")
    print(f"📊 Splitting strategy: {splitting_strategy}")
    
    try:
        # Set up credentials for the job
        credentials_b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        if credentials_b64:
            # Decode the base64 credentials
            credentials_json = base64.b64decode(credentials_b64).decode('utf-8')
            credentials_info = json.loads(credentials_json)
            
            # Create credentials object and save to temp file
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            temp_creds_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(credentials_info, temp_creds_file)
            temp_creds_file.close()
            
            # Set environment variable for credentials
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_file.name
            print("✅ Using base64 encoded service account credentials")
        else:
            print("✅ Using default service account credentials")
        
        # Initialize processor
        processor = DocumentProcessor()
        
        # Process the document from GCS
        result = processor.process_document_from_gcs(
            gcs_url=gcs_url,
            processing_mode=processing_mode,
            splitting_strategy=splitting_strategy
        )
        
        print(f"✅ Job completed successfully!")
        print(f"📊 Results: {len(result.get('deeds', []))} deeds processed")
        
        # Save results to GCS for retrieval
        # TODO: Implement result storage
        
    except Exception as e:
        print(f"❌ Job failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
