#!/usr/bin/env python3
"""
Test GCS access from the production environment
"""
import os
import sys
import tempfile
import shutil
import subprocess
import json

# Test with a small synthetic PDF
pdf_path = "data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ Test PDF not found: {pdf_path}")
    sys.exit(1)

print("🧪 Testing GCS Access from Production")
print(f"📄 Test PDF: {pdf_path}")

try:
    # Step 1: Get signed upload URL
    print("🔑 Step 1: Getting signed upload URL...")
    upload_cmd = [
        'curl', '-X', 'POST',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/get-signed-upload-url',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({
            'filename': 'test_gcs_access.pdf',
            'content_type': 'application/pdf'
        }),
        '--max-time', '30'
    ]
    
    upload_result = subprocess.run(upload_cmd, capture_output=True, text=True)
    print(f"📊 Upload URL response code: {upload_result.returncode}")
    
    if upload_result.returncode != 0:
        print(f"❌ Failed to get upload URL: {upload_result.stderr}")
        sys.exit(1)
    
    upload_data = json.loads(upload_result.stdout)
    signed_url = upload_data['signed_url']
    gcs_url = upload_data['gcs_url']
    
    print(f"✅ Signed URL obtained: {gcs_url}")
    
    # Step 2: Upload to GCS
    print("📤 Step 2: Uploading to GCS...")
    upload_file_cmd = [
        'curl', '-X', 'PUT',
        signed_url,
        '--data-binary', f'@{pdf_path}',
        '--max-time', '60'
    ]
    
    upload_file_result = subprocess.run(upload_file_cmd, capture_output=True, text=True)
    print(f"📊 Upload file response code: {upload_file_result.returncode}")
    
    if upload_file_result.returncode != 0:
        print(f"❌ Failed to upload file: {upload_file_result.stderr}")
        sys.exit(1)
    
    print("✅ File uploaded to GCS successfully")
    
    # Step 3: Test GCS access directly
    print("🔍 Step 3: Testing GCS access directly...")
    
    # Test if we can access the file via curl
    test_url_cmd = [
        'curl', '-I', gcs_url,
        '--max-time', '10'
    ]
    
    test_result = subprocess.run(test_url_cmd, capture_output=True, text=True)
    print(f"📊 Direct URL test response code: {test_result.returncode}")
    print(f"📤 Response headers: {test_result.stdout}")
    if test_result.stderr:
        print(f"❌ Error: {test_result.stderr}")
    
    # Step 4: Test a simple health check
    print("🏥 Step 4: Testing health endpoint...")
    health_cmd = [
        'curl', '-X', 'GET',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/health',
        '--max-time', '10'
    ]
    
    health_result = subprocess.run(health_cmd, capture_output=True, text=True)
    print(f"📊 Health check response code: {health_result.returncode}")
    print(f"📤 Health response: {health_result.stdout}")
    
    if health_result.returncode == 0:
        print("✅ Health check passed - service is running")
    else:
        print("❌ Health check failed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
