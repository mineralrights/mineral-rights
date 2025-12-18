#!/usr/bin/env python3
"""
Test production deployment with proper GCS upload
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

print("🧪 Testing Production with GCS Upload")
print(f"📄 Test PDF: {pdf_path}")

try:
    # Step 1: Get signed upload URL
    print("🔑 Step 1: Getting signed upload URL...")
    upload_cmd = [
        'curl', '-X', 'POST',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/get-signed-upload-url',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({
            'filename': 'test_synthetic_001.pdf',
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
    
    # Step 3: Process with page-by-page
    print("🔧 Step 3: Processing with page-by-page approach...")
    process_cmd = [
        'curl', '-X', 'POST',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/process-large-pdf',
        '-F', f'gcs_url={gcs_url}',
        '-F', 'processing_mode=page_by_page',
        '-F', 'splitting_strategy=document_ai',
        '--max-time', '300'  # 5 minute timeout
    ]
    
    print("🚀 Sending processing request...")
    process_result = subprocess.run(process_cmd, capture_output=True, text=True, timeout=300)
    
    print(f"📊 Processing response code: {process_result.returncode}")
    print(f"📤 STDOUT: {process_result.stdout}")
    if process_result.stderr:
        print(f"❌ STDERR: {process_result.stderr}")
    
    if process_result.returncode == 0:
        print("✅ Production test completed successfully!")
        try:
            response = json.loads(process_result.stdout)
            print(f"📊 Response: {json.dumps(response, indent=2)}")
        except:
            print("📄 Response (not JSON):", process_result.stdout)
    else:
        print("❌ Production test failed!")
        
except subprocess.TimeoutExpired:
    print("⏰ Request timed out after 5 minutes")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
