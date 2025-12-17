#!/usr/bin/env python3
"""
Test signed URL generation and upload process
"""
import os
import sys
import subprocess
import json
import tempfile
import shutil

# Test with a small synthetic PDF
pdf_path = "data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ Test PDF not found: {pdf_path}")
    sys.exit(1)

print("🧪 Testing Signed URL Generation and Upload Process")
print(f"📄 Test PDF: {pdf_path}")

try:
    # Step 1: Get signed upload URL
    print("\n🔑 Step 1: Getting signed upload URL...")
    upload_cmd = [
        'curl', '-X', 'POST',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/get-signed-upload-url',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({
            'filename': 'test_signed_url.pdf',
            'content_type': 'application/pdf'
        })
    ]
    
    upload_result = subprocess.run(upload_cmd, capture_output=True, text=True)
    print(f"📊 Upload URL response code: {upload_result.returncode}")
    print(f"📤 STDOUT: {upload_result.stdout}")
    if upload_result.stderr:
        print(f"📤 STDERR: {upload_result.stderr}")
    
    if upload_result.returncode != 0:
        print(f"❌ Failed to get signed URL: {upload_result.stderr}")
        sys.exit(1)
    
    upload_data = json.loads(upload_result.stdout)
    signed_url = upload_data['signed_url']
    gcs_url = upload_data['gcs_url']
    
    print(f"✅ Signed URL: {signed_url[:100]}...")
    print(f"✅ GCS URL: {gcs_url}")
    
    # Step 2: Upload to GCS using signed URL
    print("\n📤 Step 2: Uploading to GCS using signed URL...")
    upload_file_cmd = [
        'curl', '-X', 'PUT',
        signed_url,
        '--data-binary', f'@{pdf_path}',
        '-H', 'Content-Type: application/pdf',
        '-v'  # Verbose to see what's happening
    ]
    
    upload_file_result = subprocess.run(upload_file_cmd, capture_output=True, text=True)
    print(f"📊 Upload file response code: {upload_file_result.returncode}")
    print(f"📤 STDOUT: {upload_file_result.stdout}")
    if upload_file_result.stderr:
        print(f"📤 STDERR: {upload_file_result.stderr}")
    
    if upload_file_result.returncode != 0:
        print(f"❌ Upload failed: {upload_file_result.stderr}")
        sys.exit(1)
    
    print("✅ File uploaded successfully")
    
    # Step 3: Verify file exists in GCS
    print("\n🔍 Step 3: Verifying file exists in GCS...")
    verify_cmd = [
        'curl', '-I', gcs_url
    ]
    
    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
    print(f"📊 Verify response code: {verify_result.returncode}")
    print(f"📤 STDOUT: {verify_result.stdout}")
    if verify_result.stderr:
        print(f"📤 STDERR: {verify_result.stderr}")
    
    if verify_result.returncode == 0 and "200 OK" in verify_result.stdout:
        print("✅ File exists in GCS!")
    else:
        print("❌ File does not exist in GCS")
        print(f"Full response: {verify_result.stdout}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

