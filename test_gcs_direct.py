#!/usr/bin/env python3
"""
Test if uploaded files actually exist in GCS
"""
import os
import sys
import subprocess
import json
import time

# Test with a small synthetic PDF
pdf_path = "data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ Test PDF not found: {pdf_path}")
    sys.exit(1)

print("🧪 Testing GCS Upload and Immediate Verification")

try:
    # Step 1: Get signed upload URL
    print("\n🔑 Step 1: Getting signed upload URL...")
    upload_cmd = [
        'curl', '-X', 'POST',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/get-signed-upload-url',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({
            'filename': 'test_verify.pdf',
            'content_type': 'application/pdf'
        })
    ]
    
    upload_result = subprocess.run(upload_cmd, capture_output=True, text=True)
    if upload_result.returncode != 0:
        print(f"❌ Failed: {upload_result.stderr}")
        sys.exit(1)
    
    upload_data = json.loads(upload_result.stdout)
    signed_url = upload_data['signed_url']
    gcs_url = upload_data['gcs_url']
    
    print(f"✅ Signed URL: {signed_url[:80]}...")
    print(f"✅ GCS URL: {gcs_url}")
    
    # Step 2: Upload to GCS
    print("\n📤 Step 2: Uploading to GCS...")
    upload_file_cmd = [
        'curl', '-X', 'PUT',
        signed_url,
        '--data-binary', f'@{pdf_path}'
    ]
    
    upload_file_result = subprocess.run(upload_file_cmd, capture_output=True, text=True)
    if upload_file_result.returncode != 0:
        print(f"❌ Upload failed: {upload_file_result.stderr}")
        sys.exit(1)
    
    print("✅ File uploaded")
    
    # Step 3: Wait and try to access via Cloud Run
    print("\n⏳ Step 3: Waiting 5 seconds...")
    time.sleep(5)
    
    # Step 4: Try to download using the working /process-gcs endpoint
    print("\n🔍 Step 4: Testing download via /process-gcs endpoint...")
    process_cmd = [
        'curl', '-X', 'POST',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/process-gcs',
        '-F', f'gcs_url={gcs_url}',
        '-F', 'processing_mode=single_deed',
        '-F', 'splitting_strategy=simple',
        '--max-time', '60'
    ]
    
    process_result = subprocess.run(process_cmd, capture_output=True, text=True)
    print(f"📊 Response code: {process_result.returncode}")
    print(f"📤 Response: {process_result.stdout[:500]}")
    
    if "success" in process_result.stdout.lower() or "completed" in process_result.stdout.lower():
        print("\n✅ SUCCESS: File was uploaded and can be accessed!")
    else:
        print("\n❌ FAILED: File uploaded but cannot be accessed")
        print(f"Full response: {process_result.stdout}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

