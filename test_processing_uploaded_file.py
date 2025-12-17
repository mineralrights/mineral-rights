#!/usr/bin/env python3
"""
Test processing the file that was just uploaded successfully
"""
import os
import sys
import subprocess
import json

# Use the GCS URL from the previous test
gcs_url = "https://storage.googleapis.com/mineral-rights-pdfs-1759435410/uploads/7459f3cd-457a-4ba3-b142-885177313b36/test_signed_url.pdf"

print("🧪 Testing Processing of Successfully Uploaded File")
print(f"📄 GCS URL: {gcs_url}")

try:
    # Test with original working multi-deed logic
    print("\n🔍 Testing with original working multi-deed logic...")
    process_cmd = [
        'curl', '-X', 'POST',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/process-large-pdf',
        '-F', f'gcs_url={gcs_url}',
        '-F', 'processing_mode=single_deed',  # Use original working mode
        '-F', 'splitting_strategy=document_ai',
        '--max-time', '120'
    ]
    
    process_result = subprocess.run(process_cmd, capture_output=True, text=True)
    print(f"📊 Response code: {process_result.returncode}")
    print(f"📤 Response: {process_result.stdout[:1000]}")
    
    if "success" in process_result.stdout.lower() or "completed" in process_result.stdout.lower():
        print("\n✅ SUCCESS: Original working logic works with uploaded file!")
    else:
        print("\n❌ FAILED: Even with uploaded file, processing fails")
        print(f"Full response: {process_result.stdout}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

