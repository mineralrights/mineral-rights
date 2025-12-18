#!/usr/bin/env python3
"""
Test production deployment with 32GB memory limit
"""
import os
import sys
import tempfile
import shutil

# Test with a small synthetic PDF
pdf_path = "data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ Test PDF not found: {pdf_path}")
    sys.exit(1)

print("🧪 Testing Production Deployment with 32GB Memory")
print(f"📄 Test PDF: {pdf_path}")

# Create a temporary copy for testing
temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
shutil.copy2(pdf_path, temp_pdf.name)
temp_pdf.close()

print(f"📁 Temporary PDF: {temp_pdf.name}")

# Test the production endpoint
import subprocess
import json

try:
    # Test with curl
    cmd = [
        'curl', '-X', 'POST',
        'https://mineral-rights-processor-1081023230228.us-central1.run.app/process-large-pdf',
        '-F', f'gcs_url=file://{temp_pdf.name}',
        '-F', 'processing_mode=page_by_page',
        '-F', 'splitting_strategy=document_ai',
        '--max-time', '300'  # 5 minute timeout
    ]
    
    print("🚀 Sending request to production...")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    print(f"📊 Exit code: {result.returncode}")
    print(f"📤 STDOUT: {result.stdout}")
    if result.stderr:
        print(f"❌ STDERR: {result.stderr}")
    
    if result.returncode == 0:
        print("✅ Production test completed successfully!")
        try:
            response = json.loads(result.stdout)
            print(f"📊 Response: {json.dumps(response, indent=2)}")
        except:
            print("📄 Response (not JSON):", result.stdout)
    else:
        print("❌ Production test failed!")
        
except subprocess.TimeoutExpired:
    print("⏰ Request timed out after 5 minutes")
except Exception as e:
    print(f"❌ Error: {e}")

finally:
    # Clean up
    if os.path.exists(temp_pdf.name):
        os.unlink(temp_pdf.name)
        print("🧹 Cleaned up temporary file")
