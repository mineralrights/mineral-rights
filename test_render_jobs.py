#!/usr/bin/env python3
"""
Test Script for Render Jobs Setup
=================================

This script tests the Render Jobs functionality locally before deployment.
It simulates the job creation, monitoring, and result retrieval process.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from job_manager import RenderJobManager, create_processing_job, get_job_status, get_job_result
from render_jobs_solution import process_document_job

def test_job_manager():
    """Test the job manager functionality"""
    print("🧪 Testing Job Manager...")
    
    # Create a test PDF path (you can replace this with an actual PDF)
    test_pdf = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
    
    # Test job creation
    print("1. Creating test job...")
    job_id = create_processing_job(
        test_pdf,
        "multi_deed",
        "smart_detection"
    )
    print(f"   ✅ Job created: {job_id}")
    
    # Test job status monitoring
    print("2. Monitoring job status...")
    for i in range(5):
        status = get_job_status(job_id)
        if status:
            print(f"   Status: {status['status']}")
            if status['status'] == 'completed':
                break
        time.sleep(2)
    
    # Test result retrieval
    print("3. Retrieving job result...")
    result = get_job_result(job_id)
    if result:
        print(f"   ✅ Result retrieved: {result}")
    else:
        print("   ⚠️  No result yet (job may still be running)")
    
    print("✅ Job Manager test completed\n")

def test_render_job_solution():
    """Test the Render job solution script"""
    print("🧪 Testing Render Job Solution...")
    
    # Check if we have a test PDF
    test_pdf = None
    for ext in ['.pdf']:
        test_files = list(Path('.').glob(f'*{ext}'))
        if test_files:
            test_pdf = str(test_files[0])
            break
    
    if not test_pdf:
        print("   ⚠️  No PDF files found for testing")
        print("   Create a test PDF or update the path in this script")
        return
    
    print(f"   Using test PDF: {test_pdf}")
    
    try:
        # Test the job solution
        result = process_document_job(
            test_pdf,
            "multi_deed",
            "smart_detection"
        )
        print(f"   ✅ Job solution test completed: {result}")
    except Exception as e:
        print(f"   ❌ Job solution test failed: {e}")
        print("   This is expected if you don't have the full environment set up")
    
    print("✅ Render Job Solution test completed\n")

def test_api_endpoints():
    """Test the API endpoints (requires FastAPI to be running)"""
    print("🧪 Testing API Endpoints...")
    
    try:
        import requests
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/jobs/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Jobs health endpoint working")
            else:
                print(f"   ⚠️  Jobs health endpoint returned: {response.status_code}")
        except requests.exceptions.RequestException:
            print("   ⚠️  API not running (start with: uvicorn api.app:app --reload)")
        
        # Test jobs list endpoint
        try:
            response = requests.get("http://localhost:8000/jobs/", timeout=5)
            if response.status_code == 200:
                print("   ✅ Jobs list endpoint working")
            else:
                print(f"   ⚠️  Jobs list endpoint returned: {response.status_code}")
        except requests.exceptions.RequestException:
            print("   ⚠️  API not running")
        
    except ImportError:
        print("   ⚠️  requests library not available for API testing")
    
    print("✅ API Endpoints test completed\n")

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking Environment...")
    
    # Check required files
    required_files = [
        "render.yaml",
        "render_jobs_solution.py", 
        "job_manager.py",
        "job_api_endpoints.py",
        "requirements-clean.txt"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - Missing!")
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print(f"   ✅ ANTHROPIC_API_KEY is set")
    else:
        print(f"   ⚠️  ANTHROPIC_API_KEY not set (required for processing)")
    
    # Check Python path
    try:
        from src.mineral_rights.document_classifier import DocumentProcessor
        print("   ✅ DocumentProcessor can be imported")
    except ImportError as e:
        print(f"   ❌ DocumentProcessor import failed: {e}")
    
    print("✅ Environment check completed\n")

def main():
    """Run all tests"""
    print("🚀 Render Jobs Test Suite")
    print("=" * 50)
    
    check_environment()
    test_job_manager()
    test_render_job_solution()
    test_api_endpoints()
    
    print("🎉 All tests completed!")
    print("\n📋 Next Steps:")
    print("1. Fix any missing files or environment issues")
    print("2. Deploy to Render using the setup guide")
    print("3. Test with actual documents")
    print("4. Update your frontend to use the new job endpoints")

if __name__ == "__main__":
    main()
