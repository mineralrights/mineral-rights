#!/usr/bin/env python3
"""
Test script to verify deployment fixes work correctly
"""

import requests
import time
import json
import sys
from pathlib import Path

def test_api_health(api_url):
    """Test if API is healthy"""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data}")
            return True
        else:
            print(f"❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health error: {e}")
        return False

def test_job_system(api_url):
    """Test if job system is working"""
    try:
        response = requests.get(f"{api_url}/jobs/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Job System: {data}")
            return True
        else:
            print(f"❌ Job System failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Job System error: {e}")
        return False

def test_job_creation(api_url):
    """Test job creation without file upload"""
    try:
        response = requests.post(f"{api_url}/jobs/test", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Job Creation Test: {data}")
            return data.get('job_id')
        else:
            print(f"❌ Job Creation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Job Creation error: {e}")
        return None

def test_job_status(api_url, job_id):
    """Test job status retrieval"""
    try:
        response = requests.get(f"{api_url}/jobs/{job_id}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Job Status: {data}")
            return True
        else:
            print(f"❌ Job Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Job Status error: {e}")
        return False

def test_job_result(api_url, job_id):
    """Test job result retrieval"""
    try:
        response = requests.get(f"{api_url}/jobs/{job_id}/result", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Job Result: {data}")
            return True
        else:
            print(f"❌ Job Result failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Job Result error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Deployment Fixes")
    print("=" * 50)
    
    # Get API URL from environment or use default
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"Testing API at: {api_url}")
    print()
    
    # Test 1: API Health
    print("1. Testing API Health...")
    if not test_api_health(api_url):
        print("❌ API Health test failed - stopping tests")
        return False
    print()
    
    # Test 2: Job System Health
    print("2. Testing Job System Health...")
    if not test_job_system(api_url):
        print("❌ Job System test failed - stopping tests")
        return False
    print()
    
    # Test 3: Job Creation
    print("3. Testing Job Creation...")
    job_id = test_job_creation(api_url)
    if not job_id:
        print("❌ Job Creation test failed - stopping tests")
        return False
    print()
    
    # Test 4: Job Status
    print("4. Testing Job Status...")
    if not test_job_status(api_url, job_id):
        print("❌ Job Status test failed")
        return False
    print()
    
    # Test 5: Job Result
    print("5. Testing Job Result...")
    if not test_job_result(api_url, job_id):
        print("❌ Job Result test failed")
        return False
    print()
    
    print("🎉 All tests passed! Deployment fixes are working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
