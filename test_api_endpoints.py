#!/usr/bin/env python3
"""
Test script to verify API endpoints work correctly
"""
import requests
import json
import time

def test_api_endpoints(base_url):
    """Test the API endpoints to diagnose issues"""
    print(f"🧪 Testing API endpoints at: {base_url}")
    print("=" * 60)
    
    # Test health endpoint
    print("1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print()
    
    # Test debug endpoint
    print("2. Testing /debug endpoint...")
    try:
        response = requests.get(f"{base_url}/debug", timeout=10)
        if response.status_code == 200:
            debug_data = response.json()
            print(f"✅ Debug info retrieved:")
            print(f"   Imports: {debug_data.get('imports', {})}")
            print(f"   System: {debug_data.get('system', {})}")
        else:
            print(f"❌ Debug check failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Debug check error: {e}")
    
    print()
    
    # Test memory status endpoint
    print("3. Testing /memory-status endpoint...")
    try:
        response = requests.get(f"{base_url}/memory-status", timeout=10)
        if response.status_code == 200:
            memory_data = response.json()
            print(f"✅ Memory status: {memory_data}")
        else:
            print(f"❌ Memory status failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Memory status error: {e}")
    
    print()
    
    # Test predict endpoint with a small file
    print("4. Testing /predict endpoint...")
    try:
        # Create a simple test file
        test_content = b"Test PDF content"
        files = {'file': ('test.pdf', test_content, 'application/pdf')}
        data = {
            'processing_mode': 'single_deed',
            'splitting_strategy': 'smart_detection'
        }
        
        response = requests.post(f"{base_url}/predict", files=files, data=data, timeout=30)
        if response.status_code == 200:
            predict_data = response.json()
            job_id = predict_data.get('job_id')
            print(f"✅ Predict endpoint works: job_id = {job_id}")
            
            # Test stream endpoint
            if job_id:
                print(f"5. Testing /stream/{job_id} endpoint...")
                try:
                    stream_response = requests.get(f"{base_url}/stream/{job_id}", timeout=10)
                    if stream_response.status_code == 200:
                        print(f"✅ Stream endpoint accessible")
                    else:
                        print(f"❌ Stream endpoint failed: {stream_response.status_code}")
                except Exception as e:
                    print(f"❌ Stream endpoint error: {e}")
        else:
            print(f"❌ Predict endpoint failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Predict endpoint error: {e}")

if __name__ == "__main__":
    # Test with local development server
    local_url = "http://localhost:8000"
    print("Testing local development server...")
    test_api_endpoints(local_url)
    
    print("\n" + "=" * 60)
    print("To test your Render deployment, run:")
    print("python test_api_endpoints.py https://your-app.onrender.com")
