#!/usr/bin/env python3
"""
Simple test to isolate the SSL issue during file upload
"""
import requests
import time
import os

def test_simple_upload():
    """Test with a very small file to isolate the issue"""
    base_url = "https://mineral-rights.onrender.com"
    
    print("🧪 Testing Simple File Upload")
    print("=" * 50)
    
    # Create a minimal test file
    test_content = b"Test PDF content for debugging"
    
    print("1. Testing with minimal file...")
    try:
        files = {'file': ('test.pdf', test_content, 'application/pdf')}
        data = {
            'processing_mode': 'single_deed',
            'splitting_strategy': 'smart_detection'
        }
        
        print("   Sending request...")
        response = requests.post(
            f"{base_url}/predict", 
            files=files, 
            data=data, 
            timeout=60
        )
        
        print(f"   Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"   ✅ Upload successful! Job ID: {job_id}")
            
            # Test stream endpoint
            if job_id:
                print(f"2. Testing stream endpoint...")
                try:
                    stream_response = requests.get(
                        f"{base_url}/stream/{job_id}", 
                        timeout=30
                    )
                    print(f"   Stream status: {stream_response.status_code}")
                    if stream_response.status_code == 200:
                        print("   ✅ Stream endpoint accessible")
                    else:
                        print(f"   ❌ Stream failed: {stream_response.text}")
                except Exception as e:
                    print(f"   ❌ Stream error: {e}")
        else:
            print(f"   ❌ Upload failed: {response.text}")
            
    except requests.exceptions.SSLError as e:
        print(f"   ❌ SSL Error: {e}")
        print("   This confirms the SSL issue occurs during processing")
    except requests.exceptions.Timeout as e:
        print(f"   ❌ Timeout Error: {e}")
        print("   Server may be taking too long to respond")
    except Exception as e:
        print(f"   ❌ Other Error: {e}")

def test_health_after_upload():
    """Test if API is still healthy after upload attempt"""
    base_url = "https://mineral-rights.onrender.com"
    
    print("\n3. Testing API health after upload attempt...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ API still healthy: {health}")
        else:
            print(f"   ❌ API unhealthy: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")

if __name__ == "__main__":
    test_simple_upload()
    test_health_after_upload()
    
    print("\n" + "=" * 50)
    print("🔍 Analysis:")
    print("If SSL error occurs with minimal file, the issue is in processing logic")
    print("If API becomes unhealthy, the server is crashing during processing")
    print("Check Render logs for detailed error information")
