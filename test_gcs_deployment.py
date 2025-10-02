#!/usr/bin/env python3
"""
Test GCS deployment and large file handling
"""

import requests
import os

def test_api_health():
    """Test if the API is healthy"""
    url = "https://mineral-rights-processor-1081023230228.us-central1.run.app/health"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Health Check:")
            print(f"   Status: {data.get('status')}")
            print(f"   API Key Present: {data.get('api_key_present')}")
            print(f"   Document AI Present: {data.get('document_ai_endpoint_present')}")
            print(f"   GCS Available: {data.get('gcs_available', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_gcs_upload():
    """Test GCS upload endpoint"""
    url = "https://mineral-rights-processor-1081023230228.us-central1.run.app/upload-gcs"
    
    # Test with a small file first
    test_file_path = "test_small.pdf"
    if not os.path.exists(test_file_path):
        print("❌ Test file not found. Please create a small PDF for testing.")
        return False
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'processing_mode': 'multi_deed',
                'splitting_strategy': 'document_ai'
            }
            
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ GCS Upload Test:")
                print(f"   Success: {result.get('success')}")
                print(f"   File Size: {result.get('file_size_mb', 0):.1f}MB")
                print(f"   GCS URL: {result.get('gcs_url', 'N/A')}")
                return True
            else:
                print(f"❌ GCS upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
    except Exception as e:
        print(f"❌ GCS upload error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing GCS Deployment")
    print("=" * 50)
    
    # Test 1: API Health
    print("\n1. Testing API Health...")
    health_ok = test_api_health()
    
    # Test 2: GCS Upload (if health is OK)
    if health_ok:
        print("\n2. Testing GCS Upload...")
        gcs_ok = test_gcs_upload()
        
        if gcs_ok:
            print("\n🎉 All tests passed! GCS deployment is working.")
        else:
            print("\n❌ GCS upload test failed.")
    else:
        print("\n❌ API health check failed. Cannot test GCS upload.")
    
    print("\n📋 Next Steps:")
    print("1. Update Vercel environment variables")
    print("2. Redeploy Vercel frontend")
    print("3. Test with ROBERT.pdf (43MB)")
