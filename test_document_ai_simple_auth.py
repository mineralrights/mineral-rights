#!/usr/bin/env python3
"""
Simple Document AI test using gcloud authentication
"""

import os
import subprocess
import tempfile
from pathlib import Path

def test_gcloud_auth():
    """Test if gcloud authentication is working"""
    print("🧪 Testing gcloud authentication...")
    
    try:
        # Check if gcloud is authenticated
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            account = result.stdout.strip()
            print(f"✅ Authenticated as: {account}")
            return True
        else:
            print("❌ No active gcloud authentication")
            return False
            
    except Exception as e:
        print(f"❌ Error checking gcloud auth: {e}")
        return False

def test_document_ai_with_gcloud():
    """Test Document AI using gcloud authentication"""
    print("🧪 Testing Document AI with gcloud auth...")
    
    try:
        # Try to use gcloud to get an access token
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            access_token = result.stdout.strip()
            print(f"✅ Got access token: {access_token[:20]}...")
            
            # Test the Document AI endpoint
            processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
            
            print(f"📡 Testing endpoint: {processor_endpoint}")
            
            # Try a simple request to test the endpoint
            import requests
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Test with a minimal request (this will likely fail but will tell us about permissions)
            response = requests.get(processor_endpoint, headers=headers)
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Endpoint is accessible!")
                return True
            elif response.status_code == 403:
                print("⚠️ Permission denied - check if Document AI API is enabled")
                return False
            elif response.status_code == 404:
                print("⚠️ Processor not found - check the processor ID")
                return False
            else:
                print(f"⚠️ Unexpected response: {response.text[:200]}")
                return False
                
        else:
            print(f"❌ Failed to get access token: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Document AI: {e}")
        return False

def test_with_sample_pdf():
    """Find a sample PDF to test with"""
    print("🧪 Looking for sample PDFs...")
    
    sample_locations = [
        "data/multi-deed",
        "data/synthetic_dataset/test", 
        "data/synthetic_dataset/train",
        ".",
        "test_data"
    ]
    
    pdf_files = []
    for location in sample_locations:
        if os.path.exists(location):
            pdf_files.extend(Path(location).glob("*.pdf"))
    
    if pdf_files:
        print(f"📄 Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files[:3]:  # Show first 3
            print(f"   - {pdf_file}")
        return str(pdf_files[0])
    else:
        print("⚠️ No PDF files found")
        return None

def main():
    """Main test function"""
    print("🚀 Simple Document AI Authentication Test")
    print("=" * 50)
    
    # Step 1: Test gcloud auth
    if not test_gcloud_auth():
        print("\n❌ gcloud authentication failed")
        print("💡 Run: gcloud auth login")
        return False
    
    # Step 2: Test Document AI access
    if not test_document_ai_with_gcloud():
        print("\n❌ Document AI access failed")
        print("💡 Check:")
        print("   - Document AI API is enabled")
        print("   - Processor ID is correct")
        print("   - You have proper permissions")
        return False
    
    # Step 3: Find test PDF
    test_pdf = test_with_sample_pdf()
    
    if test_pdf:
        print(f"\n📄 Found test PDF: {test_pdf}")
        print("🎯 Ready to test Document AI processing!")
        print("💡 Next step: Test with actual PDF processing")
    else:
        print("\n⚠️ No PDF found for testing")
        print("💡 You can test with any PDF file")
    
    print("\n✅ Basic authentication and access is working!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Document AI is accessible!")
        print("📋 Next steps:")
        print("1. Test with a real PDF file")
        print("2. Check if deed boundaries are detected")
        print("3. Integrate into the app")
    else:
        print("\n🔧 Fix the authentication issues first")
    
    exit(0 if success else 1)



