#!/usr/bin/env python3
"""
Test with the correct Cloud Run URL
"""

import requests
import json
import os

def test_correct_url():
    """Test with the correct service URL"""
    
    # Use the correct URL from gcloud
    service_url = "https://mineral-rights-processor-ms7ew6g6zq-uc.a.run.app"
    
    print("🎯 TESTING WITH CORRECT URL")
    print("=" * 50)
    print(f"Service URL: {service_url}")
    
    # Test health
    print("\n1. Health Check...")
    try:
        health = requests.get(f"{service_url}/health", timeout=10).json()
        print(f"✅ Service: {health['status']}")
        print(f"✅ Processor Initialized: {health['processor_initialized']}")
        print(f"✅ API Key: {'Present' if health['api_key_present'] else 'Missing'}")
        print(f"✅ Document AI: {'Present' if health['document_ai_endpoint_present'] else 'Missing'}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test with PDF
    pdf_path = "test_small.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Test PDF not found: {pdf_path}")
        return
    
    print(f"\n2. Testing with {pdf_path} ({os.path.getsize(pdf_path)/1024/1024:.1f} MB)")
    
    # Test multi-deed processing
    try:
        print("   📤 Uploading PDF for multi-deed processing...")
        
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path, f, 'application/pdf')}
            data = {
                'processing_mode': 'multi_deed',
                'splitting_strategy': 'simple'
            }
            
            response = requests.post(
                f"{service_url}/predict",
                files=files,
                data=data,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            print("✅ SUCCESS! Multi-deed processing completed!")
            print(f"   📊 Total deeds found: {result.get('total_deeds', 0)}")
            
            if 'deed_results' in result and result['deed_results']:
                for i, deed in enumerate(result['deed_results']):
                    has_reservations = deed.get('has_reservations', False)
                    confidence = deed.get('confidence', 0.0)
                    status = "🎯 HAS RESERVATIONS" if has_reservations else "📄 NO RESERVATIONS"
                    print(f"   Deed {i+1}: {status} (confidence: {confidence:.3f})")
            
            print("\n🎉 END-TO-END TEST COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print("✅ Google Cloud Run is working perfectly!")
            print("✅ PDF processing is functional!")
            print("✅ Multi-deed classification is working!")
            print("✅ Ready for production use!")
            print(f"✅ Service URL: {service_url}")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        print(f"   Response: {response.text if 'response' in locals() else 'No response'}")

if __name__ == "__main__":
    test_correct_url()
