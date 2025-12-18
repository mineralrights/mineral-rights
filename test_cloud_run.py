#!/usr/bin/env python3
"""
Test script for the deployed Google Cloud Run service
"""

import requests
import json
import os
from pathlib import Path

def test_cloud_run_service():
    """Test the deployed Cloud Run service with a real PDF"""
    
    service_url = "https://mineral-rights-processor-1081023230228.us-central1.run.app"
    
    print("🚀 Testing Google Cloud Run Service")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        health_response = requests.get(f"{service_url}/health")
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"✅ Health check: {health_data}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test with a real PDF
    pdf_path = "test_small.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Test PDF not found: {pdf_path}")
        return
    
    print(f"\n2. Testing with PDF: {pdf_path}")
    print(f"   File size: {os.path.getsize(pdf_path) / 1024 / 1024:.1f} MB")
    
    try:
        # Upload PDF for processing
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path, f, 'application/pdf')}
            data = {
                'strategy': 'simple',  # Use simple splitting for testing
                'max_samples': 4,      # Fewer samples for faster testing
                'high_recall_mode': 'true'
            }
            
            print("   📤 Uploading PDF...")
            response = requests.post(
                f"{service_url}/predict",
                files=files,
                data={
                    'processing_mode': 'multi_deed',
                    'splitting_strategy': 'simple'
                },
                timeout=300  # 5 minute timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            print("✅ Processing completed!")
            print(f"   📊 Results: {json.dumps(result, indent=2)}")
            
            # Check if we got meaningful results
            if 'deed_results' in result and len(result['deed_results']) > 0:
                print(f"\n🎯 Found {len(result['deed_results'])} deed(s)")
                for i, deed_result in enumerate(result['deed_results']):
                    has_reservations = deed_result.get('has_reservations', False)
                    confidence = deed_result.get('confidence', 0.0)
                    print(f"   Deed {i+1}: {'HAS RESERVATIONS' if has_reservations else 'NO RESERVATIONS'} (confidence: {confidence:.3f})")
            else:
                print("⚠️  No results returned")
                
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - this is expected for long PDFs")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        print(f"   Response: {response.text if 'response' in locals() else 'No response'}")

if __name__ == "__main__":
    test_cloud_run_service()
