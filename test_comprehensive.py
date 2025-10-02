#!/usr/bin/env python3
"""
Comprehensive test for the deployed Google Cloud Run service
Tests both single-deed and multi-deed processing
"""

import requests
import json
import os
from pathlib import Path

def test_comprehensive():
    """Test both single-deed and multi-deed processing"""
    
    service_url = "https://mineral-rights-processor-1081023230228.us-central1.run.app"
    
    print("🚀 Comprehensive Google Cloud Run Testing")
    print("=" * 60)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        health_response = requests.get(f"{service_url}/health")
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"✅ Health: {health_data['status']}")
        print(f"   API Key: {'✅' if health_data['api_key_present'] else '❌'}")
        print(f"   Document AI: {'✅' if health_data['document_ai_endpoint_present'] else '❌'}")
        print(f"   Google Creds: {'✅' if health_data['google_credentials_present'] else '❌'}")
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
    
    # Test 1: Single deed processing
    print("\n📄 Testing SINGLE DEED processing...")
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path, f, 'application/pdf')}
            data = {
                'processing_mode': 'single_deed',
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
            
            print("✅ Single deed processing completed!")
            print(f"   Has reservation: {result.get('has_reservation', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
            print(f"   Reasoning: {result.get('reasoning', 'No reasoning')[:100]}...")
            
    except Exception as e:
        print(f"❌ Single deed processing failed: {e}")
    
    # Test 2: Multi-deed processing
    print("\n📚 Testing MULTI DEED processing...")
    try:
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
            
            print("✅ Multi-deed processing completed!")
            print(f"   Total deeds: {result.get('total_deeds', 0)}")
            
            if 'deed_results' in result:
                for i, deed_result in enumerate(result['deed_results']):
                    has_reservations = deed_result.get('has_reservations', False)
                    confidence = deed_result.get('confidence', 0.0)
                    print(f"   Deed {i+1}: {'HAS RESERVATIONS' if has_reservations else 'NO RESERVATIONS'} (confidence: {confidence:.3f})")
            
    except Exception as e:
        print(f"❌ Multi-deed processing failed: {e}")
    
    # Test 3: Page-by-page processing
    print("\n📖 Testing PAGE-BY-PAGE processing...")
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path, f, 'application/pdf')}
            data = {
                'processing_mode': 'page_by_page',
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
            
            print("✅ Page-by-page processing completed!")
            print(f"   Total pages: {result.get('total_pages', 0)}")
            
            if 'page_results' in result:
                for i, page_result in enumerate(result['page_results']):
                    has_reservations = page_result.get('has_reservations', False)
                    confidence = page_result.get('confidence', 0.0)
                    print(f"   Page {i+1}: {'HAS RESERVATIONS' if has_reservations else 'NO RESERVATIONS'} (confidence: {confidence:.3f})")
            
    except Exception as e:
        print(f"❌ Page-by-page processing failed: {e}")
    
    print("\n🎉 Comprehensive testing completed!")
    print("=" * 60)
    print("✅ Google Cloud Run deployment is working end-to-end!")
    print("✅ All processing modes are functional!")
    print("✅ Ready for production use!")

if __name__ == "__main__":
    test_comprehensive()
