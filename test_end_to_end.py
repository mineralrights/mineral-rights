#!/usr/bin/env python3
"""
End-to-end test to debug the page-by-page processing flow
"""

import os
import sys
import requests
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.append('/Users/lauragomez/Desktop/mineral-rights/src')

def test_end_to_end():
    """Test the complete page-by-page processing flow locally"""
    
    # Set up environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found in environment")
        print("Please set it with: export ANTHROPIC_API_KEY='your_key_here'")
        return False
    
    # Test file
    test_pdf = "/Users/lauragomez/Desktop/mineral-rights/data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    print(f"🔍 Testing end-to-end page-by-page processing")
    print(f"📄 Test file: {test_pdf}")
    print(f"📊 File size: {os.path.getsize(test_pdf) / 1024 / 1024:.1f} MB")
    
    try:
        # Step 1: Test the backend endpoint directly
        print(f"\n🚀 Step 1: Testing backend endpoint directly...")
        
        # Create a test GCS URL (simulate what the frontend would send)
        gcs_url = f"file://{test_pdf}"
        
        data = {
            'gcs_url': gcs_url,
            'processing_mode': 'page_by_page',
            'splitting_strategy': 'document_ai'
        }
        
        url = "http://localhost:8000/process-large-pdf"
        print(f"📡 Calling: {url}")
        print(f"📄 GCS URL: {gcs_url}")
        print(f"🔧 Processing mode: page_by_page")
        
        # Set a reasonable timeout
        response = requests.post(url, data=data, timeout=60)
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Result keys: {list(result.keys())}")
            print(f"📄 Total pages: {result.get('total_pages', 'N/A')}")
            print(f"🎯 Pages with reservations: {result.get('pages_with_reservations', 'N/A')}")
            print(f"📋 Reservation pages: {result.get('reservation_pages', [])}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"❌ Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - processing is working but taking too long")
        return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_small_pdf():
    """Test with a smaller PDF to avoid timeout"""
    
    # Create a small test PDF (just copy the first few pages)
    print(f"\n🔍 Testing with smaller PDF...")
    
    try:
        import fitz
        
        # Open the original PDF
        doc = fitz.open("/Users/lauragomez/Desktop/mineral-rights/data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf")
        
        # Create a new PDF with just the first 3 pages
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=0, to_page=2)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            new_doc.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        new_doc.close()
        doc.close()
        
        print(f"📄 Created small PDF: {tmp_file_path}")
        print(f"📊 File size: {os.path.getsize(tmp_file_path) / 1024:.1f} KB")
        
        # Test with the small PDF
        gcs_url = f"file://{tmp_file_path}"
        
        data = {
            'gcs_url': gcs_url,
            'processing_mode': 'page_by_page',
            'splitting_strategy': 'document_ai'
        }
        
        url = "http://localhost:8000/process-large-pdf"
        print(f"📡 Testing with small PDF...")
        
        response = requests.post(url, data=data, timeout=120)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success with small PDF!")
            print(f"📄 Total pages: {result.get('total_pages', 'N/A')}")
            print(f"🎯 Pages with reservations: {result.get('pages_with_reservations', 'N/A')}")
            print(f"📋 Reservation pages: {result.get('reservation_pages', [])}")
            
            # Clean up
            os.unlink(tmp_file_path)
            return True
        else:
            print(f"❌ Error with small PDF: {response.status_code}")
            print(f"❌ Response text: {response.text}")
            # Clean up
            os.unlink(tmp_file_path)
            return False
            
    except Exception as e:
        print(f"❌ Exception with small PDF: {e}")
        return False

if __name__ == "__main__":
    print("🧪 END-TO-END PAGE-BY-PAGE TEST")
    print("=" * 50)
    
    # Test 1: Full PDF (might timeout)
    print("\n🔍 Test 1: Full PDF")
    success1 = test_end_to_end()
    
    # Test 2: Small PDF (should work)
    print("\n🔍 Test 2: Small PDF (3 pages)")
    success2 = test_small_pdf()
    
    print(f"\n📊 RESULTS:")
    print(f"   - Full PDF test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"   - Small PDF test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success2:
        print(f"\n🎉 The backend is working! The issue might be:")
        print(f"   1. Production deployment not updated")
        print(f"   2. Different environment variables in production")
        print(f"   3. CORS issues in production")
    else:
        print(f"\n❌ Backend has issues that need to be fixed")
