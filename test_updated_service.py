#!/usr/bin/env python3
"""
Test the updated Document AI service

This script tests the updated Document AI service with chunking support.
"""

import os
import sys
from pathlib import Path

def test_updated_service():
    """Test the updated Document AI service"""
    print("🧪 Testing updated Document AI service...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from mineral_rights.document_ai_service import create_document_ai_service
        
        # Create the service
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        service = create_document_ai_service(processor_endpoint)
        
        print("✅ Document AI service created successfully")
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        print(f"📄 Testing with PDF: {test_pdf_path}")
        
        # Process the PDF
        result = service.split_deeds_from_pdf(test_pdf_path)
        
        print(f"\n📊 RESULTS:")
        print(f"✅ Total deeds found: {result.total_deeds}")
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        
        print(f"\n🔍 Deed details:")
        for deed in result.deeds:
            print(f"   - Deed {deed.deed_number}: pages {min(deed.pages)+1}-{max(deed.pages)+1} (confidence: {deed.confidence:.3f})")
        
        # Test creating individual PDFs
        print(f"\n📄 Creating individual deed PDFs...")
        deed_pdfs = service.create_individual_deed_pdfs(test_pdf_path, result)
        
        print(f"✅ Created {len(deed_pdfs)} individual deed PDFs:")
        for i, pdf_path in enumerate(deed_pdfs):
            print(f"   - {pdf_path}")
        
        # Clean up
        print(f"\n🧹 Cleaning up individual PDFs...")
        for pdf_path in deed_pdfs:
            try:
                os.remove(pdf_path)
                print(f"✅ Removed: {pdf_path}")
            except Exception as e:
                print(f"⚠️ Could not remove {pdf_path}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Test Updated Document AI Service")
    print("=" * 50)
    
    success = test_updated_service()
    
    if success:
        print("\n🎉 Updated Document AI service is working!")
        print("📊 Summary:")
        print("   - Service created successfully")
        print("   - PDF processed with chunking")
        print("   - Deeds detected and split")
        print("   - Individual PDFs created")
        print("\n🎯 Next steps:")
        print("1. Update the main app to use this service")
        print("2. Enable Document AI option in the UI")
        print("3. Remove non-working splitting methods")
    else:
        print("\n❌ Test failed")
        print("🔧 Need to investigate further")
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
