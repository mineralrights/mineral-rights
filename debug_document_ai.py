#!/usr/bin/env python3
"""
Debug Document AI Request Issues

This script helps diagnose why the Document AI request is failing with a 400 error.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_processor_info():
    """Debug processor information"""
    print("🔍 Debugging Document AI Processor...")
    
    try:
        from google.cloud import documentai
        
        # Your processor endpoint
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        
        # Initialize client
        client = documentai.DocumentProcessorServiceClient()
        
        # Try to get processor info
        processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878"
        
        print(f"📡 Processor endpoint: {processor_endpoint}")
        print(f"📡 Processor name: {processor_name}")
        
        try:
            # Try to get processor information
            processor = client.get_processor(name=processor_name)
            print("✅ Processor found!")
            print(f"   - Display name: {processor.display_name}")
            print(f"   - Type: {processor.type_}")
            print(f"   - State: {processor.state}")
            print(f"   - Location: {processor.location}")
            
            return True, processor
            
        except Exception as e:
            print(f"❌ Could not get processor info: {e}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None

def debug_request_format():
    """Debug the request format"""
    print("\n🔍 Debugging Request Format...")
    
    try:
        from google.cloud import documentai
        
        # Test with a minimal PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/ROBERT.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        # Read a small portion of the PDF for testing
        with open(test_pdf_path, 'rb') as f:
            pdf_content = f.read(1024 * 1024)  # Read first 1MB only
        
        print(f"📄 PDF size: {len(pdf_content)} bytes")
        
        # Create the document
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        print("✅ RawDocument created successfully")
        
        # Create the request
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        
        request = documentai.ProcessRequest(
            name=processor_endpoint,
            raw_document=raw_document
        )
        
        print("✅ ProcessRequest created successfully")
        print(f"   - Name: {request.name}")
        print(f"   - Document size: {len(request.raw_document.content)}")
        print(f"   - MIME type: {request.raw_document.mime_type}")
        
        return True, request
        
    except Exception as e:
        print(f"❌ Error creating request: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_with_smaller_pdf():
    """Test with a smaller PDF or create a test PDF"""
    print("\n🔍 Testing with smaller PDF...")
    
    try:
        # Look for smaller PDFs
        pdf_files = list(Path("/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs").glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found")
            return False
        
        # Find the smallest PDF
        smallest_pdf = min(pdf_files, key=lambda x: x.stat().st_size)
        print(f"📄 Smallest PDF: {smallest_pdf} ({smallest_pdf.stat().st_size} bytes)")
        
        return str(smallest_pdf)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_processor_types():
    """Test different processor types and configurations"""
    print("\n🔍 Testing different processor configurations...")
    
    try:
        from google.cloud import documentai
        
        # Test different request formats
        test_configs = [
            {
                "name": "Standard Request",
                "config": {
                    "name": "projects/381937358877/locations/us/processors/895767ed7f252878:process",
                    "raw_document": None  # Will be set later
                }
            },
            {
                "name": "With Skip Human Review",
                "config": {
                    "name": "projects/381937358877/locations/us/processors/895767ed7f252878:process",
                    "raw_document": None,  # Will be set later
                    "skip_human_review": True
                }
            }
        ]
        
        print("📋 Available configurations:")
        for i, config in enumerate(test_configs):
            print(f"   {i+1}. {config['name']}")
        
        return test_configs
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def main():
    """Main debug function"""
    print("🚀 Document AI Debug Tool")
    print("=" * 50)
    
    # Step 1: Debug processor info
    success, processor = debug_processor_info()
    
    if not success:
        print("\n❌ Could not access processor")
        print("💡 Check:")
        print("   - Processor ID is correct")
        print("   - You have proper permissions")
        print("   - Processor is deployed and active")
        return False
    
    # Step 2: Debug request format
    success, request = debug_request_format()
    
    if not success:
        print("\n❌ Could not create request")
        return False
    
    # Step 3: Test with smaller PDF
    smaller_pdf = test_with_smaller_pdf()
    
    if smaller_pdf:
        print(f"\n📄 Try testing with smaller PDF: {smaller_pdf}")
    
    # Step 4: Show processor details
    print(f"\n📊 Processor Details:")
    print(f"   - Display Name: {processor.display_name}")
    print(f"   - Type: {processor.type_}")
    print(f"   - State: {processor.state}")
    
    # Step 5: Suggest fixes
    print(f"\n💡 Possible fixes for 400 error:")
    print("1. Check if processor is in 'ENABLED' state")
    print("2. Verify processor type matches your request")
    print("3. Try with a smaller PDF file")
    print("4. Check if processor expects specific input format")
    print("5. Verify processor is trained for the document type")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Check processor state and type")
        print("2. Try with a smaller PDF")
        print("3. Verify processor configuration")
    else:
        print("\n🔧 Fix the processor access issues first")
    
    exit(0 if success else 1)


