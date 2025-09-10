#!/usr/bin/env python3
"""
Local Document AI Testing Script

This script helps you test your Document AI checkpoint locally before integrating it into the app.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_document_ai_connection():
    """Test basic connection to Document AI"""
    print("🧪 Testing Document AI connection...")
    
    try:
        from google.cloud import documentai
        from google.oauth2 import service_account
        print("✅ Google Cloud libraries imported successfully")
        
        # Your processor endpoint
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        print(f"📡 Processor endpoint: {processor_endpoint}")
        
        # Test client initialization
        try:
            # Try with default credentials first
            client = documentai.DocumentProcessorServiceClient()
            print("✅ Document AI client initialized with default credentials")
            return True, client, processor_endpoint
        except Exception as e:
            print(f"⚠️ Default credentials failed: {e}")
            print("💡 You may need to set up Application Default Credentials")
            print("   Run: gcloud auth application-default login")
            return False, None, processor_endpoint
            
    except ImportError as e:
        print(f"❌ Failed to import Google Cloud libraries: {e}")
        print("💡 Install with: pip install google-cloud-documentai google-auth")
        return False, None, None

def test_document_ai_with_credentials(credentials_path):
    """Test Document AI with service account file"""
    print(f"🧪 Testing Document AI with credentials: {credentials_path}")
    
    try:
        from google.cloud import documentai
        from google.oauth2 import service_account
        
        if not os.path.exists(credentials_path):
            print(f"❌ Credentials file not found: {credentials_path}")
            return False, None, None
        
        # Initialize with service account
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        client = documentai.DocumentProcessorServiceClient(credentials=credentials)
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        
        print("✅ Document AI client initialized with service account")
        return True, client, processor_endpoint
        
    except Exception as e:
        print(f"❌ Failed to initialize with credentials: {e}")
        return False, None, None

def test_document_ai_processing(client, processor_endpoint, pdf_path):
    """Test actual Document AI processing"""
    print(f"🧪 Testing Document AI processing with: {pdf_path}")
    
    try:
        from google.cloud import documentai
        
        # Read the PDF file
        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        print(f"📄 PDF loaded: {len(pdf_content)} bytes")
        
        # Create the document
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        # Create the request
        request = documentai.ProcessRequest(
            name=processor_endpoint,
            raw_document=raw_document
        )
        
        # Process the document
        print("📤 Sending request to Document AI...")
        result = client.process_document(request=request)
        document = result.document
        
        print("✅ Document AI processing completed!")
        print(f"📊 Document info:")
        print(f"   - Pages: {len(document.pages)}")
        print(f"   - Text length: {len(document.text)}")
        print(f"   - Entities: {len(document.entities)}")
        
        # Look for deed-related entities
        deed_entities = []
        for entity in document.entities:
            if entity.type_ in ['DEED_BOUNDARY', 'DEED_START', 'DEED_END', 'DOCUMENT_BOUNDARY']:
                deed_entities.append({
                    'type': entity.type_,
                    'text': entity.mention_text,
                    'confidence': entity.confidence,
                    'page_refs': [page_ref.page for page_ref in entity.page_anchor.page_refs]
                })
        
        print(f"🔍 Found {len(deed_entities)} deed-related entities:")
        for entity in deed_entities:
            print(f"   - {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.3f}, pages: {entity['page_refs']})")
        
        return True, document, deed_entities
        
    except Exception as e:
        print(f"❌ Document AI processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_with_sample_pdf():
    """Test with a sample PDF if available"""
    print("🧪 Looking for sample PDFs to test...")
    
    # Look for PDFs in common locations
    sample_locations = [
        "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs",rorr
    ]
    
    pdf_files = []
    for location in sample_locations:
        if os.path.exists(location):
            pdf_files.extend(Path(location).glob("*.pdf"))
    
    if pdf_files:
        print(f"📄 Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files[:5]:  # Show first 5
            print(f"   - {pdf_file}")
        
        # Use the first PDF for testing
        test_pdf = pdf_files[0]
        print(f"🎯 Using {test_pdf} for testing")
        return str(test_pdf)
    else:
        print("⚠️ No PDF files found for testing")
        print("💡 You can test with any PDF file by providing the path")
        return None

def main():
    """Main testing function"""
    print("🚀 Document AI Local Testing")
    print("=" * 50)
    
    # Step 1: Test connection
    success, client, processor_endpoint = test_document_ai_connection()
    
    if not success:
        print("\n💡 Trying with service account credentials...")
        credentials_path = input("Enter path to service account JSON file (or press Enter to skip): ").strip()
        
        if credentials_path:
            success, client, processor_endpoint = test_document_ai_with_credentials(credentials_path)
    
    if not success:
        print("\n❌ Could not establish Document AI connection")
        print("📋 Setup instructions:")
        print("1. Install: pip install google-cloud-documentai google-auth")
        print("2. Set up credentials:")
        print("   - Option A: gcloud auth application-default login")
        print("   - Option B: Download service account JSON and provide path")
        print("3. Ensure Document AI API is enabled in your Google Cloud project")
        return False
    
    print(f"\n✅ Document AI connection established!")
    print(f"📡 Endpoint: {processor_endpoint}")
    
    # Step 2: Find test PDF
    test_pdf = test_with_sample_pdf()
    
    if not test_pdf:
        test_pdf = input("\nEnter path to a PDF file to test (or press Enter to skip): ").strip()
    
    if not test_pdf or not os.path.exists(test_pdf):
        print("⚠️ No PDF file provided for testing")
        print("✅ Document AI connection is working - you can test with a PDF later")
        return True
    
    # Step 3: Test processing
    print(f"\n🧪 Testing Document AI processing...")
    success, document, entities = test_document_ai_processing(client, processor_endpoint, test_pdf)
    
    if success:
        print("\n🎉 Document AI is working correctly!")
        print("📊 Summary:")
        print(f"   - PDF processed successfully")
        print(f"   - Found {len(entities)} deed-related entities")
        print(f"   - Document has {len(document.pages)} pages")
        
        if entities:
            print("\n🔍 Deed boundaries detected:")
            for entity in entities:
                print(f"   - Page {entity['page_refs'][0] + 1}: {entity['type']} (confidence: {entity['confidence']:.3f})")
        else:
            print("\n⚠️ No deed boundaries detected - this might be expected for single-deed documents")
        
        return True
    else:
        print("\n❌ Document AI processing failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Document AI is working - ready for integration")
        print("2. Test with multi-deed PDFs to see boundary detection")
        print("3. Integrate into the app once you're satisfied with results")
    else:
        print("\n🔧 Fix the issues above before proceeding")
    
    sys.exit(0 if success else 1)
