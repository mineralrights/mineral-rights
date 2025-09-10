#!/usr/bin/env python3
"""
Test with the correct processor version

This script tests the custom splitting processor using the correct version.
"""

import os
import sys
from pathlib import Path

def test_with_correct_version():
    """Test with the correct processor version"""
    print("🧪 Testing with correct processor version...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Use the correct processor version that's actually trained
        processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
        
        print(f"📡 Using processor version: {processor_version}")
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        # Read PDF content
        with open(test_pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        print(f"📄 PDF loaded: {len(pdf_content)} bytes")
        
        # Create the document
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        # Create the request with the correct version
        request = documentai.ProcessRequest(
            name=processor_version,
            raw_document=raw_document,
            skip_human_review=True
        )
        
        print("📤 Sending request to correct processor version...")
        
        # Process the document
        result = client.process_document(request=request)
        document = result.document
        
        print("✅ Custom Splitting Processor completed!")
        print(f"📊 Results:")
        print(f"   - Pages: {len(document.pages)}")
        print(f"   - Text length: {len(document.text)}")
        print(f"   - Entities: {len(document.entities)}")
        
        # Look for splitting-related entities
        splitting_entities = []
        for entity in document.entities:
            if hasattr(entity, 'type_') and entity.type_:
                splitting_entities.append({
                    'type': entity.type_,
                    'text': entity.mention_text if hasattr(entity, 'mention_text') else 'N/A',
                    'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.0,
                    'page_refs': [page_ref.page for page_ref in entity.page_anchor.page_refs] if hasattr(entity, 'page_anchor') and entity.page_anchor else []
                })
        
        print(f"🔍 Found {len(splitting_entities)} splitting entities:")
        for entity in splitting_entities:
            print(f"   - {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.3f}, pages: {entity['page_refs']})")
        
        # Look for document splits
        if hasattr(document, 'revisions') and document.revisions:
            print(f"📄 Found {len(document.revisions)} document revisions/splits")
            for i, revision in enumerate(document.revisions):
                print(f"   - Revision {i+1}: {revision}")
        
        return True, document, splitting_entities
        
    except Exception as e:
        print(f"❌ Custom Splitting Processor failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_with_minimal_pdf():
    """Test with a minimal PDF first"""
    print("\n🧪 Testing with minimal PDF...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Use the correct processor version
        processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
        
        # Create a minimal PDF content
        minimal_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n174\n%%EOF"
        
        print(f"📄 Minimal PDF content: {len(minimal_pdf_content)} bytes")
        
        # Create the document
        raw_document = documentai.RawDocument(
            content=minimal_pdf_content,
            mime_type="application/pdf"
        )
        
        # Create the request
        request = documentai.ProcessRequest(
            name=processor_version,
            raw_document=raw_document,
            skip_human_review=True
        )
        
        print("📤 Sending minimal request...")
        
        # Process the document
        result = client.process_document(request=request)
        document = result.document
        
        print("✅ Minimal request succeeded!")
        print(f"📊 Results:")
        print(f"   - Pages: {len(document.pages)}")
        print(f"   - Text length: {len(document.text)}")
        print(f"   - Entities: {len(document.entities)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal request failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Test with Correct Processor Version")
    print("=" * 50)
    
    # Step 1: Test with minimal PDF first
    if test_with_minimal_pdf():
        print("\n✅ Minimal test succeeded - processor is working!")
        
        # Step 2: Test with real PDF
        success, document, entities = test_with_correct_version()
        
        if success:
            print("\n🎉 Custom Splitting Processor is working!")
            print("📊 Summary:")
            print(f"   - Document processed successfully")
            print(f"   - Found {len(entities)} splitting entities")
            print(f"   - Document has {len(document.pages)} pages")
            
            if entities:
                print("\n🔍 Splitting results:")
                for entity in entities:
                    print(f"   - {entity['type']}: confidence {entity['confidence']:.3f}")
            
            return True
        else:
            print("\n❌ Real PDF test failed")
            return False
    else:
        print("\n❌ Minimal test failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Custom Splitting Processor is working!")
        print("2. Update the app to use the correct processor version")
        print("3. Integrate into the app")
        print("4. Remove non-working splitting methods")
    else:
        print("\n🔧 Need to investigate further")
    
    exit(0 if success else 1)
