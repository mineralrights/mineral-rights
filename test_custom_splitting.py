#!/usr/bin/env python3
"""
Test Custom Splitting Processor

This script tests the custom splitting processor with the correct request format.
"""

import os
import sys
from pathlib import Path

def test_custom_splitting_processor():
    """Test the custom splitting processor"""
    print("🧪 Testing Custom Splitting Processor...")
    
    try:
        from google.cloud import documentai
        
        # Initialize client
        client = documentai.DocumentProcessorServiceClient()
        
        # Processor endpoint
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/ROBERT.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        # Read PDF content
        with open(test_pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        print(f"📄 PDF loaded: {len(pdf_content)} bytes")
        
        # For custom splitting processors, we need to use the correct request format
        # Create the document
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        # Create the request - for custom splitting processors
        request = documentai.ProcessRequest(
            name=processor_endpoint,
            raw_document=raw_document,
            # Custom splitting processors might need additional parameters
            skip_human_review=True  # Skip human review for faster processing
        )
        
        print("📤 Sending request to Custom Splitting Processor...")
        
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

def test_with_smaller_pdf():
    """Test with a smaller PDF to avoid timeout issues"""
    print("\n🧪 Testing with smaller PDF...")
    
    try:
        # Find the smallest PDF
        pdf_dir = Path("/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found")
            return None
        
        # Find the smallest PDF
        smallest_pdf = min(pdf_files, key=lambda x: x.stat().st_size)
        print(f"📄 Smallest PDF: {smallest_pdf} ({smallest_pdf.stat().st_size} bytes)")
        
        return str(smallest_pdf)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_processor_capabilities():
    """Test what the processor can do"""
    print("\n🧪 Testing processor capabilities...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878"
        
        # Get processor info
        processor = client.get_processor(name=processor_name)
        
        print(f"📊 Processor Details:")
        print(f"   - Name: {processor.display_name}")
        print(f"   - Type: {processor.type_}")
        print(f"   - State: {processor.state}")
        
        # Check if it's a custom splitting processor
        if processor.type_ == "CUSTOM_SPLITTING_PROCESSOR":
            print("✅ This is a Custom Splitting Processor - perfect for deed splitting!")
            
            # Custom splitting processors typically return:
            # - Document splits/revisions
            # - Splitting confidence scores
            # - Boundary information
            
            print("🎯 Expected outputs:")
            print("   - Document splits (multiple documents)")
            print("   - Splitting confidence scores")
            print("   - Boundary detection results")
            
            return True
        else:
            print(f"⚠️ Unexpected processor type: {processor.type_}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Custom Splitting Processor Test")
    print("=" * 50)
    
    # Step 1: Test processor capabilities
    if not test_processor_capabilities():
        print("\n❌ Processor capability test failed")
        return False
    
    # Step 2: Test with smaller PDF first
    smaller_pdf = test_with_smaller_pdf()
    
    if smaller_pdf:
        print(f"\n📄 Testing with smaller PDF: {smaller_pdf}")
        
        # Update the test to use the smaller PDF
        import test_custom_splitting
        test_custom_splitting.test_pdf_path = smaller_pdf
    
    # Step 3: Test the custom splitting processor
    success, document, entities = test_custom_splitting_processor()
    
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
        print("\n❌ Custom Splitting Processor test failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Custom Splitting Processor is working!")
        print("2. Integrate into the app")
        print("3. Remove non-working splitting methods")
    else:
        print("\n🔧 Fix the processor issues first")
    
    exit(0 if success else 1)

