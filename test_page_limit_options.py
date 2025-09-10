#!/usr/bin/env python3
"""
Test different options to extend the page limit

This script tests various Document AI configurations to see if we can increase the page limit.
"""

import os
import sys
from pathlib import Path

def test_different_configurations():
    """Test different Document AI configurations to extend page limit"""
    print("🧪 Testing different configurations to extend page limit...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Use the correct processor version
        processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        # Read PDF content
        with open(test_pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        print(f"📄 PDF loaded: {len(pdf_content)} bytes")
        
        # Test different configurations
        configurations = [
            {
                'name': 'Default',
                'request': documentai.ProcessRequest(
                    name=processor_version,
                    raw_document=documentai.RawDocument(
                        content=pdf_content,
                        mime_type="application/pdf"
                    ),
                    skip_human_review=True
                )
            },
            {
                'name': 'With OCR Config',
                'request': documentai.ProcessRequest(
                    name=processor_version,
                    raw_document=documentai.RawDocument(
                        content=pdf_content,
                        mime_type="application/pdf"
                    ),
                    skip_human_review=True,
                    process_options=documentai.ProcessOptions(
                        ocr_config=documentai.OcrConfig(
                            enable_native_pdf_parsing=True
                        )
                    )
                )
            },
            {
                'name': 'With Layout Config',
                'request': documentai.ProcessRequest(
                    name=processor_version,
                    raw_document=documentai.RawDocument(
                        content=pdf_content,
                        mime_type="application/pdf"
                    ),
                    skip_human_review=True,
                    process_options=documentai.ProcessOptions(
                        layout_config=documentai.LayoutConfig(
                            enable_layout_analysis=True
                        )
                    )
                )
            }
        ]
        
        for config in configurations:
            print(f"\n🧪 Testing configuration: {config['name']}")
            
            try:
                result = client.process_document(request=config['request'])
                document = result.document
                
                print(f"✅ {config['name']} succeeded!")
                print(f"📊 Results:")
                print(f"   - Pages: {len(document.pages)}")
                print(f"   - Text length: {len(document.text)}")
                print(f"   - Entities: {len(document.entities)}")
                
                # Look for splitting entities
                splitting_entities = []
                for entity in document.entities:
                    if hasattr(entity, 'type_') and entity.type_:
                        splitting_entities.append({
                            'type': entity.type_,
                            'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.0
                        })
                
                print(f"🔍 Found {len(splitting_entities)} splitting entities")
                for entity in splitting_entities:
                    print(f"   - {entity['type']}: confidence {entity['confidence']:.3f}")
                
                return True  # If any configuration works, we're good
                
            except Exception as e:
                print(f"❌ {config['name']} failed: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ Error testing configurations: {e}")
        return False

def test_processor_limits():
    """Check what the actual processor limits are"""
    print("\n🧪 Checking processor limits...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878"
        
        # Get processor details
        processor = client.get_processor(name=processor_name)
        
        print(f"📊 Processor Details:")
        print(f"   - Name: {processor.display_name}")
        print(f"   - Type: {processor.type_}")
        print(f"   - State: {processor.state}")
        
        # Check if there are any limit configurations
        if hasattr(processor, 'process_endpoint'):
            print(f"   - Process Endpoint: {processor.process_endpoint}")
        
        # Check processor version details
        versions = client.list_processor_versions(parent=processor_name)
        
        for version in versions:
            print(f"\n📊 Version: {version.display_name}")
            print(f"   - State: {version.state}")
            print(f"   - Create Time: {version.create_time}")
            
            # Check if there are any specific configurations
            if hasattr(version, 'deployment_state'):
                print(f"   - Deployment State: {version.deployment_state}")
            
            if hasattr(version, 'training_status'):
                print(f"   - Training Status: {version.training_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking processor limits: {e}")
        return False

def suggest_alternatives():
    """Suggest alternative approaches"""
    print("\n💡 Alternative Approaches:")
    print("=" * 40)
    
    print("1. **Smart Chunking with Overlap**:")
    print("   - Split PDF into chunks with overlapping pages")
    print("   - Process each chunk independently")
    print("   - Merge results and remove duplicates")
    print("   - Example: Chunk 1 (pages 1-15), Chunk 2 (pages 10-25), etc.")
    
    print("\n2. **Deed-Aware Chunking**:")
    print("   - First pass: Quick analysis to find deed boundaries")
    print("   - Second pass: Create chunks that respect deed boundaries")
    print("   - Ensure no deed is split across chunks")
    
    print("\n3. **Sequential Processing**:")
    print("   - Process PDF in smaller chunks sequentially")
    print("   - Keep track of deed boundaries across chunks")
    print("   - Merge results at the end")
    
    print("\n4. **Contact Google Cloud Support**:")
    print("   - Request higher page limits for your processor")
    print("   - Explain your use case for multi-deed processing")
    print("   - They might be able to increase the limit")

def main():
    """Main test function"""
    print("🚀 Test Page Limit Options")
    print("=" * 50)
    
    # Check processor limits
    if not test_processor_limits():
        print("\n❌ Could not check processor limits")
        return False
    
    # Test different configurations
    if test_different_configurations():
        print("\n✅ Found a working configuration!")
        return True
    else:
        print("\n❌ No configuration worked with the full PDF")
        suggest_alternatives()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Use the working configuration")
        print("2. Update the service to use this configuration")
    else:
        print("\n🔧 Need to implement alternative approach")
    
    exit(0 if success else 1)
