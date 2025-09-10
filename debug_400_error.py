#!/usr/bin/env python3
"""
Debug 400 Error for Custom Splitting Processor

This script helps diagnose the specific cause of the 400 error.
"""

import os
import sys
from pathlib import Path

def test_minimal_request():
    """Test with the most minimal request possible"""
    print("🧪 Testing minimal request...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        
        # Create a very small test PDF content (just a few bytes)
        minimal_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n174\n%%EOF"
        
        print(f"📄 Minimal PDF content: {len(minimal_pdf_content)} bytes")
        
        # Create minimal request
        raw_document = documentai.RawDocument(
            content=minimal_pdf_content,
            mime_type="application/pdf"
        )
        
        request = documentai.ProcessRequest(
            name=processor_endpoint,
            raw_document=raw_document
        )
        
        print("📤 Sending minimal request...")
        result = client.process_document(request=request)
        
        print("✅ Minimal request succeeded!")
        return True
        
    except Exception as e:
        print(f"❌ Minimal request failed: {e}")
        return False

def test_processor_state():
    """Check if the processor is in the right state"""
    print("\n🧪 Checking processor state...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878"
        
        processor = client.get_processor(name=processor_name)
        
        print(f"📊 Processor State Details:")
        print(f"   - Name: {processor.display_name}")
        print(f"   - Type: {processor.type_}")
        print(f"   - State: {processor.state}")
        
        # Check if processor is enabled
        if processor.state == 1:  # ENABLED
            print("✅ Processor is ENABLED")
        elif processor.state == 2:  # DISABLED
            print("❌ Processor is DISABLED")
            return False
        elif processor.state == 3:  # ENABLING
            print("⚠️ Processor is ENABLING (still starting up)")
            return False
        elif processor.state == 4:  # DISABLING
            print("⚠️ Processor is DISABLING")
            return False
        else:
            print(f"⚠️ Unknown processor state: {processor.state}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking processor state: {e}")
        return False

def test_processor_permissions():
    """Test if we have the right permissions"""
    print("\n🧪 Testing processor permissions...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878"
        
        # Try to list processors to test permissions
        parent = "projects/381937358877/locations/us"
        processors = client.list_processors(parent=parent)
        
        print("✅ Can list processors - permissions are OK")
        
        # Check if our processor is in the list
        found = False
        for processor in processors:
            if processor.name == processor_name:
                found = True
                print(f"✅ Found our processor: {processor.display_name}")
                break
        
        if not found:
            print("❌ Our processor not found in the list")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Permission test failed: {e}")
        return False

def test_processor_type_specific():
    """Test processor type-specific requirements"""
    print("\n🧪 Testing processor type-specific requirements...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878"
        
        processor = client.get_processor(name=processor_name)
        
        print(f"📊 Processor Type Analysis:")
        print(f"   - Type: {processor.type_}")
        
        if processor.type_ == "CUSTOM_SPLITTING_PROCESSOR":
            print("✅ This is a Custom Splitting Processor")
            print("💡 Custom Splitting Processors typically:")
            print("   - Expect specific input formats")
            print("   - May require training data")
            print("   - Might need specific document types")
            print("   - Could have size limitations")
            
            # Check if there are any specific requirements
            if hasattr(processor, 'process_endpoint'):
                print(f"   - Process endpoint: {processor.process_endpoint}")
            
            return True
        else:
            print(f"⚠️ Unexpected processor type: {processor.type_}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_alternative_endpoints():
    """Test alternative endpoint formats"""
    print("\n🧪 Testing alternative endpoint formats...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Test different endpoint formats
        endpoints_to_test = [
            "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process",
            "projects/381937358877/locations/us/processors/895767ed7f252878:process",
            "projects/381937358877/locations/us/processors/895767ed7f252878"
        ]
        
        for endpoint in endpoints_to_test:
            print(f"🧪 Testing endpoint: {endpoint}")
            
            try:
                # Create minimal request
                minimal_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n174\n%%EOF"
                
                raw_document = documentai.RawDocument(
                    content=minimal_pdf,
                    mime_type="application/pdf"
                )
                
                request = documentai.ProcessRequest(
                    name=endpoint,
                    raw_document=raw_document
                )
                
                result = client.process_document(request=request)
                print(f"✅ Endpoint {endpoint} worked!")
                return True
                
            except Exception as e:
                print(f"❌ Endpoint {endpoint} failed: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ Error testing endpoints: {e}")
        return False

def main():
    """Main debug function"""
    print("🚀 Debug 400 Error for Custom Splitting Processor")
    print("=" * 60)
    
    # Step 1: Check processor state
    if not test_processor_state():
        print("\n❌ Processor state issue")
        return False
    
    # Step 2: Test permissions
    if not test_processor_permissions():
        print("\n❌ Permission issue")
        return False
    
    # Step 3: Test processor type
    if not test_processor_type_specific():
        print("\n❌ Processor type issue")
        return False
    
    # Step 4: Test minimal request
    if test_minimal_request():
        print("\n✅ Minimal request works - the issue might be with the PDF content")
        return True
    
    # Step 5: Test alternative endpoints
    if test_alternative_endpoints():
        print("\n✅ Alternative endpoint works")
        return True
    
    print("\n❌ All tests failed")
    print("💡 Possible issues:")
    print("1. Processor might not be fully trained/deployed")
    print("2. Processor might expect specific input format")
    print("3. Processor might have size limitations")
    print("4. Processor might need specific document types")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Try with a different PDF")
        print("2. Check processor training status")
        print("3. Contact Google Cloud support if needed")
    else:
        print("\n🔧 Need to investigate further")
    
    exit(0 if success else 1)

