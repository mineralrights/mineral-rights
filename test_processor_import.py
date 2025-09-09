#!/usr/bin/env python3
"""
Test script to check if DocumentProcessor can be imported and initialized
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_processor_import():
    """Test if DocumentProcessor can be imported"""
    print("🧪 Testing DocumentProcessor Import")
    print("=" * 50)
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
        return True
    except Exception as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processor_initialization():
    """Test if DocumentProcessor can be initialized"""
    print("\n🧪 Testing DocumentProcessor Initialization")
    print("=" * 50)
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return False
        
        print(f"API Key present: {'Yes' if api_key else 'No'}")
        
        processor = DocumentProcessor(api_key)
        print("✅ DocumentProcessor initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ DocumentProcessor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processor_methods():
    """Test if DocumentProcessor methods exist"""
    print("\n🧪 Testing DocumentProcessor Methods")
    print("=" * 50)
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return False
        
        processor = DocumentProcessor(api_key)
        
        # Check if methods exist
        methods_to_check = [
            'process_document',
            'process_multi_deed_document',
            'process_document_memory_efficient',
            'split_pdf_by_deeds',
            'cleanup_temp_files'
        ]
        
        for method_name in methods_to_check:
            if hasattr(processor, method_name):
                print(f"✅ {method_name} method exists")
            else:
                print(f"❌ {method_name} method missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Method check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing DocumentProcessor for API Issues")
    print("=" * 60)
    
    success = True
    
    # Test import
    if not test_processor_import():
        success = False
    
    # Test initialization
    if not test_processor_initialization():
        success = False
    
    # Test methods
    if not test_processor_methods():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! DocumentProcessor should work in API.")
    else:
        print("❌ Some tests failed. This explains why processing doesn't start.")
    
    print("\n📋 If tests fail, the issue is:")
    print("1. Missing dependencies (fitz, psutil, anthropic)")
    print("2. API key not set")
    print("3. DocumentProcessor initialization error")
    print("4. Missing methods in DocumentProcessor")
