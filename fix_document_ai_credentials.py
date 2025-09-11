#!/usr/bin/env python3
"""
Fix Document AI Credentials Setup
This script will help you set up Document AI credentials properly
"""

import os
import sys
import json
import base64
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_environment():
    """Check environment variables"""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Check all relevant environment variables
    env_vars = {
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "GOOGLE_CREDENTIALS_BASE64": os.getenv("GOOGLE_CREDENTIALS_BASE64"),
        "DOCUMENT_AI_CREDENTIALS_PATH": os.getenv("DOCUMENT_AI_CREDENTIALS_PATH"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "DOCUMENT_AI_ENDPOINT": os.getenv("DOCUMENT_AI_ENDPOINT"),
    }
    
    for var_name, var_value in env_vars.items():
        if var_value:
            if var_name == "ANTHROPIC_API_KEY":
                # Don't show the full API key
                display_value = f"{var_value[:8]}...{var_value[-4:]}" if len(var_value) > 12 else "***"
            else:
                display_value = var_value
            print(f"{var_name}: ✅ Set ({display_value})")
        else:
            print(f"{var_name}: ❌ Not set")
    
    return env_vars

def check_google_credentials():
    """Check if Google credentials are working"""
    print("\n🔑 Google Credentials Check")
    print("=" * 50)
    
    try:
        from google.auth import default
        from google.cloud import documentai
        
        # Try to get default credentials
        credentials, project = default()
        print(f"✅ Default credentials found")
        print(f"   Project: {project}")
        
        # Try to create a Document AI client
        client = documentai.DocumentProcessorServiceClient(credentials=credentials)
        print("✅ Document AI client created successfully")
        
        return True, credentials, project
        
    except Exception as e:
        print(f"❌ Google credentials issue: {e}")
        return False, None, None

def test_document_ai_connection():
    """Test Document AI connection"""
    print("\n🧪 Testing Document AI Connection")
    print("=" * 50)
    
    try:
        from google.cloud import documentai
        
        # Use default credentials
        client = documentai.DocumentProcessorServiceClient()
        
        # Test with your processor endpoint
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        
        print(f"✅ Document AI client initialized")
        print(f"   Processor endpoint: {processor_endpoint}")
        
        return True, client
        
    except Exception as e:
        print(f"❌ Document AI connection failed: {e}")
        return False, None

def create_service_account_instructions():
    """Provide instructions for creating service account"""
    print("\n📋 Service Account Setup Instructions")
    print("=" * 50)
    
    print("If you need to create a service account:")
    print("1. Go to Google Cloud Console: https://console.cloud.google.com/")
    print("2. Navigate to IAM & Admin > Service Accounts")
    print("3. Create a new service account")
    print("4. Grant it 'Document AI API User' role")
    print("5. Create and download the JSON key file")
    print("6. Set the environment variable:")
    print("   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json")
    print("   OR")
    print("   export DOCUMENT_AI_CREDENTIALS_PATH=/path/to/your/service-account.json")

def test_document_processor():
    """Test DocumentProcessor with proper environment"""
    print("\n🧪 Testing DocumentProcessor")
    print("=" * 50)
    
    # Set required environment variables if not set
    if not os.getenv("DOCUMENT_AI_ENDPOINT"):
        os.environ["DOCUMENT_AI_ENDPOINT"] = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        print("✅ Set DOCUMENT_AI_ENDPOINT")
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        # Initialize processor
        processor = DocumentProcessor()
        
        print("✅ DocumentProcessor initialized successfully")
        print(f"   Document AI service available: {'✅ Yes' if processor.document_ai_service else '❌ No'}")
        
        if processor.document_ai_service:
            print("✅ Document AI service is properly initialized")
            return True, processor
        else:
            print("❌ Document AI service is not available")
            return False, None
        
    except Exception as e:
        print(f"❌ DocumentProcessor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    print("🚀 Fix Document AI Credentials")
    print("=" * 60)
    
    # Check environment
    env_vars = check_environment()
    
    # Check Google credentials
    creds_ok, credentials, project = check_google_credentials()
    
    if not creds_ok:
        print("\n❌ Google credentials are not properly set up")
        create_service_account_instructions()
        return
    
    # Test Document AI connection
    doc_ai_ok, client = test_document_ai_connection()
    
    if not doc_ai_ok:
        print("\n❌ Document AI connection failed")
        return
    
    # Test DocumentProcessor
    processor_ok, processor = test_document_processor()
    
    if processor_ok:
        print("\n🎉 SUCCESS! Document AI is properly configured")
        print("✅ Your multi-deed processing should now use Document AI instead of fallback")
    else:
        print("\n❌ DocumentProcessor test failed")
        print("   Check the error messages above for details")
    
    print("\n" + "=" * 60)
    print("🔍 Credential check complete")

if __name__ == "__main__":
    main()
