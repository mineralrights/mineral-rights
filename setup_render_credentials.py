#!/usr/bin/env python3
"""
Setup Render Credentials
This script will help you get the Google Cloud credentials for Render deployment
"""

import os
import json
import base64
from pathlib import Path

def get_application_default_credentials():
    """Get the current application default credentials"""
    print("🔍 Getting Application Default Credentials")
    print("=" * 50)
    
    # Check if credentials file exists
    creds_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    
    if os.path.exists(creds_path):
        print(f"✅ Found credentials file: {creds_path}")
        
        try:
            with open(creds_path, 'r') as f:
                creds_data = json.load(f)
            
            print("✅ Successfully read credentials file")
            print(f"   Type: {creds_data.get('type', 'unknown')}")
            print(f"   Project ID: {creds_data.get('quota_project_id', 'unknown')}")
            
            return creds_data, creds_path
            
        except Exception as e:
            print(f"❌ Error reading credentials file: {e}")
            return None, None
    else:
        print(f"❌ Credentials file not found: {creds_path}")
        return None, None

def create_service_account_instructions():
    """Provide instructions for creating a service account"""
    print("\n📋 Service Account Setup Instructions")
    print("=" * 50)
    
    print("To fix Render authentication, you need to create a service account:")
    print()
    print("1. Go to Google Cloud Console:")
    print("   https://console.cloud.google.com/")
    print()
    print("2. Navigate to IAM & Admin → Service Accounts")
    print()
    print("3. Create a new service account:")
    print("   - Name: mineral-rights-render")
    print("   - Description: Service account for Render deployment")
    print()
    print("4. Grant it these roles:")
    print("   - Document AI API User")
    print("   - Storage Object Viewer (if using GCS)")
    print()
    print("5. Create and download the JSON key file")
    print()
    print("6. Add to Render environment variables:")
    print("   GOOGLE_CREDENTIALS_BASE64=<base64-encoded-json-content>")
    print()
    print("7. Or upload the JSON file and set:")
    print("   GOOGLE_APPLICATION_CREDENTIALS=/tmp/credentials.json")

def encode_credentials_for_render(creds_data):
    """Encode credentials for Render deployment"""
    print("\n🔐 Encoding Credentials for Render")
    print("=" * 50)
    
    try:
        # Convert to JSON string
        creds_json = json.dumps(creds_data, indent=2)
        
        # Encode to base64
        creds_b64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
        
        print("✅ Credentials encoded successfully")
        print(f"   Length: {len(creds_b64)} characters")
        print()
        print("📋 Add this to your Render environment variables:")
        print("=" * 50)
        print(f"GOOGLE_CREDENTIALS_BASE64={creds_b64}")
        print("=" * 50)
        print()
        print("⚠️  Keep this value secure - it contains your Google Cloud credentials!")
        
        return creds_b64
        
    except Exception as e:
        print(f"❌ Error encoding credentials: {e}")
        return None

def main():
    print("🚀 Setup Render Google Cloud Credentials")
    print("=" * 60)
    
    # Get current credentials
    creds_data, creds_path = get_application_default_credentials()
    
    if creds_data:
        print("\n✅ You have working Google Cloud credentials!")
        
        # Check if it's a service account or user credentials
        creds_type = creds_data.get('type', '')
        
        if creds_type == 'service_account':
            print("✅ These are service account credentials - perfect for Render!")
            encode_credentials_for_render(creds_data)
        else:
            print("⚠️  These are user credentials (not ideal for production)")
            print("   Consider creating a service account for better security")
            create_service_account_instructions()
    else:
        print("\n❌ No working credentials found")
        create_service_account_instructions()
    
    print("\n" + "=" * 60)
    print("🔍 Setup complete")

if __name__ == "__main__":
    main()
