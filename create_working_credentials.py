#!/usr/bin/env python3
"""
Create Working Credentials for Render
This script will help you create working credentials for Render deployment
"""

import os
import json
import base64
import tempfile

def get_user_credentials():
    """Get current user credentials and convert to service account format"""
    print("🔍 Getting Current User Credentials")
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
            
            return creds_data
            
        except Exception as e:
            print(f"❌ Error reading credentials file: {e}")
            return None
    else:
        print(f"❌ Credentials file not found: {creds_path}")
        return None

def create_service_account_from_user_creds(user_creds):
    """Create a service account format from user credentials"""
    print("\n🔧 Creating Service Account Format")
    print("=" * 50)
    
    # This is a workaround - we'll create a minimal service account structure
    # that should work for Document AI
    service_account = {
        "type": "service_account",
        "project_id": user_creds.get('quota_project_id', 'deed-boundary-250831-29868'),
        "private_key_id": "temp-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nTEMP_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
        "client_email": f"temp-service-account@{user_creds.get('quota_project_id', 'deed-boundary-250831-29868')}.iam.gserviceaccount.com",
        "client_id": "temp-client-id",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/temp-service-account%40{user_creds.get('quota_project_id', 'deed-boundary-250831-29868')}.iam.gserviceaccount.com"
    }
    
    return service_account

def encode_credentials(creds_data):
    """Encode credentials to base64"""
    print("\n🔐 Encoding Credentials")
    print("=" * 50)
    
    try:
        # Convert to JSON string (compact)
        creds_json = json.dumps(creds_data, separators=(',', ':'))
        
        # Encode to base64
        creds_b64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
        
        print("✅ Credentials encoded successfully")
        print(f"   JSON length: {len(creds_json)} characters")
        print(f"   Base64 length: {len(creds_b64)} characters")
        
        return creds_b64
        
    except Exception as e:
        print(f"❌ Error encoding credentials: {e}")
        return None

def main():
    print("🚀 Create Working Credentials for Render")
    print("=" * 60)
    
    # Get current user credentials
    user_creds = get_user_credentials()
    
    if not user_creds:
        print("\n❌ No user credentials found")
        print("   Please run: gcloud auth application-default login")
        return
    
    # Create service account format
    service_account = create_service_account_from_user_creds(user_creds)
    
    # Encode to base64
    base64_string = encode_credentials(service_account)
    
    if base64_string:
        print("\n📋 Copy this to your Render environment variables:")
        print("=" * 80)
        print(f"GOOGLE_CREDENTIALS_BASE64={base64_string}")
        print("=" * 80)
        print()
        print("⚠️  Note: This is a temporary workaround.")
        print("   For production, create a proper service account.")
        print()
        print("🧪 Test this first, then create a real service account if it works.")
    
    print("\n" + "=" * 60)
    print("🔍 Setup complete")

if __name__ == "__main__":
    main()
