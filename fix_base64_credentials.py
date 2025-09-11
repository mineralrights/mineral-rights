#!/usr/bin/env python3
"""
Fix Base64 Credentials for Render
This script will help you create properly formatted base64 credentials
"""

import os
import json
import base64
import tempfile

def create_test_service_account():
    """Create a test service account JSON for demonstration"""
    print("🔧 Creating Test Service Account JSON")
    print("=" * 50)
    
    # This is a template - you'll need to replace with your actual service account
    test_service_account = {
        "type": "service_account",
        "project_id": "deed-boundary-250831-29868",
        "private_key_id": "your-private-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n",
        "client_email": "mineral-rights-render@deed-boundary-250831-29868.iam.gserviceaccount.com",
        "client_id": "your-client-id",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/mineral-rights-render%40deed-boundary-250831-29868.iam.gserviceaccount.com"
    }
    
    return test_service_account

def encode_credentials_properly(creds_data):
    """Encode credentials properly for Render"""
    print("🔐 Encoding Credentials Properly")
    print("=" * 50)
    
    try:
        # Convert to JSON string (compact, no extra spaces)
        creds_json = json.dumps(creds_data, separators=(',', ':'))
        
        # Encode to base64
        creds_b64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
        
        print("✅ Credentials encoded successfully")
        print(f"   JSON length: {len(creds_json)} characters")
        print(f"   Base64 length: {len(creds_b64)} characters")
        print()
        print("📋 Copy this EXACT string to your Render environment variables:")
        print("=" * 80)
        print(creds_b64)
        print("=" * 80)
        print()
        print("⚠️  Important:")
        print("   - Copy the ENTIRE string (it's very long)")
        print("   - Don't add any spaces or line breaks")
        print("   - Make sure it's all on one line")
        
        return creds_b64
        
    except Exception as e:
        print(f"❌ Error encoding credentials: {e}")
        return None

def test_base64_decoding(base64_string):
    """Test if the base64 string can be decoded properly"""
    print("\n🧪 Testing Base64 Decoding")
    print("=" * 50)
    
    try:
        # Decode base64
        decoded_bytes = base64.b64decode(base64_string)
        decoded_json = decoded_bytes.decode('utf-8')
        
        # Parse JSON
        creds_data = json.loads(decoded_json)
        
        print("✅ Base64 decoding successful")
        print(f"   Type: {creds_data.get('type', 'unknown')}")
        print(f"   Project ID: {creds_data.get('project_id', 'unknown')}")
        print(f"   Client Email: {creds_data.get('client_email', 'unknown')}")
        
        return True, creds_data
        
    except Exception as e:
        print(f"❌ Base64 decoding failed: {e}")
        return False, None

def get_instructions_for_real_service_account():
    """Provide instructions for getting real service account"""
    print("\n📋 Instructions for Real Service Account")
    print("=" * 50)
    
    print("To get your REAL service account credentials:")
    print()
    print("1. Go to Google Cloud Console:")
    print("   https://console.cloud.google.com/")
    print()
    print("2. Navigate to IAM & Admin → Service Accounts")
    print()
    print("3. Find your service account or create a new one:")
    print("   - Name: mineral-rights-render")
    print("   - Roles: Document AI API User")
    print()
    print("4. Click on the service account → Keys tab")
    print()
    print("5. Click 'Add Key' → 'Create new key' → JSON")
    print()
    print("6. Download the JSON file")
    print()
    print("7. Run this script with the real JSON file:")
    print("   python fix_base64_credentials.py /path/to/your/service-account.json")

def main():
    print("🚀 Fix Base64 Credentials for Render")
    print("=" * 60)
    
    # Check if a JSON file was provided as argument
    import sys
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        print(f"📁 Reading service account from: {json_file_path}")
        
        try:
            with open(json_file_path, 'r') as f:
                creds_data = json.load(f)
            
            print("✅ Successfully read service account file")
            base64_string = encode_credentials_properly(creds_data)
            
            if base64_string:
                test_base64_decoding(base64_string)
            
        except Exception as e:
            print(f"❌ Error reading JSON file: {e}")
            get_instructions_for_real_service_account()
    
    else:
        print("📋 No JSON file provided - showing instructions")
        get_instructions_for_real_service_account()
        
        print("\n🧪 Testing with sample data:")
        test_creds = create_test_service_account()
        base64_string = encode_credentials_properly(test_creds)
        
        if base64_string:
            test_base64_decoding(base64_string)
    
    print("\n" + "=" * 60)
    print("🔍 Fix complete")

if __name__ == "__main__":
    main()
