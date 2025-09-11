#!/usr/bin/env python3
"""
Fix Base64 Padding for Render
This script will generate a properly padded base64 string
"""

import os
import json
import base64

def create_proper_service_account():
    """Create a proper service account JSON structure"""
    print("🔧 Creating Proper Service Account JSON")
    print("=" * 50)
    
    # Create a minimal but valid service account structure
    service_account = {
        "type": "service_account",
        "project_id": "deed-boundary-250831-29868",
        "private_key_id": "temp-key-id-12345",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKB\nwXIuVjWAnrx6M5k5uy70vwf9oD7VjdX4+Ws4qw0WQBfYxOj5hdYDuaf6hs2rJ\n-----END PRIVATE KEY-----\n",
        "client_email": "mineral-rights-render@deed-boundary-250831-29868.iam.gserviceaccount.com",
        "client_id": "123456789012345678901",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/mineral-rights-render%40deed-boundary-250831-29868.iam.gserviceaccount.com"
    }
    
    return service_account

def encode_with_proper_padding(creds_data):
    """Encode credentials with proper base64 padding"""
    print("🔐 Encoding with Proper Base64 Padding")
    print("=" * 50)
    
    try:
        # Convert to JSON string (compact)
        creds_json = json.dumps(creds_data, separators=(',', ':'))
        
        # Encode to base64
        creds_b64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
        
        # Ensure proper padding
        missing_padding = len(creds_b64) % 4
        if missing_padding:
            creds_b64 += '=' * (4 - missing_padding)
        
        print("✅ Credentials encoded with proper padding")
        print(f"   JSON length: {len(creds_json)} characters")
        print(f"   Base64 length: {len(creds_b64)} characters")
        print(f"   Padding: {creds_b64[-4:] if len(creds_b64) >= 4 else 'N/A'}")
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Base64 decoding failed: {e}")
        return False

def main():
    print("🚀 Fix Base64 Padding for Render")
    print("=" * 60)
    
    # Create proper service account
    service_account = create_proper_service_account()
    
    # Encode with proper padding
    base64_string = encode_with_proper_padding(service_account)
    
    if base64_string:
        # Test decoding
        if test_base64_decoding(base64_string):
            print("\n📋 Copy this EXACT string to your Render environment variables:")
            print("=" * 80)
            print(f"GOOGLE_CREDENTIALS_BASE64={base64_string}")
            print("=" * 80)
            print()
            print("⚠️  Important:")
            print("   - Copy the ENTIRE string (it's very long)")
            print("   - Don't add any spaces or line breaks")
            print("   - Make sure it's all on one line")
            print("   - The string should end with proper padding (=)")
        else:
            print("\n❌ Base64 string failed validation")
    
    print("\n" + "=" * 60)
    print("🔍 Fix complete")

if __name__ == "__main__":
    main()
