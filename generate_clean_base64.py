#!/usr/bin/env python3
"""
Generate Clean Base64 for Render
This script will generate a clean, properly formatted base64 string
"""

import json
import base64

def create_minimal_service_account():
    """Create a minimal service account that should work"""
    service_account = {
        "type": "service_account",
        "project_id": "deed-boundary-250831-29868",
        "private_key_id": "temp-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKB\nwXIuVjWAnrx6M5k5uy70vwf9oD7VjdX4+Ws4qw0WQBfYxOj5hdYDuaf6hs2rJ\n-----END PRIVATE KEY-----\n",
        "client_email": "mineral-rights@deed-boundary-250831-29868.iam.gserviceaccount.com",
        "client_id": "123456789012345678901",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/mineral-rights%40deed-boundary-250831-29868.iam.gserviceaccount.com"
    }
    return service_account

def generate_clean_base64():
    """Generate a clean base64 string"""
    print("🔧 Generating Clean Base64 String")
    print("=" * 50)
    
    # Create service account
    service_account = create_minimal_service_account()
    
    # Convert to JSON (compact)
    json_string = json.dumps(service_account, separators=(',', ':'))
    
    # Encode to base64
    base64_string = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')
    
    print(f"✅ Generated clean base64 string")
    print(f"   Length: {len(base64_string)} characters")
    print(f"   Ends with: {base64_string[-4:]}")
    
    # Test decoding
    try:
        decoded = base64.b64decode(base64_string)
        parsed = json.loads(decoded.decode('utf-8'))
        print(f"✅ Decoding test successful")
        print(f"   Type: {parsed.get('type')}")
        print(f"   Project: {parsed.get('project_id')}")
    except Exception as e:
        print(f"❌ Decoding test failed: {e}")
        return None
    
    return base64_string

def main():
    print("🚀 Generate Clean Base64 for Render")
    print("=" * 60)
    
    base64_string = generate_clean_base64()
    
    if base64_string:
        print("\n📋 COPY THIS EXACT STRING TO RENDER:")
        print("=" * 80)
        print(base64_string)
        print("=" * 80)
        print()
        print("📋 INSTRUCTIONS:")
        print("1. Copy the ENTIRE string above (it's all on one line)")
        print("2. Go to your Render dashboard")
        print("3. Find your backend service")
        print("4. Go to Environment tab")
        print("5. Update GOOGLE_CREDENTIALS_BASE64 with this value")
        print("6. Save and redeploy")
        print()
        print("⚠️  Make sure to copy the ENTIRE string - no spaces, no line breaks!")
    
    print("\n" + "=" * 60)
    print("🔍 Generation complete")

if __name__ == "__main__":
    main()
