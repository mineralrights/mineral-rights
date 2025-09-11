#!/usr/bin/env python3
"""
Create a proper service account JSON for Document AI
This will create a service account with the correct format
"""

import json
import base64

def create_service_account_json():
    """Create a proper service account JSON file"""
    
    # This is a template - you'll need to replace with your actual service account details
    service_account = {
        "type": "service_account",
        "project_id": "deed-boundary-250831-29868",
        "private_key_id": "temp-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n-----END PRIVATE KEY-----\n",
        "client_email": "mineral-rights@deed-boundary-250831-29868.iam.gserviceaccount.com",
        "client_id": "123456789012345678901",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/mineral-rights%40deed-boundary-250831-29868.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }
    
    # Save the service account JSON
    with open('proper-service-account.json', 'w') as f:
        json.dump(service_account, f, indent=2)
    
    print("✅ Created proper-service-account.json")
    print("⚠️  You need to replace the private_key and other values with your actual service account details")
    
    # Create base64 version
    with open('proper-service-account.json', 'r') as f:
        content = f.read()
    
    base64_content = base64.b64encode(content.encode()).decode()
    
    with open('proper-service-account-base64.txt', 'w') as f:
        f.write(base64_content)
    
    print("✅ Created proper-service-account-base64.txt")
    print(f"📋 Base64 length: {len(base64_content)} characters")
    print(f"📋 First 100 chars: {base64_content[:100]}...")

if __name__ == "__main__":
    create_service_account_json()
