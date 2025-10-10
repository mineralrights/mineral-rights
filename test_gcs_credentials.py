#!/usr/bin/env python3
"""
Test GCS credentials and bucket access
"""
import os
import base64
import json
from google.cloud import storage
from google.oauth2 import service_account

# Set the credentials from the environment variable
GOOGLE_CREDENTIALS_BASE64 = "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAiZGVlZC1ib3VuZGFyeS0yNTA4MzEtMjk4NjgiLAogICJwcml2YXRlX2tleV9pZCI6ICIxNTg2NjdiYjdmMTQzMGQ0YTAzYzIxYTkwOTY0MDBkM2E3MTRkMGUxIiwKICAicHJpdmF0ZV9rZXkiOiAiLS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tXG5NSUlFdlFJQkFEQU5CZ2txaGtpRzl3MEJBUUVGQUFTQ0JLY3dnZ1NqQWdFQUFvSUJBUUNVdjE2Qm4rRFJQWHlKXG5YRTEzRnlOWVBBYnV3VHpmQ2EzVFFKODR4aUhJa0lOcWltWGViZWtBWDhDaUZoRkJwL1E2TkJsK0UxYkVwdlBFXG5BcjBtVkgrRG1RaVVSRnpPdUhLWUhOb2Jsd1JPZDNXb3ErT09sNnBhRFZQZjVOSGU3QVZqaUVKWEFBNnFGODhjXG5hS1VlcFhyQWxHUmVPdVBqNVFsNkk4WTJBVlRaYnlHdEV5MlRtRTdLdGk5NXZmMm9IK3R5UXVyQVdGd2R4VEJhXG5IZ3lrZFQzNUJ0L1VQWkw3WFo4SXRIaXZCcUlvY2RYV3pOM29QcllDYzE0d3ZHWGY4aVZaWm9YenFpb1gwaFZ3XG4yaFVCTTBNeGN5aW9TM3dZeDVzams4Qlo2dXpjSkFPWUtkcE9VbW9FTUlBMERLSFhKYkROUkNWU053VGRmc3VuXG5nVndzeS9wMUFnTUJBQUVDZ2dFQUNpUkRia3Bzbi9lSzZTdjQ2czRTRG9SYjQxQWJpckxKNk9XMnRXay9pUmJlXG5wRUgydTI2ay9KaUt0VzBVYWg4OXdKdDliWGQ3YXF4ZURHdzB0MHRRUFFtNXZhK0NOYm42WGpLc0VFWXJUZitlXG5id3loTEQ5Yy9GelNRMzZzd2RUM3g0SFZKc0lONkpKTkFGNkdKalp1emMzbG9LTWFsMEdEZER6Q0xwRWRWdUcrXG5oQy8xZ1pEdnZ1Z1RLbUZ6NkJldFpHdERDc0YySEdlejZ3ZkdobHdIMUJCNC9wdnJpRnVIRTlqMnJhZ1ZKYi94XG43TG9MamIyMFFPdDg4bWk2RUIvYjBWNk5MQU9idzZtRThyRm9ZTHpYR20wUlp1YnhDenBBY2Z1d1lqWG1Ydzc2XG5FQXlmenFsUDh5Q2tJVWxpTWVBejRvQ0ZUcmxZZ254WTI5TW9LYU5SdFFLQmdRRFJCeU5vSkU1TFB1cVB1cWhHXG5SVXlOK3VqSXVRZW9xUVFlV2x1UnJaMjFlTkdtbDZoQ2JDMXNOSCt0MjBXTGRRZ3lZQ2piZ3RlUVFGQ0R2NGxYXG5QVW05TkRJYWVkOUR3d0VvcEhJME9weWZrajVoVGdlTHQ4Wk1vOXIxTjUra1R5TDBKM25VclBZSFdLM2pKc25UXG52RWlIZk9qSHI1MGFqSUZpb2JiNitTN3Nvd0tCZ1FDMkxITkJSU0tQV1FHNUFWWVluaHZLN1QyaDRtUHJxczl6XG5nWjdCYlB3SlhVTDRaWDZqVkpnWjlqZGNBZ1dRWEUzb29qM2NWQWRXSFlNK3lScXBjbkVISGlMOXVxUUMyT1Z3XG5WQy9JK052NHZoMXV1VGVDZTZyTGw1cmRzOHkyQk9waWd4MkkwQkR2V3FNai8zRjRkU0lubks5VUlUZXFSZ1JoXG5VUWlCREhTV0J3S0JnQnBqdmxLZGtzenBLby91enVQZ1IrUDg4M3F6OFlXWG9ROTc5T2VWZGIyOWZTcDlKeDhMXG5yVWhsOEdDd3VEejlENFhjb2d2a252WjFTRFQ1NzRyMkQrTTFQY1lkOE5RTFFKQXpBc0FaVGtEUEk5VUJGdTdLXG55dHhPSzR5ZDh5ZTVYZjVaSllaMk43R0J2cUpZK1U0a0RUd2R4djIrN0NTMzdIWFpXTktpdHd2bEFvR0FZNEsrXG5abjMwVmVkaHlJUlJXbHNyaFZxNFd2Q0djbG80dmJpbFZyVklxM2pWTjBpQnQ0aGpHWE5rWkE3NnFST3l2d3U0XG56Tnpkc1Eybi9xanR0bmU4QkE1VFFOUXQrUnd2b0g5c0p0VEJuQXVGbWxEMFlJTmJGYUUzeURrSjdyZWFyTHRBXG5hOVowR2JzaCtHejF0NzRNTFNVcXBNTU1YQ2VwQnR1ejBJSlVRUXNDZ1lFQTBHQ09aZ2VObUpCYjFwd1hBTkFQXG40OUlDM3FUVkE3Qm0rQVhMdUJKUkhEUlNlTUtwMTJSRW5palZNcEg3MVpHSW40bDJuODYzeGRjQ3hWMXBFcHpyXG4yS2M5K1E0cW1mWkZqdG1FWWIwNVhjRHpFaGM4b0lPRmVQZ0c5UEM4bXQwekh6bTF6TmdLeTN4SnU3UHN2SE10XG5CY3V3ZzkyN0ZZMVdXNGw2bW8vNUl1TT1cbi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS1cbiIsCiAgImNsaWVudF9lbWFpbCI6ICJtaW5lcmFsLXJpZ2h0cy1zYUBkZWVkLWJvdW5kYXJ5LTI1MDgzMS0yOTg2OC5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsCiAgImNsaWVudF9pZCI6ICIxMDY1NTA5ODExNDA2MzY5Njk1OTMiLAogICJhdXRoX3VyaSI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20vby9vYXV0aDIvYXV0aCIsCiAgInRva2VuX3VyaSI6ICJodHRwczovL29hdXRoMi5nb29nbGVhcGlzLmNvbS90b2tlbiIsCiAgImF1dGhfcHJvdmlkZXJfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9vYXV0aDIvdjEvY2VydHMiLAogICJjbGllbnRfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9yb2JvdC92MS9tZXRhZGF0YS94NTA5L21pbmVyYWwtcmlnaHRzLXNhJTQwZGVlZC1ib3VuZGFyeS0yNTA4MzEtMjk4NjguaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICJ1bml2ZXJzZV9kb21haW4iOiAiZ29vZ2xlYXBpcy5jb20iCn0K"

BUCKET_NAME = "mineral-rights-pdfs-1759435410"

def test_gcs_credentials():
    """Test GCS credentials and bucket access"""
    print("🔍 Testing GCS credentials...")
    
    try:
        # Decode base64 credentials
        credentials_json = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
        credentials_info = json.loads(credentials_json)
        
        print(f"✅ Credentials decoded successfully")
        print(f"📧 Service account email: {credentials_info.get('client_email')}")
        print(f"🆔 Project ID: {credentials_info.get('project_id')}")
        
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        client = storage.Client(credentials=credentials)
        
        print(f"✅ GCS client created successfully")
        
        # Test bucket access
        bucket = client.bucket(BUCKET_NAME)
        print(f"🪣 Testing bucket: {BUCKET_NAME}")
        
        # Check if bucket exists and is accessible
        if bucket.exists():
            print(f"✅ Bucket {BUCKET_NAME} exists and is accessible")
            
            # List some objects to test permissions
            blobs = list(bucket.list_blobs(max_results=5))
            print(f"📁 Found {len(blobs)} objects in bucket")
            
            # Test creating a signed URL
            from datetime import datetime, timedelta
            blob_name = f"test-uploads/test-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
            blob = bucket.blob(blob_name)
            
            expiration = datetime.utcnow() + timedelta(hours=1)
            signed_url = blob.generate_signed_url(
                expiration=expiration,
                method="PUT",
                content_type="application/pdf"
            )
            
            print(f"✅ Signed URL generated successfully")
            print(f"🔗 URL: {signed_url[:100]}...")
            
            return True
            
        else:
            print(f"❌ Bucket {BUCKET_NAME} does not exist or is not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GCS credentials: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gcs_credentials()
    if success:
        print("\n🎉 GCS credentials test PASSED")
    else:
        print("\n💥 GCS credentials test FAILED")
