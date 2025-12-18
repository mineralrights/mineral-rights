#!/usr/bin/env python3
"""
Test the new service account credentials
"""
import base64
import json
import tempfile
import os

# New credentials from the downloaded JSON
NEW_CREDENTIALS_BASE64 = "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAibWluZXJhbC1yaWdodHMtYXBwIiwKICAicHJpdmF0ZV9rZXlfaWQiOiAiMmM5NDQyNGMzYWFiM2Q2OGE4NDA3MmE0Yzk5ODU2ZTIyYTBmMjQwNCIsCiAgInByaXZhdGVfa2V5IjogIi0tLS0tQkVHSU4gUFJJVkFURSBLRVktLS0tLVxuTUlJRXZRSUJBREFOQmdrcWhraUc5dzBCQVFFRkFBU0NCS2N3Z2dTakFnRUFBb0lCQVFDZG13RndFZWhEMXpCL1xua1ZDZU82OGJFa0ZSbkc4SVovOTlWUGNJcEdYL1libnZ6NlFkbmllWU1oYkptZWN4d2JoS3VWN2FSTStFVnlaTFxubE5Hd0t5dUh2VHhaUy8rQXBhMkV3SFJtbUZGNmdPaDRacHlWTWhlQkxlby83SVZPTnVhY0NKQWsrcjRMTWlBRFxuOGpNa3g4TTltTUtVeWxTQk5henFDbytOZGdxd0hOSXBwdVh2UDhVUjlDL3NnMWZWQ1gzUmJRZXllZkNYSEQ5WFxuTkhOZUxpQ3F4Z01yYVdzaDZsOHI4ODRORStHZzdqVHAwcTNNY080TGNxTmFUbjBtbHh0d01FQm4va254c1pTelxuclU2c2lnbXFUUkJ3eHJ1UG5HaWNGWFJ3a21MQ25zZnp2Y0FGQ0EyWkFjK1FMc0licGJPbStSb0lSSzNOT21OYlxucHFhVmlmU2ZBZ01CQUFFQ2dnRUFFMWhKZldReW9sSnIrVm5iVDRqR09LaEN6STNMSjg0b0V5TldMUmFNdWM0dVxuSzBGMzJmWWwwd01NUHFqb01xTWVCL2RTZ3ErMmhGUEpkMkVvRHh2Rlp6bXkzM0preFU3UXRST0ZxMnJLZVhobFxuZXhNbWsyamN4TFkwNDFnWWtWMzJGclNMQ05iMEhsNEQydjE1WDBHR2RVTG1XS3F3WTZtMGRJZTJVTDAvN2tJY1xubjRrdVJPVk5zQWtycFBMMG1qZ0xnaGN2djBpRmtGUmU0MDBFV3VTUCtYSC9nNmgwcEtUZk0wL0xHNmFDd05YUVxuUm0yZzlQQU1YeXFhZFpDOFF5Q1lHeW1za2hVa1BmazJ6MFpJcXFweEVHVXQzWWwxbGk2ckkrVll0aFBVS0dQTFxuVGVFcllDWDNrNTlVUU40S3Q1UzlsTzN6Z3d0QW41YitWNkR5NnM4cHlRS0JnUURXTlhac2RKVkIycFdSZ3ZvaVxubndOd29yb1QxZkFoK3o4V2NuaUM1WFd2cHlQUklwVGMxVXN0UFcvR0pDcHZyaDlJdjN4TlY5R1k3YXI2MnIvdFxuWU9ubW1IY3lXb3J1ZmIyT1BDNGgvVG9PbWZKeG1zMUwrSDFqRVJUUjFEK0RIc2cwc2ZuekNTWHpmSjUxSm9SdVxuY2FuNXFTWHMrSktibjdxMWM3Tk9DMWdxWndLQmdRQzhXb1YyVk5pTE5EazRNYSs2d3UvYzI3Tkt1aFIzRWZvQlxud1RkSHN4VmsxSjZOaDVwMkxGNFE5aEJJZlFqS01RbmcrMmVwT1pxam9CY3k5SVAvQWY1WElXY3pYblU2MXdqR1xudTlJUmNJT3FxNWNySXhNR0pZYmRYTnpyYS9pQm5RbzRTczNFL3h1WE93aFlUUURpMysvOUlVd2U3OUNzNUN2OVxuSm1XQTQ5OXhDUUtCZ0VYZjFxdDJOQ0h4TFl6enpxaHdlbXpKaUMxa1FocXpuRmEwTEg5MlhqZFlMQ1RTUlFEc1xuU3NPTklPTGZkVUJNNmtPT3d2dHZ4QjFBbWQrT2I4RDlOZzlVZUwxaUw2T3dQSjhqSG1GVCt4WThQWXUxVlhhTVxucmtvY2psQU1EbE8xUE5XRG9PY1lldHE4TWV4QkRqNEFzZE9ReTZCTFRYZWFXUXRMbkplK3Q1bk5Bb0dBZjZlV1xuSnFIUWRWLzZtOVJJOW5uaDJUenBvZTdGcWdGOEFLNTBDZHNjMTg2bWV1TjUwemUwdFNnZjF4RXU0T0lsZ043Q1xuM2RWVnNpbnhMeTY3T3h5ZHhXMjFKUUtTejBNb0JwRUxDWmpKRStYaHVYRzNGZ1pmQmk1RzZDT3dOQ0E3NmZVQVxueXMvZllqcTNLQ2xnUFdOcW9wTnJwTmdDQlB0THVQSEovM1h4WFFrQ2dZRUFrVWdDTnFmS2FhOG9rYVZ4cG9KTVxub0dwNjBzVWtlV0JNZ1ErMnZNb0E0SnlKUE1SbjJnVG9vaklaeXE1QkNvSFY2N1UxZTN2UlRpTVRLT0tCWERidVxuOFZQZzljcklrNW9DVyt4eklqUXp3RFNLVUF1VDIxMGZNTnp0cHcyTW8xS2lHSjRqV3o1c2lTQlo2UTJqZEVqRlxuNmduRmxYMC9VK3lXand1MEVNK2l6VG89XG4tLS0tLUVORCBQUklWQVRFIEtFWS0tLS0tXG4iLAogICJjbGllbnRfZW1haWwiOiAibWluZXJhbC1yaWdodHMtc2EtdjJAbWluZXJhbC1yaWdodHMtYXBwLmlhbS5nc2VydmljZWFjY291bnQuY29tIiwKICAiY2xpZW50X2lkIjogIjExMDM3NDAzNzYxNjA3Nzk3MzYxOSIsCiAgImF1dGhfdXJpIjogImh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbS9vL29hdXRoMi9hdXRoIiwKICAidG9rZW5fdXJpIjogImh0dHBzOi8vb2F1dGgyLmdvb2dsZWFwaXMuY29tL3Rva2VuIiwKICAiYXV0aF9wcm92aWRlcl94NTA5X2NlcnRfdXJsIjogImh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL29hdXRoMi92MS9jZXJ0cyIsCiAgImNsaWVudF94NTA5X2NlcnRfdXJsIjogImh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL3JvYm90L3YxL21ldGFkYXRhL3g1MDkvbWluZXJhbC1yaWdodHMtc2EtdjIlNDBtaW5lcmFsLXJpZ2h0cy1hcHAuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICJ1bml2ZXJzZV9kb21haW4iOiAiZ29vZ2xlYXBpcy5jb20iCn0K"

BUCKET_NAME = "mineral-rights-pdfs-1759435410"

def test_new_credentials():
    """Test the new service account credentials"""
    print("🔍 Testing new service account credentials...")
    
    try:
        # Decode credentials
        credentials_json = base64.b64decode(NEW_CREDENTIALS_BASE64).decode('utf-8')
        credentials_info = json.loads(credentials_json)
        
        print(f"✅ Credentials decoded successfully")
        print(f"📧 Service account email: {credentials_info.get('client_email')}")
        print(f"🆔 Project ID: {credentials_info.get('project_id')}")
        
        # Create temporary credentials file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(credentials_info, f)
            credentials_path = f.name
        
        print(f"📁 Created temporary credentials file: {credentials_path}")
        
        # Set environment variable
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Test with GCS
        from google.cloud import storage
        client = storage.Client()
        
        print(f"✅ GCS client created successfully")
        
        # Test bucket access
        bucket = client.bucket(BUCKET_NAME)
        print(f"🪣 Testing bucket: {BUCKET_NAME}")
        
        if bucket.exists():
            print(f"✅ Bucket {BUCKET_NAME} exists and is accessible")
            
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
        print(f"❌ Error testing new credentials: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if 'credentials_path' in locals():
            try:
                os.unlink(credentials_path)
                print(f"🧹 Cleaned up temporary file: {credentials_path}")
            except:
                pass

if __name__ == "__main__":
    success = test_new_credentials()
    if success:
        print("\n🎉 New credentials test PASSED")
        print("\n📋 Copy this to your Vercel environment variables:")
        print(f"GOOGLE_CREDENTIALS_BASE64={NEW_CREDENTIALS_BASE64}")
    else:
        print("\n💥 New credentials test FAILED")
