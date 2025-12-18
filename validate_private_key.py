#!/usr/bin/env python3
"""
Validate the private key format
"""
import base64
import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

GOOGLE_CREDENTIALS_BASE64 = "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAiZGVlZC1ib3VuZGFyeS0yNTA4MzEtMjk4NjgiLAogICJwcml2YXRlX2tleV9pZCI6ICIxNTg2NjdiYjdmMTQzMGQ0YTAzYzIxYTkwOTY0MDBkM2E3MTRkMGUxIiwKICAicHJpdmF0ZV9rZXkiOiAiLS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tXG5NSUlFdlFJQkFEQU5CZ2txaGtpRzl3MEJBUUVGQUFTQ0JLY3dnZ1NqQWdFQUFvSUJBUUNVdjE2Qm4rRFJQWHlKXG5YRTEzRnlOWVBBYnV3VHpmQ2EzVFFKODR4aUhJa0lOcWltWGViZWtBWDhDaUZoRkJwL1E2TkJsK0UxYkVwdlBFXG5BcjBtVkgrRG1RaVVSRnpPdUhLWUhOb2Jsd1JPZDNXb3ErT09sNnBhRFZQZjVOSGU3QVZqaUVKWEFBNnFGODhjXG5hS1VlcFhyQWxHUmVPdVBqNVFsNkk4WTJBVlRaYnlHdEV5MlRtRTdLdGk5NXZmMm9IK3R5UXVyQVdGd2R4VEJhXG5IZ3lrZFQzNUJ0L1VQWkw3WFo4SXRIaXZCcUlvY2RYV3pOM29QcllDYzE0d3ZHWGY4aVZaWm9YenFpb1gwaFZ3XG4yaFVCTTBNeGN5aW9TM3dZeDVzams4Qlo2dXpjSkFPWUtkcE9VbW9FTUlBMERLSFhKYkROUkNWU053VGRmc3VuXG5nVndzeS9wMUFnTUJBQUVDZ2dFQUNpUkRia3Bzbi9lSzZTdjQ2czRTRG9SYjQxQWJpckxKNk9XMnRXay9pUmJlXG5wRUgydTI2ay9KaUt0VzBVYWg4OXdKdDliWGQ3YXF4ZURHdzB0MHRRUFFtNXZhK0NOYm42WGpLc0VFWXJUZitlXG5id3loTEQ5Yy9GelNRMzZzd2RUM3g0SFZKc0lONkpKTkFGNkdKalp1emMzbG9LTWFsMEdEZER6Q0xwRWRWdUcrXG5oQy8xZ1pEdnZ1Z1RLbUZ6NkJldFpHdERDc0YySEdlejZ3ZkdobHdIMUJCNC9wdnJpRnVIRTlqMnJhZ1ZKYi94XG43TG9MamIyMFFPdDg4bWk2RUIvYjBWNk5MQU9idzZtRThyRm9ZTHpYR20wUlp1YnhDenBBY2Z1d1lqWG1Ydzc2XG5FQXlmenFsUDh5Q2tJVWxpTWVBejRvQ0ZUcmxZZ254WTI5TW9LYU5SdFFLQmdRRFJCeU5vSkU1TFB1cVB1cWhHXG5SVXlOK3VqSXVRZW9xUVFlV2x1UnJaMjFlTkdtbDZoQ2JDMXNOSCt0MjBXTGRRZ3lZQ2piZ3RlUVFGQ0R2NGxYXG5QVW05TkRJYWVkOUR3d0VvcEhJME9weWZrajVoVGdlTHQ4Wk1vOXIxTjUra1R5TDBKM25VclBZSFdLM2pKc25UXG52RWlIZk9qSHI1MGFqSUZpb2JiNitTN3Nvd0tCZ1FDMkxITkJSU0tQV1FHNUFWWVluaHZLN1QyaDRtUHJxczl6XG5nWjdCYlB3SlhVTDRaWDZqVkpnWjlqZGNBZ1dRWEUzb29qM2NWQWRXSFlNK3lScXBjbkVISGlMOXVxUUMyT1Z3XG5WQy9JK052NHZoMXV1VGVDZTZyTGw1cmRzOHkyQk9waWd4MkkwQkR2V3FNai8zRjRkU0lubks5VUlUZXFSZ1JoXG5VUWlCREhTV0J3S0JnQnBqdmxLZGtzenBLby91enVQZ1IrUDg4M3F6OFlXWG9ROTc5T2VWZGIyOWZTcDlKeDhMXG5yVWhsOEdDd3VEejlENFhjb2d2a252WjFTRFQ1NzRyMkQrTTFQY1lkOE5RTFFKQXpBc0FaVGtEUEk5VUJGdTdLXG55dHhPSzR5ZDh5ZTVYZjVaSllaMk43R0J2cUpZK1U0a0RUd2R4djIrN0NTMzdIWFpXTktpdHd2bEFvR0FZNEsrXG5abjMwVmVkaHlJUlJXbHNyaFZxNFd2Q0djbG80dmJpbFZyVklxM2pWTjBpQnQ0aGpHWE5rWkE3NnFST3l2d3U0XG56Tnpkc1Eybi9xanR0bmU4QkE1VFFOUXQrUnd2b0g5c0p0VEJuQXVGbWxEMFlJTmJGYUUzeURrSjdyZWFyTHRBXG5hOVowR2JzaCtHejF0NzRNTFNVcXBNTU1YQ2VwQnR1ejBJSlVRUXNDZ1lFQTBHQ09aZ2VObUpCYjFwd1hBTkFQXG40OUlDM3FUVkE3Qm0rQVhMdUJKUkhEUlNlTUtwMTJSRW5palZNcEg3MVpHSW40bDJuODYzeGRjQ3hWMXBFcHpyXG4yS2M5K1E0cW1mWkZqdG1FWWIwNVhjRHpFaGM4b0lPRmVQZ0c5UEM4bXQwekh6bTF6TmdLeTN4SnU3UHN2SE10XG5CY3V3ZzkyN0ZZMVdXNGw2bW8vNUl1TT1cbi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS1cbiIsCiAgImNsaWVudF9lbWFpbCI6ICJtaW5lcmFsLXJpZ2h0cy1zYUBkZWVkLWJvdW5kYXJ5LTI1MDgzMS0yOTg2OC5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsCiAgImNsaWVudF9pZCI6ICIxMDY1NTA5ODExNDA2MzY5Njk1OTMiLAogICJhdXRoX3VyaSI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20vby9vYXV0aDIvYXV0aCIsCiAgInRva2VuX3VyaSI6ICJodHRwczovL29hdXRoMi5nb29nbGVhcGlzLmNvbS90b2tlbiIsCiAgImF1dGhfcHJvdmlkZXJfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9vYXV0aDIvdjEvY2VydHMiLAogICJjbGllbnRfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9yb2JvdC92MS9tZXRhZGF0YS94NTA5L21pbmVyYWwtcmlnaHRzLXNhJTQwZGVlZC1ib3VuZGFyeS0yNTA4MzEtMjk4NjguaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICJ1bml2ZXJzZV9kb21haW4iOiAiZ29vZ2xlYXBpcy5jb20iCn0K"

def validate_private_key():
    """Validate the private key format"""
    print("🔍 Validating private key...")
    
    try:
        # Decode credentials
        credentials_json = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
        credentials_info = json.loads(credentials_json)
        
        private_key = credentials_info.get('private_key', '')
        print(f"🔑 Private key length: {len(private_key)}")
        print(f"🔑 First 50 chars: {private_key[:50]}")
        print(f"🔑 Last 50 chars: {private_key[-50:]}")
        
        # Check if it starts and ends correctly
        if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
            print("❌ Private key does not start with -----BEGIN PRIVATE KEY-----")
            return False
            
        if not private_key.endswith('-----END PRIVATE KEY-----'):
            print("❌ Private key does not end with -----END PRIVATE KEY-----")
            return False
            
        print("✅ Private key has correct PEM headers")
        
        # Try to parse the private key
        try:
            key_bytes = private_key.encode('utf-8')
            private_key_obj = serialization.load_pem_private_key(
                key_bytes,
                password=None
            )
            print("✅ Private key parsed successfully")
            print(f"🔑 Key type: {type(private_key_obj)}")
            print(f"🔑 Key size: {private_key_obj.key_size if hasattr(private_key_obj, 'key_size') else 'Unknown'}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to parse private key: {e}")
            
            # Try to identify the issue
            lines = private_key.split('\n')
            print(f"🔍 Key has {len(lines)} lines")
            print(f"🔍 First line: {lines[0]}")
            print(f"🔍 Last line: {lines[-1]}")
            
            # Check for common issues
            if '\\n' in private_key:
                print("⚠️  Key contains \\n (escaped newlines) - this is likely the issue!")
                print("🔧 Trying to fix by replacing \\n with actual newlines...")
                
                fixed_key = private_key.replace('\\n', '\n')
                print(f"🔧 Fixed key length: {len(fixed_key)}")
                
                try:
                    key_bytes = fixed_key.encode('utf-8')
                    private_key_obj = serialization.load_pem_private_key(
                        key_bytes,
                        password=None
                    )
                    print("✅ Fixed private key parsed successfully!")
                    return True
                except Exception as e2:
                    print(f"❌ Fixed key still doesn't work: {e2}")
                    return False
            else:
                print("🔍 No escaped newlines found")
                return False
                
    except Exception as e:
        print(f"❌ Error validating private key: {e}")
        return False

if __name__ == "__main__":
    success = validate_private_key()
    if success:
        print("\n🎉 Private key validation PASSED")
    else:
        print("\n💥 Private key validation FAILED")
