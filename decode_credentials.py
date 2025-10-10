#!/usr/bin/env python3
"""
Decode and inspect the base64 credentials
"""
import base64
import json

GOOGLE_CREDENTIALS_BASE64 = "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAiZGVlZC1ib3VuZGFyeS0yNTA4MzEtMjk4NjgiLAogICJwcml2YXRlX2tleV9pZCI6ICIxNTg2NjdiYjdmMTQzMGQ0YTAzYzIxYTkwOTY0MDBkM2E3MTRkMGUxIiwKICAicHJpdmF0ZV9rZXkiOiAiLS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tXG5NSUlFdlFJQkFEQU5CZ2txaGtpRzl3MEJBUUVGQUFTQ0JLY3dnZ1NqQWdFQUFvSUJBUUNVdjE2Qm4rRFJQWHlKXG5YRTEzRnlOWVBBYnV3VHpmQ2EzVFFKODR4aUhJa0lOcWltWGViZWtBWDhDaUZoRkJwL1E2TkJsK0UxYkVwdlBFXG5BcjBtVkgrRG1RaVVSRnpPdUhLWUhOb2Jsd1JPZDNXb3ErT09sNnBhRFZQZjVOSGU3QVZqaUVKWEFBNnFGODhjXG5hS1VlcFhyQWxHUmVPdVBqNVFsNkk4WTJBVlRaYnlHdEV5MlRtRTdLdGk5NXZmMm9IK3R5UXVyQVdGd2R4VEJhXG5IZ3lrZFQzNUJ0L1VQWkw3WFo4SXRIaXZCcUlvY2RYV3pOM29QcllDYzE0d3ZHWGY4aVZaWm9YenFpb1gwaFZ3XG4yaFVCTTBNeGN5aW9TM3dZeDVzams4Qlo2dXpjSkFPWUtkcE9VbW9FTUlBMERLSFhKYkROUkNWU053VGRmc3VuXG5nVndzeS9wMUFnTUJBQUVDZ2dFQUNpUkRia3Bzbi9lSzZTdjQ2czRTRG9SYjQxQWJpckxKNk9XMnRXay9pUmJlXG5wRUgydTI2ay9KaUt0VzBVYWg4OXdKdDliWGQ3YXF4ZURHdzB0MHRRUFFtNXZhK0NOYm42WGpLc0VFWXJUZitlXG5id3loTEQ5Yy9GelNRMzZzd2RUM3g0SFZKc0lONkpKTkFGNkdKalp1emMzbG9LTWFsMEdEZER6Q0xwRWRWdUcrXG5oQy8xZ1pEdnZ1Z1RLbUZ6NkJldFpHdERDc0YySEdlejZ3ZkdobHdIMUJCNC9wdnJpRnVIRTlqMnJhZ1ZKYi94XG43TG9MamIyMFFPdDg4bWk2RUIvYjBWNk5MQU9idzZtRThyRm9ZTHpYR20wUlp1YnhDenBBY2Z1d1lqWG1Ydzc2XG5FQXlmenFsUDh5Q2tJVWxpTWVBejRvQ0ZUcmxZZ254WTI5TW9LYU5SdFFLQmdRRFJCeU5vSkU1TFB1cVB1cWhHXG5SVXlOK3VqSXVRZW9xUVFlV2x1UnJaMjFlTkdtbDZoQ2JDMXNOSCt0MjBXTGRRZ3lZQ2piZ3RlUVFGQ0R2NGxYXG5QVW05TkRJYWVkOUR3d0VvcEhJME9weWZrajVoVGdlTHQ4Wk1vOXIxTjUra1R5TDBKM25VclBZSFdLM2pKc25UXG52RWlIZk9qSHI1MGFqSUZpb2JiNitTN3Nvd0tCZ1FDMkxITkJSU0tQV1FHNUFWWVluaHZLN1QyaDRtUHJxczl6XG5nWjdCYlB3SlhVTDRaWDZqVkpnWjlqZGNBZ1dRWEUzb29qM2NWQWRXSFlNK3lScXBjbkVISGlMOXVxUUMyT1Z3XG5WQy9JK052NHZoMXV1VGVDZTZyTGw1cmRzOHkyQk9waWd4MkkwQkR2V3FNai8zRjRkU0lubks5VUlUZXFSZ1JoXG5VUWlCREhTV0J3S0JnQnBqdmxLZGtzenBLby91enVQZ1IrUDg4M3F6OFlXWG9ROTc5T2VWZGIyOWZTcDlKeDhMXG5yVWhsOEdDd3VEejlENFhjb2d2a252WjFTRFQ1NzRyMkQrTTFQY1lkOE5RTFFKQXpBc0FaVGtEUEk5VUJGdTdLXG55dHhPSzR5ZDh5ZTVYZjVaSllaMk43R0J2cUpZK1U0a0RUd2R4djIrN0NTMzdIWFpXTktpdHd2bEFvR0FZNEsrXG5abjMwVmVkaHlJUlJXbHNyaFZxNFd2Q0djbG80dmJpbFZyVklxM2pWTjBpQnQ0aGpHWE5rWkE3NnFST3l2d3U0XG56Tnpkc1Eybi9xanR0bmU4QkE1VFFOUXQrUnd2b0g5c0p0VEJuQXVGbWxEMFlJTmJGYUUzeURrSjdyZWFyTHRBXG5hOVowR2JzaCtHejF0NzRNTFNVcXBNTU1YQ2VwQnR1ejBJSlVRUXNDZ1lFQTBHQ09aZ2VObUpCYjFwd1hBTkFQXG40OUlDM3FUVkE3Qm0rQVhMdUJKUkhEUlNlTUtwMTJSRW5palZNcEg3MVpHSW40bDJuODYzeGRjQ3hWMXBFcHpyXG4yS2M5K1E0cW1mWkZqdG1FWWIwNVhjRHpFaGM4b0lPRmVQZ0c5UEM4bXQwekh6bTF6TmdLeTN4SnU3UHN2SE10XG5CY3V3ZzkyN0ZZMVdXNGw2bW8vNUl1TT1cbi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS1cbiIsCiAgImNsaWVudF9lbWFpbCI6ICJtaW5lcmFsLXJpZ2h0cy1zYUBkZWVkLWJvdW5kYXJ5LTI1MDgzMS0yOTg2OC5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsCiAgImNsaWVudF9pZCI6ICIxMDY1NTA5ODExNDA2MzY5Njk1OTMiLAogICJhdXRoX3VyaSI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20vby9vYXV0aDIvYXV0aCIsCiAgInRva2VuX3VyaSI6ICJodHRwczovL29hdXRoMi5nb29nbGVhcGlzLmNvbS90b2tlbiIsCiAgImF1dGhfcHJvdmlkZXJfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9vYXV0aDIvdjEvY2VydHMiLAogICJjbGllbnRfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9yb2JvdC92MS9tZXRhZGF0YS94NTA5L21pbmVyYWwtcmlnaHRzLXNhJTQwZGVlZC1ib3VuZGFyeS0yNTA4MzEtMjk4NjguaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICJ1bml2ZXJzZV9kb21haW4iOiAiZ29vZ2xlYXBpcy5jb20iCn0K"

def decode_credentials():
    """Decode and inspect the credentials"""
    try:
        credentials_json = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
        credentials_info = json.loads(credentials_json)
        
        print("🔍 Decoded credentials:")
        print(f"Type: {credentials_info.get('type')}")
        print(f"Project ID: {credentials_info.get('project_id')}")
        print(f"Client Email: {credentials_info.get('client_email')}")
        print(f"Client ID: {credentials_info.get('client_id')}")
        print(f"Private Key ID: {credentials_info.get('private_key_id')}")
        
        private_key = credentials_info.get('private_key', '')
        print(f"\n🔑 Private Key (first 100 chars):")
        print(private_key[:100])
        print(f"\n🔑 Private Key (last 100 chars):")
        print(private_key[-100:])
        
        # Check if it looks like a valid PEM key
        if private_key.startswith('-----BEGIN PRIVATE KEY-----'):
            print("✅ Private key appears to be in correct PEM format")
        else:
            print("❌ Private key does not start with -----BEGIN PRIVATE KEY-----")
            
        if private_key.endswith('-----END PRIVATE KEY-----'):
            print("✅ Private key appears to end correctly")
        else:
            print("❌ Private key does not end with -----END PRIVATE KEY-----")
            
        # Check for newlines
        if '\\n' in private_key:
            print("⚠️  Private key contains \\n (escaped newlines) - this might be the issue")
            print("🔧 Trying to fix by replacing \\n with actual newlines...")
            
            fixed_key = private_key.replace('\\n', '\n')
            print(f"Fixed key (first 100 chars):")
            print(fixed_key[:100])
            
            # Test if this fixes it
            try:
                from google.oauth2 import service_account
                from google.auth.crypt import _cryptography_rsa
                
                # Create a copy of credentials with fixed key
                fixed_credentials = credentials_info.copy()
                fixed_credentials['private_key'] = fixed_key
                
                credentials = service_account.Credentials.from_service_account_info(fixed_credentials)
                print("✅ Fixed credentials work!")
                return fixed_credentials
                
            except Exception as e:
                print(f"❌ Fixed credentials still don't work: {e}")
                return None
        else:
            print("✅ Private key appears to have proper newlines")
            return credentials_info
            
    except Exception as e:
        print(f"❌ Error decoding credentials: {e}")
        return None

if __name__ == "__main__":
    result = decode_credentials()
    if result:
        print("\n🎉 Credentials decoded successfully")
    else:
        print("\n💥 Failed to decode credentials")
