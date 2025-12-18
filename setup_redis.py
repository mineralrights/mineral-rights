#!/usr/bin/env python3
"""
Setup Upstash Redis for job persistence
"""

import os
import requests
import json

def setup_upstash_redis():
    """Setup Upstash Redis for job persistence"""
    
    print("🚀 Setting up Upstash Redis for job persistence...")
    
    # Check if we already have Redis credentials
    upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
    upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
    if upstash_url and upstash_token:
        print("✅ Redis credentials already configured")
        print(f"URL: {upstash_url}")
        print(f"Token: {upstash_token[:10]}...")
        return True
    
    print("❌ No Redis credentials found")
    print("\n📋 To set up Redis job persistence:")
    print("1. Go to https://console.upstash.com/")
    print("2. Create a new Redis database")
    print("3. Copy the REST URL and REST Token")
    print("4. Add them to Railway environment variables:")
    print("   - UPSTASH_REDIS_REST_URL")
    print("   - UPSTASH_REDIS_REST_TOKEN")
    print("\n🔄 After adding the credentials, Railway will automatically redeploy")
    
    return False

def test_redis_connection():
    """Test Redis connection"""
    
    upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
    upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
    if not upstash_url or not upstash_token:
        print("❌ No Redis credentials found")
        return False
    
    try:
        # Test Redis connection
        response = requests.get(
            f"{upstash_url}/ping",
            headers={'Authorization': f"Bearer {upstash_token}"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Redis connection successful")
            return True
        else:
            print(f"❌ Redis connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Redis connection error: {e}")
        return False

if __name__ == "__main__":
    setup_upstash_redis()
    test_redis_connection()
