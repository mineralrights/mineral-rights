#!/usr/bin/env python3
"""
Check Batch Processing Status

This script checks the status of the running batch processing job.
"""

import os
import json
import time
from setup_gcs_batch import GCSBatchProcessingService

def check_job_status():
    """Check the status of the current batch processing job"""
    try:
        # Load job info
        if not os.path.exists('batch_job_info.json'):
            print("❌ No batch job found. Run setup_gcs_batch.py first.")
            return False
        
        with open('batch_job_info.json', 'r') as f:
            job_info = json.load(f)
        
        print("📊 Checking Batch Processing Status")
        print("=" * 40)
        print(f"📋 Job ID: {job_info['job_id']}")
        print(f"📁 Input: {job_info['input_uri']}")
        print(f"📁 Output: {job_info['output_uri']}")
        print(f"⏰ Started: {time.ctime(job_info['created_at'])}")
        
        # Create service
        batch_service = GCSBatchProcessingService(
            project_id="381937358877",
            location="us",
            bucket_name=job_info['bucket_name']
        )
        
        # Check status
        print(f"\n🔍 Checking job status...")
        status = batch_service.check_job_status(job_info['job_id'])
        
        print(f"📊 Status: {status}")
        
        if status == "COMPLETED":
            print(f"\n🎉 Job completed! Downloading results...")
            results = batch_service.download_results(job_info['output_uri'])
            
            print(f"\n📊 Results Summary:")
            print(f"   - Total entities found: {len(results)}")
            
            # Show first few results
            print(f"\n🔍 Sample Results:")
            for i, result in enumerate(results[:10]):
                print(f"   - {i+1}. {result['type']}: confidence {result['confidence']:.3f}")
                if result['page_refs']:
                    pages = [ref.get('page', '?') for ref in result['page_refs']]
                    print(f"     Pages: {pages}")
            
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more")
            
            # Save results
            with open('batch_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: batch_results.json")
            
            return True
            
        elif status == "RUNNING":
            print(f"\n⏳ Job is still running...")
            print(f"   - This is normal for large documents")
            print(f"   - Check again in a few minutes")
            return False
            
        elif "FAILED" in status:
            print(f"\n❌ Job failed: {status}")
            return False
            
        else:
            print(f"\n❓ Unknown status: {status}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking job status: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_job():
    """Monitor the job until completion"""
    print("🔄 Monitoring batch processing job...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            completed = check_job_status()
            if completed:
                print("\n✅ Job completed successfully!")
                break
            
            print(f"\n⏳ Waiting 30 seconds before next check...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped by user")
        print(f"💡 You can run 'python check_batch_status.py' later to check status")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_job()
    else:
        check_job_status()
