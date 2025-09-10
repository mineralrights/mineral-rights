#!/usr/bin/env python3
"""
Check Processor Status and Training

This script checks the detailed status of the custom splitting processor.
"""

import os
import sys
from pathlib import Path

def check_processor_versions():
    """Check processor versions and training status"""
    print("🧪 Checking processor versions...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878"
        
        # List processor versions
        versions = client.list_processor_versions(parent=processor_name)
        
        print(f"📊 Processor Versions:")
        version_count = 0
        for version in versions:
            version_count += 1
            print(f"   Version {version_count}:")
            print(f"     - Name: {version.name}")
            print(f"     - Display Name: {version.display_name}")
            print(f"     - State: {version.state}")
            print(f"     - Create Time: {version.create_time}")
            
            # Check if this version is deployed
            if hasattr(version, 'deployment_state'):
                print(f"     - Deployment State: {version.deployment_state}")
            
            # Check training status
            if hasattr(version, 'training_status'):
                print(f"     - Training Status: {version.training_status}")
            
            print()
        
        if version_count == 0:
            print("❌ No processor versions found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking versions: {e}")
        return False

def check_processor_details():
    """Check detailed processor information"""
    print("🧪 Checking processor details...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878"
        
        # Get processor details
        processor = client.get_processor(name=processor_name)
        
        print(f"📊 Processor Details:")
        print(f"   - Name: {processor.display_name}")
        print(f"   - Type: {processor.type_}")
        print(f"   - State: {processor.state}")
        print(f"   - Create Time: {processor.create_time}")
        # print(f"   - Update Time: {processor.update_time}")  # This field might not exist
        
        # Check if there's a default processor version
        if hasattr(processor, 'default_processor_version'):
            print(f"   - Default Version: {processor.default_processor_version}")
        
        # Check process endpoint
        if hasattr(processor, 'process_endpoint'):
            print(f"   - Process Endpoint: {processor.process_endpoint}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking processor details: {e}")
        return False

def check_processor_operations():
    """Check if there are any ongoing operations"""
    print("\n🧪 Checking processor operations...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        parent = "projects/381937358877/locations/us"
        
        # List operations
        operations = client.list_operations(parent=parent)
        
        print(f"📊 Recent Operations:")
        operation_count = 0
        for operation in operations:
            operation_count += 1
            print(f"   Operation {operation_count}:")
            print(f"     - Name: {operation.name}")
            print(f"     - Done: {operation.done}")
            if hasattr(operation, 'metadata'):
                print(f"     - Metadata: {operation.metadata}")
            print()
        
        if operation_count == 0:
            print("   No recent operations found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking operations: {e}")
        return False

def suggest_solutions():
    """Suggest solutions based on the findings"""
    print("\n💡 Suggested Solutions:")
    print("=" * 40)
    
    print("1. **Check Training Status**:")
    print("   - Go to Google Cloud Console")
    print("   - Navigate to Document AI > Processors")
    print("   - Check if your custom processor is fully trained")
    print("   - Look for any training errors or warnings")
    
    print("\n2. **Verify Processor Deployment**:")
    print("   - Ensure the processor is deployed and active")
    print("   - Check if there are any deployment errors")
    print("   - Verify the processor version is correct")
    
    print("\n3. **Check Training Data**:")
    print("   - Verify your training data is properly formatted")
    print("   - Ensure you have enough training examples")
    print("   - Check if the training completed successfully")
    
    print("\n4. **Test with Google Cloud Console**:")
    print("   - Try processing a document directly in the console")
    print("   - This will help identify if it's a code issue or processor issue")
    
    print("\n5. **Contact Support**:")
    print("   - If the processor appears to be trained but still fails")
    print("   - Contact Google Cloud support with the processor ID")
    print("   - Provide the error details and processor configuration")

def main():
    """Main function"""
    print("🚀 Processor Status Check")
    print("=" * 50)
    
    # Check processor details
    if not check_processor_details():
        print("\n❌ Could not get processor details")
        return False
    
    # Check processor versions
    if not check_processor_versions():
        print("\n❌ Could not get processor versions")
        return False
    
    # Check operations
    check_processor_operations()
    
    # Suggest solutions
    suggest_solutions()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Check the processor training status in Google Cloud Console")
        print("2. Verify the processor is fully deployed")
        print("3. Test with a simple document in the console first")
    else:
        print("\n🔧 Need to investigate further")
    
    exit(0 if success else 1)
