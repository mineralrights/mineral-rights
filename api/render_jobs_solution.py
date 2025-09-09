#!/usr/bin/env python3
"""
Render Jobs Solution for Long-Running Processing
===============================================

This script demonstrates how to convert your current web service 
into a Render Job that can run for 8+ hours without timeout issues.

Render Jobs are designed for:
- Long-running batch processing
- Tasks that run to completion
- No web service timeout constraints
- Better resource allocation for intensive tasks
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add your project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.mineral_rights.document_classifier import DocumentProcessor

def process_document_job(pdf_path: str, processing_mode: str = "multi_deed", 
                        splitting_strategy: str = "smart_detection"):
    """
    Process a document as a Render Job - can run for 8+ hours without timeouts
    """
    print(f"🚀 Starting Render Job for long-running processing")
    print(f"📁 File: {pdf_path}")
    print(f"🎯 Processing mode: {processing_mode}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize processor
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    processor = DocumentProcessor(api_key)
    
    # Process the document
    if processing_mode == "single_deed":
        print("📄 Using single deed processing")
        result = processor.process_document(pdf_path)
    elif processing_mode == "multi_deed":
        print(f"📑 Using multi-deed processing with strategy: {splitting_strategy}")
        result = processor.process_multi_deed_document(
            pdf_path, 
            strategy=splitting_strategy
        )
    else:
        raise ValueError(f"Unknown processing_mode: {processing_mode}")
    
    # Save results to file for retrieval
    output_file = f"job_result_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Processing completed successfully")
    print(f"📄 Results saved to: {output_file}")
    print(f"⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Process document as Render Job")
    parser.add_argument("pdf_path", help="Path to PDF file to process")
    parser.add_argument("--mode", default="multi_deed", 
                       choices=["single_deed", "multi_deed"],
                       help="Processing mode")
    parser.add_argument("--strategy", default="smart_detection",
                       choices=["smart_detection", "ai_assisted"],
                       help="Splitting strategy for multi-deed mode")
    
    args = parser.parse_args()
    
    try:
        result = process_document_job(
            args.pdf_path, 
            args.mode, 
            args.strategy
        )
        print("🎉 Job completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Job failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
