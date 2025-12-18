#!/usr/bin/env python3
"""
Diagnostic script to identify why long PDFs are failing
"""

import os
import sys
import time
import psutil
import gc
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def check_system_limits():
    """Check system memory and process limits"""
    print("🔍 SYSTEM DIAGNOSTICS")
    print("=" * 50)
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"💾 Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"💾 Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"💾 Memory usage: {memory.percent:.1f}%")
    
    # Process limits
    process = psutil.Process()
    print(f"🔧 Process memory: {process.memory_info().rss / (1024**2):.1f} MB")
    print(f"🔧 Process CPU: {process.cpu_percent():.1f}%")
    
    # Python limits
    import sys
    print(f"🐍 Python version: {sys.version}")
    print(f"🐍 Max recursion depth: {sys.getrecursionlimit()}")
    
    return memory.available / (1024**3)  # Return available GB

def test_pdf_processing_limits(pdf_path: str):
    """Test PDF processing with different configurations"""
    
    print(f"\n🧪 TESTING PDF PROCESSING LIMITS")
    print("=" * 50)
    print(f"📄 PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False
    
    # Get file info
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    print(f"📊 File size: {file_size:.1f} MB")
    
    # Check if we have API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        # Initialize processor
        print("🔧 Initializing processor...")
        processor = DocumentProcessor(api_key=api_key)
        
        # Test different chunk sizes
        chunk_sizes = [10, 25, 50, 100]
        
        for chunk_size in chunk_sizes:
            print(f"\n📊 Testing chunk size: {chunk_size} pages")
            
            # Monitor memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)
            print(f"💾 Memory before: {memory_before:.1f} MB")
            
            try:
                start_time = time.time()
                
                result = processor.process_document_memory_efficient(
                    pdf_path,
                    chunk_size=chunk_size,
                    max_samples=2,  # Minimal samples for speed
                    high_recall_mode=True
                )
                
                processing_time = time.time() - start_time
                memory_after = process.memory_info().rss / (1024**2)
                memory_used = memory_after - memory_before
                
                print(f"✅ Chunk size {chunk_size}: {processing_time:.1f}s, {memory_used:.1f}MB memory")
                print(f"   Classification: {result['classification']}")
                print(f"   Pages processed: {result['pages_processed']}")
                
                # Force garbage collection
                gc.collect()
                memory_after_gc = process.memory_info().rss / (1024**2)
                print(f"🧹 After GC: {memory_after_gc:.1f} MB (freed {memory_after - memory_after_gc:.1f} MB)")
                
            except Exception as e:
                print(f"❌ Chunk size {chunk_size} failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Check if it's a memory issue
                memory_after = process.memory_info().rss / (1024**2)
                if memory_after > 2000:  # More than 2GB
                    print(f"⚠️ High memory usage detected: {memory_after:.1f} MB")
                
                # Force cleanup
                gc.collect()
                continue
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_failure_patterns():
    """Analyze common failure patterns"""
    
    print(f"\n🔍 FAILURE PATTERN ANALYSIS")
    print("=" * 50)
    
    print("Common failure patterns for long PDFs:")
    print("1. 🚨 Memory exhaustion (>2GB RAM usage)")
    print("2. ⏰ Timeout issues (processing takes >30 minutes)")
    print("3. 🔄 Connection drops (SSL errors, network issues)")
    print("4. 📄 PDF parsing errors (corrupted or complex PDFs)")
    print("5. 🤖 AI API rate limits (too many requests)")
    print("6. 💾 Disk space issues (temporary files)")
    
    print("\nSolutions by failure type:")
    print("1. Memory: Use smaller chunks, more aggressive GC")
    print("2. Timeout: Use async processing, progress saving")
    print("3. Connection: Use retry logic, fallback endpoints")
    print("4. PDF: Use different PDF libraries, error handling")
    print("5. API: Implement rate limiting, request queuing")
    print("6. Disk: Clean up temp files, use streaming")

def recommend_platforms():
    """Recommend platforms based on requirements"""
    
    print(f"\n🏗️ PLATFORM RECOMMENDATIONS")
    print("=" * 50)
    
    print("For LONG PDF processing (8+ hours):")
    print()
    print("🥇 AWS ECS Fargate:")
    print("   ✅ No timeout limits")
    print("   ✅ Up to 30GB RAM")
    print("   ✅ Auto-scaling")
    print("   ✅ Cost: ~$100-200/month")
    print("   ❌ More complex setup")
    print()
    print("🥈 Google Cloud Run Jobs:")
    print("   ✅ No timeout limits")
    print("   ✅ Up to 32GB RAM")
    print("   ✅ Serverless")
    print("   ✅ Cost: ~$50-150/month")
    print("   ❌ Cold starts")
    print()
    print("🥉 Fly.io:")
    print("   ✅ No timeout limits")
    print("   ✅ Up to 16GB RAM")
    print("   ✅ Simple deployment")
    print("   ✅ Cost: ~$20-50/month")
    print("   ❌ Less memory than others")
    print()
    print("❌ NOT SUITABLE:")
    print("   ❌ AWS Lambda (15min limit)")
    print("   ❌ Vercel (30s limit)")
    print("   ❌ Render (timeout issues)")

def main():
    """Main diagnostic function"""
    
    print("🔍 LONG PDF FAILURE DIAGNOSTIC")
    print("=" * 60)
    
    # Check system
    available_gb = check_system_limits()
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Look for test PDFs
        test_paths = [
            "data/reservs/Washington DB 405_547.pdf",
            "data/no-reservs/sample.pdf", 
            "test_small.pdf"
        ]
        
        pdf_path = None
        for path in test_paths:
            if os.path.exists(path):
                pdf_path = path
                break
        
        if not pdf_path:
            print("❌ No test PDF found. Usage: python diagnose_long_pdf.py /path/to/pdf")
            return
    
    # Test PDF processing
    if available_gb < 2:
        print("⚠️ Low memory available. Results may be unreliable.")
    
    success = test_pdf_processing_limits(pdf_path)
    
    # Analyze patterns
    analyze_failure_patterns()
    
    # Recommend platforms
    recommend_platforms()
    
    if success:
        print(f"\n✅ DIAGNOSTIC COMPLETE")
        print("The system can handle the PDF, but check the recommendations above.")
    else:
        print(f"\n❌ DIAGNOSTIC FAILED")
        print("The system cannot handle this PDF. Consider platform migration.")

if __name__ == "__main__":
    main()



