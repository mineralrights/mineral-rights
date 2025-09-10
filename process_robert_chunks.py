#!/usr/bin/env python3
"""
Process all chunks for ROBERT.pdf automatically
"""

import os
import subprocess
import time

def calculate_chunks(total_pages, chunk_size=15, overlap=3):
    """Calculate chunk boundaries"""
    chunks = []
    start = 0
    
    while start < total_pages:
        end = min(start + chunk_size - 1, total_pages - 1)  # 0-indexed
        chunks.append((start + 1, end + 1))  # Convert to 1-indexed
        
        # Move start to create overlap
        start = end - overlap + 1
        
        # Ensure we don't go backwards
        if start <= chunks[-1][0]:
            start = chunks[-1][1] - overlap + 1
    
    return chunks

def process_robert_pdf():
    """Process ROBERT.pdf in chunks"""
    print("🧪 Processing ROBERT.pdf with Smart Chunking")
    print("=" * 50)
    
    # ROBERT.pdf has 101 pages
    total_pages = 101
    chunks = calculate_chunks(total_pages, chunk_size=15, overlap=3)
    
    print(f"📄 ROBERT.pdf: {total_pages} pages")
    print(f"📦 Created {len(chunks)} chunks:")
    for i, (start, end) in enumerate(chunks):
        print(f"   Chunk {i+1}: Pages {start}-{end} ({end-start+1} pages)")
    
    print(f"\n🚀 Starting chunk processing...")
    start_time = time.time()
    
    # Process each chunk
    for i, (start_page, end_page) in enumerate(chunks):
        chunk_id = i + 1
        print(f"\n🔄 Processing chunk {chunk_id}/{len(chunks)}...")
        
        # Run the single chunk processor
        cmd = [
            "python", "process_single_chunk.py", 
            str(start_page), str(end_page), str(chunk_id)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
            
            if result.returncode == 0:
                print(f"   ✅ Chunk {chunk_id} completed successfully")
                # Print the output
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
            else:
                print(f"   ❌ Chunk {chunk_id} failed:")
                print(f"   Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Chunk {chunk_id} timed out after 5 minutes")
        except Exception as e:
            print(f"   ❌ Chunk {chunk_id} error: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total processing time: {total_time:.2f}s")
    
    # Check results
    chunk_files = [f for f in os.listdir('.') if f.startswith('chunk_') and f.endswith('_results.json')]
    print(f"\n📊 Results: {len(chunk_files)} chunk files created")
    
    if chunk_files:
        print("✅ Ready to merge results with: python merge_chunk_results.py")
    else:
        print("❌ No chunk results found")

if __name__ == "__main__":
    process_robert_pdf()
