#!/usr/bin/env python3
"""
Reprocess chunks that were processed with the old buggy version
"""

import os
import subprocess
import time

def reprocess_buggy_chunks():
    """Reprocess chunks that have small file sizes (indicating 0 deeds from old buggy version)"""
    print("🔄 Reprocessing chunks with old buggy version...")
    
    # Chunks that need reprocessing (based on file sizes and timestamps)
    chunks_to_reprocess = [
        (1, 15, 1),    # Chunk 1
        (13, 27, 2),   # Chunk 2  
        (25, 39, 3),   # Chunk 3
        (37, 51, 4),   # Chunk 4
        (49, 63, 5),   # Chunk 5
        (61, 75, 6),   # Chunk 6
        (73, 87, 7),   # Chunk 7
        (85, 99, 8),   # Chunk 8
        (97, 111, 9),  # Chunk 9
        (121, 135, 11), # Chunk 11
        (133, 147, 12), # Chunk 12
        (145, 159, 13), # Chunk 13
        (157, 171, 14), # Chunk 14
        (181, 195, 16), # Chunk 16
        (205, 219, 18), # Chunk 18
        (217, 231, 19), # Chunk 19
        (229, 243, 20), # Chunk 20
        (241, 255, 21), # Chunk 21
        (253, 267, 22), # Chunk 22
        (265, 278, 23)  # Chunk 23
    ]
    
    print(f"📄 Reprocessing {len(chunks_to_reprocess)} chunks...")
    
    for i, (start_page, end_page, chunk_id) in enumerate(chunks_to_reprocess):
        print(f"\n🔄 Reprocessing chunk {chunk_id}/{len(chunks_to_reprocess)}...")
        
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
    
    print(f"\n✅ Reprocessing completed!")
    print("📊 Ready to merge results with: python merge_chunk_results.py")

if __name__ == "__main__":
    reprocess_buggy_chunks()
