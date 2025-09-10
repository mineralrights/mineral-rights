#!/usr/bin/env python3
"""
Merge results from individual chunk processing files
"""

import os
import json
import glob
from typing import List, Dict, Any

def load_chunk_results():
    """Load all chunk result files"""
    chunk_files = glob.glob("chunk_*_results.json")
    chunk_files.sort(key=lambda x: int(x.split('_')[1]))
    
    all_results = []
    for file in chunk_files:
        with open(file, 'r') as f:
            results = json.load(f)
            all_results.append(results)
            print(f"📄 Loaded {file}: {len(results['deeds'])} deeds")
    
    return all_results

def merge_deeds(all_results: List[Dict]):
    """Merge deeds from all chunks"""
    all_deeds = []
    
    for result in all_results:
        for deed in result['deeds']:
            all_deeds.append(deed)
    
    print(f"\n📊 Total deeds before merge: {len(all_deeds)}")
    
    # Sort by starting page
    all_deeds.sort(key=lambda x: min(x['pages']) if x['pages'] else 0)
    
    merged_deeds = []
    
    for deed in all_deeds:
        if not deed['pages']:
            continue
            
        deed_start = min(deed['pages'])
        deed_end = max(deed['pages'])
        
        # Check if this deed should be merged with any existing merged deed
        merged = False
        
        for i, merged_deed in enumerate(merged_deeds):
            merged_start = min(merged_deed['pages'])
            merged_end = max(merged_deed['pages'])
            
            # Check for significant overlap (at least 2 pages)
            overlap_start = max(deed_start, merged_start)
            overlap_end = min(deed_end, merged_end)
            overlap_pages = max(0, overlap_end - overlap_start + 1)
            
            # Merge if there's significant overlap
            if overlap_pages >= 2:
                # Merge pages
                merged_pages = sorted(list(set(merged_deed['pages'] + deed['pages'])))
                merged_deeds[i] = {
                    'type': 'DEED',
                    'confidence': max(merged_deed['confidence'], deed['confidence']),
                    'pages': merged_pages,
                    'chunk_id': f"{merged_deed.get('chunk_id', 'unknown')}+{deed.get('chunk_id', 'unknown')}"
                }
                print(f"   Merged deed: Pages {merged_pages} (overlap: {overlap_pages} pages)")
                merged = True
                break
        
        if not merged:
            merged_deeds.append(deed)
            print(f"   New deed: Pages {deed['pages']}")
    
    return merged_deeds

def apply_offset_correction(merged_deeds, offset: int = 1):
    """Apply systematic offset correction"""
    corrected_deeds = []
    
    for deed in merged_deeds:
        corrected_pages = [p + offset for p in deed['pages']]
        corrected_deed = deed.copy()
        corrected_deed['pages'] = corrected_pages
        corrected_deed['original_pages'] = deed['pages']  # Keep original for reference
        corrected_deeds.append(corrected_deed)
    
    return corrected_deeds

def compare_with_ground_truth(final_deeds, pdf_name: str = "FRANCO"):
    """Compare results with ground truth"""
    gt_path = f"data/multi-deed/normalized_boundaries/{pdf_name}.normalized.json"
    
    if not os.path.exists(gt_path):
        print(f"❌ Ground truth file not found: {gt_path}")
        return
    
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    
    gt_deed_count = gt['deed_count']
    gt_page_starts = gt['page_starts']
    
    print(f"\n📈 Comparison with Ground Truth:")
    print(f"   - Ground truth deeds: {gt_deed_count}")
    print(f"   - Detected deeds: {len(final_deeds)}")
    print(f"   - Detection ratio: {len(final_deeds)/gt_deed_count:.2f}x")
    
    # Calculate accuracy metrics
    ai_starts = [min(d['pages']) for d in final_deeds if d['pages']]
    if ai_starts and gt_page_starts:
        # Check matches within 1 page tolerance
        matches = 0
        for gt_start in gt_page_starts:
            closest_ai_start = min(ai_starts, key=lambda x: abs(x - gt_start))
            if abs(closest_ai_start - gt_start) <= 1:
                matches += 1
        
        print(f"   - Matches (within 1 page): {matches}/{gt_deed_count}")
        print(f"   - Match rate: {matches/gt_deed_count:.2f}")
        
        # Check for systematic offset
        if len(ai_starts) >= 3 and len(gt_page_starts) >= 3:
            offsets = [ai - gt for ai, gt in zip(ai_starts[:3], gt_page_starts[:3])]
            print(f"   - First 3 offsets: {offsets}")
            if len(set(offsets)) == 1:
                print(f"   - Systematic offset: {offsets[0]} pages")
            else:
                print(f"   - No systematic offset detected")
    
    # Show first few deed ranges
    print(f"\n📄 First 5 Detected Deeds:")
    for i, deed in enumerate(final_deeds[:5]):
        pages = deed['pages']
        if pages:
            print(f"   Deed {i+1}: Pages {min(pages)}-{max(pages)} (Confidence: {deed['confidence']:.3f})")
    
    # Show ground truth for comparison
    print(f"\n📄 Ground Truth Deed Starts:")
    for i, start in enumerate(gt_page_starts[:5]):
        print(f"   Deed {i+1}: Page {start}")

def main():
    """Main function"""
    print("🧪 Merging Chunk Results")
    print("=" * 50)
    
    # Load chunk results
    all_results = load_chunk_results()
    
    if not all_results:
        print("❌ No chunk result files found")
        return
    
    # Calculate total processing time
    total_time = sum(result['processing_time'] for result in all_results)
    print(f"\n⏱️  Total processing time: {total_time:.2f}s")
    
    # Merge deeds
    print(f"\n📊 Merging deeds...")
    merged_deeds = merge_deeds(all_results)
    
    # Apply offset correction
    print(f"\n🔧 Applying +1 page offset correction...")
    final_deeds = apply_offset_correction(merged_deeds, offset=1)
    
    print(f"\n📊 Final Results:")
    print(f"   - Chunks processed: {len(all_results)}")
    print(f"   - Deeds after merge: {len(merged_deeds)}")
    print(f"   - Final deeds: {len(final_deeds)}")
    
    # Compare with ground truth - need to determine which PDF we processed
    # Check if we have THOMAS chunk results (278 pages)
    if any('chunk_23_results.json' in f for f in os.listdir('.')):
        pdf_name = "THOMAS"
    elif any('chunk_9_results.json' in f for f in os.listdir('.')):
        pdf_name = "ROBERT" 
    else:
        pdf_name = "FRANCO"
    
    print(f"🔍 Detected PDF: {pdf_name}")
    compare_with_ground_truth(final_deeds, pdf_name)
    
    # Save final results
    final_results = {
        'chunks_processed': len(all_results),
        'total_processing_time': total_time,
        'deeds_before_merge': sum(len(result['deeds']) for result in all_results),
        'deeds_after_merge': len(merged_deeds),
        'final_deeds': len(final_deeds),
        'deeds': final_deeds
    }
    
    with open('smart_chunking_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Final results saved to smart_chunking_final_results.json")

if __name__ == "__main__":
    main()
