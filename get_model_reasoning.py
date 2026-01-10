#!/usr/bin/env python3
"""
Script to get the model's reasoning for a specific PDF classification.
This will help us understand why the model is misclassifying documents.
"""

import os
import json
from pathlib import Path
from src.mineral_rights.document_classifier import DocumentProcessor

def get_model_reasoning(pdf_path: str):
    """Process a PDF and extract all model reasoning"""
    
    print("=" * 80)
    print("MODEL REASONING ANALYSIS")
    print("=" * 80)
    print(f"\n📄 Processing: {pdf_path}\n")
    
    # Initialize processor
    print("🔧 Initializing DocumentProcessor...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("\n💡 Please set it with:")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("\n   Or run:")
        print("   ANTHROPIC_API_KEY='your-key-here' python get_model_reasoning.py")
        return
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    processor = DocumentProcessor(api_key=api_key)
    print("✅ Processor initialized\n")
    
    # Process document with detailed output
    print("🔍 Processing document...")
    print("-" * 80)
    
    result = processor.process_document(
        pdf_path,
        max_samples=8,
        confidence_threshold=0.7,
        page_strategy="sequential_early_stop",
        high_recall_mode=True  # Use the same mode as production
    )
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULT")
    print("=" * 80)
    
    classification = result['classification']
    confidence = result['confidence']
    
    print(f"\n🎯 Final Classification: {'HAS RESERVATIONS (1)' if classification == 1 else 'NO RESERVATIONS (0)'}")
    print(f"📊 Confidence: {confidence:.3f}")
    print(f"📈 Votes: {result.get('votes', {})}")
    print(f"🔢 Samples Used: {result.get('samples_used', 0)}")
    
    # Display chunk analysis if available
    if 'chunk_analysis' in result:
        print(f"\n📑 Pages Processed: {len(result['chunk_analysis'])}")
        for i, chunk in enumerate(result['chunk_analysis'], 1):
            print(f"\n  Page {chunk.get('page_number', i)}:")
            print(f"    Classification: {chunk.get('classification', 'N/A')}")
            print(f"    Confidence: {chunk.get('confidence', 0):.3f}")
            print(f"    Samples Used: {chunk.get('samples_used', 0)}")
    
    # Display all samples with full reasoning
    print("\n" + "=" * 80)
    print("DETAILED MODEL REASONING (All Samples)")
    print("=" * 80)
    
    # Get all samples from chunk analysis
    all_samples = []
    if 'chunk_analysis' in result:
        for chunk in result['chunk_analysis']:
            if 'all_samples' in chunk:
                all_samples.extend(chunk['all_samples'])
    
    if not all_samples:
        print("\n⚠️ No detailed samples found in chunk_analysis")
        print("   Trying to extract from result directly...")
        # Try alternative location
        if 'detailed_samples' in result:
            all_samples = result['detailed_samples']
    
    if all_samples:
        print(f"\n📝 Found {len(all_samples)} classification samples\n")
        
        for i, sample in enumerate(all_samples, 1):
            print("-" * 80)
            print(f"SAMPLE {i}/{len(all_samples)}")
            print("-" * 80)
            
            pred_class = sample.get('predicted_class', 'N/A')
            confidence_score = sample.get('confidence_score', 0)
            reasoning = sample.get('reasoning', 'N/A')
            
            print(f"\n🔹 Classification: {pred_class} ({'HAS RESERVATIONS' if pred_class == 1 else 'NO RESERVATIONS'})")
            print(f"🔹 Confidence Score: {confidence_score:.3f}")
            print(f"\n💭 REASONING:")
            print("-" * 80)
            print(reasoning)
            print("-" * 80)
            
            # Also show raw response if available
            if 'raw_response' in sample:
                print(f"\n📋 RAW RESPONSE:")
                print("-" * 80)
                print(sample['raw_response'])
                print("-" * 80)
            
            # Show features if available
            if 'features' in sample:
                print(f"\n📊 FEATURES:")
                for key, value in sample['features'].items():
                    print(f"  {key}: {value:.3f}")
            
            print("\n")
    else:
        print("\n⚠️ No detailed samples found. The result structure may be different.")
        print("\n📋 Full Result Structure:")
        print(json.dumps(result, indent=2, default=str))
    
    # Save to file for reference
    output_file = Path("model_reasoning_output.json")
    with open(output_file, 'w') as f:
        json.dump({
            'pdf_path': pdf_path,
            'classification': classification,
            'confidence': confidence,
            'result': result,
            'all_samples': all_samples
        }, f, indent=2, default=str)
    
    print(f"\n💾 Full output saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "/Users/lauragomez/Downloads/1.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: File not found: {pdf_path}")
        exit(1)
    
    get_model_reasoning(pdf_path)
