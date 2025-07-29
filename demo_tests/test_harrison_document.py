#!/usr/bin/env python3
"""
Test script to run the updated classification logic on Harrison DV 202-635.pdf
to verify that the new logic correctly detects mineral rights reservations.
"""

import os
from document_classifier import DocumentProcessor

def test_harrison_document():
    """Test the Harrison document with the updated classification logic"""
    
    # Check if API key is available
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key with: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Initialize processor
    print("🔧 Initializing document processor with updated logic...")
    try:
        processor = DocumentProcessor(api_key)
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return
    
    # Path to Harrison document
    harrison_path = "data/reservs/Harrison DV 202-635.pdf"
    
    # Check if file exists
    if not os.path.exists(harrison_path):
        print(f"❌ Error: File not found at {harrison_path}")
        return
    
    print(f"📄 Testing Harrison document: {harrison_path}")
    print("🎯 Using UPDATED logic: General mineral rights include oil/gas unless coal-only")
    print("-" * 80)
    
    # Process the document
    try:
        result = processor.process_document(
            harrison_path,
            max_samples=6,  # Fewer samples for faster testing
            confidence_threshold=0.7,
            page_strategy="first_few",  # Process first few pages
            max_pages=3  # Limit to 3 pages for faster processing
        )
        
        # Print results
        print(f"\n{'='*60}")
        print(f"HARRISON DOCUMENT TEST RESULTS")
        print(f"{'='*60}")
        print(f"Document: {result['document_path']}")
        print(f"Pages processed: {result['pages_processed']}")
        print(f"Classification: {result['classification']} ({'✅ HAS mineral rights reservations' if result['classification'] == 1 else '❌ NO mineral rights reservations'})")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Samples used: {result['samples_used']}")
        print(f"Early stopped: {result['early_stopped']}")
        print(f"Vote distribution: {result['votes']}")
        
        # Show reasoning from samples
        if 'detailed_samples' in result and result['detailed_samples']:
            print(f"\n📋 SAMPLE REASONING:")
            for i, sample in enumerate(result['detailed_samples'][:3], 1):  # Show first 3 samples
                print(f"\nSample {i}:")
                print(f"  Classification: {sample['predicted_class']}")
                print(f"  Confidence: {sample['confidence_score']:.3f}")
                print(f"  Reasoning: {sample['reasoning'][:300]}...")
        
        # Expected vs Actual
        print(f"\n🎯 ANALYSIS:")
        print(f"Expected: 1 (HAS reservations) - Document contains 'coal and mining rights' + 'minerals'")
        print(f"Actual:   {result['classification']} ({'✅ CORRECT' if result['classification'] == 1 else '❌ STILL INCORRECT'})")
        
        if result['classification'] == 1:
            print("🎉 SUCCESS! The updated logic correctly detected mineral rights reservations!")
            print("   The model now understands that general mineral language includes oil/gas")
        else:
            print("⚠️  The document is still being misclassified. This suggests:")
            print("   - The OCR might not be extracting the key text properly")
            print("   - The model might need additional prompt refinement")
            print("   - We should examine the extracted text to see what the model is seeing")
        
        # Show a sample of the extracted text
        if len(result['ocr_text']) > 0:
            print(f"\n📝 SAMPLE OF EXTRACTED TEXT (first 500 chars):")
            print("-" * 40)
            print(result['ocr_text'][:500])
            print("..." if len(result['ocr_text']) > 500 else "")
            print("-" * 40)
            
            # Look for key phrases
            text_lower = result['ocr_text'].lower()
            key_phrases = [
                'excepting and reserving',
                'coal and mining rights',
                'minerals',
                'mineral rights',
                'one-half of the income',
                'any minerals'
            ]
            
            print(f"\n🔍 KEY PHRASES FOUND IN TEXT:")
            for phrase in key_phrases:
                if phrase in text_lower:
                    print(f"   ✅ Found: '{phrase}'")
                else:
                    print(f"   ❌ Missing: '{phrase}'")
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_harrison_document() 