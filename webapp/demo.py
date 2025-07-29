#!/usr/bin/env python3
"""
Mineral Rights Classification Demo
=================================

Interactive demonstration of the document classification pipeline.
Shows the complete process from PDF to final classification with detailed analysis.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from document_classifier import DocumentProcessor

def print_banner(title: str, char: str = "="):
    """Print a formatted banner"""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}")

def print_section(title: str):
    """Print a section header"""
    print(f"\n{'─' * 60}")
    print(f"📋 {title}")
    print(f"{'─' * 60}")

def format_confidence(confidence: float) -> str:
    """Format confidence with color-coded emoji"""
    if confidence >= 0.8:
        return f"🟢 {confidence:.3f} (High)"
    elif confidence >= 0.6:
        return f"🟡 {confidence:.3f} (Medium)"
    else:
        return f"🔴 {confidence:.3f} (Low)"

def format_classification(classification: int, confidence: float) -> str:
    """Format classification result with appropriate emoji"""
    if classification == 1:
        emoji = "⚠️" if confidence >= 0.7 else "❓"
        return f"{emoji} HAS MINERAL RIGHTS RESERVATIONS"
    else:
        emoji = "✅" if confidence >= 0.7 else "❓"
        return f"{emoji} NO MINERAL RIGHTS RESERVATIONS"

def display_chunk_analysis(chunk_analysis: list):
    """Display detailed chunk-by-chunk analysis"""
    if not chunk_analysis:
        print("No chunk analysis available (legacy processing mode)")
        return
    
    print("\n📄 PAGE-BY-PAGE ANALYSIS:")
    print("─" * 50)
    
    for i, chunk in enumerate(chunk_analysis, 1):
        page_num = chunk['page_number']
        classification = chunk['classification']
        confidence = chunk['confidence']
        text_length = chunk['text_length']
        samples_used = chunk['samples_used']
        
        status = "🎯 RESERVATIONS FOUND!" if classification == 1 else "✅ No reservations"
        
        print(f"Page {page_num}:")
        print(f"  Result: {status}")
        print(f"  Confidence: {format_confidence(confidence)}")
        print(f"  Text extracted: {text_length:,} characters")
        print(f"  AI samples: {samples_used}")
        
        if classification == 1:
            print(f"  🛑 Analysis stopped here (reservations found)")
            break
        print()

def display_sample_reasoning(samples: list, max_samples: int = 3):
    """Display reasoning from classification samples"""
    if not samples:
        print("No detailed samples available")
        return
    
    print(f"\n🤖 AI REASONING (showing up to {max_samples} samples):")
    print("─" * 50)
    
    for i, sample in enumerate(samples[:max_samples], 1):
        classification = sample['predicted_class']
        confidence = sample['confidence_score']
        reasoning = sample['reasoning']
        
        result_text = "HAS RESERVATIONS" if classification == 1 else "NO RESERVATIONS"
        
        print(f"Sample {i}: {result_text} ({format_confidence(confidence)})")
        print(f"Reasoning: {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}")
        print()

def save_demo_results(result: dict, output_dir: Path):
    """Save demo results to files"""
    output_dir.mkdir(exist_ok=True)
    
    # Save full result as JSON
    doc_name = Path(result['document_path']).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"demo_result_{doc_name}_{timestamp}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    # Save extracted text
    with open(output_dir / f"demo_text_{doc_name}_{timestamp}.txt", "w") as f:
        f.write(result['ocr_text'])
    
    # Save summary report
    with open(output_dir / f"demo_summary_{doc_name}_{timestamp}.txt", "w") as f:
        f.write(generate_summary_report(result))
    
    return output_dir / f"demo_result_{doc_name}_{timestamp}.json"

def generate_summary_report(result: dict) -> str:
    """Generate a human-readable summary report"""
    doc_path = result['document_path']
    classification = result['classification']
    confidence = result['confidence']
    pages_processed = result['pages_processed']
    samples_used = result['samples_used']
    early_stopped = result['early_stopped']
    strategy = result.get('page_strategy', 'unknown')
    
    report = f"""MINERAL RIGHTS CLASSIFICATION DEMO REPORT
{'=' * 50}

DOCUMENT INFORMATION:
- File: {doc_path}
- Pages processed: {pages_processed}
- Processing strategy: {strategy}
- Text extracted: {result['ocr_text_length']:,} characters

CLASSIFICATION RESULT:
- Final decision: {format_classification(classification, confidence)}
- Confidence score: {confidence:.3f}
- AI samples used: {samples_used}
- Early stopping: {'Yes' if early_stopped else 'No'}

PROCESSING DETAILS:
- Vote distribution: {result['votes']}
- Stopped at chunk: {result.get('stopped_at_chunk', 'N/A')}
- Total pages in document: {result.get('total_pages_in_document', 'Unknown')}

ANALYSIS SUMMARY:
"""
    
    if result.get('chunk_analysis'):
        report += "This document was processed page-by-page with early stopping.\n"
        if classification == 1:
            stopped_page = result.get('stopped_at_chunk')
            report += f"Mineral rights reservations were detected on page {stopped_page}, so analysis stopped there.\n"
        else:
            report += f"All {pages_processed} pages were analyzed and no reservations were found.\n"
    else:
        report += "This document was processed using legacy batch mode.\n"
    
    return report

def run_demo(pdf_path: str, output_dir: str = "demo_results"):
    """Run the complete demo pipeline"""
    
    print_banner("🏛️  MINERAL RIGHTS CLASSIFICATION DEMO")
    
    # Validate input
    if not Path(pdf_path).exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return None
    
    print(f"📁 Document: {pdf_path}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize processor
    print_section("Initializing AI Processor")
    print("🤖 Loading document classifier...")
    print("🧠 Initializing confidence scoring model...")
    print("📊 Setting up self-consistent sampling...")
    
    try:
        processor = DocumentProcessor()
        print("✅ Processor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return None
    
    # Process document
    print_section("Processing Document")
    print("📄 Converting PDF pages to images...")
    print("👁️  Extracting text with Claude OCR...")
    print("🔍 Analyzing for mineral rights reservations...")
    print("🎯 Using chunk-by-chunk early stopping...")
    
    try:
        result = processor.process_document(
            pdf_path,
            max_samples=8,  # Good balance of accuracy and speed for demo
            confidence_threshold=0.7,
            page_strategy="sequential_early_stop"  # Use the smart early stopping
        )
        print("✅ Document processing completed")
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return None
    
    # Display results
    print_banner("📊 CLASSIFICATION RESULTS", "=")
    
    classification = result['classification']
    confidence = result['confidence']
    
    print(f"🎯 FINAL DECISION: {format_classification(classification, confidence)}")
    print(f"📈 Confidence Score: {format_confidence(confidence)}")
    print(f"📄 Pages Processed: {result['pages_processed']} of {result.get('total_pages_in_document', '?')}")
    print(f"🤖 AI Samples Used: {result['samples_used']}")
    print(f"⚡ Early Stopping: {'Yes' if result['early_stopped'] else 'No'}")
    print(f"📊 Vote Distribution: Class 0: {result['votes'].get(0, 0):.2f}, Class 1: {result['votes'].get(1, 0):.2f}")
    
    # Show chunk analysis
    if result.get('chunk_analysis'):
        display_chunk_analysis(result['chunk_analysis'])
    
    # Show AI reasoning
    if result.get('detailed_samples'):
        display_sample_reasoning(result['detailed_samples'])
    
    # Show text sample
    print_section("Extracted Text Sample")
    ocr_text = result['ocr_text']
    print(f"📝 Total text length: {len(ocr_text):,} characters")
    print(f"📄 First 500 characters:")
    print("─" * 50)
    print(ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text)
    
    # Save results
    print_section("Saving Results")
    output_path = Path(output_dir)
    saved_file = save_demo_results(result, output_path)
    
    print(f"💾 Results saved to: {output_path}")
    print(f"📄 Main result file: {saved_file.name}")
    print(f"📝 Text file: demo_text_*.txt")
    print(f"📊 Summary report: demo_summary_*.txt")
    
    print_banner("✅ DEMO COMPLETED SUCCESSFULLY")
    
    return result

def main():
    """Main demo function with command line support"""
    
    if len(sys.argv) < 2:
        print("Usage: python demo.py <pdf_path> [output_dir]")
        print("\nExample documents to try:")
        print("  python demo.py 'data/reservs/Washington DB 475_646 - 4.23.2025.pdf'")
        print("  python demo.py 'data/no-reservs/Butler DB 1895_80 - 4.23.2025.pdf'")
        print("  python demo.py 'data/reservs/Indiana Co. PA DB 550_322.pdf'")
        return
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "demo_results"
    
    result = run_demo(pdf_path, output_dir)
    
    if result:
        print(f"\n🎉 Demo completed! Check the '{output_dir}' folder for detailed results.")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 