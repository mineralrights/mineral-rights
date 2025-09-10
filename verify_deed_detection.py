#!/usr/bin/env python3
"""
Verify Deed Detection Accuracy

This script analyzes the PDF to verify the accuracy of deed detection.
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DeedInfo:
    """Information about a detected deed"""
    deed_number: int
    start_page: int
    end_page: int
    confidence: float
    chunk_id: int
    text_preview: str = ""

def analyze_pdf_structure(pdf_path: str) -> Dict[str, Any]:
    """Analyze the PDF structure to understand deed boundaries"""
    print(f"📄 Analyzing PDF structure: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"📊 PDF Analysis:")
        print(f"   - Total pages: {total_pages}")
        
        # Look for common deed indicators
        deed_indicators = []
        page_texts = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            page_texts.append(text)
            
            # Look for deed-related keywords
            deed_keywords = [
                r'DEED\s+OF\s+TRUST',
                r'WARRANTY\s+DEED',
                r'QUITCLAIM\s+DEED',
                r'MINERAL\s+DEED',
                r'DEED\s+OF\s+CONVEYANCE',
                r'THIS\s+DEED',
                r'KNOW\s+ALL\s+MEN\s+BY\s+THESE\s+PRESENTS',
                r'GRANTOR',
                r'GRANTEE',
                r'CONSIDERATION',
                r'IN\s+WITNESS\s+WHEREOF'
            ]
            
            found_keywords = []
            for keyword in deed_keywords:
                if re.search(keyword, text, re.IGNORECASE):
                    found_keywords.append(keyword)
            
            if found_keywords:
                deed_indicators.append({
                    'page': page_num + 1,  # 1-indexed
                    'keywords': found_keywords,
                    'text_preview': text[:200].replace('\n', ' ').strip()
                })
        
        doc.close()
        
        print(f"🔍 Found {len(deed_indicators)} pages with deed indicators:")
        for indicator in deed_indicators:
            print(f"   - Page {indicator['page']}: {', '.join(indicator['keywords'][:2])}")
            print(f"     Preview: {indicator['text_preview'][:100]}...")
        
        return {
            'total_pages': total_pages,
            'deed_indicators': deed_indicators,
            'page_texts': page_texts
        }
        
    except Exception as e:
        print(f"❌ Error analyzing PDF: {e}")
        return {}

def get_detected_deeds() -> List[DeedInfo]:
    """Get the detected deeds from our Document AI processing"""
    print(f"\n🤖 Getting detected deeds from Document AI...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from mineral_rights.document_ai_service import create_document_ai_service
        from simple_chunking_solution import SimpleChunkingService
        
        # Create services
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        document_ai_service = create_document_ai_service(processor_endpoint)
        simple_chunking_service = SimpleChunkingService(document_ai_service)
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return []
        
        # Process with simple chunking
        deeds = simple_chunking_service.process_pdf_simple(test_pdf_path)
        
        # Convert to DeedInfo objects
        deed_infos = []
        for deed in deeds:
            deed_infos.append(DeedInfo(
                deed_number=deed.deed_number,
                start_page=deed.start_page + 1,  # Convert to 1-indexed
                end_page=deed.end_page + 1,      # Convert to 1-indexed
                confidence=deed.confidence,
                chunk_id=deed.chunk_id
            ))
        
        return deed_infos
        
    except Exception as e:
        print(f"❌ Error getting detected deeds: {e}")
        return []

def compare_results(pdf_analysis: Dict[str, Any], detected_deeds: List[DeedInfo]):
    """Compare PDF analysis with detected deeds"""
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"=" * 60)
    
    if not pdf_analysis or not detected_deeds:
        print("❌ Cannot compare - missing data")
        return
    
    # Basic statistics
    total_pages = pdf_analysis['total_pages']
    deed_indicators = pdf_analysis['deed_indicators']
    detected_count = len(detected_deeds)
    
    print(f"📄 PDF Analysis:")
    print(f"   - Total pages: {total_pages}")
    print(f"   - Pages with deed indicators: {len(deed_indicators)}")
    
    print(f"\n🤖 Document AI Detection:")
    print(f"   - Total deeds detected: {detected_count}")
    print(f"   - Average confidence: {sum(d.confidence for d in detected_deeds) / len(detected_deeds):.3f}")
    
    # Check if detected deeds align with indicators
    print(f"\n🔍 Deed Start Pages (Detected vs Indicators):")
    
    detected_start_pages = [d.start_page for d in detected_deeds]
    indicator_pages = [i['page'] for i in deed_indicators]
    
    print(f"   - Detected start pages: {sorted(detected_start_pages)}")
    print(f"   - Indicator pages: {sorted(indicator_pages)}")
    
    # Find matches
    matches = 0
    for detected_page in detected_start_pages:
        if any(abs(detected_page - indicator_page) <= 1 for indicator_page in indicator_pages):
            matches += 1
    
    print(f"\n📈 Accuracy Analysis:")
    print(f"   - Matches found: {matches}/{len(detected_start_pages)}")
    print(f"   - Match rate: {matches/len(detected_start_pages)*100:.1f}%")
    
    # Detailed comparison
    print(f"\n📋 Detailed Deed List:")
    for i, deed in enumerate(detected_deeds[:10]):  # Show first 10
        print(f"   - Deed {deed.deed_number}: Pages {deed.start_page}-{deed.end_page} (confidence: {deed.confidence:.3f})")
    
    if len(detected_deeds) > 10:
        print(f"   ... and {len(detected_deeds) - 10} more deeds")

def main():
    """Main verification function"""
    print("🔍 Verifying Deed Detection Accuracy")
    print("=" * 50)
    
    # Test PDF
    test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Test PDF not found: {test_pdf_path}")
        return False
    
    # Step 1: Analyze PDF structure
    pdf_analysis = analyze_pdf_structure(test_pdf_path)
    
    # Step 2: Get detected deeds
    detected_deeds = get_detected_deeds()
    
    # Step 3: Compare results
    compare_results(pdf_analysis, detected_deeds)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Verification completed!")
    else:
        print("\n❌ Verification failed")
    
    exit(0 if success else 1)
