#!/usr/bin/env python3
"""
Analyze PDF Content

This script analyzes the actual content of the PDF to understand what's being detected.
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any

def analyze_pdf_content(pdf_path: str, sample_pages: List[int] = [1, 2, 3, 30, 31, 60, 61]):
    """Analyze the content of specific pages"""
    print(f"📄 Analyzing PDF content: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"📊 PDF has {total_pages} pages")
        
        for page_num in sample_pages:
            if page_num <= total_pages:
                page = doc[page_num - 1]  # Convert to 0-indexed
                text = page.get_text()
                
                print(f"\n📄 PAGE {page_num}:")
                print(f"   Text length: {len(text)} characters")
                print(f"   First 300 chars: {text[:300].replace(chr(10), ' ').strip()}")
                
                # Look for specific patterns
                patterns = [
                    r'DEED',
                    r'TRUST',
                    r'CONVEYANCE',
                    r'GRANTOR',
                    r'GRANTEE',
                    r'MINERAL',
                    r'OIL',
                    r'GAS',
                    r'RESERVATION',
                    r'EXCEPT',
                    r'EXCEPTING'
                ]
                
                found_patterns = []
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        found_patterns.append(pattern)
                
                if found_patterns:
                    print(f"   Found patterns: {', '.join(found_patterns)}")
                else:
                    print(f"   No specific patterns found")
        
        doc.close()
        
    except Exception as e:
        print(f"❌ Error analyzing PDF content: {e}")

def main():
    """Main analysis function"""
    print("🔍 Analyzing PDF Content")
    print("=" * 40)
    
    # Test PDF
    test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Test PDF not found: {test_pdf_path}")
        return False
    
    # Analyze content
    analyze_pdf_content(test_pdf_path)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Content analysis completed!")
    else:
        print("\n❌ Content analysis failed")
    
    exit(0 if success else 1)
