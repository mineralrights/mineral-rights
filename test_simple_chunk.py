#!/usr/bin/env python3
"""
Simple test to create a small PDF with just a few pages
"""
import fitz
import os

def create_small_test_pdf():
    """Create a small test PDF with just 3 pages"""
    
    # Open the original PDF
    doc = fitz.open('data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf')
    
    # Create a new PDF with just the first 3 pages
    small_doc = fitz.open()
    small_doc.insert_pdf(doc, from_page=0, to_page=2)  # Pages 0, 1, 2 (first 3 pages)
    
    # Save the small PDF
    small_pdf_path = 'test_small.pdf'
    small_doc.save(small_pdf_path)
    small_doc.close()
    doc.close()
    
    print(f"✅ Created small test PDF: {small_pdf_path}")
    print(f"📄 Pages: 3")
    
    return small_pdf_path

if __name__ == "__main__":
    create_small_test_pdf()
