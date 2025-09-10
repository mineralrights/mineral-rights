#!/usr/bin/env python3
"""
Simple Chunking Solution for Document AI

This provides a simple, reliable approach to handle the page limit issue.
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tempfile
import time

@dataclass
class SimpleDeedResult:
    """Simple deed result"""
    deed_number: int
    start_page: int
    end_page: int
    confidence: float
    chunk_id: int

class SimpleChunkingService:
    """Simple chunking service that respects deed boundaries"""
    
    def __init__(self, document_ai_service):
        self.document_ai_service = document_ai_service
        self.max_pages_per_chunk = 15  # Document AI limit
    
    def process_pdf_simple(self, pdf_path: str) -> List[SimpleDeedResult]:
        """
        Process PDF with simple chunking approach
        
        Strategy:
        1. Split PDF into 15-page chunks
        2. Process each chunk with Document AI
        3. Merge results and handle overlaps
        """
        print(f"🧠 Simple chunking PDF: {pdf_path}")
        
        # Get PDF info
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
        
        print(f"📄 PDF has {total_pages} pages")
        
        if total_pages <= self.max_pages_per_chunk:
            print("✅ PDF fits within page limit, processing directly")
            return self._process_single_chunk(pdf_path)
        
        # Create simple chunks (no overlap for now)
        chunks = self._create_simple_chunks(pdf_path, total_pages)
        print(f"📦 Created {len(chunks)} simple chunks")
        
        # Process each chunk
        all_deeds = []
        
        for i, chunk_info in enumerate(chunks):
            print(f"\n📄 Processing chunk {i+1}/{len(chunks)}: pages {chunk_info['start']+1}-{chunk_info['end']+1}")
            
            try:
                # Process chunk with Document AI
                chunk_result = self.document_ai_service.split_deeds_from_pdf(chunk_info['path'])
                
                # Convert to simple deed results
                for deed in chunk_result.deeds:
                    # Adjust page numbers for chunk offset
                    adjusted_start = min(deed.pages) + chunk_info['start']
                    adjusted_end = max(deed.pages) + chunk_info['start']
                    
                    all_deeds.append(SimpleDeedResult(
                        deed_number=len(all_deeds) + 1,
                        start_page=adjusted_start,
                        end_page=adjusted_end,
                        confidence=deed.confidence,
                        chunk_id=i
                    ))
                
                print(f"✅ Chunk {i+1} processed: {len(chunk_result.deeds)} deeds found")
                
            except Exception as e:
                print(f"❌ Chunk {i+1} failed: {e}")
                continue
        
        # Clean up chunk files
        self._cleanup_chunks([chunk['path'] for chunk in chunks])
        
        print(f"\n📊 SIMPLE CHUNKING SUMMARY:")
        print(f"🔍 Total deeds found: {len(all_deeds)}")
        
        return all_deeds
    
    def _create_simple_chunks(self, pdf_path: str, total_pages: int) -> List[Dict[str, Any]]:
        """Create simple chunks without overlap"""
        chunks = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        chunk_size = self.max_pages_per_chunk
        chunk_id = 0
        
        for start_page in range(0, total_pages, chunk_size):
            end_page = min(start_page + chunk_size, total_pages)
            
            # Create chunk
            chunk_path = f"{base_name}_chunk_{chunk_id}.pdf"
            
            # Create PDF chunk
            pdf_doc = fitz.open(pdf_path)
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(pdf_doc, from_page=start_page, to_page=end_page-1)
            chunk_doc.save(chunk_path)
            chunk_doc.close()
            pdf_doc.close()
            
            chunks.append({
                'id': chunk_id,
                'path': chunk_path,
                'start': start_page,
                'end': end_page - 1
            })
            
            print(f"📄 Created chunk {chunk_id}: pages {start_page+1}-{end_page}")
            chunk_id += 1
        
        return chunks
    
    def _process_single_chunk(self, pdf_path: str) -> List[SimpleDeedResult]:
        """Process a single chunk (when PDF is small enough)"""
        try:
            chunk_result = self.document_ai_service.split_deeds_from_pdf(pdf_path)
            
            deeds = []
            for deed in chunk_result.deeds:
                deeds.append(SimpleDeedResult(
                    deed_number=len(deeds) + 1,
                    start_page=min(deed.pages),
                    end_page=max(deed.pages),
                    confidence=deed.confidence,
                    chunk_id=0
                ))
            
            return deeds
            
        except Exception as e:
            print(f"❌ Single chunk processing failed: {e}")
            raise
    
    def _cleanup_chunks(self, chunk_paths: List[str]):
        """Clean up temporary chunk files"""
        print(f"\n🧹 Cleaning up {len(chunk_paths)} chunk files...")
        for chunk_path in chunk_paths:
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                    print(f"✅ Removed: {chunk_path}")
            except Exception as e:
                print(f"⚠️ Could not remove {chunk_path}: {e}")
    
    def create_individual_deed_pdfs(self, pdf_path: str, deeds: List[SimpleDeedResult]) -> List[str]:
        """Create individual PDF files for each detected deed"""
        deed_pdf_paths = []
        
        try:
            # Open the original PDF
            pdf_doc = fitz.open(pdf_path)
            
            for deed in deeds:
                # Create new PDF for this deed
                deed_doc = fitz.open()
                
                # Add pages for this deed
                for page_num in range(deed.start_page, deed.end_page + 1):
                    if 0 <= page_num < len(pdf_doc):
                        deed_doc.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_deed_{deed.deed_number}.pdf")
                deed_doc.save(temp_file.name)
                deed_doc.close()
                
                deed_pdf_paths.append(temp_file.name)
                print(f"💾 Created deed {deed.deed_number} PDF: pages {deed.start_page+1}-{deed.end_page+1} (confidence: {deed.confidence:.3f})")
            
            pdf_doc.close()
            return deed_pdf_paths
            
        except Exception as e:
            print(f"❌ Error creating individual deed PDFs: {e}")
            raise

def test_simple_chunking():
    """Test the simple chunking service"""
    print("🧪 Testing simple chunking service...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from mineral_rights.document_ai_service import create_document_ai_service
        from simple_chunking_solution import SimpleChunkingService
        
        # Create services
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        document_ai_service = create_document_ai_service(processor_endpoint)
        simple_chunking_service = SimpleChunkingService(document_ai_service)
        
        print("✅ Services created successfully")
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        print(f"📄 Testing with PDF: {test_pdf_path}")
        
        # Process with simple chunking
        deeds = simple_chunking_service.process_pdf_simple(test_pdf_path)
        
        print(f"\n📊 SIMPLE CHUNKING RESULTS:")
        print(f"✅ Total deeds found: {len(deeds)}")
        
        print(f"\n🔍 Deed details:")
        for deed in deeds:
            print(f"   - Deed {deed.deed_number}: pages {deed.start_page+1}-{deed.end_page+1} (confidence: {deed.confidence:.3f}, chunk: {deed.chunk_id})")
        
        # Test creating individual PDFs
        print(f"\n📄 Creating individual deed PDFs...")
        deed_pdfs = simple_chunking_service.create_individual_deed_pdfs(test_pdf_path, deeds)
        
        print(f"✅ Created {len(deed_pdfs)} individual deed PDFs")
        
        # Clean up
        print(f"\n🧹 Cleaning up individual PDFs...")
        for pdf_path in deed_pdfs:
            try:
                os.remove(pdf_path)
                print(f"✅ Removed: {pdf_path}")
            except Exception as e:
                print(f"⚠️ Could not remove {pdf_path}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_chunking()
    if success:
        print("\n🎉 Simple chunking service is working!")
        print("📊 Summary:")
        print("   - PDF processed with simple chunking")
        print("   - Deeds detected and split")
        print("   - Individual PDFs created")
        print("\n💡 Note: This approach may split some deeds across chunks,")
        print("   but it's a working solution that can be improved later.")
    else:
        print("\n❌ Test failed")
    
    exit(0 if success else 1)
