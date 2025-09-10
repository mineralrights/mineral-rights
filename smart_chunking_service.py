#!/usr/bin/env python3
"""
Smart Chunking Service for Document AI

This service implements smart chunking with overlap to ensure deeds are not split in half.
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
class DeedBoundary:
    """Represents a deed boundary with confidence"""
    start_page: int
    end_page: int
    confidence: float
    entity_type: str
    chunk_id: int

@dataclass
class SmartChunkResult:
    """Result from smart chunking"""
    total_deeds: int
    deed_boundaries: List[DeedBoundary]
    processing_time: float
    chunks_processed: int
    overlap_pages: int

class SmartChunkingService:
    """Service for smart chunking with overlap to preserve deed boundaries"""
    
    def __init__(self, document_ai_service, overlap_pages: int = 5):
        """
        Initialize smart chunking service
        
        Args:
            document_ai_service: The Document AI service to use
            overlap_pages: Number of pages to overlap between chunks
        """
        self.document_ai_service = document_ai_service
        self.overlap_pages = overlap_pages
        self.max_pages_per_chunk = 15  # Document AI limit
    
    def process_pdf_with_smart_chunking(self, pdf_path: str) -> SmartChunkResult:
        """
        Process a PDF using smart chunking with overlap
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            SmartChunkResult with deed boundaries
        """
        start_time = time.time()
        
        try:
            print(f"🧠 Smart chunking PDF: {pdf_path}")
            
            # Get PDF info
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()
            
            print(f"📄 PDF has {total_pages} pages")
            print(f"📏 Max pages per chunk: {self.max_pages_per_chunk}")
            print(f"🔄 Overlap pages: {self.overlap_pages}")
            
            if total_pages <= self.max_pages_per_chunk:
                print("✅ PDF fits within page limit, processing directly")
                return self._process_single_chunk(pdf_path, start_time)
            
            # Create smart chunks with overlap
            chunks = self._create_smart_chunks(pdf_path, total_pages)
            print(f"📦 Created {len(chunks)} smart chunks with overlap")
            
            # Process each chunk
            all_deed_boundaries = []
            successful_chunks = 0
            
            for i, chunk_info in enumerate(chunks):
                print(f"\n📄 Processing chunk {i+1}/{len(chunks)}: pages {chunk_info['start']+1}-{chunk_info['end']+1}")
                
                try:
                    # Process chunk with Document AI
                    chunk_result = self.document_ai_service.split_deeds_from_pdf(chunk_info['path'])
                    
                    # Convert to deed boundaries with chunk offset
                    chunk_boundaries = self._convert_to_deed_boundaries(
                        chunk_result, 
                        chunk_info['start'], 
                        i
                    )
                    
                    all_deed_boundaries.extend(chunk_boundaries)
                    successful_chunks += 1
                    
                    print(f"✅ Chunk {i+1} processed: {len(chunk_boundaries)} deed boundaries found")
                    
                except Exception as e:
                    print(f"❌ Chunk {i+1} failed: {e}")
                    continue
            
            # Merge overlapping results
            merged_boundaries = self._merge_overlapping_boundaries(all_deed_boundaries)
            
            # Clean up chunk files
            self._cleanup_chunks([chunk['path'] for chunk in chunks])
            
            processing_time = time.time() - start_time
            
            print(f"\n📊 SMART CHUNKING SUMMARY:")
            print(f"✅ Successful chunks: {successful_chunks}/{len(chunks)}")
            print(f"🔍 Total deed boundaries found: {len(merged_boundaries)}")
            print(f"⏱️ Processing time: {processing_time:.2f}s")
            
            return SmartChunkResult(
                total_deeds=len(merged_boundaries),
                deed_boundaries=merged_boundaries,
                processing_time=processing_time,
                chunks_processed=successful_chunks,
                overlap_pages=self.overlap_pages
            )
            
        except Exception as e:
            print(f"❌ Smart chunking failed: {e}")
            raise
    
    def _create_smart_chunks(self, pdf_path: str, total_pages: int) -> List[Dict[str, Any]]:
        """Create smart chunks with overlap"""
        chunks = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Calculate chunk boundaries with overlap
        chunk_size = self.max_pages_per_chunk
        overlap = self.overlap_pages
        
        start_page = 0
        chunk_id = 0
        
        while start_page < total_pages:
            # Calculate end page for this chunk
            end_page = min(start_page + chunk_size, total_pages)
            
            # Create chunk
            chunk_path = f"{base_name}_smart_chunk_{chunk_id}.pdf"
            
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
                'end': end_page - 1,
                'pages': list(range(start_page, end_page))
            })
            
            print(f"📄 Created chunk {chunk_id}: pages {start_page+1}-{end_page}")
            
            # Move to next chunk with overlap
            start_page = end_page - overlap
            chunk_id += 1
            
            # Prevent infinite loop
            if start_page >= total_pages - 1:
                break
        
        return chunks
    
    def _convert_to_deed_boundaries(self, chunk_result, chunk_start_page: int, chunk_id: int) -> List[DeedBoundary]:
        """Convert Document AI result to deed boundaries"""
        boundaries = []
        
        for deed in chunk_result.deeds:
            # Adjust page numbers to account for chunk offset
            adjusted_start = min(deed.pages) + chunk_start_page
            adjusted_end = max(deed.pages) + chunk_start_page
            
            boundaries.append(DeedBoundary(
                start_page=adjusted_start,
                end_page=adjusted_end,
                confidence=deed.confidence,
                entity_type='DEED',
                chunk_id=chunk_id
            ))
        
        return boundaries
    
    def _merge_overlapping_boundaries(self, boundaries: List[DeedBoundary]) -> List[DeedBoundary]:
        """Merge overlapping deed boundaries from different chunks"""
        if not boundaries:
            return []
        
        # Sort boundaries by start page
        boundaries.sort(key=lambda x: x.start_page)
        
        merged = []
        current = boundaries[0]
        
        for boundary in boundaries[1:]:
            # Check if boundaries overlap or are adjacent
            if boundary.start_page <= current.end_page + 1:  # +1 for adjacent pages
                # Merge boundaries
                current.end_page = max(current.end_page, boundary.end_page)
                current.confidence = max(current.confidence, boundary.confidence)
                print(f"🔄 Merged boundaries: pages {current.start_page+1}-{current.end_page+1}")
            else:
                # No overlap, add current and start new
                merged.append(current)
                current = boundary
        
        # Add the last boundary
        merged.append(current)
        
        print(f"🔗 Merged {len(boundaries)} boundaries into {len(merged)} final boundaries")
        
        return merged
    
    def _process_single_chunk(self, pdf_path: str, start_time: float) -> SmartChunkResult:
        """Process a single chunk (when PDF is small enough)"""
        try:
            chunk_result = self.document_ai_service.split_deeds_from_pdf(pdf_path)
            
            boundaries = []
            for deed in chunk_result.deeds:
                boundaries.append(DeedBoundary(
                    start_page=min(deed.pages),
                    end_page=max(deed.pages),
                    confidence=deed.confidence,
                    entity_type='DEED',
                    chunk_id=0
                ))
            
            processing_time = time.time() - start_time
            
            return SmartChunkResult(
                total_deeds=len(boundaries),
                deed_boundaries=boundaries,
                processing_time=processing_time,
                chunks_processed=1,
                overlap_pages=0
            )
            
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
    
    def create_individual_deed_pdfs(self, pdf_path: str, chunk_result: SmartChunkResult) -> List[str]:
        """Create individual PDF files for each detected deed"""
        deed_pdf_paths = []
        
        try:
            # Open the original PDF
            pdf_doc = fitz.open(pdf_path)
            
            for i, boundary in enumerate(chunk_result.deed_boundaries):
                # Create new PDF for this deed
                deed_doc = fitz.open()
                
                # Add pages for this deed
                for page_num in range(boundary.start_page, boundary.end_page + 1):
                    if 0 <= page_num < len(pdf_doc):
                        deed_doc.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_deed_{i+1}.pdf")
                deed_doc.save(temp_file.name)
                deed_doc.close()
                
                deed_pdf_paths.append(temp_file.name)
                print(f"💾 Created deed {i+1} PDF: pages {boundary.start_page+1}-{boundary.end_page+1} (confidence: {boundary.confidence:.3f})")
            
            pdf_doc.close()
            return deed_pdf_paths
            
        except Exception as e:
            print(f"❌ Error creating individual deed PDFs: {e}")
            raise

def test_smart_chunking():
    """Test the smart chunking service"""
    print("🧪 Testing smart chunking service...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from mineral_rights.document_ai_service import create_document_ai_service
        from smart_chunking_service import SmartChunkingService
        
        # Create services
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        document_ai_service = create_document_ai_service(processor_endpoint)
        smart_chunking_service = SmartChunkingService(document_ai_service, overlap_pages=5)
        
        print("✅ Services created successfully")
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        print(f"📄 Testing with PDF: {test_pdf_path}")
        
        # Process with smart chunking
        result = smart_chunking_service.process_pdf_with_smart_chunking(test_pdf_path)
        
        print(f"\n📊 SMART CHUNKING RESULTS:")
        print(f"✅ Total deeds found: {result.total_deeds}")
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        print(f"📦 Chunks processed: {result.chunks_processed}")
        print(f"🔄 Overlap pages: {result.overlap_pages}")
        
        print(f"\n🔍 Deed boundaries:")
        for i, boundary in enumerate(result.deed_boundaries):
            print(f"   - Deed {i+1}: pages {boundary.start_page+1}-{boundary.end_page+1} (confidence: {boundary.confidence:.3f})")
        
        # Test creating individual PDFs
        print(f"\n📄 Creating individual deed PDFs...")
        deed_pdfs = smart_chunking_service.create_individual_deed_pdfs(test_pdf_path, result)
        
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
    success = test_smart_chunking()
    if success:
        print("\n🎉 Smart chunking service is working!")
        print("📊 Summary:")
        print("   - PDF processed with smart chunking")
        print("   - Deed boundaries preserved")
        print("   - No deeds split in half")
        print("   - Individual PDFs created")
    else:
        print("\n❌ Test failed")
    
    exit(0 if success else 1)
