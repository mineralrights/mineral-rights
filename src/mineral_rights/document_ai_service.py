"""
Google Cloud Document AI Service for Deed Splitting

This service integrates with Google Cloud Document AI to split multi-deed PDFs
using a custom trained model for deed boundary detection with smart chunking.
"""

import os
import json
import tempfile
import base64
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from google.cloud import documentai
from google.oauth2 import service_account
import fitz  # PyMuPDF
from .smart_chunking_service import SmartChunkingService, SmartChunkingResult, DeedDetectionResult


@dataclass
class DeedSplitResult:
    """Result from Document AI deed splitting"""
    deed_number: int
    pages: List[int]  # Page numbers (0-indexed)
    confidence: float
    bounding_box: Optional[Dict[str, float]] = None
    extracted_text: Optional[str] = None


@dataclass
class DocumentAISplitResult:
    """Complete result from Document AI processing"""
    total_deeds: int
    deeds: List[DeedSplitResult]
    processing_time: float
    raw_response: Optional[Dict] = None


class DocumentAIService:
    """Service for interacting with Google Cloud Document AI"""
    
    def __init__(self, processor_endpoint: str, credentials_path: Optional[str] = None):
        """
        Initialize Document AI service
        
        Args:
            processor_endpoint: The Document AI processor endpoint URL
            credentials_path: Path to Google Cloud service account JSON file
        """
        self.processor_endpoint = processor_endpoint
        self.credentials_path = credentials_path
        
        # Initialize Document AI client
        self._initialize_client()
        
        # Initialize smart chunking service
        self.smart_chunking_service = SmartChunkingService(
            project_id="381937358877",
            location="us",
            processor_id="895767ed7f252878",
            processor_version="106a39290d05efaf",  # The working processor version
            credentials_path=credentials_path
        )
    
    def _initialize_client(self):
        """Initialize the Document AI client with credentials"""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                # Use service account file
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                self.client = documentai.DocumentProcessorServiceClient(credentials=credentials)
                print(f"✅ Document AI client initialized with credentials: {self.credentials_path}")
            else:
                # Use default credentials (e.g., from environment)
                self.client = documentai.DocumentProcessorServiceClient()
                print("✅ Document AI client initialized with default credentials")
                
        except Exception as e:
            print(f"❌ Failed to initialize Document AI client: {e}")
            raise
    
    def split_deeds_with_smart_chunking(self, pdf_path: str) -> DocumentAISplitResult:
        """
        Split deeds from PDF using smart chunking approach
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DocumentAISplitResult with detected deeds
        """
        print(f"🚀 Starting Smart Chunking Deed Detection for {pdf_path}")
        
        # Use smart chunking service
        result = self.smart_chunking_service.process_pdf(pdf_path)
        
        # Convert to DocumentAISplitResult format
        deeds = []
        for detection in result.deed_detections:
            deed = DeedSplitResult(
                deed_number=detection.deed_number,
                pages=detection.pages,
                confidence=detection.confidence,
                bounding_box=None,
                extracted_text=None
            )
            deeds.append(deed)
        
        return DocumentAISplitResult(
            total_deeds=result.total_deeds,
            deeds=deeds,
            processing_time=result.processing_time,
            raw_response={
                'chunks_processed': result.chunks_processed,
                'systematic_offset': result.systematic_offset,
                'raw_deeds_before_merge': result.raw_deeds_before_merge
            }
        )
    
    def split_deeds_from_pdf(self, pdf_path: str, force_single_chunk: bool = False) -> DocumentAISplitResult:
        """
        Split a multi-deed PDF using Document AI custom processor with chunking support
        
        Args:
            pdf_path: Path to the PDF file to process
            force_single_chunk: If True, process entire PDF without chunking (for batch processing)
            
        Returns:
            DocumentAISplitResult with deed boundaries and metadata
        """
        start_time = time.time()
        
        try:
            print(f"🔧 Processing PDF with Document AI: {pdf_path}")
            
            # Check PDF size and determine if chunking is needed
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()
            
            print(f"📄 PDF has {total_pages} pages")
            
            # Use chunking if PDF is too large (30 pages max per chunk) and not forced to single chunk
            if total_pages > 30 and not force_single_chunk:
                print("📦 PDF is large, using chunking approach")
                return self._process_with_chunking(pdf_path, start_time)
            else:
                if force_single_chunk:
                    print("🔄 Force single chunk mode - processing entire PDF")
                else:
                    print("📄 PDF is small enough, processing directly")
                return self._process_single_chunk(pdf_path, start_time)
            
        except Exception as e:
            print(f"❌ Document AI processing failed: {e}")
            raise
    
    def _process_single_chunk(self, pdf_path: str, start_time: float) -> DocumentAISplitResult:
        """Process a single PDF chunk"""
        try:
            # Read the PDF file
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
            
            # Use the correct processor version
            processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
            
            # Create the document
            raw_document = documentai.RawDocument(
                content=pdf_content,
                mime_type="application/pdf"
            )
            
            # Create the request
            request = documentai.ProcessRequest(
                name=processor_version,
                raw_document=raw_document,
                skip_human_review=True
            )
            
            # Process the document
            print("📤 Sending request to Document AI...")
            result = self.client.process_document(request=request)
            document = result.document
            
            # Parse the results
            deeds = self._parse_document_ai_response(document, pdf_path)
            
            processing_time = time.time() - start_time
            
            print(f"✅ Document AI processing completed in {processing_time:.2f}s")
            print(f"📄 Found {len(deeds)} deeds")
            
            return DocumentAISplitResult(
                total_deeds=len(deeds),
                deeds=deeds,
                processing_time=processing_time,
                raw_response={
                    'text': document.text,
                    'pages': len(document.pages),
                    'entities': [{'type': e.type_, 'mention_text': e.mention_text, 'confidence': e.confidence} 
                               for e in document.entities]
                }
            )
            
        except Exception as e:
            print(f"❌ Single chunk processing failed: {e}")
            raise
    
    def _process_with_chunking(self, pdf_path: str, start_time: float) -> DocumentAISplitResult:
        """Process a large PDF by splitting it into chunks"""
        try:
            # Split PDF into chunks
            chunks = self._split_pdf_into_chunks(pdf_path, max_pages=15)
            
            print(f"📦 Created {len(chunks)} chunks")
            
            # Process each chunk
            all_deeds = []
            successful_chunks = 0
            
            for i, chunk_path in enumerate(chunks):
                print(f"\n📄 Processing chunk {i+1}/{len(chunks)}: {chunk_path}")
                
                try:
                    # Read chunk content
                    with open(chunk_path, 'rb') as f:
                        chunk_content = f.read()
                    
                    # Use the correct processor version
                    processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
                    
                    # Create the document
                    raw_document = documentai.RawDocument(
                        content=chunk_content,
                        mime_type="application/pdf"
                    )
                    
                    # Create the request
                    request = documentai.ProcessRequest(
                        name=processor_version,
                        raw_document=raw_document,
                        skip_human_review=True
                    )
                    
                    # Process the document
                    result = self.client.process_document(request=request)
                    document = result.document
                    
                    # Parse the results for this chunk
                    chunk_deeds = self._parse_document_ai_response(document, chunk_path, chunk_offset=i*15)
                    
                    # Adjust page numbers to account for chunk offset
                    for deed in chunk_deeds:
                        deed.pages = [p + (i * 15) for p in deed.pages]
                        deed.deed_number = len(all_deeds) + deed.deed_number
                    
                    all_deeds.extend(chunk_deeds)
                    successful_chunks += 1
                    
                    print(f"✅ Chunk {i+1} processed: {len(chunk_deeds)} deeds found")
                    
                except Exception as e:
                    print(f"❌ Chunk {i+1} failed: {e}")
                    continue
            
            # Clean up chunk files
            self._cleanup_chunks(chunks)
            
            processing_time = time.time() - start_time
            
            print(f"\n📊 FINAL SUMMARY:")
            print(f"✅ Successful chunks: {successful_chunks}/{len(chunks)}")
            print(f"🔍 Total deeds found: {len(all_deeds)}")
            print(f"⏱️ Processing time: {processing_time:.2f}s")
            
            return DocumentAISplitResult(
                total_deeds=len(all_deeds),
                deeds=all_deeds,
                processing_time=processing_time,
                raw_response={
                    'chunks_processed': successful_chunks,
                    'total_chunks': len(chunks),
                    'entities': [{'type': 'DEED', 'confidence': deed.confidence, 'pages': deed.pages} 
                               for deed in all_deeds]
                }
            )
            
        except Exception as e:
            print(f"❌ Chunking processing failed: {e}")
            raise
    
    def _split_pdf_into_chunks(self, pdf_path: str, max_pages: int = 15) -> List[str]:
        """Split a PDF into chunks that fit within the page limit"""
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            print(f"📄 PDF has {total_pages} pages")
            print(f"📏 Max pages per chunk: {max_pages}")
            
            if total_pages <= max_pages:
                print("✅ PDF fits within page limit, no splitting needed")
                return [pdf_path]
            
            # Calculate number of chunks needed
            num_chunks = (total_pages + max_pages - 1) // max_pages
            print(f"📦 Will create {num_chunks} chunks")
            
            chunks = []
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            for i in range(num_chunks):
                start_page = i * max_pages
                end_page = min((i + 1) * max_pages, total_pages)
                
                print(f"📄 Creating chunk {i+1}/{num_chunks}: pages {start_page+1}-{end_page}")
                
                # Create new document for this chunk
                chunk_doc = fitz.open()
                chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
                
                # Save chunk
                chunk_path = f"{base_name}_chunk_{i+1}.pdf"
                chunk_doc.save(chunk_path)
                chunk_doc.close()
                
                chunks.append(chunk_path)
                print(f"✅ Saved: {chunk_path}")
            
            doc.close()
            return chunks
            
        except Exception as e:
            print(f"❌ Error splitting PDF: {e}")
            return []
    
    def _cleanup_chunks(self, chunks: List[str]):
        """Clean up temporary chunk files"""
        print(f"\n🧹 Cleaning up {len(chunks)} chunk files...")
        for chunk in chunks:
            try:
                if os.path.exists(chunk) and chunk != chunks[0]:  # Don't delete original file
                    os.remove(chunk)
                    print(f"✅ Removed: {chunk}")
            except Exception as e:
                print(f"⚠️ Could not remove {chunk}: {e}")
    
    def _parse_document_ai_response(self, document, pdf_path: str, chunk_offset: int = 0) -> List[DeedSplitResult]:
        """
        Parse Document AI response to extract deed boundaries
        
        Args:
            document: Document AI document object
            pdf_path: Original PDF path for reference
            chunk_offset: Page offset for chunked processing
            
        Returns:
            List of DeedSplitResult objects
        """
        deeds = []
        
        try:
            # Get total pages from the document
            total_pages = len(document.pages)
            
            # Look for deed-related entities in the response
            deed_entities = []
            for entity in document.entities:
                if entity.type_ in ['DEED', 'COVER', 'DEED_BOUNDARY', 'DEED_START', 'DEED_END', 'DOCUMENT_BOUNDARY']:
                    page_refs = []
                    if hasattr(entity, 'page_anchor') and entity.page_anchor:
                        page_refs = [page_ref.page for page_ref in entity.page_anchor.page_refs]
                    
                    deed_entities.append({
                        'type': entity.type_,
                        'text': entity.mention_text if hasattr(entity, 'mention_text') else '',
                        'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.0,
                        'page_refs': page_refs
                    })
            
            print(f"🔍 Found {len(deed_entities)} deed-related entities")
            
            if deed_entities:
                # Sort entities by page number
                deed_entities.sort(key=lambda x: x['page_refs'][0] if x['page_refs'] else 0)
                
                # Group pages into deeds based on DEED entities
                current_deed_pages = []
                deed_number = 1
                
                for page_num in range(total_pages):
                    current_deed_pages.append(page_num)
                    
                    # Check if this page has a DEED entity
                    has_deed = any(
                        page_num in entity['page_refs'] 
                        for entity in deed_entities 
                        if entity['type'] == 'DEED' and entity['page_refs']
                    )
                    
                    # If we hit a new DEED entity or this is the last page, create a deed
                    if has_deed or page_num == total_pages - 1:
                        if current_deed_pages:  # Only create deed if we have pages
                            # Calculate confidence based on entity confidences
                            page_confidences = [
                                entity['confidence'] 
                                for entity in deed_entities 
                                if any(p in entity['page_refs'] for p in current_deed_pages)
                            ]
                            avg_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0.8
                            
                            # Adjust page numbers for chunk offset
                            adjusted_pages = [p + chunk_offset for p in current_deed_pages]
                            
                            deeds.append(DeedSplitResult(
                                deed_number=deed_number,
                                pages=adjusted_pages,
                                confidence=avg_confidence
                            ))
                            
                            print(f"📄 Deed {deed_number}: pages {min(adjusted_pages)+1}-{max(adjusted_pages)+1} (confidence: {avg_confidence:.3f})")
                            deed_number += 1
                            current_deed_pages = []
            else:
                # Fallback: if no entities found, treat as single deed
                print("⚠️ No deed boundaries detected, treating as single deed")
                adjusted_pages = [p + chunk_offset for p in range(total_pages)]
                deeds.append(DeedSplitResult(
                    deed_number=1,
                    pages=adjusted_pages,
                    confidence=0.5  # Lower confidence for fallback
                ))
            
            return deeds
            
        except Exception as e:
            print(f"❌ Error parsing Document AI response: {e}")
            # Fallback to single deed
            adjusted_pages = [p + chunk_offset for p in range(total_pages)]
            
            return [DeedSplitResult(
                deed_number=1,
                pages=adjusted_pages,
                confidence=0.3  # Very low confidence for error fallback
            )]
    
    def create_individual_deed_pdfs(self, pdf_path: str, split_result: DocumentAISplitResult) -> List[str]:
        """
        Create individual PDF files for each detected deed
        
        Args:
            pdf_path: Original PDF path
            split_result: Result from Document AI splitting
            
        Returns:
            List of paths to individual deed PDF files
        """
        import tempfile
        
        deed_pdf_paths = []
        
        try:
            # Open the original PDF
            pdf_doc = fitz.open(pdf_path)
            
            for deed in split_result.deeds:
                # Create new PDF for this deed
                deed_doc = fitz.open()
                
                # Add pages for this deed
                for page_num in deed.pages:
                    if 0 <= page_num < len(pdf_doc):
                        deed_doc.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_deed_{deed.deed_number}.pdf")
                deed_doc.save(temp_file.name)
                deed_doc.close()
                
                deed_pdf_paths.append(temp_file.name)
                print(f"💾 Created deed {deed.deed_number} PDF: {temp_file.name}")
            
            pdf_doc.close()
            return deed_pdf_paths
            
        except Exception as e:
            print(f"❌ Error creating individual deed PDFs: {e}")
            raise


class DocumentAIServiceFallback:
    """Fallback service when Document AI is not available"""
    
    def __init__(self):
        print("⚠️ Using Document AI fallback service")
    
    def split_deeds_from_pdf(self, pdf_path: str) -> DocumentAISplitResult:
        """Fallback to simple page-based splitting"""
        start_time = time.time()
        
        # Simple fallback: split every 3 pages
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
        
        deeds = []
        deed_number = 1
        pages_per_deed = 3
        
        for start_page in range(0, total_pages, pages_per_deed):
            end_page = min(start_page + pages_per_deed, total_pages)
            deed_pages = list(range(start_page, end_page))
            
            deeds.append(DeedSplitResult(
                deed_number=deed_number,
                pages=deed_pages,
                confidence=0.4  # Low confidence for fallback
            ))
            
            deed_number += 1
        
        processing_time = time.time() - start_time
        
        return DocumentAISplitResult(
            total_deeds=len(deeds),
            deeds=deeds,
            processing_time=processing_time
        )
    
    def create_individual_deed_pdfs(self, pdf_path: str, split_result: DocumentAISplitResult) -> List[str]:
        """Create individual PDFs using fallback method"""
        import tempfile
        
        deed_pdf_paths = []
        pdf_doc = fitz.open(pdf_path)
        
        for deed in split_result.deeds:
            deed_doc = fitz.open()
            
            for page_num in deed.pages:
                if 0 <= page_num < len(pdf_doc):
                    deed_doc.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_deed_{deed.deed_number}.pdf")
            deed_doc.save(temp_file.name)
            deed_doc.close()
            
            deed_pdf_paths.append(temp_file.name)
        
        pdf_doc.close()
        return deed_pdf_paths


def create_document_ai_service(processor_endpoint: str, credentials_path: Optional[str] = None) -> DocumentAIService:
    """
    Factory function to create Document AI service with fallback
    
    Args:
        processor_endpoint: Document AI processor endpoint
        credentials_path: Optional path to credentials file
        
    Returns:
        DocumentAIService or DocumentAIServiceFallback
    """
    try:
        return DocumentAIService(processor_endpoint, credentials_path)
    except Exception as e:
        print(f"⚠️ Failed to create Document AI service: {e}")
        print("🔄 Falling back to simple page-based splitting")
        return DocumentAIServiceFallback()
