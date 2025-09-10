"""
Deed Tracking Service

This service handles saving and tracking identified deeds from the Document AI splitting step.
It provides functionality to save deed boundaries, metadata, and results for later analysis.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile


@dataclass
class DeedBoundary:
    """Represents a deed boundary detected by Document AI"""
    deed_number: int
    pages: List[int]
    confidence: float
    page_range: str
    detected_at: float
    splitting_strategy: str
    document_ai_used: bool


@dataclass
class DeedClassificationResult:
    """Represents the classification result for a specific deed"""
    deed_number: int
    classification: int  # 0 or 1
    confidence: float
    pages_in_deed: int
    processing_time: float
    error: Optional[str] = None
    deed_boundary_info: Optional[Dict] = None


@dataclass
class MultiDeedProcessingSession:
    """Represents a complete multi-deed processing session"""
    session_id: str
    original_filename: str
    total_pages: int
    splitting_strategy: str
    document_ai_used: bool
    deed_boundaries: List[DeedBoundary]
    classification_results: List[DeedClassificationResult]
    processing_start_time: float
    processing_end_time: Optional[float] = None
    total_processing_time: Optional[float] = None
    summary: Optional[Dict] = None


class DeedTracker:
    """Service for tracking and saving deed processing results"""
    
    def __init__(self, output_dir: str = "deed_tracking"):
        """
        Initialize the deed tracker
        
        Args:
            output_dir: Directory to save tracking data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "sessions").mkdir(exist_ok=True)
        (self.output_dir / "deed_boundaries").mkdir(exist_ok=True)
        (self.output_dir / "classification_results").mkdir(exist_ok=True)
        
        print(f"✅ Deed tracker initialized with output directory: {self.output_dir}")
    
    def create_session(self, original_filename: str, total_pages: int, 
                      splitting_strategy: str, document_ai_used: bool = False) -> str:
        """
        Create a new processing session
        
        Args:
            original_filename: Name of the original PDF file
            total_pages: Total number of pages in the document
            splitting_strategy: Strategy used for splitting
            document_ai_used: Whether Document AI was used
            
        Returns:
            Session ID
        """
        session_id = f"session_{int(time.time())}_{hash(original_filename) % 10000}"
        
        session = MultiDeedProcessingSession(
            session_id=session_id,
            original_filename=original_filename,
            total_pages=total_pages,
            splitting_strategy=splitting_strategy,
            document_ai_used=document_ai_used,
            deed_boundaries=[],
            classification_results=[],
            processing_start_time=time.time()
        )
        
        # Save initial session
        self._save_session(session)
        
        print(f"📊 Created processing session: {session_id}")
        return session_id
    
    def add_deed_boundaries(self, session_id: str, deed_boundaries: List[Dict]) -> None:
        """
        Add deed boundaries to a session
        
        Args:
            session_id: Session ID
            deed_boundaries: List of deed boundary information
        """
        session = self._load_session(session_id)
        if not session:
            print(f"❌ Session {session_id} not found")
            return
        
        # Convert to DeedBoundary objects
        for boundary_data in deed_boundaries:
            boundary = DeedBoundary(
                deed_number=boundary_data['deed_number'],
                pages=boundary_data['pages'],
                confidence=boundary_data['confidence'],
                page_range=boundary_data['page_range'],
                detected_at=time.time(),
                splitting_strategy=session.splitting_strategy,
                document_ai_used=session.document_ai_used
            )
            session.deed_boundaries.append(boundary)
        
        # Save updated session
        self._save_session(session)
        
        # Also save individual boundary files
        self._save_deed_boundaries(session_id, session.deed_boundaries)
        
        print(f"📄 Added {len(deed_boundaries)} deed boundaries to session {session_id}")
    
    def add_classification_results(self, session_id: str, results: List[Dict]) -> None:
        """
        Add classification results to a session
        
        Args:
            session_id: Session ID
            results: List of classification results
        """
        session = self._load_session(session_id)
        if not session:
            print(f"❌ Session {session_id} not found")
            return
        
        # Convert to DeedClassificationResult objects
        for result_data in results:
            result = DeedClassificationResult(
                deed_number=result_data['deed_number'],
                classification=result_data.get('classification', 0),
                confidence=result_data.get('confidence', 0.0),
                pages_in_deed=result_data.get('pages_in_deed', 0),
                processing_time=result_data.get('processing_time', 0.0),
                error=result_data.get('error'),
                deed_boundary_info=result_data.get('deed_boundary_info')
            )
            session.classification_results.append(result)
        
        # Calculate summary
        session.summary = self._calculate_summary(session)
        
        # Save updated session
        self._save_session(session)
        
        # Also save individual result files
        self._save_classification_results(session_id, session.classification_results)
        
        print(f"🎯 Added {len(results)} classification results to session {session_id}")
    
    def finalize_session(self, session_id: str) -> Dict:
        """
        Finalize a processing session and return summary
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary
        """
        session = self._load_session(session_id)
        if not session:
            print(f"❌ Session {session_id} not found")
            return {}
        
        # Update timing
        session.processing_end_time = time.time()
        session.total_processing_time = session.processing_end_time - session.processing_start_time
        
        # Calculate final summary
        session.summary = self._calculate_summary(session)
        
        # Save final session
        self._save_session(session)
        
        # Save summary file
        self._save_session_summary(session)
        
        print(f"✅ Finalized session {session_id} in {session.total_processing_time:.2f}s")
        return session.summary
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get summary for a session"""
        session = self._load_session(session_id)
        return session.summary if session else None
    
    def list_sessions(self) -> List[Dict]:
        """List all processing sessions"""
        sessions_dir = self.output_dir / "sessions"
        sessions = []
        
        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    sessions.append({
                        'session_id': session_data['session_id'],
                        'original_filename': session_data['original_filename'],
                        'created_at': session_data['processing_start_time'],
                        'total_deeds': len(session_data.get('deed_boundaries', [])),
                        'splitting_strategy': session_data['splitting_strategy'],
                        'document_ai_used': session_data['document_ai_used']
                    })
            except Exception as e:
                print(f"⚠️ Error reading session file {session_file}: {e}")
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        return sessions
    
    def _calculate_summary(self, session: MultiDeedProcessingSession) -> Dict:
        """Calculate summary statistics for a session"""
        total_deeds = len(session.deed_boundaries)
        results = session.classification_results
        
        if not results:
            return {
                'total_deeds': total_deeds,
                'deeds_processed': 0,
                'deeds_with_reservations': 0,
                'deeds_without_reservations': 0,
                'deeds_with_errors': 0,
                'average_confidence': 0.0,
                'splitting_confidence': 0.0
            }
        
        deeds_with_reservations = sum(1 for r in results if r.classification == 1)
        deeds_without_reservations = sum(1 for r in results if r.classification == 0)
        deeds_with_errors = sum(1 for r in results if r.error is not None)
        
        # Calculate average confidence
        valid_confidences = [r.confidence for r in results if r.error is None]
        avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
        
        # Calculate average splitting confidence
        splitting_confidences = [b.confidence for b in session.deed_boundaries]
        avg_splitting_confidence = sum(splitting_confidences) / len(splitting_confidences) if splitting_confidences else 0.0
        
        return {
            'total_deeds': total_deeds,
            'deeds_processed': len(results),
            'deeds_with_reservations': deeds_with_reservations,
            'deeds_without_reservations': deeds_without_reservations,
            'deeds_with_errors': deeds_with_errors,
            'average_confidence': avg_confidence,
            'splitting_confidence': avg_splitting_confidence,
            'splitting_strategy': session.splitting_strategy,
            'document_ai_used': session.document_ai_used,
            'processing_time': session.total_processing_time
        }
    
    def _save_session(self, session: MultiDeedProcessingSession) -> None:
        """Save session to file"""
        session_file = self.output_dir / "sessions" / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(asdict(session), f, indent=2)
    
    def _load_session(self, session_id: str) -> Optional[MultiDeedProcessingSession]:
        """Load session from file"""
        session_file = self.output_dir / "sessions" / f"{session_id}.json"
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
                
                # Convert deed boundaries back to DeedBoundary objects
                deed_boundaries = []
                for boundary_data in session_data.get('deed_boundaries', []):
                    if isinstance(boundary_data, dict):
                        boundary = DeedBoundary(**boundary_data)
                    else:
                        boundary = boundary_data  # Already a DeedBoundary object
                    deed_boundaries.append(boundary)
                
                # Convert classification results back to DeedClassificationResult objects
                classification_results = []
                for result_data in session_data.get('classification_results', []):
                    if isinstance(result_data, dict):
                        result = DeedClassificationResult(**result_data)
                    else:
                        result = result_data  # Already a DeedClassificationResult object
                    classification_results.append(result)
                
                # Create session with converted objects
                session = MultiDeedProcessingSession(
                    session_id=session_data['session_id'],
                    original_filename=session_data['original_filename'],
                    total_pages=session_data['total_pages'],
                    splitting_strategy=session_data['splitting_strategy'],
                    document_ai_used=session_data['document_ai_used'],
                    deed_boundaries=deed_boundaries,
                    classification_results=classification_results,
                    processing_start_time=session_data['processing_start_time'],
                    processing_end_time=session_data.get('processing_end_time'),
                    total_processing_time=session_data.get('total_processing_time'),
                    summary=session_data.get('summary')
                )
                
                return session
        except Exception as e:
            print(f"❌ Error loading session {session_id}: {e}")
            return None
    
    def _save_deed_boundaries(self, session_id: str, boundaries: List[DeedBoundary]) -> None:
        """Save deed boundaries to separate file"""
        boundaries_file = self.output_dir / "deed_boundaries" / f"{session_id}_boundaries.json"
        with open(boundaries_file, 'w') as f:
            json.dump([asdict(b) for b in boundaries], f, indent=2)
    
    def _save_classification_results(self, session_id: str, results: List[DeedClassificationResult]) -> None:
        """Save classification results to separate file"""
        results_file = self.output_dir / "classification_results" / f"{session_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    
    def _save_session_summary(self, session: MultiDeedProcessingSession) -> None:
        """Save session summary to separate file"""
        summary_file = self.output_dir / f"{session.session_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(session.summary, f, indent=2)


# Global deed tracker instance
_deed_tracker = None

def get_deed_tracker(output_dir: str = "deed_tracking") -> DeedTracker:
    """Get or create global deed tracker instance"""
    global _deed_tracker
    if _deed_tracker is None:
        _deed_tracker = DeedTracker(output_dir)
    return _deed_tracker
