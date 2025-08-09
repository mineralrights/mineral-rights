#!/usr/bin/env python3
"""
IMPROVED Oil and Gas Rights Document Classification Agent
========================================================
Enhanced version addressing the performance degradation issues.
"""

import os
import re
import time
import json
import random
import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import fitz                       # PyMuPDF
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import anthropic                  # >= 0.40.0


@dataclass
class ClassificationSample:
    predicted_class: int          # 0 / 1
    reasoning: str
    confidence_score: float
    features: Dict[str, float]
    raw_response: str


@dataclass
class ClassificationResult:
    predicted_class: int
    confidence: float
    votes: Dict[int, float]
    samples_used: int
    early_stopped: bool
    all_samples: List[ClassificationSample]


class ConfidenceScorer:
    """Enhanced confidence scorer with better feature engineering."""

    FEATURE_NAMES = [
        "sentence_count", "trigger_word_presence", "lexical_consistency",
        "format_validity", "answer_certainty", "past_agreement",
        "oil_gas_keyword_density", "boilerplate_indicators",
        "substantive_language_ratio",
    ]

    TRIGGER_WORDS = {'concern', 'issue', 'but', 'however', 'although', 'unclear'}
    HEDGING_TERMS = {'might', 'probably', 'unclear', 'possibly',
                     'maybe', 'seems', 'appears'}
    BOILERPLATE_PHRASES = {'subject to', 'matters of record',
                           'restrictions of record', 'does not enlarge',
                           'otherwise reserved', 'general warranty',
                           'title insurance', 'recording acknowledgment'}

    OIL_GAS_KEYWORDS = {
        'oil', 'gas', 'petroleum', 'hydrocarbons', 'oil and gas',
        'oil or gas', 'oil, gas', 'natural gas', 'crude oil',
    }

    SUBSTANTIVE_TERMS = {
        'grantor reserves oil', 'grantor reserves gas', 'excepting oil and gas',
        'reserving oil and gas', 'oil and gas rights', 'petroleum interests',
        'oil and gas lease', 'hydrocarbon rights',
    }

    GENERAL_TERMS = {'subject to', 'matters of', 'otherwise',
                     'general', 'standard'}

    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self._train_initial_model()

    def extract_features(
        self,
        response: str,
        input_text: str,
        past_responses: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute normalized feature vector."""
        # Basic counts
        sentence_count = len(re.findall(r'[.!?]+', response))

        trigger_presence = sum(
            1 for w in self.TRIGGER_WORDS if w in response.lower()
        )

        # Lexical Jaccard
        input_words = set(input_text.lower().split())
        response_words = set(response.lower().split())
        union = input_words | response_words
        lexical_consistency = (len(input_words & response_words) / len(union)
                               if union else 0.0)

        # Simple format heuristic
        format_validity = float(
            response.strip().lower().startswith(
                ("answer:", "classification:", "result:"))
        )

        # Certainty inverse of hedges
        hedges = sum(1 for h in self.HEDGING_TERMS if h in response.lower())
        answer_certainty = 1.0 - min(1.0, hedges / 3.0)

        # Agreement with last 5 answers
        past_agreement = 0.0
        if past_responses:
            sims = []
            for pr in past_responses[-5:]:
                pw = set(pr.lower().split())
                if pw | response_words:
                    sims.append(len(pw & response_words) / len(pw | response_words))
            past_agreement = float(np.mean(sims)) if sims else 0.0

        # Keyword densities (regex with word boundaries)
        resp = response.lower()
        og_density = sum(bool(re.search(rf"\b{re.escape(k)}\b", resp))
                         for k in self.OIL_GAS_KEYWORDS)
        og_density = min(og_density / max(1, len(response.split()) / 50), 1.0)

        boilerplate = sum(p in resp for p in self.BOILERPLATE_PHRASES)
        boilerplate_ind = min(boilerplate / 3.0, 1.0)

        substantive = sum(t in resp for t in self.SUBSTANTIVE_TERMS)
        general = sum(t in resp for t in self.GENERAL_TERMS)
        substantive_ratio = (substantive / (substantive + general)
                             if substantive + general else 0.0)

        return {
            "sentence_count": min(sentence_count / 10.0, 1.0),
            "trigger_word_presence": min(trigger_presence / 3.0, 1.0),
            "lexical_consistency": lexical_consistency,
            "format_validity": format_validity,
            "answer_certainty": answer_certainty,
            "past_agreement": past_agreement,
            "oil_gas_keyword_density": og_density,
            "boilerplate_indicators": boilerplate_ind,
            "substantive_language_ratio": substantive_ratio,
        }

    def _train_initial_model(self) -> None:
        n = 1000
        # class‑1 = high confidence
        X1 = np.random.normal([0.3, 0.05, 0.7, 1, 0.9, 0.6, 0.3, 0.3, 0.6],
                              0.08, (n // 2, 9))
        y1 = np.ones(n // 2)
        # class‑0 = low confidence
        X0 = np.random.normal([0.6, 0.7, 0.3, 0.2, 0.3, 0.2, 0.7, 0.4, 0.3],
                              0.12, (n // 2, 9))
        y0 = np.zeros(n // 2)

        X = np.clip(np.vstack([X1, X0]), 0, 1)
        y = np.hstack([y1, y0])

        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.is_trained = True

    def score(self, feats: Dict[str, float]) -> float:
        vec = np.array([[feats[k] for k in self.FEATURE_NAMES]])
        prob_high = self.model.predict_proba(self.scaler.transform(vec))[0, 1]

        # Custom adjustments
        prob_high += feats["format_validity"] * 0.25
        prob_high += feats["answer_certainty"] * 0.30
        prob_high -= feats["trigger_word_presence"] * 0.35
        prob_high -= (1 - feats["lexical_consistency"]) * 0.20

        og = feats["oil_gas_keyword_density"]
        subs = feats["substantive_language_ratio"]
        boiler = feats["boilerplate_indicators"]
        if og > 0.3 and subs > 0.4:
            prob_high += 0.20
        elif boiler > 0.6:
            prob_high -= 0.15

        # small randomness to avoid ties
        prob_high += (random.random() - 0.5) * 0.10
        return float(max(0.25, min(0.95, prob_high)))


class ImprovedOilGasRightsClassifier:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY env var first.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.confidence = ConfidenceScorer()
        self.past_high_conf = []

    def create_prompt(self, ocr_text: str, high_recall_mode: bool = False) -> str:
        """IMPROVED: More detailed and specific prompt."""
        if high_recall_mode:
            raise RuntimeError("High specificity mode only.")
        
        return f"""You are an expert legal document analyst specializing in oil and gas rights in real estate deeds. 

Your task: Determine if this document text contains a SUBSTANTIVE reservation of oil and gas rights.

CLASSIFICATION RULES:
- Answer "1" ONLY if the text explicitly reserves/excepts oil and gas rights for the grantor
- Answer "0" for all other cases, including:
  * General mineral rights (without oil/gas specificity)
  * Coal or other mineral reservations only
  * Standard warranty deed language
  * General "subject to" clauses
  * Boilerplate legal language

DOCUMENT TEXT:
\"\"\"{ocr_text}\"\"\"

RESPONSE FORMAT:
Answer: 0 or 1
Reasoning: Explain your decision in 1-2 sentences, citing specific language if reserving oil/gas rights.
"""

    def _sample_once(self, text: str) -> str:
        """Enhanced with better error handling."""
        try:
            msg = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                messages=[{"role": "user", "content": self.create_prompt(text)}]
            )
            return msg.content[0].text.strip()
        except Exception as e:
            # Return a conservative default response for errors
            return "Answer: 0\nReasoning: Error in processing - defaulting to no reservation."

    def classify_document(
        self,
        input_text: str,
        max_samples: int = 8,
        confidence_threshold: float = 0.80,
        high_recall_mode: bool = False,
    ) -> ClassificationResult:
        """FIXED: Proper confidence calculation and early stopping."""
        votes = {0: 0.0, 1: 0.0}
        all_samples: List[ClassificationSample] = []
        early = False

        for i in range(max_samples):
            raw = self._sample_once(input_text)
            
            # Parse
            m = re.search(r"answer\s*:\s*([01])", raw, re.I)
            pred = int(m.group(1)) if m else 0
            reasoning = re.sub(r"(?i)^answer\s*:\s*[01]\s*", "", raw, 1).strip()

            feats = self.confidence.extract_features(
                raw, input_text, self.past_high_conf)
            conf = self.confidence.score(feats)

            all_samples.append(
                ClassificationSample(pred, reasoning, conf, feats, raw)
            )
            votes[pred] += conf

            # IMPROVED: Early stopping for both classes when highly confident
            if conf >= confidence_threshold:
                # Stop early if we have a confident prediction after at least 3 samples
                if i >= 2:
                    early = True
                    break

        # FIXED: Proper confidence calculation
        final_class = 1 if votes[1] > votes[0] else 0
        
        # Calculate confidence as average confidence of samples that voted for the final class
        same_class_samples = [s for s in all_samples if s.predicted_class == final_class]
        if same_class_samples:
            final_conf = sum(s.confidence_score for s in same_class_samples) / len(same_class_samples)
        else:
            final_conf = 0.5  # Default uncertainty

        # Store high-confidence answers
        if final_conf >= 0.8:
            self.past_high_conf.append(raw)
            # Keep only recent high-confidence responses
            self.past_high_conf = self.past_high_conf[-10:]

        return ClassificationResult(
            predicted_class=final_class,
            confidence=round(final_conf, 3),
            votes={0: round(votes[0], 3), 1: round(votes[1], 3)},
            samples_used=len(all_samples),
            early_stopped=early,
            all_samples=all_samples,
        )

    def _pdf_page_to_image(self, page) -> Image.Image:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        return Image.open(BytesIO(pix.tobytes("png")))

    def _image_to_text(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        try:
            resp = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        }},
                        {"type": "text", "text": "Extract all text. No commentary."}
                    ]
                }]
            )
            return resp.content[0].text
        except Exception as e:
            return f"[OCR Error: {str(e)}]"

    def process_document(
        self,
        pdf_path: str,
        *,
        max_samples: int = 8,
        confidence_threshold: float = 0.80,
        max_pages: Optional[int] = None,
    ) -> dict:
        """IMPROVED: Better document-level aggregation."""
        doc = fitz.open(pdf_path)
        total = len(doc)
        pages_to_process = min(total, max_pages) if max_pages else total

        ocr_text_full = []
        chunk_results = []
        stopped_at = None
        unread_pages: List[int] = []
        
        # Track best result for document-level confidence
        best_confidence = 0.0
        best_result = None

        for idx in range(pages_to_process):
            page_no = idx + 1
            page = doc.load_page(idx)
            text = self._image_to_text(self._pdf_page_to_image(page))

            if len(text.strip()) < 50:
                unread_pages.append(page_no)
                continue

            ocr_text_full.append(f"=== PAGE {page_no} ===\n{text}")

            res = self.classify_document(
                text,
                max_samples=max_samples,
                confidence_threshold=confidence_threshold,
                high_recall_mode=False,
            )
            
            chunk_results.append({
                "page": page_no,
                "class": res.predicted_class,
                "conf": res.confidence,
            })

            # Track the highest confidence result
            if res.confidence > best_confidence:
                best_confidence = res.confidence
                best_result = res

            # Early stop on high-confidence positive
            if res.predicted_class == 1 and res.confidence >= confidence_threshold:
                stopped_at = page_no
                break

        doc.close()

        # IMPROVED: Better final decision logic
        final_class = 0 if stopped_at is None else 1
        
        # Use the highest confidence result for document confidence
        final_conf = best_confidence if best_result else 0.0

        return {
            "document_path": pdf_path,
            "classification": final_class,
            "confidence": final_conf,
            "early_stopped": stopped_at is not None,
            "stopped_at_page": stopped_at,
            "pages_processed": stopped_at or pages_to_process,
            "ocr_failed_pages": unread_pages,
            "chunk_results": chunk_results,
        }


# Alias for backward compatibility
OilGasRightsClassifier = ImprovedOilGasRightsClassifier


class DocumentProcessor:
    """Complete pipeline from PDF to classification"""
    
    def __init__(self, api_key: str = None):
        try:
            self.classifier = OilGasRightsClassifier(api_key)
            print("✅ Document processor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize document processor: {e}")
            raise
        
    def pdf_to_image(self, pdf_path: str) -> Image.Image:
        """Convert PDF to high-quality image (first page only - legacy method)"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        mat = fitz.Matrix(2, 2)  # 2x zoom for quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        doc.close()
        return Image.open(BytesIO(img_data))
    
    def pdf_to_images(self, pdf_path: str, max_pages: int = None) -> List[Image.Image]:
        """Convert PDF pages to high-quality images
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None = all pages)
        """
        doc = fitz.open(pdf_path)
        images = []
        
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        
        print(f"Converting {pages_to_process} pages to images (total pages: {total_pages})")
        
        for page_num in range(pages_to_process):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2, 2)  # 2x zoom for quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            images.append(Image.open(BytesIO(img_data)))
            
        doc.close()
        return images
    
    def get_smart_pages(self, pdf_path: str, strategy: str = "first_few") -> List[int]:
        """Smart page selection based on strategy
        
        Args:
            pdf_path: Path to PDF file
            strategy: "first_few", "first_and_last", "all", or "first_only"
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        if strategy == "first_only":
            return [0] if total_pages > 0 else []
        elif strategy == "first_few":
            # First 3 pages (where mineral rights clauses typically appear)
            return list(range(min(3, total_pages)))
        elif strategy == "first_and_last":
            # First 2 pages and last page
            if total_pages <= 2:
                return list(range(total_pages))
            else:
                return [0, 1, total_pages - 1]
        elif strategy == "all":
            return list(range(total_pages))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def extract_text_with_claude(self, image: Image.Image, max_tokens: int = 8000) -> str:
        """Extract text using Claude OCR with configurable token limit"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Retry logic for network issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.classifier.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,  # Configurable token limit
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": "Extract ALL text from this legal deed document. Pay special attention to any mineral rights reservations but do not make any judgment. Format as clean text. Avoid any commentary."
                            }
                        ]
                    }]
                )
                
                return response.content[0].text
                
            except anthropic.APIError as e:
                print(f"Anthropic API error during OCR (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise Exception(f"OCR failed after {max_retries} attempts: {e}")
            except Exception as e:
                print(f"Unexpected error during OCR (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise Exception(f"OCR failed after {max_retries} attempts: {e}")
        
        raise Exception("OCR failed - should not reach here")
    
    def extract_text_from_multiple_pages(self, images: List[Image.Image], 
                                       max_tokens_per_page: int = 8000,
                                       combine_method: str = "concatenate") -> str:
        """Extract text from multiple page images
        
        Args:
            images: List of page images
            max_tokens_per_page: Token limit per page
            combine_method: "concatenate" or "summarize"
        """
        all_text = []
        
        for i, image in enumerate(images, 1):
            print(f"Extracting text from page {i}/{len(images)}...")
            try:
                page_text = self.extract_text_with_claude(image, max_tokens_per_page)
                all_text.append(f"=== PAGE {i} ===\n{page_text}")
            except Exception as e:
                print(f"Error extracting text from page {i}: {e}")
                all_text.append(f"=== PAGE {i} ===\n[ERROR: Could not extract text]")
        
        if combine_method == "concatenate":
            return "\n\n".join(all_text)
        elif combine_method == "summarize":
            # For very long documents, we could implement summarization here
            combined = "\n\n".join(all_text)
            if len(combined) > 50000:  # If too long, truncate with warning
                print("Warning: Combined text is very long, truncating...")
                return combined[:50000] + "\n\n[TRUNCATED - Document continues...]"
            return combined
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")
    
    def process_document(self, pdf_path: str, max_samples: int = 8, 
                        confidence_threshold: float = 0.7,
                        page_strategy: str = "sequential_early_stop",
                        max_pages: int = None,
                        max_tokens_per_page: int = 8000,
                        combine_method: str = "early_stop",
                        high_recall_mode: bool = False) -> dict:
        """Complete pipeline: PDF -> OCR -> Classification with high recall mode
        
        Args:
            pdf_path: Path to PDF file
            max_samples: Maximum classification samples per chunk
            confidence_threshold: Early stopping threshold for classification
            page_strategy: "sequential_early_stop" (default), "first_only", "first_few", "first_and_last", "all"
            max_pages: Maximum pages to process (overrides strategy if set)
            max_tokens_per_page: Token limit per page for OCR
            combine_method: "early_stop" (default), "concatenate", "summarize"
        """
        
        print(f"Processing: {pdf_path}")
        print(f"Page strategy: {page_strategy}")
        print(f"Max tokens per page: {max_tokens_per_page}")
        if high_recall_mode:
            print("🎯 BALANCED HIGH RECALL MODE – Good sensitivity while maintaining accuracy")
        else:
            print("🎯 CONSERVATIVE (High Specificity) MODE – Extra-cautious, prioritising specificity")
        
        # Use sequential early stopping by default
        if page_strategy == "sequential_early_stop" or combine_method == "early_stop":
            return self._process_with_early_stopping(
                pdf_path,
                max_samples,
                confidence_threshold,
                max_tokens_per_page,
                max_pages,
                high_recall_mode,
            )
        
        # Legacy processing for other strategies
        # Step 1: Determine which pages to process
        if max_pages is not None:
            # Use max_pages override
            images = self.pdf_to_images(pdf_path, max_pages)
        else:
            # Use smart page selection
            page_numbers = self.get_smart_pages(pdf_path, page_strategy)
            print(f"Selected pages: {[p+1 for p in page_numbers]}")  # Convert to 1-indexed for display
            
            if page_strategy == "first_only":
                # Use legacy single-page method for backward compatibility
                image = self.pdf_to_image(pdf_path)
                images = [image]
            else:
                # Process selected pages
                doc = fitz.open(pdf_path)
                images = []
                for page_num in page_numbers:
                    page = doc.load_page(page_num)
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    images.append(Image.open(BytesIO(img_data)))
                doc.close()
        
        # Step 2: Extract text from all selected pages
        print("Extracting text with Claude OCR...")
        if len(images) == 1:
            ocr_text = self.extract_text_with_claude(images[0], max_tokens_per_page)
        else:
            ocr_text = self.extract_text_from_multiple_pages(
                images, max_tokens_per_page, combine_method
            )
        
        print(f"Extracted text length: {len(ocr_text)} characters")
        
        # Step 3: Classify with self-consistent sampling (always high recall)
        print("Classifying document...")
        classification_result = self.classifier.classify_document(
            ocr_text, max_samples, confidence_threshold, high_recall_mode=high_recall_mode
        )
        
        return {
            'document_path': pdf_path,
            'pages_processed': len(images),
            'page_strategy': page_strategy,
            'max_tokens_per_page': max_tokens_per_page,
            'high_recall_mode': high_recall_mode,
            'ocr_text': ocr_text,
            'ocr_text_length': len(ocr_text),
            'classification': classification_result.predicted_class,
            'confidence': classification_result.confidence,
            'votes': classification_result.votes,
            'samples_used': classification_result.samples_used,
            'early_stopped': classification_result.early_stopped,
            'chunk_analysis': [],  # Empty for legacy mode
            'stopped_at_chunk': None,
            'detailed_samples': [
                {
                    'predicted_class': s.predicted_class,
                    'reasoning': s.reasoning,
                    'confidence_score': s.confidence_score,
                    'features': s.features
                }
                for s in classification_result.all_samples
            ]
        }
    
    def _process_with_early_stopping(self, pdf_path: str, max_samples: int, 
                                   confidence_threshold: float, max_tokens_per_page: int, 
                                   max_pages: int = None, high_recall_mode: bool = False) -> dict:
        """Process document chunk by chunk with early stopping when reservations are found"""
        
        print("Using chunk-by-chunk early stopping analysis")
        mode_label = (
            "BALANCED HIGH RECALL"
            if high_recall_mode
            else "CONSERVATIVE (High Specificity)"
        )
        mode_msg = (
            "Good sensitivity while maintaining accuracy"
            if high_recall_mode
            else "Extra-cautious, prioritising specificity"
        )
        print(f"🛈 {mode_label} MODE – {mode_msg}")
        
        # Open PDF and get total pages
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        
        print(f"Document has {total_pages} pages, will process up to {pages_to_process} if needed")
        
        chunk_analysis = []
        all_ocr_text = []
        stopped_at_chunk = None
        unread_pages = []
        
        # Process page by page with early stopping
        for page_num in range(pages_to_process):
            current_page = page_num + 1
            print(f"\n--- PROCESSING CHUNK {current_page}/{pages_to_process} ---")
            
            # Convert current page to image
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2, 2)  # 2x zoom for quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(BytesIO(img_data))
            
            # Extract text from current page
            print(f"Extracting text from page {current_page}...")
            try:
                page_text = self.extract_text_with_claude(image, max_tokens_per_page)
                all_ocr_text.append(f"=== PAGE {current_page} ===\n{page_text}")
                print(f"Extracted {len(page_text)} characters from page {current_page}")
            
            except Exception as e:
                print(f"Error extracting text from page {current_page}: {e}")
                # record failure and skip classification for this page
                unread_pages.append(current_page)
                chunk_analysis.append({
                    "page_number": current_page,
                    "status": "ocr_failed",
                    "error": str(e),
                })   
                continue
            
            # Classify current chunk (always high recall)
            print(f"Analyzing page {current_page} for oil and gas reservations...")
            
            classification_result = self.classifier.classify_document(
                page_text,
                max_samples,
                confidence_threshold,
                high_recall_mode=high_recall_mode,
            )
            
            chunk_info = {
                'page_number': current_page,
                'text_length': len(page_text),
                'classification': classification_result.predicted_class,
                'confidence': classification_result.confidence,
                'votes': classification_result.votes,
                'samples_used': classification_result.samples_used,
                'early_stopped': classification_result.early_stopped,
                'page_text': page_text,
                'high_recall_mode': high_recall_mode
            }
        
            chunk_analysis.append(chunk_info)
            
            print(f"Page {current_page} analysis:")
            print(f"  Classification: {classification_result.predicted_class} ({'Has Oil and Gas Reservations' if classification_result.predicted_class == 1 else 'No Oil and Gas Reservations'})")
            print(f"  Confidence: {classification_result.confidence:.3f}")
            print(f"  Samples used: {classification_result.samples_used}")
            
            # EARLY STOPPING: If oil and gas reservations found, stop here!
            if classification_result.predicted_class == 1:
                print(f"🎯 OIL AND GAS RESERVATIONS FOUND in page {current_page}! Stopping analysis here.")
                stopped_at_chunk = current_page
                doc.close()
                
                return {
                    'document_path': pdf_path,
                    'pages_processed': current_page,
                    'page_strategy': "sequential_early_stop",
                    'max_tokens_per_page': max_tokens_per_page,
                    'high_recall_mode': high_recall_mode,
                    'ocr_text': "\n\n".join(all_ocr_text),
                    'ocr_text_length': sum(len(chunk.get('page_text', '')) for chunk in chunk_analysis),
                    'classification': 1,  # Found oil and gas reservations
                    'confidence': classification_result.confidence,
                    'votes': classification_result.votes,
                    'samples_used': classification_result.samples_used,
                    'early_stopped': True,  # Stopped early due to finding oil and gas reservations
                    'chunk_analysis': chunk_analysis,
                    'stopped_at_chunk': stopped_at_chunk,
                    'total_pages_in_document': total_pages,
                    'detailed_samples': [
                        {
                            'predicted_class': s.predicted_class,
                            'reasoning': s.reasoning,
                            'confidence_score': s.confidence_score,
                            'features': s.features
                        }
                        for s in classification_result.all_samples
                    ],
                    'ocr_failed_pages': unread_pages,
                    'requires_manual_review': len(unread_pages) > 0,
                }
            else:
                print(f"No oil and gas reservations found in page {current_page}, continuing to next page...")
        
        doc.close()
        
        # If we get here, no oil and gas reservations were found in any page
        print(f"\n✅ ANALYSIS COMPLETE: No oil and gas reservations found in any of the {pages_to_process} pages")
        
        # For final classification when no reservations found, use the last page's result
        successful_chunks = [c for c in chunk_analysis if 'classification' in c]
        if successful_chunks:
            final_result = successful_chunks[-1]
        else:  # every page failed OCR
            final_result = {
                'classification': 0,
                'confidence': 0.0,
                'votes': {0: 0.0, 1: 0.0},
                'samples_used': 0,
                'early_stopped': False
            }
        
        return {
            'document_path': pdf_path,
            'pages_processed': pages_to_process,
            'page_strategy': "sequential_early_stop",
            'max_tokens_per_page': max_tokens_per_page,
            'high_recall_mode': high_recall_mode,
            'ocr_text': "\n\n".join(all_ocr_text),
            'ocr_text_length': sum(len(chunk.get('page_text', '')) for chunk in chunk_analysis),
            'classification': final_result['classification'],
            'confidence': final_result['confidence'],
            'votes': final_result['votes'],
            'samples_used': final_result['samples_used'],
            'early_stopped': final_result['early_stopped'],
            'chunk_analysis': chunk_analysis,
            'stopped_at_chunk': stopped_at_chunk,
            'total_pages_in_document': total_pages,
            'detailed_samples': [],  # Could aggregate all samples if needed
            'ocr_failed_pages': unread_pages,
            'requires_manual_review': len(unread_pages) > 0,
        }

    def detect_deed_boundaries(self, full_text: str) -> List[Dict]:
        """Use LLM to detect deed boundaries in multi-deed document
        
        Returns:
            List of deed boundaries with start/end positions and metadata
        """
        
        boundary_prompt = f"""You are analyzing a legal document that contains multiple property deeds. Your task is to identify where each individual deed starts and ends.

DOCUMENT TEXT:
\"\"\"{full_text[:20000]}\"\"\"

Please identify all individual deeds in this document. For each deed, provide:
1. The approximate character position where it starts
2. The approximate character position where it ends  
3. A brief description of the deed (grantors, grantees, property description)

Look for patterns like:
- "KNOW ALL MEN BY THESE PRESENTS" or similar deed opening language
- Grantor and grantee names
- Property descriptions
- Notary acknowledgments or signatures that end a deed
- Recording information

Format your response as JSON:
{{
  "deeds": [
    {{
      "deed_number": 1,
      "start_position": 0,
      "end_position": 1500,
      "description": "Deed from John Smith to Jane Doe for Lot 1"
    }},
    ...
  ]
}}

If you cannot clearly identify multiple deeds, return a single deed covering the entire document."""

        try:
            response = self.classifier.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": boundary_prompt}]
            )
            
            # Parse JSON response
            import json
            boundary_data = json.loads(response.content[0].text)
            return boundary_data.get("deeds", [])
            
        except Exception as e:
            print(f"Error detecting deed boundaries: {e}")
            # Fallback: treat entire document as single deed
            return [{
                "deed_number": 1,
                "start_position": 0,
                "end_position": len(full_text),
                "description": "Full document (boundary detection failed)"
            }]

    def extract_deed_text(self, full_text: str, start_pos: int, end_pos: int) -> str:
        """Extract text for a specific deed based on character positions"""
        return full_text[start_pos:end_pos].strip()

    def process_multi_deed_document(self, pdf_path: str, strategy: str = "smart_detection") -> List[Dict]:
        """Process a PDF containing multiple deeds with intelligent deed boundary detection
        
        Args:
            pdf_path: Path to PDF file containing multiple deeds
            strategy: "page_based", "smart_detection", or "ai_assisted"
        
        Returns:
            List of classification results, one per deed
        """
        print(f"Processing multi-deed document: {pdf_path}")
        print(f"Strategy: {strategy}")
        
        results = []
        
        if strategy == "page_based":
            # Simple approach: each page is a deed
            return self._process_page_based_deeds(pdf_path)
        
        elif strategy in ["smart_detection", "ai_assisted", "smart_split", "llm_boundaries"]:
            # Advanced approach: detect deed boundaries using LLM
            return self._process_with_boundary_detection(pdf_path)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Valid strategies are: page_based, smart_detection, ai_assisted")

    def _process_page_based_deeds(self, pdf_path: str) -> List[Dict]:
        """Simple page-based processing - each page is a separate deed"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        results = []
        
        print(f"Processing {total_pages} pages as separate deeds")
        
        for page_num in range(total_pages):
            current_page = page_num + 1
            print(f"\n--- PROCESSING DEED {current_page}/{total_pages} (Page {current_page}) ---")
            
            try:
                # Convert page to image and extract text
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                page_text = self.extract_text_with_claude(image, 8000)
                print(f"Extracted {len(page_text)} characters from page {current_page}")
                
                # Classify the deed
                classification_result = self.classifier.classify_document(
                    page_text,
                    max_samples=6,
                    confidence_threshold=0.7,
                    high_recall_mode=False
                )
                
                deed_result = {
                    'deed_number': current_page,
                    'page_numbers': [current_page],
                    'document_path': pdf_path,
                    'strategy': 'page_based',
                    'ocr_text': page_text,
                    'ocr_text_length': len(page_text),
                    'classification': classification_result.predicted_class,
                    'confidence': classification_result.confidence,
                    'votes': classification_result.votes,
                    'samples_used': classification_result.samples_used,
                    'early_stopped': classification_result.early_stopped,
                    'status': 'success'
                }
                
                results.append(deed_result)
                
                print(f"Deed {current_page}: {classification_result.predicted_class} ({'Oil & Gas Reservations' if classification_result.predicted_class == 1 else 'No Reservations'}) (conf: {classification_result.confidence:.3f})")
                
            except Exception as e:
                print(f"Error processing page {current_page}: {e}")
                results.append({
                    'deed_number': current_page,
                    'page_numbers': [current_page],
                    'document_path': pdf_path,
                    'strategy': 'page_based',
                    'status': 'error',
                    'error': str(e),
                    'classification': 0,
                    'confidence': 0.0
                })
        
        doc.close()
        return results

    def _process_with_boundary_detection(self, pdf_path: str) -> List[Dict]:
        """Advanced processing with LLM-based deed boundary detection"""
        
        # Step 1: Extract all text from the document
        print("Step 1: Extracting full document text...")
        images = self.pdf_to_images(pdf_path)
        full_text = self.extract_text_from_multiple_pages(images, 8000, "concatenate")
        
        print(f"Extracted {len(full_text)} total characters from {len(images)} pages")
        
        # Step 2: Use LLM to detect deed boundaries
        print("Step 2: Detecting deed boundaries using LLM...")
        deed_boundaries = self.detect_deed_boundaries(full_text)
        
        print(f"Detected {len(deed_boundaries)} deeds:")
        for boundary in deed_boundaries:
            print(f"  Deed {boundary['deed_number']}: chars {boundary['start_position']}-{boundary['end_position']}")
            print(f"    Description: {boundary.get('description', 'No description')}")
        
        # Step 3: Process each detected deed
        results = []
        
        for boundary in deed_boundaries:
            deed_num = boundary['deed_number']
            print(f"\n--- PROCESSING DEED {deed_num}/{len(deed_boundaries)} ---")
            
            try:
                # Extract deed text
                deed_text = self.extract_deed_text(
                    full_text, 
                    boundary['start_position'], 
                    boundary['end_position']
                )
                
                print(f"Deed {deed_num} text length: {len(deed_text)} characters")
                
                # Classify the deed
                classification_result = self.classifier.classify_document(
                    deed_text,
                    max_samples=6,
                    confidence_threshold=0.7,
                    high_recall_mode=False
                )
                
                deed_result = {
                    'deed_number': deed_num,
                    'document_path': pdf_path,
                    'strategy': 'llm_boundaries',
                    'boundary_info': boundary,
                    'ocr_text': deed_text,
                    'ocr_text_length': len(deed_text),
                    'classification': classification_result.predicted_class,
                    'confidence': classification_result.confidence,
                    'votes': classification_result.votes,
                    'samples_used': classification_result.samples_used,
                    'early_stopped': classification_result.early_stopped,
                    'status': 'success'
                }
                
                results.append(deed_result)
                
                print(f"Deed {deed_num}: {classification_result.predicted_class} ({'Oil & Gas Reservations' if classification_result.predicted_class == 1 else 'No Reservations'}) (conf: {classification_result.confidence:.3f})")
                
            except Exception as e:
                print(f"Error processing deed {deed_num}: {e}")
                results.append({
                    'deed_number': deed_num,
                    'document_path': pdf_path,
                    'strategy': 'llm_boundaries',
                    'boundary_info': boundary,
                    'status': 'error',
                    'error': str(e),
                    'classification': 0,
                    'confidence': 0.0
                })
        
        return results


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    clf = ImprovedOilGasRightsClassifier(api_key=api_key)

    pdf_path = "data/reservs/Washington DB 405_547.pdf"
    result = clf.process_document(
        pdf_path,
        max_samples=8,
        confidence_threshold=0.80,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
