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


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    clf = OilGasRightsClassifier(api_key=api_key)

    pdf_path = "data/reservs/Washington DB 405_547.pdf"
    result = clf.process_document(
        pdf_path,
        max_samples=8,
        confidence_threshold=0.80,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
