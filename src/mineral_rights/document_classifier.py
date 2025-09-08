#!/usr/bin/env python3
"""
Oil and Gas Rights Document Classification Agent
===============================================

Self-consistent sampling with confidence scoring for binary classification.
Specifically detects oil and gas reservations (not coal or other minerals).
"""

import os
import json
import re
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import anthropic
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import base64
import random

# Remove hardcoded API key - use environment variable only
# os.environ['ANTHROPIC_API_KEY'] = "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"

@dataclass
class ClassificationSample:
    """Single classification attempt with metadata"""
    predicted_class: int
    reasoning: str
    confidence_score: float
    features: Dict[str, float]
    raw_response: str

@dataclass
class ClassificationResult:
    """Final classification result with metadata"""
    predicted_class: int
    confidence: float
    votes: Dict[int, float]
    samples_used: int
    early_stopped: bool
    all_samples: List[ClassificationSample]

class ConfidenceScorer:
    """Lightweight confidence scoring using logistic regression"""
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'sentence_count',
            'trigger_word_presence', 
            'lexical_consistency',
            'format_validity',
            'answer_certainty',
            'past_agreement',
            'oil_gas_keyword_density',
            'boilerplate_indicators',
            'substantive_language_ratio'
        ]
        
    def extract_features(self, response: str, input_text: str, 
                        past_responses: List[str] = None) -> Dict[str, float]:
        """Extract confidence features from a response"""
        
        # Sentence count
        sentence_count = len(re.findall(r'[.!?]+', response))
        
        # Trigger word presence (uncertainty indicators)
        trigger_words = ['concern', 'issue', 'but', 'however', 'although', 'unclear']
        trigger_presence = sum(1 for word in trigger_words if word.lower() in response.lower())
        
        # Lexical consistency (Jaccard similarity)
        input_words = set(input_text.lower().split())
        response_words = set(response.lower().split())
        if len(input_words.union(response_words)) > 0:
            lexical_consistency = len(input_words.intersection(response_words)) / len(input_words.union(response_words))
        else:
            lexical_consistency = 0.0
            
        # Format validity (structured response)
        format_validity = 1.0 if response.strip().startswith(('Answer:', 'Classification:', 'Result:')) else 0.0
        
        # Answer certainty (hedging terms)
        hedging_terms = ['might', 'probably', 'unclear', 'possibly', 'maybe', 'seems', 'appears']
        answer_certainty = 1.0 - min(1.0, sum(1 for term in hedging_terms if term.lower() in response.lower()) / 3.0)
        
        # Past agreement (similarity to previous high-confidence responses)
        past_agreement = 0.0
        if past_responses:
            similarities = []
            for past_resp in past_responses[-5:]:  # Last 5 responses
                past_words = set(past_resp.lower().split())
                if len(response_words.union(past_words)) > 0:
                    sim = len(response_words.intersection(past_words)) / len(response_words.union(past_words))
                    similarities.append(sim)
            past_agreement = np.mean(similarities) if similarities else 0.0
        
        # NEW FEATURES FOR BETTER NO-RESERVATION DETECTION
        
        # Oil and gas keyword density (normalized) - UPDATED FOR OIL/GAS FOCUS
        oil_gas_keywords = ['oil', 'gas', 'petroleum', 'hydrocarbons', 'oil and gas', 
                           'oil or gas', 'oil, gas', 'natural gas', 'crude oil']
        response_lower = response.lower()
        oil_gas_count = sum(1 for keyword in oil_gas_keywords if keyword in response_lower)
        oil_gas_keyword_density = min(oil_gas_count / max(1, len(response.split()) / 50), 1.0)
        
        # Boilerplate indicators (high score = likely boilerplate language)
        boilerplate_phrases = ['subject to', 'matters of record', 'restrictions of record',
                              'does not enlarge', 'otherwise reserved', 'general warranty',
                              'title insurance', 'recording acknowledgment']
        boilerplate_count = sum(1 for phrase in boilerplate_phrases if phrase in response_lower)
        boilerplate_indicators = min(boilerplate_count / 3.0, 1.0)
        
        # Substantive language ratio (specific vs general language) - UPDATED FOR OIL/GAS
        substantive_terms = ['grantor reserves oil', 'grantor reserves gas', 'excepting oil and gas', 
                           'reserving oil and gas', 'oil and gas rights', 'petroleum interests',
                           'oil and gas lease', 'hydrocarbon rights']
        substantive_count = sum(1 for term in substantive_terms if term in response_lower)
        general_terms = ['subject to', 'matters of', 'otherwise', 'general', 'standard']
        general_count = sum(1 for term in general_terms if term in response_lower)
        
        if substantive_count + general_count > 0:
            substantive_language_ratio = substantive_count / (substantive_count + general_count)
        else:
            substantive_language_ratio = 0.0
        
        return {
            'sentence_count': min(sentence_count / 10.0, 1.0),  # Normalize
            'trigger_word_presence': min(trigger_presence / 3.0, 1.0),
            'lexical_consistency': lexical_consistency,
            'format_validity': format_validity,
            'answer_certainty': answer_certainty,
            'past_agreement': past_agreement,
            'oil_gas_keyword_density': oil_gas_keyword_density,
            'boilerplate_indicators': boilerplate_indicators,
            'substantive_language_ratio': substantive_language_ratio
        }
    
    def train_initial_model(self):
        """Train with synthetic data for bootstrap"""
        # Generate synthetic training data for initial model
        np.random.seed(42)
        n_samples = 1000
        
        # IMPROVED: High confidence samples (clear, decisive classifications)
        # Features: good format, low trigger words, high certainty, good lexical consistency
        # [sentence_count, trigger_presence, lexical_consistency, format_validity, answer_certainty, 
        #  past_agreement, oil_gas_density, boilerplate_indicators, substantive_ratio]
        X_high_confidence = np.random.normal([0.4, 0.05, 0.7, 1.0, 0.95, 0.6, 0.3, 0.4, 0.6], 0.08, (n_samples//2, 9))
        y_high_confidence = np.ones(n_samples//2)  # Label 1 = high confidence
        
        # IMPROVED: Low confidence samples (uncertain, unclear classifications)
        # Features: poor format, high trigger words, low certainty, poor consistency
        X_low_confidence = np.random.normal([0.6, 0.7, 0.3, 0.2, 0.3, 0.2, 0.7, 0.3, 0.3], 0.12, (n_samples//2, 9))
        y_low_confidence = np.zeros(n_samples//2)  # Label 0 = low confidence
        
        X = np.vstack([X_high_confidence, X_low_confidence])
        y = np.hstack([y_high_confidence, y_low_confidence])
        
        # Clip to valid ranges
        X = np.clip(X, 0, 1)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
    def score_confidence(self, features: Dict[str, float]) -> float:
        """Score confidence for a response with enhanced variability"""
        if not self.is_trained:
            self.train_initial_model()
            
        feature_vector = np.array([[features[name] for name in self.feature_names]])
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get base probability of high confidence (class 1)
        base_confidence = self.model.predict_proba(feature_vector_scaled)[0][1]
        
        # ENHANCED: Create more realistic confidence variation based on key indicators
        
        # Strong indicators of high confidence
        format_bonus = features.get('format_validity', 0.0) * 0.25  # Well-formatted response
        certainty_bonus = features.get('answer_certainty', 0.0) * 0.30  # No hedging language
        
        # Strong indicators of low confidence  
        uncertainty_penalty = features.get('trigger_word_presence', 0.0) * 0.35  # Uncertainty words
        poor_consistency_penalty = (1.0 - features.get('lexical_consistency', 0.5)) * 0.20
        
        # Content-specific adjustments
        oil_gas_density = features.get('oil_gas_keyword_density', 0.0)
        boilerplate_ratio = features.get('boilerplate_indicators', 0.0)
        substantive_ratio = features.get('substantive_language_ratio', 0.0)
        
        # If response discusses specific oil/gas terms substantively, higher confidence
        content_confidence_bonus = 0.0
        if oil_gas_density > 0.3 and substantive_ratio > 0.4:
            content_confidence_bonus = 0.20
        elif boilerplate_ratio > 0.6:  # Mostly boilerplate = lower confidence
            content_confidence_bonus = -0.15
        
        # Calculate final confidence with realistic bounds
        final_confidence = (
            base_confidence + 
            format_bonus + 
            certainty_bonus + 
            content_confidence_bonus -
            uncertainty_penalty - 
            poor_consistency_penalty
        )
        
        # Add small random variation to prevent identical scores (±0.05)
        random_variation = (random.random() - 0.5) * 0.10  # ±0.05 variation
        final_confidence += random_variation
        
        # Ensure confidence is in realistic range [0.25, 0.95] with good spread
        final_confidence = max(0.25, min(0.95, final_confidence))
        
        return float(final_confidence)

class OilGasRightsClassifier:
    """Main classification agent with self-consistent sampling for oil and gas reservations"""
    
    def __init__(self, api_key: str):
        """Initialize the oil and gas rights classifier with Anthropic API key."""
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        print("Initializing Anthropic client...")
        try:
            # Simple, standard initialization that works with anthropic 0.40.0
            self.client = anthropic.Anthropic(api_key=api_key)
            print("✓ Anthropic client initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Anthropic client: {e}")
            raise ValueError(f"Failed to initialize Anthropic client: {e}")
        
        # Initialize other components
        self.confidence_scorer = ConfidenceScorer()
        self.past_high_confidence_responses = []
    
    def create_classification_prompt(self, ocr_text: str, high_recall_mode: bool = True) -> str:
        """Create the prompt template for classification with balanced high-recall mode"""
        
        if high_recall_mode:
            # UPDATED PROMPT: Handle broader mineral rights language that includes oil/gas
            prompt = f"""You are a legal document analyst specializing in detecting mineral rights reservations that include oil and gas. Your goal is to identify documents that reserve mineral rights (which typically include oil and gas unless specifically excluded).

UPDATED LOGIC: Mineral rights generally include oil and gas unless explicitly limited to coal only.

DOCUMENT TEXT (from OCR):
{ocr_text}

CLASSIFICATION GUIDELINES:

CLASSIFY AS 1 (HAS mineral rights reservations that include oil/gas) if you find:

EXPLICIT OIL AND GAS LANGUAGE:
- "oil", "gas", "petroleum", "hydrocarbons", "natural gas", "crude oil"
- "oil and gas rights", "oil and gas interests", "petroleum interests"
- "oil and gas lease", "oil and gas royalty", "gas lease", "oil lease"
- "hydrocarbon rights", "petroleum rights"

GENERAL MINERAL RIGHTS LANGUAGE (includes oil/gas unless coal-only):
- "mineral rights" (general - includes oil and gas)
- "mining rights" (general - includes oil and gas)
- "EXCEPTING AND RESERVING the minerals" 
- "RESERVING unto grantors all minerals"
- "EXCEPTING AND RESERVING coal and mining rights" (includes other minerals beyond coal)
- "one-half of the income from any minerals" (includes oil and gas)
- "all subsurface rights"
- "all mineral interests"

RESERVATION PATTERNS THAT INCLUDE OIL/GAS:
- "Grantor reserves..." + any mineral language (unless coal-only)
- "Excepting and reserving..." + mineral/mining rights
- "Subject to mineral lease to [company]" (unless coal-only)
- "Reserving unto grantor mineral rights"

CLASSIFY AS 0 (NO mineral rights reservations that include oil/gas) if:

EXPLICIT EXCLUSIONS:
- "excepting oil and gas" or "excluding oil and gas"
- "reserving all minerals EXCEPT oil and gas"
- "coal rights only" or "ONLY the coal"

COAL-ONLY RESERVATIONS:
- "EXCEPTING AND RESERVING ONLY the coal"
- "coal rights only"
- "reserving the coal seam only"
- Language that specifically limits reservation to coal alone

NO MINERAL RESERVATIONS:
- Pure real estate transactions with no mineral language
- Standard legal boilerplate without substantive mineral reservations
- Historical references only ("subject to coal rights previously reserved")

KEY PRINCIPLE:
- General "mineral rights" or "minerals" = INCLUDES oil and gas (classify as 1)
- "Coal and mining rights" = INCLUDES other minerals beyond coal (classify as 1)  
- "ONLY coal" or "coal only" = EXCLUDES oil and gas (classify as 0)
- No mineral language = NO reservations (classify as 0)

RESPONSE FORMAT:
Answer: [0 or 1]
Reasoning: [Explain your analysis. If you found mineral rights language, explain whether it's general (includes oil/gas) or coal-only. If classifying as 0, explain why there are no mineral reservations or why they're explicitly limited to coal only.]

Where:
- 0 = NO mineral rights reservations that include oil and gas
- 1 = HAS mineral rights reservations that include oil and gas

Goal: Properly classify mineral rights reservations, understanding that general mineral language includes oil and gas unless specifically limited to coal only."""

        else:
            # STANDARD PROMPT (conservative) - keeping this for potential future use
            prompt = f"""You are a conservative legal document analyst specializing in OIL AND GAS rights. Your primary task is to identify documents WITHOUT oil and gas reservations while being extremely cautious about false positives.

CRITICAL: You are ONLY looking for OIL AND GAS reservations. Documents that reserve only coal, other minerals, or general "mineral rights" without specific mention of oil and gas should be classified as 0 (NO oil and gas reservations).

DOCUMENT TEXT (from OCR):
{ocr_text}

CLASSIFICATION APPROACH:
Default assumption: This document has NO oil and gas reservations (classify as 0)
Only classify as 1 if you find CLEAR, SUBSTANTIVE oil and gas reservation language.

STEP-BY-STEP ANALYSIS:

1. FIRST SCAN - Look specifically for OIL AND GAS keywords:
   - "oil", "gas", "petroleum", "hydrocarbons"
   - "oil and gas", "oil or gas", "oil, gas"
   - "oil and gas rights", "oil and gas interests"
   - Royalty percentages specifically tied to oil and gas

2. SECOND ANALYSIS - For each oil/gas keyword found, determine:
   - Is this in a SUBSTANTIVE OIL AND GAS RESERVATION CLAUSE? (classify as 1)
   - OR is this BOILERPLATE/DISCLAIMER text? (classify as 0)
   - OR does it only mention coal/other minerals without oil and gas? (classify as 0)

STRONG EVIDENCE FOR OIL AND GAS RESERVATIONS (classify as 1):
- "Grantor reserves all oil and gas rights"
- "Excepting and reserving unto grantor all oil and gas"
- "Subject to oil and gas rights reserved in prior deed to [specific party]"
- "Reserving 1/2 of all oil and gas rights"
- "Grantor retains all oil, gas and petroleum interests"
- Named grantors retaining specific oil and gas rights with operational details
- "Subject to oil and gas lease to [specific company]"

IMPORTANT: IGNORE THESE (classify as 0 - NO oil and gas reservations):
- "Grantor reserves all coal rights" (coal only, no oil/gas mentioned)
- "Excepting and reserving all coal and mining rights" (no oil/gas)
- "Subject to mineral rights" (general minerals, no specific oil/gas mention)
- "Reserving all minerals except oil and gas" (actually GRANTS oil and gas)
- "All mineral rights excluding petroleum" (excludes oil/gas, so no reservation)

CONSERVATIVE BIAS:
When in doubt between 0 and 1, choose 0. It's better to miss an actual oil and gas reservation than to create a false positive.
Be especially skeptical of language that appears to be standard legal boilerplate or mentions only coal/other minerals.

RESPONSE FORMAT:
Answer: [0 or 1]
Reasoning: [Explain your analysis step by step. If you found oil/gas keywords, explain why they are/aren't substantive OIL AND GAS reservations. Be specific about whether the language mentions oil and gas specifically or just general minerals/coal. If you suspect boilerplate, explain why.]

Where:
- 0 = NO substantive oil and gas reservations (default assumption)
- 1 = CLEAR, substantive oil and gas reservations present

Remember: Your goal is to confidently identify documents WITHOUT oil and gas reservations. Only classify as 1 when you're certain there's a substantive reservation that specifically mentions oil, gas, petroleum, or hydrocarbons."""

        return prompt
    
    def extract_classification(self, response: str) -> Tuple[Optional[int], str]:
        """Extract classification and reasoning from model response"""
        
        # Look for Answer: pattern
        answer_match = re.search(r'Answer:\s*([01])', response, re.IGNORECASE)
        if answer_match:
            classification = int(answer_match.group(1))
        else:
            # Fallback: look for standalone 0 or 1
            number_matches = re.findall(r'\b([01])\b', response)
            if number_matches:
                classification = int(number_matches[0])
            else:
                return None, response
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            reasoning = response
            
        return classification, reasoning
    
    def generate_sample(self, ocr_text: str, temperature: float = 0.1, 
                       high_recall_mode: bool = True) -> Optional[ClassificationSample]:
        """Generate a single classification sample - always use high recall mode by default"""
        
        prompt = self.create_classification_prompt(ocr_text, high_recall_mode)
        
        # Retry logic for network issues with timeout
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=temperature,  # Keep temperature as provided (0.1)
                    messages=[{
                        "role": "user", 
                        "content": prompt
                    }],
                    timeout=30.0  # 30 second timeout for API calls
                )
                
                raw_response = response.content[0].text
                predicted_class, reasoning = self.extract_classification(raw_response)
                
                if predicted_class is None:
                    print(f"Warning: Could not extract classification from response (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief pause before retry
                        continue
                    return None
                    
                # Extract features for confidence scoring
                features = self.confidence_scorer.extract_features(
                    raw_response, 
                    ocr_text, 
                    self.past_high_confidence_responses
                )
                
                # Score confidence
                confidence_score = self.confidence_scorer.score_confidence(features)
                
                return ClassificationSample(
                    predicted_class=predicted_class,
                    reasoning=reasoning,
                    confidence_score=confidence_score,
                    features=features,
                    raw_response=raw_response
                )
                
            except anthropic.APIError as e:
                print(f"Anthropic API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
            except Exception as e:
                print(f"Unexpected error generating sample (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None
        
        return None
    
    def classify_document(self, ocr_text: str, max_samples: int = 8, 
                         confidence_threshold: float = 0.7,
                         high_recall_mode: bool = True) -> ClassificationResult:
        """Classify document using self-consistent sampling - balanced high recall mode"""
        
        votes = {0: 0.0, 1: 0.0}
        all_samples = []
        early_stopped = False
        
        # mode_label = "BALANCED HIGH RECALL" if high_recall_mode else "CONSERVATIVE (High Specificity)"
        # mode_msg   = "Good sensitivity while maintaining accuracy" \
        #             if high_recall_mode else "Extra-cautious, prioritising specificity"
        
        
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
        print(f"   - Max samples: {max_samples}")
        print(f"   - Confidence threshold: {confidence_threshold}")
        
        for i in range(max_samples):
            print(f"Generating sample {i+1}/{max_samples}...")
            
            # Keep temperature constant at 0.1 as requested
            sample = self.generate_sample(ocr_text, temperature=0.1, high_recall_mode=high_recall_mode)
            
            if sample is None:
                continue
                
            all_samples.append(sample)
            
            # Add weighted vote
            votes[sample.predicted_class] += sample.confidence_score
            
            # Store high-confidence responses for future reference
            if sample.confidence_score > 0.8:
                self.past_high_confidence_responses.append(sample.raw_response)
                # Keep only recent high-confidence responses
                if len(self.past_high_confidence_responses) > 20:
                    self.past_high_confidence_responses.pop(0)
            
            # BALANCED HIGH RECALL EARLY STOPPING LOGIC
            total_votes = sum(votes.values())
            if total_votes > 0:
                leading_class = max(votes.keys(), key=lambda k: votes[k])
                leading_proportion = votes[leading_class] / total_votes
                
                if leading_class == 1:  # Positive classification (has reservations)
                    # Moderate threshold for positive classifications - still favor recall but not extreme
                    required_confidence = 0.65  # Reasonable threshold (was 0.55, too low)
                    min_samples_positive = 4    # More samples needed for confidence (was 3)
                    
                    if leading_proportion >= required_confidence and i >= min_samples_positive - 1:
                        print(f"BALANCED Early stopping: Positive classification with {leading_proportion:.3f} confidence after {i+1} samples")
                        early_stopped = True
                        break
                else:  # Negative classification (no reservations)
                    # Moderate threshold for negative classifications - not too high to maintain recall
                    required_confidence = 0.75  # Reasonable threshold (was 0.85, too high)
                    min_samples_negative = 5     # Moderate samples required (was 6)
                    
                    if leading_proportion >= required_confidence and i >= min_samples_negative - 1:
                        print(f"BALANCED Early stopping: Negative classification with {leading_proportion:.3f} confidence after {i+1} samples")
                        early_stopped = True
                        break
        
        # Determine final classification
        if sum(votes.values()) > 0:
            predicted_class = max(votes.keys(), key=lambda k: votes[k])
            
            # Use vote proportion (mathematically sound for consensus confidence)
            final_confidence = votes[predicted_class] / sum(votes.values())
            
            # BALANCED HIGH RECALL MODE: Apply tie-breaking bias only in very close cases
            if abs(votes[0] - votes[1]) < 0.2:  # Only very close votes (was 0.3, too aggressive)
                print(f"🎯 BALANCED RECALL: Very close vote detected ({votes}), applying slight positive bias")
                predicted_class = 1  # Bias toward positive classification
                final_confidence = max(final_confidence, 0.55)  # Modest boost (was 0.6)
                
        else:
            predicted_class = 0  # Default to no reservations
            final_confidence = 0.0
            
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=final_confidence,
            votes=votes,
            samples_used=len(all_samples),
            early_stopped=early_stopped,
            all_samples=all_samples
        )

class DocumentProcessor:
    """Complete pipeline from PDF to classification"""
    
    def __init__(self, api_key: str = None):
        try:
            self.classifier = OilGasRightsClassifier(api_key)
            print("✅ Document processor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize document processor: {e}")
            raise

    def split_pdf_by_deeds(self, pdf_path: str, strategy: str = "smart_detection") -> List[str]:
        """Split PDF into individual deed PDFs based on strategy"""
        
        import tempfile
        import os
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"Splitting PDF with {total_pages} pages using strategy: {strategy}")
        
        if strategy == "page_based":
            # Simple page-based splitting: every 3 pages = 1 deed
            deed_pdfs = []
            pages_per_deed = 3
            
            for i in range(0, total_pages, pages_per_deed):
                end_page = min(i + pages_per_deed, total_pages)
                
                # Create new PDF with these pages
                new_doc = fitz.open()
                for page_num in range(i, end_page):
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                new_doc.save(temp_file.name)
                new_doc.close()
                
                deed_pdfs.append(temp_file.name)
                print(f"Created deed {len(deed_pdfs)}: pages {i+1}-{end_page}")
            
            doc.close()
            return deed_pdfs
            
        elif strategy == "smart_detection":
            # Smart detection: look for deed boundaries in text
            deed_pdfs = []
            current_deed_start = 0
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                
                # Look for deed boundary indicators
                boundary_indicators = [
                    "this deed", "witnesseth", "know all men by these presents",
                    "grantor", "grantee", "deed of", "warranty deed",
                    "quitclaim deed", "special warranty deed"
                ]
                
                is_boundary = any(indicator in text for indicator in boundary_indicators)
                
                # If this looks like a new deed and we have pages, create a deed
                if is_boundary and page_num > current_deed_start:
                    # Create deed from current_deed_start to page_num-1
                    new_doc = fitz.open()
                    for p in range(current_deed_start, page_num):
                        new_doc.insert_pdf(doc, from_page=p, to_page=p)
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    new_doc.save(temp_file.name)
                    new_doc.close()
                    
                    deed_pdfs.append(temp_file.name)
                    print(f"Created deed {len(deed_pdfs)}: pages {current_deed_start+1}-{page_num}")
                    current_deed_start = page_num
            
            # Add the last deed (from current_deed_start to end)
            if current_deed_start < total_pages:
                new_doc = fitz.open()
                for p in range(current_deed_start, total_pages):
                    new_doc.insert_pdf(doc, from_page=p, to_page=p)
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                new_doc.save(temp_file.name)
                new_doc.close()
                
                deed_pdfs.append(temp_file.name)
                print(f"Created deed {len(deed_pdfs)}: pages {current_deed_start+1}-{total_pages}")
            
            doc.close()
            
            # If no boundaries found, treat as single deed
            if not deed_pdfs:
                print("No deed boundaries detected, treating as single deed")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                doc.save(temp_file.name)
                doc.close()
                return [temp_file.name]
            
            return deed_pdfs
            
        elif strategy == "ai_assisted":
            # AI-assisted splitting: use Claude to analyze document structure
            print("AI-assisted splitting not yet implemented, falling back to smart detection")
            return self.split_pdf_by_deeds(pdf_path, "smart_detection")
            
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")

    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files created during processing"""
        import os
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"Warning: Could not clean up {file_path}: {e}")

    def process_multi_deed_document(self, pdf_path: str, strategy: str = "smart_detection") -> List[Dict]:
        """Process PDF with multiple deeds using memory-efficient processing"""
        
        print(f"🔧 Starting multi-deed processing with strategy: {strategy}")
        
        # 1. Split PDF into individual deed PDFs based on strategy
        deed_pdfs = self.split_pdf_by_deeds(pdf_path, strategy)
        print(f"📄 Split into {len(deed_pdfs)} deed files")
        
        try:
            # 2. Process each deed separately using memory-efficient processing
            results = []
            for i, deed_pdf_path in enumerate(deed_pdfs):
                print(f"Processing deed {i+1}/{len(deed_pdfs)}...")
                try:
                    # Use memory-efficient processing for each deed
                    result = self.process_document_memory_efficient(
                        deed_pdf_path,
                        chunk_size=25,  # Smaller chunks for individual deeds
                        max_samples=2,  # Very few samples for speed in multi-deed processing
                        confidence_threshold=0.5,  # Lower threshold for faster processing
                        high_recall_mode=True
                    )
                    result['deed_number'] = i + 1
                    result['deed_file'] = deed_pdf_path
                    result['pages_in_deed'] = result.get('pages_processed', 0)
                    results.append(result)
                    print(f"✅ Deed {i+1} completed: {result['classification']} (confidence: {result['confidence']:.3f})")
                except Exception as e:
                    print(f"❌ Error processing deed {i+1}: {e}")
                    # Add error result
                    error_result = {
                        'deed_number': i + 1,
                        'deed_file': deed_pdf_path,
                        'classification': 0,
                        'confidence': 0.0,
                        'pages_in_deed': 0,
                        'error': str(e)
                    }
                    results.append(error_result)
            
            print(f"🎯 Multi-deed processing completed: {len(results)} deeds processed")
            return results
        finally:
            # 3. Clean up temporary files
            print("🧹 Cleaning up temporary deed files...")
            self.cleanup_temp_files(deed_pdfs)

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
        
        # Retry logic for network issues with timeout
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
                    }],
                    timeout=45.0  # 45 second timeout for OCR calls (longer than classification)
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
    
    def process_document(
        self,
        pdf_path: str,
        *,
        log_fn=None,                    # ← NEW
        max_samples: int = 8,
        confidence_threshold: float = 0.7,
        page_strategy: str = "sequential_early_stop",
        max_pages: int = None,
        max_tokens_per_page: int = 8000,
        combine_method: str = "early_stop",
        high_recall_mode: bool = False,
    ) -> Dict:
        # OPTIONAL: simple helper so we don’t have to touch every print
        def _log(msg: str):
            print(msg)
            if log_fn:
                log_fn(msg)

        _log(f"Processing: {pdf_path}")
        _log(f"Page strategy: {page_strategy}")
        if high_recall_mode:
            _log("🎯 BALANCED HIGH RECALL MODE – Good sensitivity while maintaining accuracy")
        else:
            _log("🎯 CONSERVATIVE (High Specificity) MODE – Extra-cautious, prioritising specificity")
        
        # Use sequential early stopping by default
        if page_strategy == "sequential_early_stop" or combine_method == "early_stop":
            # return self._process_with_early_stopping(
            #     pdf_path, max_samples, confidence_threshold, max_tokens_per_page, max_pages
            # )
            return self._process_with_early_stopping(
                pdf_path,
                max_samples,
                confidence_threshold,
                max_tokens_per_page,
                max_pages,
                high_recall_mode,                     # pass flag down
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
                                   max_pages: int = None, high_recall_mode: bool = False, ) -> Dict:
 

        """Process document chunk by chunk with early stopping when reservations are found"""
        
        import time
        import gc
        
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
        print(f"⏰ Session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        chunk_analysis = []
        all_ocr_text = []
        stopped_at_chunk = None
        unread_pages: list[int] = []
        
        # Track timing for progress estimates
        start_time = time.time()
        page_times = []
        last_gc_time = start_time
        
        # Process page by page with early stopping
        for page_num in range(pages_to_process):
            page_start_time = time.time()
            current_page = page_num + 1
            print(f"\n--- PROCESSING CHUNK {current_page}/{pages_to_process} ---")
            
            # Send progress update every few pages to keep connection alive
            if current_page % 3 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_page = elapsed_time / current_page if current_page > 0 else 0
                remaining_pages = pages_to_process - current_page
                estimated_remaining_time = avg_time_per_page * remaining_pages
                
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                
                print(f"📊 Progress: {current_page}/{pages_to_process} pages processed ({current_page/pages_to_process*100:.1f}%)")
                print(f"⏱️  Elapsed: {hours}h {minutes}m | Est. remaining: {estimated_remaining_time/60:.1f} minutes")
                print(f"📈 Avg time per page: {avg_time_per_page:.1f} seconds")
                
                # Force garbage collection every 30 minutes for long sessions
                if elapsed_time - last_gc_time > 1800:  # 30 minutes
                    print("🧹 Running garbage collection...")
                    gc.collect()
                    last_gc_time = elapsed_time
            
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
                # page_text = f"[ERROR: Could not extract text from page {current_page}]"
                # all_ocr_text.append(f"=== PAGE {current_page} ===\n{page_text}")
                
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
            
            
            # classification_result = self.classifier.classify_document(
            #     page_text, max_samples, confidence_threshold
            # )

            classification_result = self.classifier.classify_document(
                page_text,
                max_samples,
                confidence_threshold,
                high_recall_mode=high_recall_mode,    # use flag
            )

            # Track page processing time
            page_time = time.time() - page_start_time
            page_times.append(page_time)
            print(f"Page {current_page} processed in {page_time:.1f} seconds")

            # ADD reasoning to chunk_info  🚀
            first_reasoning = (
                classification_result.all_samples[0].reasoning
                if classification_result.all_samples else ""
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
                'high_recall_mode': high_recall_mode,
                'reasoning': first_reasoning,              #  ← new
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
        else:                          # every page failed OCR
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
            'detailed_samples': [
                { "predicted_class": 0, "reasoning": successful_chunks[-1]["reasoning"] }
            ] if successful_chunks else [],
            'ocr_failed_pages': unread_pages,
            'requires_manual_review': len(unread_pages) > 0,
        }

    def process_document_memory_efficient(self, pdf_path: str, max_samples: int = 8, 
                                        confidence_threshold: float = 0.7,
                                        max_tokens_per_page: int = 8000, 
                                        chunk_size: int = 50, high_recall_mode: bool = False) -> Dict:
        """
        Process document in memory-efficient chunks to prevent memory leaks during long sessions
        """
        import time
        import gc
        import psutil
        import os
        
        print("Using memory-efficient chunked processing")
        print(f"📊 Chunk size: {chunk_size} pages")
        
        # Memory monitoring
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
        
        # Open PDF and get total pages
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"Document has {total_pages} pages")
        print(f"⏰ Session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Process in chunks
        all_results = []
        chunk_results = []
        current_chunk = 1
        total_chunks = (total_pages + chunk_size - 1) // chunk_size
        
        start_time = time.time()
        
        for chunk_start in range(0, total_pages, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pages)
            chunk_pages = list(range(chunk_start, chunk_end))
            
            print(f"\n--- PROCESSING CHUNK {current_chunk}/{total_chunks} (pages {chunk_start+1}-{chunk_end}) ---")
            
            # Process this chunk
            chunk_result = self._process_chunk_memory_efficient(
                doc, chunk_pages, max_samples, confidence_threshold, 
                max_tokens_per_page, high_recall_mode, current_chunk
            )
            
            chunk_results.append(chunk_result)
            
            # Check for early stopping
            if chunk_result.get('classification') == 1:
                print(f"🎯 Oil and gas reservations found in chunk {current_chunk}!")
                print(f"✅ Early stopping - no need to process remaining chunks")
                
                # Combine results
                final_result = self._combine_chunk_results(chunk_results, pdf_path, total_pages)
                final_result['early_stopped'] = True
                final_result['stopped_at_chunk'] = current_chunk
                
                break
            
            # Memory management between chunks
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            print(f"💾 Memory after chunk {current_chunk}: {current_memory:.1f} MB (change: {memory_increase:+.1f} MB)")
            
            # Force garbage collection if memory increase is significant
            if memory_increase > 200:  # More than 200MB increase
                print("🧹 Running garbage collection between chunks...")
                gc.collect()
                after_gc_memory = process.memory_info().rss / 1024 / 1024
                gc_reduction = current_memory - after_gc_memory
                print(f"🧹 GC freed: {gc_reduction:.1f} MB")
            
            current_chunk += 1
            
            # Progress update
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            print(f"📊 Progress: {current_chunk-1}/{total_chunks} chunks completed ({((current_chunk-1)/total_chunks*100):.1f}%)")
            print(f"⏱️  Elapsed: {hours}h {minutes}m")
        
        # Close document
        doc.close()
        
        # If we get here, no reservations were found
        if 'final_result' not in locals():
            final_result = self._combine_chunk_results(chunk_results, pdf_path, total_pages)
            final_result['early_stopped'] = False
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_change = final_memory - initial_memory
        print(f"💾 Final memory: {final_memory:.1f} MB (change: {memory_change:+.1f} MB)")
        
        # Final garbage collection
        print("🧹 Running final garbage collection...")
        gc.collect()
        
        return final_result

    def _process_chunk_memory_efficient(self, doc, chunk_pages, max_samples, confidence_threshold, 
                                      max_tokens_per_page, high_recall_mode, chunk_number):
        """Process a single chunk of pages with memory management"""
        
        chunk_start_time = time.time()
        chunk_analysis = []
        chunk_ocr_text = []
        
        print(f"Processing {len(chunk_pages)} pages in chunk {chunk_number}")
        
        for page_num in chunk_pages:
            page_start_time = time.time()
            current_page = page_num + 1
            
            print(f"  Processing page {current_page}...")
            
            try:
                # Convert page to image
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                # Clear page objects immediately
                page = None
                pix = None
                del page, pix
                
                # Extract text
                page_text = self.extract_text_with_claude(image, max_tokens_per_page)
                chunk_ocr_text.append(f"=== PAGE {current_page} ===\n{page_text}")
                
                # Clear image immediately
                image.close()
                image = None
                del image, img_data
                
                # Classify page
                classification_result = self.classifier.classify_document(
                    page_text, max_samples, confidence_threshold, high_recall_mode
                )
                
                # Track page info
                page_time = time.time() - page_start_time
                chunk_analysis.append({
                    'page_number': current_page,
                    'text_length': len(page_text),
                    'classification': classification_result.predicted_class,
                    'confidence': classification_result.confidence,
                    'processing_time': page_time
                })
                
                # Clear page text
                page_text = None
                del page_text
                
                print(f"    Page {current_page} processed in {page_time:.1f}s")
                
            except Exception as e:
                print(f"    Error processing page {current_page}: {e}")
                chunk_analysis.append({
                    'page_number': current_page,
                    'status': 'error',
                    'error': str(e)
                })
                continue
        
        chunk_time = time.time() - chunk_start_time
        print(f"Chunk {chunk_number} completed in {chunk_time:.1f}s")
        
        return {
            'chunk_number': chunk_number,
            'pages': chunk_pages,
            'analysis': chunk_analysis,
            'ocr_text': chunk_ocr_text,
            'processing_time': chunk_time,
            'classification': max([p.get('classification', 0) for p in chunk_analysis if 'classification' in p], default=0)
        }

    def _combine_chunk_results(self, chunk_results, pdf_path, total_pages):
        """Combine results from multiple chunks into final result"""
        
        # Combine all OCR text
        all_ocr_text = []
        for chunk in chunk_results:
            all_ocr_text.extend(chunk['ocr_text'])
        
        # Find overall classification
        overall_classification = max([chunk['classification'] for chunk in chunk_results], default=0)
        
        # Calculate overall confidence (average of chunk confidences)
        chunk_confidences = []
        for chunk in chunk_results:
            for page in chunk['analysis']:
                if 'confidence' in page:
                    chunk_confidences.append(page['confidence'])
        
        overall_confidence = sum(chunk_confidences) / len(chunk_confidences) if chunk_confidences else 0.0
        
        # Combine all page analysis
        all_page_analysis = []
        for chunk in chunk_results:
            all_page_analysis.extend(chunk['analysis'])
        
        return {
            'document_path': pdf_path,
            'pages_processed': total_pages,
            'page_strategy': "memory_efficient_chunked",
            'chunks_processed': len(chunk_results),
            'classification': overall_classification,
            'confidence': overall_confidence,
            'ocr_text': "\n\n".join(all_ocr_text),
            'page_analysis': all_page_analysis,
            'total_processing_time': sum(chunk['processing_time'] for chunk in chunk_results)
        }

def main():
    """Example usage"""
    
    # Initialize processor
    api_key = os.getenv("ANTHROPIC_API_KEY")
    processor = DocumentProcessor(api_key=api_key)
    
    # Process single document
    pdf_path = "data/reservs/Washington DB 405_547.pdf"
    result = processor.process_document(pdf_path)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION RESULT")
    print(f"{'='*60}")
    print(f"Document: {result['document_path']}")
    print(f"Classification: {result['classification']} ({'Has Oil and Gas Reservations' if result['classification'] == 1 else 'No Oil and Gas Reservations'})")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Samples Used: {result['samples_used']}")
    print(f"Early Stopped: {result['early_stopped']}")
    print(f"Vote Distribution: {result['votes']}")
    
    # Save detailed results
    output_dir = Path("classification_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / f"result_{Path(pdf_path).stem}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_dir}")


if __name__ == "__main__":
    main()
