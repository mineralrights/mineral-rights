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
from .document_ai_service import create_document_ai_service, DocumentAISplitResult
from .deed_tracker import get_deed_tracker
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
    
    def __init__(self, api_key: str, model_name: str = None):
        """Initialize the oil and gas rights classifier with Anthropic API key."""
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        # Strip whitespace/newlines from API key (common issue with secrets)
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("Anthropic API key is empty after stripping whitespace")
        
        # Get model name from parameter or environment variable, with fallback
        import os
        self.model_name = model_name or os.getenv("CLAUDE_MODEL_NAME", "claude-opus-4-5-20251101")
        print(f"🔧 Using Claude model: {self.model_name}")
        
        print("Initializing Anthropic client...")
        try:
            # Initialize with increased timeout for Cloud Run environments
            # Default httpx timeout is 10s, increase to 60s for large image processing
            import httpx
            timeout = httpx.Timeout(60.0, connect=30.0)  # 60s total, 30s connect
            self.client = anthropic.Anthropic(
                api_key=api_key,
                timeout=timeout,
                max_retries=2  # Let our code handle retries
            )
            print("✓ Anthropic client initialized successfully with 60s timeout")
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

CRITICAL: Check these exceptions FIRST before classifying as 1. If any of these apply, classify as 0:

1. CONVEYANCES/GRANTS (grantor transferring rights - NOT reservations):
   - If you see "Grantor does hereby grant, sell, convey, assign and transfer unto Grantee..." with mineral rights
   - This is a TRANSFER where grantor gives rights away, NOT a reservation where grantor keeps them
   - Classify as 0 even if it mentions "oil, gas, coal and other minerals"
   - EXCEPTION: Only if there's ALSO actual reservation language elsewhere (e.g., "Grantor reserves..."), then classify based on the reservation

2. "SUBJECT TO... OF RECORD" CLAUSES (acknowledging existing reservations - NOT creating new ones):
   - If you see "Subject to... reservations... of record" or "Subject to... exceptions... of record"
   - The phrase "of record" means it's referring to things ALREADY recorded in PRIOR deeds
   - This is NOT creating a new reservation, just acknowledging what may exist from prior conveyances
   - Classify as 0 if this is the ONLY reservation-related language
   - EXCEPTION: Only if there's ALSO actual reservation language elsewhere (e.g., "Grantor reserves..."), then classify based on the actual reservation

ONLY IF NONE OF THE ABOVE EXCEPTIONS APPLY, THEN:

CLASSIFY AS 1 (HAS mineral rights reservations that include oil/gas) if you find:

EXPLICIT OIL AND GAS LANGUAGE:
- "oil", "gas", "petroleum", "hydrocarbons", "natural gas", "crude oil"
- "oil and gas rights", "oil and gas interests", "petroleum interests"
- "oil and gas lease", "oil and gas royalty", "gas lease", "oil lease"
- "hydrocarbon rights", "petroleum rights"
- "rentals" (when associated with oil/gas/mineral reservations)
- "royalty" or "royalties" (when associated with oil/gas/mineral reservations)

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
- "Reserves" + "oil," "gas," "oil and gas," "petroleum," "natural gas," "minerals," "rentals," "royalty," or "royalties"
- "Excepting and reserving" + "oil," "gas," "oil and gas," "petroleum," "natural gas," "minerals," "rentals," "royalty," or "royalties"
- "Excepts and reserves" + "oil," "gas," "oil and gas," "petroleum," "natural gas," "minerals," "rentals," "royalty," or "royalties"
- "Reserving" + "oil," "gas," "oil and gas," "petroleum," "natural gas," "minerals," "rentals," "royalty," or "royalties"
- "Excepting" + "oil," "gas," "oil and gas," "petroleum," "natural gas," "minerals," "rentals," "royalty," or "royalties"
- "There is reserved from" + "oil," "gas," "oil and gas," "petroleum," "natural gas," "minerals," "rentals," "royalty," or "royalties"
- "There is excepted from" + "oil," "gas," "oil and gas," "petroleum," "natural gas," "minerals," "rentals," "royalty," or "royalties"
- "Excepting and reserving..." + mineral/mining rights
- "Subject to mineral lease to [company]" (unless coal-only)
- "Reserving unto grantor mineral rights"

CLASSIFY AS 0 (NO mineral rights reservations that include oil/gas) if:

CONVEYANCES/GRANTS (grantor transferring rights to another party - NOT reservations):
- "Grantor does hereby grant, sell, convey, assign and transfer unto Grantee... all of the oil, gas, coal and other minerals"
- "Grantor does hereby grant, sell, convey, assign and transfer unto Grantee, its successors and assigns... 100% of Grantor's right, title and interest in and to all of the oil, gas, coal and other minerals"
- Any language where grantor is TRANSFERRING/GIVING mineral rights TO another party (grantee, buyer, etc.)
- These are CONVEYANCES where grantor gives rights away, not RESERVATIONS where grantor keeps them
- IMPORTANT: Classify as 0 ONLY if this conveyance language is the ONLY evidence of mineral rights. If the document ALSO contains actual reservation language (e.g., "Grantor reserves..." or "Excepting and reserving..."), then classify based on the actual reservation, not the conveyance.
- If the ONLY mineral rights language is a conveyance (grantor giving rights away), classify as 0 because it's a transfer, not a reservation

"SUBJECT TO" CLAUSES (acknowledging existing reservations from prior deeds - NOT creating new reservations):
- "Subject to all encumbrances, reservations and exceptions, including but not limited to coal, minerals, oil and gas, right of ways, easements and leases, if any, of record"
- "Subject to... reservations... of record" - This acknowledges existing reservations from PRIOR deeds in the chain of title
- "Subject to... exceptions... of record" - This acknowledges existing exceptions from PRIOR deeds
- "Subject to... reservations and exceptions... of record" - This is NOT creating a new reservation, just acknowledging what may already exist
- Key indicator: "of record" or similar language showing it refers to things already recorded in prior deeds
- These clauses make the buyer aware of potential existing reservations but do NOT create new ones in this document
- IMPORTANT: Classify as 0 ONLY if this "subject to... of record" language is the ONLY evidence of reservations. If the document ALSO contains actual reservation language (e.g., "Grantor reserves..." or "Excepting and reserving unto grantor..."), then classify based on the actual reservation, not the "subject to" clause.
- If the ONLY reservation-related language is a "subject to... of record" clause, classify as 0 because this document is not reserving anything - it's just acknowledging what might exist from prior conveyances

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
Quoted Text: [IF you classified as 1 (HAS reservations), you MUST quote the exact text from the document where the reservations are stated. Include the full sentence or clause that contains the reservation language. If you classified as 0, you can leave this blank or write "N/A".]

Where:
- 0 = NO mineral rights reservations that include oil and gas
- 1 = HAS mineral rights reservations that include oil and gas

Goal: Properly classify mineral rights reservations, understanding that general mineral language includes oil and gas unless specifically limited to coal only. If reservations are found, you must explicitly quote WHERE they appear in the document."""

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
Quoted Text: [IF you classified as 1 (HAS reservations), you MUST quote the exact text from the document where the oil and gas reservations are stated. Include the full sentence or clause that contains the reservation language. If you classified as 0, you can leave this blank or write "N/A".]

Where:
- 0 = NO substantive oil and gas reservations (default assumption)
- 1 = CLEAR, substantive oil and gas reservations present

Remember: Your goal is to confidently identify documents WITHOUT oil and gas reservations. Only classify as 1 when you're certain there's a substantive reservation that specifically mentions oil, gas, petroleum, or hydrocarbons. If reservations are found, you must explicitly quote WHERE they appear in the document."""

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
        
        # Extract reasoning - capture everything after "Reasoning:" until end of response
        # Use greedy match to capture full reasoning even if it contains semicolons, colons, or single newlines
        # This includes the "Quoted Text:" section which contains important reservation quotes
        reasoning_match = re.search(r'Reasoning:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Remove any trailing "Answer:" or "Classification:" sections that might appear after reasoning
            # But keep "Quoted Text:" as it's part of the reasoning
            reasoning = re.sub(r'\s*(?:Answer|Classification):\s*.*$', '', reasoning, flags=re.IGNORECASE | re.DOTALL).strip()
        else:
            reasoning = response
            
        return classification, reasoning
    
    def generate_sample(self, ocr_text: str, temperature: float = 0.1, 
                       high_recall_mode: bool = True) -> Optional[ClassificationSample]:
        """Generate a single classification sample - always use high recall mode by default"""
        
        prompt = self.create_classification_prompt(ocr_text, high_recall_mode)
        
        # Retry logic for network issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,  # Use configurable model name
                    max_tokens=8000,  # Increased to 8000 to ensure full explanations are not truncated
                    temperature=temperature,  # Keep temperature as provided (0.1)
                    messages=[{
                        "role": "user", 
                        "content": prompt
                    }]
                )
                
                # Check if response was truncated
                stop_reason = getattr(response, 'stop_reason', None)
                if stop_reason == 'max_tokens':
                    print(f"⚠️ WARNING: LLM response was truncated due to max_tokens limit. Consider increasing max_tokens.")
                
                raw_response = response.content[0].text
                                # Check if response appears truncated (ends with colon or incomplete sentence)
                response_ends_with_colon = raw_response.rstrip().endswith(':')
                if stop_reason == 'max_tokens' or response_ends_with_colon:
                    print(f"⚠️ WARNING: LLM response may be truncated. Stop reason: {stop_reason}, Ends with colon: {response_ends_with_colon}")
                    print(f"🔍 Response length: {len(raw_response)} characters")
                    print(f"🔍 Response ends with: ...{raw_response[-150:]}")
                    # If truncated, we still extract what we have, but log the issue
                
                predicted_class, reasoning = self.extract_classification(raw_response)
                
                # Additional check: if reasoning ends with colon, it's likely truncated
                if reasoning.rstrip().endswith(':'):
                    print(f"⚠️ WARNING: Extracted reasoning ends with colon - likely truncated!")
                    print(f"🔍 Reasoning length: {len(reasoning)} characters")
                    print(f"🔍 Reasoning ends with: ...{reasoning[-100:]}")
                
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
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"❌ Anthropic API error (attempt {attempt + 1}): {error_type}: {error_msg}")
                # Check for common API key errors
                if "401" in error_msg or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                    print("🔑 API KEY ERROR DETECTED - Authentication failed. Please check your ANTHROPIC_API_KEY.")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                import traceback
                print(f"❌ Unexpected error generating sample (attempt {attempt + 1}): {error_type}: {error_msg}")
                print(f"📋 Full error details:")
                traceback.print_exc()
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
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
    
    def __init__(self, api_key: str = None, document_ai_endpoint: str = None, document_ai_credentials: str = None, model_name: str = None):
        try:
            self.classifier = OilGasRightsClassifier(api_key, model_name=model_name)
            
            # Initialize Document AI service if endpoint is provided
            self.document_ai_service = None
            if document_ai_endpoint:
                try:
                    self.document_ai_service = create_document_ai_service(
                        document_ai_endpoint, 
                        document_ai_credentials
                    )
                    print("✅ Document AI service initialized successfully")
                except Exception as e:
                    print(f"⚠️ Document AI service initialization failed: {e}")
                    print("🔄 Will use fallback splitting methods")
            
            # Initialize deed tracker
            self.deed_tracker = get_deed_tracker()
            
            print("✅ Document processor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize document processor: {e}")
            raise

    def split_pdf_by_deeds(self, pdf_path: str, strategy: str = "document_ai") -> List[str]:
        """Split PDF into individual deed PDFs using the specified strategy"""
        
        import tempfile
        import os
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"🚀 Splitting PDF with {total_pages} pages using strategy: {strategy}")
        
        if strategy == "document_ai":
            # Use Document AI smart chunking
            if self.document_ai_service:
                print("✅ Document AI service is available - starting smart chunking")
                return self._split_with_smart_chunking(pdf_path)
            else:
                print("⚠️ Document AI service not available, falling back to simple splitting")
                return self._split_with_simple_strategy(pdf_path)
        elif strategy == "simple":
            # Use simple page-based splitting
            print("📄 Using simple page-based splitting")
            return self._split_with_simple_strategy(pdf_path)
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
    
    def _split_with_simple_strategy(self, pdf_path: str) -> List[str]:
        """Split PDF using simple page-based strategy (every 3 pages = 1 deed)"""
        import tempfile
        import os
        
        print(f"🔧 Using simple page-based splitting...")
        
        # Open PDF to get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Simple strategy: every 3 pages = 1 deed
        pages_per_deed = 3
        deed_pdfs = []
        
        for start_page in range(0, total_pages, pages_per_deed):
            end_page = min(start_page + pages_per_deed, total_pages)
            deed_number = len(deed_pdfs) + 1
            
            # Create temporary PDF for this deed
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_deed_{deed_number}.pdf")
            
            # Extract pages for this deed
            doc = fitz.open(pdf_path)
            deed_doc = fitz.open()
            deed_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
            deed_doc.save(temp_file.name)
            deed_doc.close()
            doc.close()
            
            deed_pdfs.append(temp_file.name)
            print(f"💾 Created deed {deed_number} PDF (pages {start_page+1}-{end_page}): {temp_file.name}")
        
        print(f"📄 Simple splitting created {len(deed_pdfs)} deed files")
        return deed_pdfs
    
    def _split_with_smart_chunking(self, pdf_path: str) -> List[str]:
        """Split PDF using smart chunking Document AI approach"""
        import tempfile
        import os
        
        print(f"🔧 Using Smart Chunking Document AI for deed detection...")
        
        # Use smart chunking service
        print("📡 Calling Document AI service for smart chunking...")
        result = self.document_ai_service.split_deeds_with_smart_chunking(pdf_path)
        print(f"🔍 DEBUG: Document AI result type: {type(result)}")
        print(f"🔍 DEBUG: Document AI result: {result}")
        if result is None:
            raise ValueError("Document AI service returned None result")
        self._last_split_result = result
        print("✅ Document AI smart chunking completed successfully")
        
        print(f"📊 Smart Chunking Results:")
        print(f"   - Total deeds detected: {result.total_deeds}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        chunks_processed = "N/A"
        if hasattr(result, 'raw_response') and result.raw_response is not None:
            chunks_processed = result.raw_response.get('chunks_processed', 'N/A')
        print(f"   - Chunks processed: {chunks_processed}")
        systematic_offset = "N/A"
        raw_deeds_before_merge = "N/A"
        if hasattr(result, 'raw_response') and result.raw_response is not None:
            systematic_offset = result.raw_response.get('systematic_offset', 'N/A')
            raw_deeds_before_merge = result.raw_response.get('raw_deeds_before_merge', 'N/A')
        print(f"   - Systematic offset: {systematic_offset}")
        print(f"   - Raw deeds before merge: {raw_deeds_before_merge}")
        
        # Log deed boundaries for interpretability
        print(f"\n📋 Deed Boundaries Detected:")
        for i, deed in enumerate(result.deeds):
            start_page = deed.start_page + 1  # Convert to 1-indexed
            end_page = deed.end_page + 1      # Convert to 1-indexed
            print(f"   Deed {i+1}: Pages {start_page}-{end_page} (Confidence: {deed.confidence:.3f})")
        
        # Create individual deed PDFs
        deed_pdfs = []
        doc = fitz.open(pdf_path)
        
        for i, deed in enumerate(result.deeds):
            # Create new PDF with deed pages
            new_doc = fitz.open()
            
            # Handle both old format (start_page, end_page) and new format (pages list)
            if hasattr(deed, 'start_page') and hasattr(deed, 'end_page'):
                # Old format
                for page_num in range(deed.start_page, deed.end_page + 1):
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            elif hasattr(deed, 'pages'):
                # New format
                for page_num in deed.pages:
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            else:
                # Fallback - use first page only
                new_doc.insert_pdf(doc, from_page=0, to_page=0)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_deed_{i+1}.pdf")
            new_doc.save(temp_file.name)
            new_doc.close()
            
            deed_pdfs.append(temp_file.name)
            print(f"✅ Created deed {i+1} PDF: {temp_file.name}")
        
        doc.close()
        return deed_pdfs
    

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

    def process_multi_deed_document(self, pdf_path: str, strategy: str = "document_ai") -> List[Dict]:
        """Process PDF with multiple deeds using memory-efficient processing"""
        
        print(f"🔧 Starting multi-deed processing with strategy: {strategy}")
        
        # Initialize tracking variables
        self._last_split_result = None
        deed_boundaries = []
        
        # Get document info for tracking
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Create tracking session
        session_id = self.deed_tracker.create_session(
            original_filename=os.path.basename(pdf_path),
            total_pages=total_pages,
            splitting_strategy=strategy,
            document_ai_used=(strategy == "document_ai" and self.document_ai_service is not None)
        )
        
        # 1. Split PDF into individual deed PDFs based on strategy
        try:
            print(f"🔧 Attempting to split PDF with strategy: {strategy}")
            deed_pdfs = self.split_pdf_by_deeds(pdf_path, strategy)
            print(f"📄 Split into {len(deed_pdfs)} deed files")
        except Exception as e:
            print(f"⚠️ Document AI splitting failed: {e}")
            print("🔄 Falling back to simple page-based splitting")
            deed_pdfs = self.split_pdf_by_deeds(pdf_path, "simple")
            print(f"📄 Fallback split into {len(deed_pdfs)} deed files")
        
        # 2. Extract deed boundary information if available
        if hasattr(self, '_last_split_result') and self._last_split_result:
            deed_boundaries = []
            for deed in self._last_split_result.deeds:
                # Handle both old format (start_page, end_page) and new format (pages list)
                if hasattr(deed, 'start_page') and hasattr(deed, 'end_page'):
                    # Old format
                    pages = list(range(deed.start_page, deed.end_page + 1))
                    page_range = f"{deed.start_page+1}-{deed.end_page+1}"
                elif hasattr(deed, 'pages'):
                    # New format
                    pages = deed.pages
                    if pages:
                        page_range = f"{min(pages)+1}-{max(pages)+1}"
                    else:
                        page_range = "0-0"
                else:
                    # Fallback
                    pages = [0]
                    page_range = "1-1"
                
                deed_boundaries.append({
                    'deed_number': deed.deed_number,
                    'pages': pages,
                    'confidence': deed.confidence,
                    'page_range': page_range
                })
            print(f"📊 Deed boundaries tracked: {len(deed_boundaries)} deeds")
            
            # Save deed boundaries to tracker
            self.deed_tracker.add_deed_boundaries(session_id, deed_boundaries)
        
        try:
            # 3. Process each deed separately using single-deed classification
            results = []
            for i, deed_pdf_path in enumerate(deed_pdfs):
                print(f"Processing deed {i+1}/{len(deed_pdfs)}...")
                try:
                    # #region agent log
                    with open('/Users/lauragomez/Desktop/mineral-rights/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'B',
                            'location': 'document_classifier.py:842',
                            'message': 'Starting deed processing - logging input file',
                            'data': {
                                'deed_number': i + 1,
                                'deed_pdf_path': deed_pdf_path,
                                'deed_pdf_exists': os.path.exists(deed_pdf_path)
                            },
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                    # #endregion
                    
                    # Use regular single-deed processing for each deed
                    result = self.process_document(
                        deed_pdf_path,
                        max_samples=6,  # Fewer samples for speed
                        confidence_threshold=0.7,
                        page_strategy="first_few",  # Process first few pages of each deed
                        high_recall_mode=True
                    )
                    result['deed_number'] = i + 1
                    result['deed_file'] = deed_pdf_path
                    result['pages_in_deed'] = result.get('pages_processed', 0)
                    
                    # Add deed boundary information if available
                    if i < len(deed_boundaries):
                        result['deed_boundary_info'] = deed_boundaries[i]
                        result['splitting_confidence'] = deed_boundaries[i]['confidence']
                    
                    # #region agent log
                    with open('/Users/lauragomez/Desktop/mineral-rights/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'A',
                            'location': 'document_classifier.py:859',
                            'message': 'Deed processing completed - checking detailed_samples',
                            'data': {
                                'deed_number': i + 1,
                                'has_detailed_samples': 'detailed_samples' in result,
                                'detailed_samples_count': len(result.get('detailed_samples', [])),
                                'first_sample_reasoning_preview': result.get('detailed_samples', [{}])[0].get('reasoning', '')[:100] if result.get('detailed_samples') else 'N/A',
                                'classification': result.get('classification'),
                                'confidence': result.get('confidence')
                            },
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                    # #endregion
                    
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
                    # Add boundary info if available
                    if i < len(deed_boundaries):
                        error_result['deed_boundary_info'] = deed_boundaries[i]
                        error_result['splitting_confidence'] = deed_boundaries[i]['confidence']
                    
                    results.append(error_result)
            
            # 4. Save classification results to tracker
            self.deed_tracker.add_classification_results(session_id, results)
            
            # 5. Finalize session and get summary
            summary = self.deed_tracker.finalize_session(session_id)
            
            # 6. Add summary information to results
            for result in results:
                result['session_id'] = session_id
                result['tracking_summary'] = summary
            
            print(f"🎯 Multi-deed processing completed: {len(results)} deeds processed")
            print(f"📊 Session {session_id} summary: {summary}")
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
        
        # Retry logic for network issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.classifier.client.messages.create(
                    model=self.classifier.model_name,  # Use configurable model name
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
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"❌ Anthropic API error during OCR (attempt {attempt + 1}): {error_type}: {error_msg}")
                # Check for common API key errors
                if "401" in error_msg or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                    print("🔑 API KEY ERROR DETECTED - Authentication failed. Please check your ANTHROPIC_API_KEY.")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise Exception(f"OCR failed after {max_retries} attempts: {error_type}: {error_msg}")
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                import traceback
                print(f"❌ Unexpected error during OCR (attempt {attempt + 1}): {error_type}: {error_msg}")
                print(f"📋 Full error details:")
                traceback.print_exc()
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise Exception(f"OCR failed after {max_retries} attempts: {error_type}: {error_msg}")
        
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
        # Debug logging to see what parameters are received
        print(f"🔍 DEBUG process_document called with:")
        print(f"🔍   pdf_path = '{pdf_path}'")
        print(f"🔍   page_strategy = '{page_strategy}'")
        print(f"🔍   max_samples = {max_samples}")
        print(f"🔍   high_recall_mode = {high_recall_mode}")
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
        
        # #region agent log
        import json
        import time
        with open('/Users/lauragomez/Desktop/mineral-rights/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'E',
                'location': 'document_classifier.py:1142',
                'message': 'About to classify document - logging OCR text preview',
                'data': {
                    'ocr_text_length': len(ocr_text),
                    'ocr_text_preview': ocr_text[:300],
                    'ocr_text_hash': hash(ocr_text[:1000])  # Hash to detect duplicates
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        
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
        
        # #region agent log
        import json
        import time
        with open('/Users/lauragomez/Desktop/mineral-rights/.cursor/debug.log', 'a') as f:
            reasoning_previews = [s.reasoning[:100] for s in classification_result.all_samples[:3]]
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'C',
                'location': 'document_classifier.py:1169',
                'message': 'process_document returning - logging reasoning samples',
                'data': {
                    'samples_count': len(classification_result.all_samples),
                    'first_3_reasoning_previews': reasoning_previews,
                    'ocr_text_length': len(ocr_text),
                    'ocr_text_preview': ocr_text[:200]
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
    
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
                'all_samples': [                          #  ← NEW: Store all samples for detailed_samples
                    {
                        'predicted_class': s.predicted_class,
                        'reasoning': s.reasoning,
                        'confidence_score': s.confidence_score,
                        'features': s.features
                    }
                    for s in classification_result.all_samples
                ]
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
            # Get detailed_samples from the last chunk's stored all_samples
            last_chunk = successful_chunks[-1]
            detailed_samples_list = last_chunk.get('all_samples', [])
            # Fallback: if all_samples not stored, create from reasoning
            if not detailed_samples_list:
                detailed_samples_list = [{
                    'predicted_class': last_chunk.get('classification', 0),
                    'reasoning': last_chunk.get('reasoning', '') or "No reasoning provided",
                    'confidence_score': last_chunk.get('confidence', 0.0),
                    'features': {}
                }]
        else:                          # every page failed OCR
            final_result = {
                'classification': 0,
                'confidence': 0.0,
                'votes': {0: 0.0, 1: 0.0},
                'samples_used': 0,
                'early_stopped': False
            }
            detailed_samples_list = []
        
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
            'detailed_samples': detailed_samples_list,
            'ocr_failed_pages': unread_pages,
            'requires_manual_review': len(unread_pages) > 0,
        }

    def process_document_page_by_page(self, pdf_path: str, max_samples: int = 6, 
                                    confidence_threshold: float = 0.7,
                                    max_tokens_per_page: int = 8000, 
                                    high_recall_mode: bool = True) -> Dict:
        """
        Process document page by page, treating each page as a separate deed.
        Returns which pages contain mineral rights reservations.
        """
        import time
        import gc
        import psutil
        import os
        
        print("Using page-by-page classification (treating each page as a deed)")
        print(f"📊 Max samples per page: {max_samples}")
        
        # Memory monitoring
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
        
        # Open PDF and get total pages
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"Document has {total_pages} pages")
        print(f"⏰ Session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Process each page individually
        page_results = []
        pages_with_reservations = []
        start_time = time.time()
        
        for page_num in range(total_pages):
            page_start_time = time.time()
            current_page = page_num + 1
            
            print(f"\n--- PROCESSING PAGE {current_page}/{total_pages} ---")
            
            try:
                # Convert page to image
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2, 2)  # 2x zoom for quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                # Clear page objects immediately
                page = None
                pix = None
                del page, pix
                
                # Extract text from current page
                print(f"Extracting text from page {current_page}...")
                page_text = self.extract_text_with_claude(image, max_tokens_per_page)
                print(f"Extracted {len(page_text)} characters from page {current_page}")
                
                # Clear image immediately
                image.close()
                image = None
                del image, img_data
                
                # Classify current page as if it were a deed
                print(f"Classifying page {current_page} for oil and gas reservations...")
                classification_result = self.classifier.classify_document(
                    page_text, max_samples, confidence_threshold, high_recall_mode
                )
                
                # Track page processing time
                page_time = time.time() - page_start_time
                
                # Get reasoning from first sample
                first_reasoning = (
                    classification_result.all_samples[0].reasoning
                    if classification_result.all_samples else "No reasoning available"
                )
                
                page_result = {
                    'page_number': current_page,
                    'text_length': len(page_text),
                    'classification': classification_result.predicted_class,
                    'confidence': classification_result.confidence,
                    'votes': classification_result.votes,
                    'samples_used': classification_result.samples_used,
                    'early_stopped': classification_result.early_stopped,
                    'page_text': page_text,
                    'reasoning': first_reasoning,
                    'processing_time': page_time,
                    'has_reservations': classification_result.predicted_class == 1
                }
                
                page_results.append(page_result)
                
                # Track pages with reservations
                if classification_result.predicted_class == 1:
                    pages_with_reservations.append(current_page)
                    print(f"🎯 PAGE {current_page}: HAS OIL AND GAS RESERVATIONS (confidence: {classification_result.confidence:.3f})")
                else:
                    print(f"📄 PAGE {current_page}: No oil and gas reservations (confidence: {classification_result.confidence:.3f})")
                
                print(f"Page {current_page} processed in {page_time:.1f} seconds")
                
                # Clear page text to save memory
                page_text = None
                del page_text
                
                # Memory management every 10 pages
                if current_page % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    print(f"💾 Memory after page {current_page}: {current_memory:.1f} MB (change: {memory_increase:+.1f} MB)")
                    
                    # Force garbage collection if memory increase is significant
                    if memory_increase > 200:  # More than 200MB increase
                        print("🧹 Running garbage collection...")
                        gc.collect()
                        after_gc_memory = process.memory_info().rss / 1024 / 1024
                        gc_reduction = current_memory - after_gc_memory
                        print(f"🧹 GC freed: {gc_reduction:.1f} MB")
                
                # Progress update every 5 pages
                if current_page % 5 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_page = elapsed_time / current_page
                    remaining_pages = total_pages - current_page
                    estimated_remaining_time = avg_time_per_page * remaining_pages
                    
                    hours = int(elapsed_time // 3600)
                    minutes = int((elapsed_time % 3600) // 60)
                    
                    print(f"📊 Progress: {current_page}/{total_pages} pages processed ({current_page/total_pages*100:.1f}%)")
                    print(f"⏱️  Elapsed: {hours}h {minutes}m | Est. remaining: {estimated_remaining_time/60:.1f} minutes")
                    print(f"📈 Avg time per page: {avg_time_per_page:.1f} seconds")
                    print(f"🎯 Pages with reservations so far: {len(pages_with_reservations)}")
                
            except Exception as e:
                print(f"❌ Error processing page {current_page}: {e}")
                # Add error result
                error_result = {
                    'page_number': current_page,
                    'text_length': 0,
                    'classification': 0,
                    'confidence': 0.0,
                    'votes': {0: 0.0, 1: 0.0},
                    'samples_used': 0,
                    'early_stopped': False,
                    'page_text': f"[ERROR: Could not process page {current_page}]",
                    'reasoning': f"Error: {str(e)}",
                    'processing_time': 0.0,
                    'has_reservations': False,
                    'error': str(e)
                }
                page_results.append(error_result)
                continue
        
        # Close document
        doc.close()
        
        # Calculate final statistics
        total_processing_time = time.time() - start_time
        successful_pages = [p for p in page_results if 'error' not in p]
        failed_pages = [p for p in page_results if 'error' in p]
        
        # Overall classification: 1 if ANY page has reservations, 0 otherwise
        overall_classification = 1 if pages_with_reservations else 0
        
        # Overall confidence: average of all page confidences
        page_confidences = [p['confidence'] for p in successful_pages if 'confidence' in p]
        overall_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_change = final_memory - initial_memory
        print(f"💾 Final memory: {final_memory:.1f} MB (change: {memory_change:+.1f} MB)")
        
        # Final garbage collection
        print("🧹 Running final garbage collection...")
        gc.collect()
        
        print(f"\n✅ PAGE-BY-PAGE ANALYSIS COMPLETE:")
        print(f"📊 Total pages processed: {total_pages}")
        print(f"🎯 Pages with oil and gas reservations: {len(pages_with_reservations)}")
        print(f"📄 Pages without reservations: {total_pages - len(pages_with_reservations)}")
        print(f"❌ Pages with errors: {len(failed_pages)}")
        print(f"⏱️  Total processing time: {total_processing_time/60:.1f} minutes")
        
        if pages_with_reservations:
            print(f"🎯 RESERVATIONS FOUND ON PAGES: {', '.join(map(str, pages_with_reservations))}")
        else:
            print("📄 NO OIL AND GAS RESERVATIONS FOUND ON ANY PAGE")
        
        return {
            'document_path': pdf_path,
            'processing_mode': 'page_by_page',
            'total_pages': total_pages,
            'pages_processed': len(successful_pages),
            'pages_with_errors': len(failed_pages),
            'classification': overall_classification,
            'confidence': overall_confidence,
            'pages_with_reservations': pages_with_reservations,
            'total_pages_with_reservations': len(pages_with_reservations),
            'page_results': page_results,
            'total_processing_time': total_processing_time,
            'average_time_per_page': total_processing_time / total_pages if total_pages > 0 else 0,
            'memory_usage_mb': final_memory,
            'memory_change_mb': memory_change,
            'high_recall_mode': high_recall_mode,
            'max_samples_per_page': max_samples,
            'summary': {
                'has_reservations': len(pages_with_reservations) > 0,
                'reservation_pages': pages_with_reservations,
                'total_pages': total_pages,
                'success_rate': len(successful_pages) / total_pages if total_pages > 0 else 0
            }
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

    def process_document_from_gcs(self, gcs_url: str, processing_mode: str = "single_deed", splitting_strategy: str = "document_ai"):
        """Process a document from a GCS URL"""
        try:
            # Download file from GCS
            from google.cloud import storage
            import tempfile
            
            # Initialize GCS client
            client = storage.Client()
            
            # Parse GCS URL
            bucket_name = gcs_url.split('/')[3]
            blob_name = '/'.join(gcs_url.split('/')[4:])
            
            # Download to temp file
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                blob.download_to_filename(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            # Process the file
            result = self.process_document(tmp_file_path, processing_mode, splitting_strategy)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing document from GCS: {e}")
            raise

    def process_large_document_chunked(self, gcs_url: str, processing_mode: str = "single_deed", splitting_strategy: str = "document_ai"):
        """Process large documents using Document AI smart chunking approach"""
        try:
            print(f"🚀 Processing large document with Document AI smart chunking...")
            print(f"🔧 GCS URL: {gcs_url}")
            print(f"🔧 Processing mode: {processing_mode}")
            print(f"🔧 Splitting strategy: {splitting_strategy}")
            
            # Download file from GCS
            from google.cloud import storage
            import tempfile
            
            # Initialize GCS client with credentials
            try:
                # Try to use base64 encoded credentials first
                credentials_b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
                if credentials_b64:
                    import base64
                    import json
                    from google.oauth2 import service_account
                    
                    # Decode the base64 credentials
                    credentials_json = base64.b64decode(credentials_b64).decode('utf-8')
                    credentials_info = json.loads(credentials_json)
                    
                    # Create credentials object
                    credentials = service_account.Credentials.from_service_account_info(credentials_info)
                    client = storage.Client(credentials=credentials)
                    print("✅ Using base64 encoded service account credentials for GCS")
                else:
                    # Fallback to default credentials
                    client = storage.Client()
                    print("✅ Using default service account credentials for GCS")
            except Exception as e:
                print(f"❌ GCS client initialization failed: {e}")
                raise
            
            # Parse GCS URL
            bucket_name = gcs_url.split('/')[3]
            blob_name = '/'.join(gcs_url.split('/')[4:])
            print(f"🔧 Bucket: {bucket_name}, Blob: {blob_name}")
            
            # Download to temp file
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            print(f"🔧 Downloading file from GCS...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                blob.download_to_filename(tmp_file.name)
                tmp_file_path = tmp_file.name
                print(f"✅ File downloaded to: {tmp_file_path}")
            
            # Get PDF info
            doc = fitz.open(tmp_file_path)
            total_pages = len(doc)
            doc.close()
            
            print(f"📄 Large PDF detected: {total_pages} pages")
            print(f"🔧 Using Document AI smart chunking for deed detection")
            
            # Use Document AI smart chunking service for deed detection
            print(f"🔍 DEBUG: document_ai_service = {self.document_ai_service is not None}")
            print(f"🔍 DEBUG: splitting_strategy = {splitting_strategy}")
            if self.document_ai_service:
                print(f"🔍 DEBUG: smart_chunking_service = {self.document_ai_service.smart_chunking_service}")
            else:
                print(f"🔍 DEBUG: document_ai_service is None - will use fallback")
            
            if self.document_ai_service and splitting_strategy == "document_ai":
                print("📡 Using Document AI smart chunking service...")
                try:
                    # Use smart chunking service to process the entire PDF
                    smart_result = self.document_ai_service.smart_chunking_service.process_pdf(tmp_file_path)
                    
                    print(f"📊 Smart chunking results:")
                    print(f"   - Total deeds detected: {smart_result.total_deeds}")
                    print(f"   - Chunks processed: {smart_result.chunks_processed}")
                    print(f"   - Processing time: {smart_result.processing_time:.2f}s")
                    
                    # Convert smart chunking results to our format
                    deeds = []
                    for i, deed in enumerate(smart_result.deeds):
                        deed_result = {
                            'deed_number': i + 1,
                            'classification': 0,  # Will be determined by processing
                            'confidence': 0.0,
                            'reasoning': 'Deed detected by Document AI',
                            'pages': deed.pages,
                            'pages_in_deed': len(deed.pages),
                            'deed_boundary_info': {
                                'deed_number': i + 1,
                                'pages': deed.pages,
                                'confidence': deed.confidence,
                                'page_range': f"{min(deed.pages)+1}-{max(deed.pages)+1}" if deed.pages else "0-0"
                            }
                        }
                        deeds.append(deed_result)
                    
                    # Clean up original temp file
                    os.unlink(tmp_file_path)
                    
                    result = {
                        "deeds": deeds,
                        "total_pages": total_pages,
                        "chunks_processed": smart_result.chunks_processed,
                        "processing_method": "document_ai_smart_chunking",
                        "message": f"Very large file processed using Document AI smart chunking. {smart_result.chunks_processed} chunks processed, {len(deeds)} deeds detected."
                    }
                    
                    print(f"✅ Document AI smart chunking completed: {len(deeds)} deeds found")
                    return result
                    
                except Exception as e:
                    print(f"⚠️ Document AI smart chunking failed: {e}")
                    print(f"🔍 DEBUG: Exception details: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    print("🔄 Falling back to regular multi-deed processing...")
            
            # Fallback to regular multi-deed processing
            print("🔄 Using regular multi-deed processing as fallback...")
            print(f"🔍 DEBUG: Processing mode = {processing_mode}")
            print(f"🔍 DEBUG: Splitting strategy = {splitting_strategy}")
            try:
                if processing_mode == "multi_deed":
                    results = self.process_multi_deed_document(tmp_file_path, splitting_strategy)
                else:
                    # For single deed mode, process the entire document
                    result = self.process_document(tmp_file_path)
                    results = [result] if isinstance(result, dict) else result
                
                # Clean up original temp file
                os.unlink(tmp_file_path)
                
                # Convert results to expected format
                deeds = []
                if isinstance(results, list):
                    for i, result in enumerate(results):
                        if isinstance(result, dict):
                            deed_result = {
                                'deed_number': result.get('deed_number', i + 1),
                                'classification': result.get('classification', 0),
                                'confidence': result.get('confidence', 0.0),
                                'reasoning': result.get('reasoning', 'No reasoning provided'),
                                'pages': result.get('pages', []),
                                'pages_in_deed': result.get('pages_in_deed', 0),
                                'deed_boundary_info': result.get('deed_boundary_info', {})
                            }
                            deeds.append(deed_result)
                elif isinstance(results, dict):
                    deed_result = {
                        'deed_number': 1,
                        'classification': results.get('classification', 0),
                        'confidence': results.get('confidence', 0.0),
                        'reasoning': results.get('reasoning', 'No reasoning provided'),
                        'pages': results.get('pages', []),
                        'pages_in_deed': results.get('pages_in_deed', 0),
                        'deed_boundary_info': results.get('deed_boundary_info', {})
                    }
                    deeds.append(deed_result)
                
                result = {
                    "deeds": deeds,
                    "total_pages": total_pages,
                    "chunks_processed": 1,  # Single processing
                    "processing_method": "fallback_processing",
                    "message": f"Very large file processed using fallback approach. {len(deeds)} deeds found."
                }
                
                print(f"🔍 DEBUG: Fallback result = {result}")
                
                print(f"✅ Fallback processing completed: {len(deeds)} deeds found")
                print(f"🔍 DEBUG: Final result = {result}")
                return result
                
            except Exception as e:
                print(f"❌ Fallback processing also failed: {e}")
                # Clean up original temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                raise
            
        except Exception as e:
            print(f"❌ Error in chunked processing: {e}")
            raise

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
