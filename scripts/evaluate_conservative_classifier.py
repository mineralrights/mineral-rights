#!/usr/bin/env python3
"""
Run the classifier in CONSERVATIVE mode
(bias toward ‘no reservations’, minimum false positives)
"""
import os, time, json
import sys
from pathlib import Path

# add "<repo_root>/src" to sys.path at runtime
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mineral_rights.document_classifier import DocumentProcessor

proc = DocumentProcessor(os.getenv("ANTHROPIC_API_KEY"))

# folders to scan
folders = [Path("data/reservs"), Path("data/no-reservs")]

results = []
for folder in folders:
    for pdf in folder.glob("*.pdf"):
        print(f"\n🔍 {pdf.name}")
        txt = proc.extract_text_from_multiple_pages(        # OCR every page
                  proc.pdf_to_images(str(pdf)),              #   (no early stop)
                  max_tokens_per_page=8000,
                  combine_method="concatenate")
        res = proc.classifier.classify_document(
                  txt,
                  max_samples= 6,             # keep it small for speed
                  confidence_threshold=0.8,  # stricter
                  high_recall_mode=False)    # ← THE IMPORTANT BIT
        results.append({
            "file": pdf.name,
            "folder": folder.name,
            "prediction": res.predicted_class,
            "confidence": res.confidence})
        print(f" → class={res.predicted_class}  conf={res.confidence:.3f}")

# dump a quick JSON log
stamp = time.strftime("%Y%m%d_%H%M%S")
Path("evaluation_results").mkdir(exist_ok=True)
with open(f"evaluation_results/conservative_run_{stamp}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ finished – results saved to evaluation_results/conservative_run_{stamp}.json")
