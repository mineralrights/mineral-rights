#!/usr/bin/env python3
"""
Dataset Summary Generator

This script provides a comprehensive summary of the generated synthetic dataset,
including statistics, sample analysis, and recommendations for Google Cloud Document AI.

Usage:
    python scripts/dataset_summary.py --dataset_dir data/synthetic_dataset
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import fitz  # PyMuPDF


class DatasetSummarizer:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.stats = {
            "train": {"docs": 0, "pages": 0, "deeds": 0, "reservations": 0},
            "test": {"docs": 0, "pages": 0, "deeds": 0, "reservations": 0}
        }
        self.sample_analysis = []
    
    def analyze_document(self, pdf_path: Path, label_path: Path, split: str) -> Dict[str, Any]:
        """Analyze a single document and its label."""
        try:
            # Read PDF info
            doc = fitz.open(str(pdf_path))
            actual_pages = len(doc)
            doc.close()
            
            # Read label
            with open(label_path, 'r') as f:
                label = json.load(f)
            
            # Extract statistics
            doc_stats = {
                "doc_id": label["doc_id"],
                "split": split,
                "actual_pages": actual_pages,
                "label_pages": label["attributes"]["total_pages"],
                "deed_count": label["deed_count"],
                "has_reservations": label["attributes"]["has_oil_gas_reservations"],
                "reservation_count": label["attributes"]["reservation_count"],
                "page_consistency": actual_pages == label["attributes"]["total_pages"]
            }
            
            return doc_stats
        except Exception as e:
            print(f"Error analyzing {pdf_path}: {e}")
            return None
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive dataset summary."""
        print("📊 Analyzing synthetic dataset...")
        
        # Analyze train split
        train_pdfs = list((self.dataset_dir / "train" / "pdfs").glob("*.pdf"))
        train_labels = list((self.dataset_dir / "train" / "labels").glob("*.json"))
        
        for pdf_file in train_pdfs:
            label_file = self.dataset_dir / "train" / "labels" / f"{pdf_file.stem}.json"
            if label_file.exists():
                doc_stats = self.analyze_document(pdf_file, label_file, "train")
                if doc_stats:
                    self.sample_analysis.append(doc_stats)
                    self.stats["train"]["docs"] += 1
                    self.stats["train"]["pages"] += doc_stats["actual_pages"]
                    self.stats["train"]["deeds"] += doc_stats["deed_count"]
                    if doc_stats["has_reservations"]:
                        self.stats["train"]["reservations"] += 1
        
        # Analyze test split
        test_pdfs = list((self.dataset_dir / "test" / "pdfs").glob("*.pdf"))
        test_labels = list((self.dataset_dir / "test" / "labels").glob("*.json"))
        
        for pdf_file in test_pdfs:
            label_file = self.dataset_dir / "test" / "labels" / f"{pdf_file.stem}.json"
            if label_file.exists():
                doc_stats = self.analyze_document(pdf_file, label_file, "test")
                if doc_stats:
                    self.sample_analysis.append(doc_stats)
                    self.stats["test"]["docs"] += 1
                    self.stats["test"]["pages"] += doc_stats["actual_pages"]
                    self.stats["test"]["deeds"] += doc_stats["deed_count"]
                    if doc_stats["has_reservations"]:
                        self.stats["test"]["reservations"] += 1
        
        # Calculate totals and averages
        total_docs = self.stats["train"]["docs"] + self.stats["test"]["docs"]
        total_pages = self.stats["train"]["pages"] + self.stats["test"]["pages"]
        total_deeds = self.stats["train"]["deeds"] + self.stats["test"]["deeds"]
        total_reservations = self.stats["train"]["reservations"] + self.stats["test"]["reservations"]
        
        summary = {
            "dataset_overview": {
                "total_documents": total_docs,
                "total_pages": total_pages,
                "total_deeds": total_deeds,
                "documents_with_reservations": total_reservations,
                "average_deeds_per_document": total_deeds / total_docs if total_docs > 0 else 0,
                "average_pages_per_document": total_pages / total_docs if total_docs > 0 else 0,
                "reservation_rate": total_reservations / total_docs if total_docs > 0 else 0
            },
            "train_split": {
                "documents": self.stats["train"]["docs"],
                "pages": self.stats["train"]["pages"],
                "deeds": self.stats["train"]["deeds"],
                "documents_with_reservations": self.stats["train"]["reservations"],
                "average_deeds_per_document": self.stats["train"]["deeds"] / self.stats["train"]["docs"] if self.stats["train"]["docs"] > 0 else 0,
                "average_pages_per_document": self.stats["train"]["pages"] / self.stats["train"]["docs"] if self.stats["train"]["docs"] > 0 else 0
            },
            "test_split": {
                "documents": self.stats["test"]["docs"],
                "pages": self.stats["test"]["pages"],
                "deeds": self.stats["test"]["deeds"],
                "documents_with_reservations": self.stats["test"]["reservations"],
                "average_deeds_per_document": self.stats["test"]["deeds"] / self.stats["test"]["docs"] if self.stats["test"]["docs"] > 0 else 0,
                "average_pages_per_document": self.stats["test"]["pages"] / self.stats["test"]["docs"] if self.stats["test"]["docs"] > 0 else 0
            },
            "quality_metrics": {
                "page_consistency_rate": sum(1 for doc in self.sample_analysis if doc["page_consistency"]) / len(self.sample_analysis) if self.sample_analysis else 0,
                "label_completeness": len(self.sample_analysis) / total_docs if total_docs > 0 else 0
            }
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary to console."""
        print("\n" + "="*60)
        print("🎯 SYNTHETIC MULTI-DEED DATASET SUMMARY")
        print("="*60)
        
        # Overview
        overview = summary["dataset_overview"]
        print(f"\n📈 DATASET OVERVIEW")
        print(f"   Total Documents: {overview['total_documents']}")
        print(f"   Total Pages: {overview['total_pages']:,}")
        print(f"   Total Deeds: {overview['total_deeds']}")
        print(f"   Documents with Reservations: {overview['documents_with_reservations']} ({overview['reservation_rate']:.1%})")
        print(f"   Average Deeds per Document: {overview['average_deeds_per_document']:.1f}")
        print(f"   Average Pages per Document: {overview['average_pages_per_document']:.1f}")
        
        # Train split
        train = summary["train_split"]
        print(f"\n🚂 TRAINING SPLIT")
        print(f"   Documents: {train['documents']}")
        print(f"   Pages: {train['pages']:,}")
        print(f"   Deeds: {train['deeds']}")
        print(f"   Documents with Reservations: {train['documents_with_reservations']}")
        print(f"   Average Deeds per Document: {train['average_deeds_per_document']:.1f}")
        print(f"   Average Pages per Document: {train['average_pages_per_document']:.1f}")
        
        # Test split
        test = summary["test_split"]
        print(f"\n🧪 TEST SPLIT")
        print(f"   Documents: {test['documents']}")
        print(f"   Pages: {test['pages']:,}")
        print(f"   Deeds: {test['deeds']}")
        print(f"   Documents with Reservations: {test['documents_with_reservations']}")
        print(f"   Average Deeds per Document: {test['average_deeds_per_document']:.1f}")
        print(f"   Average Pages per Document: {test['average_pages_per_document']:.1f}")
        
        # Quality metrics
        quality = summary["quality_metrics"]
        print(f"\n✅ QUALITY METRICS")
        print(f"   Page Consistency Rate: {quality['page_consistency_rate']:.1%}")
        print(f"   Label Completeness: {quality['label_completeness']:.1%}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR GOOGLE CLOUD DOCUMENT AI")
        print(f"   • Dataset size is {'adequate' if overview['total_documents'] >= 50 else 'small - consider generating more documents'} for training")
        print(f"   • {'Good' if overview['reservation_rate'] >= 0.3 else 'Consider increasing'} reservation rate for balanced training")
        print(f"   • Average {overview['average_deeds_per_document']:.1f} deeds per document provides {'good' if overview['average_deeds_per_document'] >= 5 else 'limited'} complexity")
        print(f"   • {'Excellent' if quality['page_consistency_rate'] >= 0.95 else 'Check'} page consistency for reliable training")
        
        print(f"\n📋 NEXT STEPS")
        print(f"   1. Upload PDFs and labels to Google Cloud Document AI")
        print(f"   2. Train custom model for deed boundary detection")
        print(f"   3. Evaluate model performance on test set")
        print(f"   4. Deploy for production use")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset summary")
    parser.add_argument("--dataset_dir", default="data/synthetic_dataset",
                       help="Path to the synthetic dataset directory")
    
    args = parser.parse_args()
    
    summarizer = DatasetSummarizer(args.dataset_dir)
    summary = summarizer.generate_summary()
    summarizer.print_summary(summary)
    
    # Save summary to file
    summary_file = Path(args.dataset_dir) / "detailed_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Detailed summary saved to: {summary_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
