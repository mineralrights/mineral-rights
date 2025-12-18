#!/usr/bin/env python3
"""
Synthetic Dataset Validator

This script validates the generated synthetic dataset to ensure:
1. All PDFs are readable and properly merged
2. JSON labels are valid and match the PDFs
3. Page boundaries are correct
4. Document AI format compliance

Usage:
    python scripts/validate_synthetic_dataset.py --dataset_dir data/synthetic_dataset
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import fitz  # PyMuPDF


class DatasetValidator:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.errors = []
        self.warnings = []
        self.stats = {
            "train_docs": 0,
            "test_docs": 0,
            "total_pages": 0,
            "total_deeds": 0,
            "docs_with_reservations": 0
        }
    
    def validate_pdf(self, pdf_path: Path) -> bool:
        """Validate that a PDF is readable and has content."""
        try:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            
            if page_count == 0:
                self.errors.append(f"PDF {pdf_path} has no pages")
                return False
            
            return True
        except Exception as e:
            self.errors.append(f"Cannot read PDF {pdf_path}: {e}")
            return False
    
    def validate_label(self, label_path: Path) -> Dict[str, Any]:
        """Validate JSON label format and content."""
        try:
            with open(label_path, 'r') as f:
                label = json.load(f)
        except Exception as e:
            self.errors.append(f"Cannot read label {label_path}: {e}")
            return {}
        
        # Check required fields
        required_fields = ["doc_id", "deed_count", "deeds", "attributes"]
        for field in required_fields:
            if field not in label:
                self.errors.append(f"Label {label_path} missing required field: {field}")
                return {}
        
        # Validate deeds structure
        if not isinstance(label["deeds"], list):
            self.errors.append(f"Label {label_path} deeds field is not a list")
            return {}
        
        if len(label["deeds"]) != label["deed_count"]:
            self.errors.append(f"Label {label_path} deed_count doesn't match deeds list length")
            return {}
        
        # Validate each deed
        for i, deed in enumerate(label["deeds"]):
            deed_required = ["index", "page_start", "page_end", "has_oil_gas_reservations"]
            for field in deed_required:
                if field not in deed:
                    self.errors.append(f"Label {label_path} deed {i} missing field: {field}")
        
        return label
    
    def validate_pdf_label_consistency(self, pdf_path: Path, label: Dict[str, Any]) -> bool:
        """Validate that PDF page count matches label expectations."""
        try:
            doc = fitz.open(str(pdf_path))
            actual_pages = len(doc)
            doc.close()
            
            # Get expected page count from label
            if not label.get("deeds"):
                return True
            
            last_deed = max(label["deeds"], key=lambda x: x["page_end"])
            expected_pages = last_deed["page_end"]
            
            if actual_pages != expected_pages:
                self.errors.append(
                    f"PDF {pdf_path} has {actual_pages} pages but label expects {expected_pages}"
                )
                return False
            
            return True
        except Exception as e:
            self.errors.append(f"Error validating PDF {pdf_path}: {e}")
            return False
    
    def validate_split(self, split: str) -> int:
        """Validate all documents in a split (train/test)."""
        split_dir = self.dataset_dir / split
        pdfs_dir = split_dir / "pdfs"
        labels_dir = split_dir / "labels"
        
        if not pdfs_dir.exists():
            self.errors.append(f"PDFs directory not found: {pdfs_dir}")
            return 0
        
        if not labels_dir.exists():
            self.errors.append(f"Labels directory not found: {labels_dir}")
            return 0
        
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        label_files = list(labels_dir.glob("*.json"))
        
        if len(pdf_files) != len(label_files):
            self.warnings.append(
                f"Split {split}: {len(pdf_files)} PDFs but {len(label_files)} labels"
            )
        
        valid_docs = 0
        for pdf_file in pdf_files:
            # Validate PDF
            if not self.validate_pdf(pdf_file):
                continue
            
            # Find corresponding label
            label_file = labels_dir / f"{pdf_file.stem}.json"
            if not label_file.exists():
                self.errors.append(f"No label found for PDF: {pdf_file}")
                continue
            
            # Validate label
            label = self.validate_label(label_file)
            if not label:
                continue
            
            # Validate consistency
            if not self.validate_pdf_label_consistency(pdf_file, label):
                continue
            
            # Update statistics
            valid_docs += 1
            self.stats[f"{split}_docs"] = self.stats.get(f"{split}_docs", 0) + 1
            self.stats["total_pages"] += label["attributes"]["total_pages"]
            self.stats["total_deeds"] += label["deed_count"]
            
            if label["attributes"]["has_oil_gas_reservations"]:
                self.stats["docs_with_reservations"] += 1
        
        return valid_docs
    
    def validate_dataset(self) -> bool:
        """Validate the entire dataset."""
        print("Validating synthetic dataset...")
        
        # Check dataset structure
        if not self.dataset_dir.exists():
            self.errors.append(f"Dataset directory not found: {self.dataset_dir}")
            return False
        
        # Validate train and test splits
        train_valid = self.validate_split("train")
        test_valid = self.validate_split("test")
        
        # Check for dataset summary
        summary_file = self.dataset_dir / "dataset_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                print(f"Dataset summary found: {summary['statistics']}")
            except Exception as e:
                self.warnings.append(f"Cannot read dataset summary: {e}")
        
        # Print results
        print(f"\nValidation Results:")
        print(f"Valid training documents: {train_valid}")
        print(f"Valid test documents: {test_valid}")
        print(f"Total valid documents: {train_valid + test_valid}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"  Total pages: {self.stats['total_pages']}")
        print(f"  Total deeds: {self.stats['total_deeds']}")
        print(f"  Documents with reservations: {self.stats['docs_with_reservations']}")
        print(f"  Average deeds per document: {self.stats['total_deeds'] / (train_valid + test_valid):.1f}")
        print(f"  Average pages per document: {self.stats['total_pages'] / (train_valid + test_valid):.1f}")
        
        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic multi-deed dataset")
    parser.add_argument("--dataset_dir", default="data/synthetic_dataset",
                       help="Path to the synthetic dataset directory")
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.dataset_dir)
    is_valid = validator.validate_dataset()
    
    if is_valid:
        print("\n✅ Dataset validation passed!")
        return 0
    else:
        print("\n❌ Dataset validation failed!")
        return 1


if __name__ == "__main__":
    exit(main())
