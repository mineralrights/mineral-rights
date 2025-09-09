#!/usr/bin/env python3
"""
Synthetic Multi-Deed Dataset Generator

This script creates a robust dataset for training deed detection models by:
1. Randomly sampling single deeds from no-reservs and reservs directories
2. Merging them into multi-deed PDFs with varying numbers of deeds
3. Generating Google Cloud Document AI compatible JSON labels
4. Creating train/test splits with known ground truth

Usage:
    python scripts/generate_synthetic_dataset.py --output_dir data/synthetic_dataset --num_documents 100
"""

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import fitz  # PyMuPDF
import hashlib


class SyntheticDatasetGenerator:
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        (self.output_dir / "train" / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "train" / "labels").mkdir(exist_ok=True)
        (self.output_dir / "test" / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "test" / "labels").mkdir(exist_ok=True)
        
        # Load single deed files
        self.no_reservs_files = list(Path("data/no-reservs").glob("*.pdf"))
        self.reservs_files = list(Path("data/reservs").glob("*.pdf"))
        self.all_single_deeds = self.no_reservs_files + self.reservs_files
        
        print(f"Loaded {len(self.no_reservs_files)} no-reservs deeds")
        print(f"Loaded {len(self.reservs_files)} reservs deeds")
        print(f"Total single deeds: {len(self.all_single_deeds)}")
    
    def get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF file."""
        try:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return 0
    
    def sample_deeds_for_document(self, min_deeds: int = 3, max_deeds: int = 15) -> List[Tuple[Path, int, bool]]:
        """
        Sample random deeds for a single multi-deed document.
        Returns list of (file_path, page_count, has_reservations)
        """
        num_deeds = random.randint(min_deeds, max_deeds)
        sampled_deeds = random.sample(self.all_single_deeds, num_deeds)
        
        deed_info = []
        for deed_path in sampled_deeds:
            page_count = self.get_pdf_page_count(deed_path)
            if page_count > 0:  # Only include valid PDFs
                has_reservations = deed_path in self.reservs_files
                deed_info.append((deed_path, page_count, has_reservations))
        
        return deed_info
    
    def merge_pdfs(self, deed_info: List[Tuple[Path, int, bool]], output_path: Path) -> bool:
        """Merge multiple PDFs into a single multi-deed document."""
        try:
            merged_doc = fitz.open()
            
            for deed_path, _, _ in deed_info:
                doc = fitz.open(str(deed_path))
                merged_doc.insert_pdf(doc)
                doc.close()
            
            merged_doc.save(str(output_path))
            merged_doc.close()
            return True
        except Exception as e:
            print(f"Error merging PDFs to {output_path}: {e}")
            return False
    
    def generate_document_ai_label(self, doc_id: str, deed_info: List[Tuple[Path, int, bool]], 
                                 output_pdf_path: Path) -> Dict[str, Any]:
        """
        Generate Google Cloud Document AI compatible label format.
        """
        current_page = 1
        deeds = []
        page_starts = []
        
        # Calculate page boundaries for each deed
        for i, (deed_path, page_count, has_reservations) in enumerate(deed_info):
            page_start = current_page
            page_end = current_page + page_count - 1
            
            deeds.append({
                "index": i + 1,
                "page_start": page_start,
                "page_end": page_end,
                "source_file": deed_path.name,
                "has_oil_gas_reservations": has_reservations,
                "page_count": page_count
            })
            
            page_starts.append(page_start)
            current_page = page_end + 1
        
        # Calculate document-level attributes
        total_pages = current_page - 1
        has_any_reservations = any(has_reservations for _, _, has_reservations in deed_info)
        reservation_count = sum(1 for _, _, has_reservations in deed_info if has_reservations)
        
        # Generate SHA256 hash of the merged PDF
        try:
            with open(output_pdf_path, 'rb') as f:
                pdf_hash = hashlib.sha256(f.read()).hexdigest()
        except:
            pdf_hash = "unknown"
        
        label = {
            "doc_id": doc_id,
            "source_pdf": str(output_pdf_path.relative_to(self.output_dir)),
            "attributes": {
                "has_oil_gas_reservations": has_any_reservations,
                "reservation_count": reservation_count,
                "total_deeds": len(deeds),
                "total_pages": total_pages
            },
            "deed_count": len(deeds),
            "page_starts": page_starts,
            "deeds": deeds,
            "validation": {
                "status": "synthetic",
                "issues": [],
                "n_starts_raw": len(page_starts),
                "n_starts_normalized": len(page_starts),
                "duplicates": {}
            },
            "qc": {
                "status": "synthetic",
                "issues": []
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "synthetic_dataset_generator",
                "seed": self.seed,
                "pdf_hash": pdf_hash
            }
        }
        
        return label
    
    def generate_document(self, doc_id: str, split: str = "train") -> bool:
        """Generate a single multi-deed document with its label."""
        # Sample deeds for this document
        deed_info = self.sample_deeds_for_document()
        if not deed_info:
            return False
        
        # Create output paths
        pdf_path = self.output_dir / split / "pdfs" / f"{doc_id}.pdf"
        label_path = self.output_dir / split / "labels" / f"{doc_id}.json"
        
        # Merge PDFs
        if not self.merge_pdfs(deed_info, pdf_path):
            return False
        
        # Generate label
        label = self.generate_document_ai_label(doc_id, deed_info, pdf_path)
        
        # Save label
        with open(label_path, 'w') as f:
            json.dump(label, f, indent=2)
        
        return True
    
    def generate_dataset(self, num_train: int = 80, num_test: int = 20):
        """Generate the complete synthetic dataset."""
        print(f"Generating {num_train} training documents and {num_test} test documents...")
        
        # Generate training documents
        train_success = 0
        for i in range(num_train):
            doc_id = f"synthetic_train_{i+1:03d}"
            if self.generate_document(doc_id, "train"):
                train_success += 1
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{num_train} training documents")
        
        # Generate test documents
        test_success = 0
        for i in range(num_test):
            doc_id = f"synthetic_test_{i+1:03d}"
            if self.generate_document(doc_id, "test"):
                test_success += 1
                if (i + 1) % 5 == 0:
                    print(f"Generated {i + 1}/{num_test} test documents")
        
        print(f"\nDataset generation complete!")
        print(f"Training documents: {train_success}/{num_train}")
        print(f"Test documents: {test_success}/{num_test}")
        
        # Generate dataset summary
        self.generate_dataset_summary(train_success, test_success)
    
    def generate_dataset_summary(self, train_count: int, test_count: int):
        """Generate a summary of the created dataset."""
        summary = {
            "dataset_info": {
                "name": "synthetic_multi_deed_dataset",
                "generated_at": datetime.now().isoformat(),
                "generator": "synthetic_dataset_generator",
                "seed": self.seed
            },
            "statistics": {
                "train_documents": train_count,
                "test_documents": test_count,
                "total_documents": train_count + test_count,
                "source_single_deeds": len(self.all_single_deeds),
                "no_reservs_source": len(self.no_reservs_files),
                "reservs_source": len(self.reservs_files)
            },
            "structure": {
                "train_pdfs": str(self.output_dir / "train" / "pdfs"),
                "train_labels": str(self.output_dir / "train" / "labels"),
                "test_pdfs": str(self.output_dir / "test" / "pdfs"),
                "test_labels": str(self.output_dir / "test" / "labels")
            },
            "label_format": {
                "compatible_with": "Google Cloud Document AI",
                "format_version": "1.0",
                "description": "Each label file contains deed boundaries, page numbers, and metadata"
            }
        }
        
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Dataset summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-deed dataset")
    parser.add_argument("--output_dir", default="data/synthetic_dataset", 
                       help="Output directory for the dataset")
    parser.add_argument("--num_train", type=int, default=80,
                       help="Number of training documents to generate")
    parser.add_argument("--num_test", type=int, default=20,
                       help="Number of test documents to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate input directories exist
    if not Path("data/no-reservs").exists():
        print("Error: data/no-reservs directory not found")
        return 1
    
    if not Path("data/reservs").exists():
        print("Error: data/reservs directory not found")
        return 1
    
    # Generate dataset
    generator = SyntheticDatasetGenerator(args.output_dir, args.seed)
    generator.generate_dataset(args.num_train, args.num_test)
    
    print(f"\nSynthetic dataset created successfully in: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the generated PDFs and labels")
    print("2. Upload to Google Cloud Document AI for training")
    print("3. Use the test set for model evaluation")
    
    return 0


if __name__ == "__main__":
    exit(main())
