#!/usr/bin/env python3
"""
Quick Dataset Generation Runner

This script provides a simple interface to generate the synthetic dataset
with reasonable defaults for testing and production use.

Usage:
    python scripts/run_dataset_generation.py --quick    # Generate small test dataset
    python scripts/run_dataset_generation.py --full     # Generate full dataset
    python scripts/run_dataset_generation.py --custom 80 20  # Custom train/test split
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-deed dataset")
    parser.add_argument("--quick", action="store_true",
                       help="Generate a small test dataset (10 train, 5 test)")
    parser.add_argument("--full", action="store_true",
                       help="Generate a full dataset (100 train, 25 test)")
    parser.add_argument("--custom", nargs=2, type=int, metavar=("TRAIN", "TEST"),
                       help="Generate custom dataset with specified train/test counts")
    parser.add_argument("--output_dir", default="data/synthetic_dataset",
                       help="Output directory for the dataset")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the generated dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Determine dataset size
    if args.quick:
        num_train, num_test = 10, 5
        print("🚀 Generating quick test dataset...")
    elif args.full:
        num_train, num_test = 100, 25
        print("🚀 Generating full dataset...")
    elif args.custom:
        num_train, num_test = args.custom
        print(f"🚀 Generating custom dataset ({num_train} train, {num_test} test)...")
    else:
        print("Please specify --quick, --full, or --custom")
        return 1
    
    # Check if source data exists
    if not Path("data/no-reservs").exists() or not Path("data/reservs").exists():
        print("❌ Error: Source data directories not found!")
        print("Please ensure data/no-reservs and data/reservs directories exist")
        return 1
    
    # Generate dataset
    cmd = [
        sys.executable, "scripts/generate_synthetic_dataset.py",
        "--output_dir", args.output_dir,
        "--num_train", str(num_train),
        "--num_test", str(num_test),
        "--seed", str(args.seed)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    if not run_command(cmd):
        return 1
    
    # Validate dataset if requested
    if args.validate:
        print("\n🔍 Validating generated dataset...")
        validate_cmd = [
            sys.executable, "scripts/validate_synthetic_dataset.py",
            "--dataset_dir", args.output_dir
        ]
        
        if not run_command(validate_cmd):
            print("⚠️  Dataset validation failed, but generation completed")
            return 1
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Dataset location: {args.output_dir}")
    print(f"📊 Generated {num_train} training documents and {num_test} test documents")
    
    print("\n📋 Next steps:")
    print("1. Review the generated PDFs in the train/pdfs and test/pdfs directories")
    print("2. Check the JSON labels in the train/labels and test/labels directories")
    print("3. Upload to Google Cloud Document AI for training")
    print("4. Use the test set for model evaluation")
    
    return 0


if __name__ == "__main__":
    exit(main())
