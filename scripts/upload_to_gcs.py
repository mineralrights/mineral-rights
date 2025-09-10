#!/usr/bin/env python3
"""
Upload Synthetic Dataset to Google Cloud Storage

This script uploads the generated synthetic dataset to Google Cloud Storage
for training with Google Cloud Document AI, following the same pattern
as shown in your example.

Usage:
    python scripts/upload_to_gcs.py --dataset_dir data/synthetic_dataset --bucket my-deed-bucket
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class GCSUploader:
    def __init__(self, project_id: str, bucket: str, dataset_dir: Path):
        self.project_id = project_id
        self.bucket = bucket
        self.dataset_dir = Path(dataset_dir)
        
        # Set up environment
        os.environ["PROJECT_ID"] = project_id
        os.environ["BUCKET"] = bucket
        
        print(f"🚀 Setting up GCS upload for project: {project_id}")
        print(f"📦 Target bucket: {bucket}")
        print(f"📁 Dataset directory: {dataset_dir}")
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a gcloud command and return success status."""
        print(f"\n🔄 {description}")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error: {e}")
            if e.stderr:
                print(f"   Error output: {e.stderr}")
            return False
    
    def setup_gcloud(self) -> bool:
        """Set up gcloud configuration."""
        print("\n" + "="*60)
        print("🔧 SETTING UP GCLOUD CONFIGURATION")
        print("="*60)
        
        # Set project
        if not self.run_command(
            ["gcloud", "config", "set", "project", self.project_id],
            f"Setting project to {self.project_id}"
        ):
            return False
        
        # Verify bucket exists
        if not self.run_command(
            ["gcloud", "storage", "ls", f"gs://{self.bucket}/"],
            f"Verifying bucket {self.bucket} exists"
        ):
            print(f"   ⚠️  Bucket {self.bucket} might not exist or you don't have access")
            print(f"   💡 You may need to create it first:")
            print(f"      gcloud storage buckets create gs://{self.bucket}")
            return False
        
        return True
    
    def upload_pdfs(self, split: str) -> bool:
        """Upload PDFs for a specific split (train/test)."""
        pdfs_dir = self.dataset_dir / split / "pdfs"
        if not pdfs_dir.exists():
            print(f"   ❌ PDFs directory not found: {pdfs_dir}")
            return False
        
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"   ❌ No PDF files found in {pdfs_dir}")
            return False
        
        print(f"\n📤 Uploading {len(pdf_files)} {split} PDFs...")
        
        success_count = 0
        for pdf_file in pdf_files:
            gcs_path = f"gs://{self.bucket}/synthetic_dataset/{split}/pdfs/{pdf_file.name}"
            
            if self.run_command(
                ["gcloud", "storage", "cp", str(pdf_file), gcs_path],
                f"Uploading {pdf_file.name}"
            ):
                success_count += 1
        
        print(f"   📊 Uploaded {success_count}/{len(pdf_files)} {split} PDFs")
        return success_count == len(pdf_files)
    
    def upload_labels(self, split: str) -> bool:
        """Upload JSON labels for a specific split (train/test)."""
        labels_dir = self.dataset_dir / split / "labels"
        if not labels_dir.exists():
            print(f"   ❌ Labels directory not found: {labels_dir}")
            return False
        
        label_files = list(labels_dir.glob("*.json"))
        if not label_files:
            print(f"   ❌ No label files found in {labels_dir}")
            return False
        
        print(f"\n📤 Uploading {len(label_files)} {split} labels...")
        
        success_count = 0
        for label_file in label_files:
            gcs_path = f"gs://{self.bucket}/synthetic_dataset/{split}/labels/{label_file.name}"
            
            if self.run_command(
                ["gcloud", "storage", "cp", str(label_file), gcs_path],
                f"Uploading {label_file.name}"
            ):
                success_count += 1
        
        print(f"   📊 Uploaded {success_count}/{len(label_files)} {split} labels")
        return success_count == len(label_files)
    
    def upload_dataset_summary(self) -> bool:
        """Upload dataset summary files."""
        summary_files = [
            "dataset_summary.json",
            "detailed_summary.json"
        ]
        
        print(f"\n📤 Uploading dataset summary files...")
        
        success_count = 0
        for summary_file in summary_files:
            local_path = self.dataset_dir / summary_file
            if local_path.exists():
                gcs_path = f"gs://{self.bucket}/synthetic_dataset/{summary_file}"
                
                if self.run_command(
                    ["gcloud", "storage", "cp", str(local_path), gcs_path],
                    f"Uploading {summary_file}"
                ):
                    success_count += 1
        
        return success_count > 0
    
    def verify_upload(self) -> bool:
        """Verify the upload by listing GCS contents."""
        print(f"\n🔍 Verifying upload...")
        
        # List the synthetic_dataset directory
        if not self.run_command(
            ["gcloud", "storage", "ls", f"gs://{self.bucket}/synthetic_dataset/"],
            "Listing synthetic_dataset directory"
        ):
            return False
        
        # List train PDFs
        if not self.run_command(
            ["gcloud", "storage", "ls", f"gs://{self.bucket}/synthetic_dataset/train/pdfs/"],
            "Listing train PDFs"
        ):
            return False
        
        # List test PDFs
        if not self.run_command(
            ["gcloud", "storage", "ls", f"gs://{self.bucket}/synthetic_dataset/test/pdfs/"],
            "Listing test PDFs"
        ):
            return False
        
        return True
    
    def upload_dataset(self) -> bool:
        """Upload the complete synthetic dataset."""
        print("\n" + "="*60)
        print("📤 UPLOADING SYNTHETIC DATASET TO GCS")
        print("="*60)
        
        # Setup gcloud
        if not self.setup_gcloud():
            return False
        
        # Upload training data
        print(f"\n🚂 Uploading training data...")
        if not self.upload_pdfs("train"):
            return False
        if not self.upload_labels("train"):
            return False
        
        # Upload test data
        print(f"\n🧪 Uploading test data...")
        if not self.upload_pdfs("test"):
            return False
        if not self.upload_labels("test"):
            return False
        
        # Upload summary files
        if not self.upload_dataset_summary():
            print("   ⚠️  Warning: Could not upload summary files")
        
        # Verify upload
        if not self.verify_upload():
            print("   ⚠️  Warning: Upload verification failed")
        
        print(f"\n✅ Dataset upload complete!")
        print(f"📁 GCS Location: gs://{self.bucket}/synthetic_dataset/")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Upload synthetic dataset to Google Cloud Storage")
    parser.add_argument("--project_id", required=True,
                       help="Google Cloud project ID")
    parser.add_argument("--bucket", required=True,
                       help="Google Cloud Storage bucket name")
    parser.add_argument("--dataset_dir", default="data/synthetic_dataset",
                       help="Local dataset directory to upload")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.dataset_dir).exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        print("   Please generate the dataset first using:")
        print("   python scripts/run_dataset_generation.py --full --validate")
        return 1
    
    # Upload dataset
    uploader = GCSUploader(args.project_id, args.bucket, args.dataset_dir)
    success = uploader.upload_dataset()
    
    if success:
        print(f"\n🎯 NEXT STEPS FOR GOOGLE CLOUD DOCUMENT AI:")
        print(f"   1. Go to Google Cloud Console > Document AI")
        print(f"   2. Create a new dataset or use existing one")
        print(f"   3. Import training data from: gs://{args.bucket}/synthetic_dataset/train/")
        print(f"   4. Import test data from: gs://{args.bucket}/synthetic_dataset/test/")
        print(f"   5. Train your custom model for deed boundary detection")
        print(f"   6. Evaluate on the test set")
        return 0
    else:
        print(f"\n❌ Upload failed!")
        return 1


if __name__ == "__main__":
    exit(main())







