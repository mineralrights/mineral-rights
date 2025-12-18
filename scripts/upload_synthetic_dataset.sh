#!/bin/bash

# Upload Synthetic Dataset to Google Cloud Storage
# Following the same pattern as your existing upload process

echo "🚀 Uploading Synthetic Dataset to Google Cloud Storage"
echo "====================================================="

# Set up variables (same pattern as your example)
export PROJECT_ID="deed-boundary-250831-29868"  # Update this to your project ID
export BUCKET="my-deed-bucket-$PROJECT_ID"      # Update this to your bucket name

echo "📋 Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Bucket: $BUCKET"
echo ""

# Set the project
echo "🔧 Setting up gcloud configuration..."
gcloud config set project "$PROJECT_ID"

# Verify bucket exists
echo "🔍 Verifying bucket exists..."
gcloud storage ls "gs://$BUCKET/"

if [ $? -ne 0 ]; then
    echo "❌ Bucket $BUCKET not found or no access!"
    echo "   Please create the bucket first:"
    echo "   gcloud storage buckets create gs://$BUCKET"
    exit 1
fi

echo ""
echo "📤 Uploading synthetic dataset PDFs and labels..."

# Upload training PDFs
echo "🚂 Uploading training PDFs..."
for pdf in data/synthetic_dataset/train/pdfs/*.pdf; do
    if [ -f "$pdf" ]; then
        filename=$(basename "$pdf")
        echo "   Uploading $filename..."
        gcloud storage cp "$pdf" \
            "gs://$BUCKET/synthetic_dataset/train/pdfs/$filename"
    fi
done

# Upload training labels
echo "🏷️  Uploading training labels..."
for json in data/synthetic_dataset/train/labels/*.json; do
    if [ -f "$json" ]; then
        filename=$(basename "$json")
        echo "   Uploading $filename..."
        gcloud storage cp "$json" \
            "gs://$BUCKET/synthetic_dataset/train/labels/$filename"
    fi
done

# Upload test PDFs
echo "🧪 Uploading test PDFs..."
for pdf in data/synthetic_dataset/test/pdfs/*.pdf; do
    if [ -f "$pdf" ]; then
        filename=$(basename "$pdf")
        echo "   Uploading $filename..."
        gcloud storage cp "$pdf" \
            "gs://$BUCKET/synthetic_dataset/test/pdfs/$filename"
    fi
done

# Upload test labels
echo "🏷️  Uploading test labels..."
for json in data/synthetic_dataset/test/labels/*.json; do
    if [ -f "$json" ]; then
        filename=$(basename "$json")
        echo "   Uploading $filename..."
        gcloud storage cp "$json" \
            "gs://$BUCKET/synthetic_dataset/test/labels/$filename"
    fi
done

# Upload summary files
echo "📊 Uploading summary files..."
if [ -f "data/synthetic_dataset/dataset_summary.json" ]; then
    gcloud storage cp "data/synthetic_dataset/dataset_summary.json" \
        "gs://$BUCKET/synthetic_dataset/dataset_summary.json"
fi

if [ -f "data/synthetic_dataset/detailed_summary.json" ]; then
    gcloud storage cp "data/synthetic_dataset/detailed_summary.json" \
        "gs://$BUCKET/synthetic_dataset/detailed_summary.json"
fi

echo ""
echo "🔍 Verifying upload..."

# List the synthetic_dataset directory
echo "📁 Contents of gs://$BUCKET/synthetic_dataset/:"
gcloud storage ls "gs://$BUCKET/synthetic_dataset/"

echo ""
echo "🚂 Training PDFs:"
gcloud storage ls "gs://$BUCKET/synthetic_dataset/train/pdfs/"

echo ""
echo "🧪 Test PDFs:"
gcloud storage ls "gs://$BUCKET/synthetic_dataset/test/pdfs/"

echo ""
echo "✅ Upload complete!"
echo ""
echo "📋 GCS Structure:"
echo "   gs://$BUCKET/synthetic_dataset/"
echo "   ├── train/"
echo "   │   ├── pdfs/          # Training PDF files"
echo "   │   └── labels/        # Training JSON labels"
echo "   ├── test/"
echo "   │   ├── pdfs/          # Test PDF files"
echo "   │   └── labels/        # Test JSON labels"
echo "   ├── dataset_summary.json"
echo "   └── detailed_summary.json"
echo ""
echo "🎯 Next Steps for Google Cloud Document AI:"
echo "   1. Go to Google Cloud Console > Document AI"
echo "   2. Create a new dataset or use existing one"
echo "   3. Import training data from: gs://$BUCKET/synthetic_dataset/train/"
echo "   4. Import test data from: gs://$BUCKET/synthetic_dataset/test/"
echo "   5. Train your custom model for deed boundary detection"
echo "   6. Evaluate on the test set"








