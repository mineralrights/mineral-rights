#!/bin/bash

# Setup script for GCS upload configuration
# This helps you configure the upload with your specific project details

echo "🔧 Setting up GCS Upload Configuration"
echo "====================================="

# Check if dataset exists
if [ ! -d "data/synthetic_dataset" ]; then
    echo "❌ Synthetic dataset not found!"
    echo "   Please generate the dataset first:"
    echo "   python scripts/run_dataset_generation.py --full --validate"
    exit 1
fi

echo "✅ Synthetic dataset found!"
echo ""

# Get current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -n "$CURRENT_PROJECT" ]; then
    echo "📋 Current gcloud project: $CURRENT_PROJECT"
    echo ""
    read -p "Do you want to use this project? (y/n): " use_current
    
    if [ "$use_current" = "y" ] || [ "$use_current" = "Y" ]; then
        PROJECT_ID="$CURRENT_PROJECT"
    else
        read -p "Enter your Google Cloud project ID: " PROJECT_ID
    fi
else
    read -p "Enter your Google Cloud project ID: " PROJECT_ID
fi

echo ""
echo "📦 Setting up bucket name..."
echo "   Common patterns:"
echo "   1. my-deed-bucket-$PROJECT_ID"
echo "   2. deed-training-data"
echo "   3. mineral-rights-dataset"
echo ""

read -p "Enter your bucket name (or press Enter for 'my-deed-bucket-$PROJECT_ID'): " BUCKET_NAME

if [ -z "$BUCKET_NAME" ]; then
    BUCKET_NAME="my-deed-bucket-$PROJECT_ID"
fi

echo ""
echo "📋 Configuration Summary:"
echo "   Project ID: $PROJECT_ID"
echo "   Bucket: $BUCKET_NAME"
echo "   Dataset: data/synthetic_dataset"
echo ""

read -p "Proceed with upload? (y/n): " proceed

if [ "$proceed" = "y" ] || [ "$proceed" = "Y" ]; then
    echo ""
    echo "🚀 Starting upload..."
    
    # Update the upload script with the correct values
    sed -i.bak "s/export PROJECT_ID=\"deed-boundary-250831-29868\"/export PROJECT_ID=\"$PROJECT_ID\"/" scripts/upload_synthetic_dataset.sh
    sed -i.bak "s/export BUCKET=\"my-deed-bucket-\$PROJECT_ID\"/export BUCKET=\"$BUCKET_NAME\"/" scripts/upload_synthetic_dataset.sh
    
    # Run the upload
    ./scripts/upload_synthetic_dataset.sh
    
    # Restore original script
    mv scripts/upload_synthetic_dataset.sh.bak scripts/upload_synthetic_dataset.sh 2>/dev/null || true
else
    echo "❌ Upload cancelled"
    echo ""
    echo "💡 To upload manually later:"
    echo "   1. Edit scripts/upload_synthetic_dataset.sh"
    echo "   2. Update PROJECT_ID and BUCKET variables"
    echo "   3. Run: ./scripts/upload_synthetic_dataset.sh"
fi




