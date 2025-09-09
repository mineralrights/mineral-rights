#!/bin/bash

# Production Dataset Generation Script
# This script generates a full production-ready dataset for Google Cloud Document AI

echo "🚀 Generating Production Synthetic Multi-Deed Dataset"
echo "=================================================="

# Configuration
OUTPUT_DIR="data/production_dataset"
TRAIN_COUNT=100
TEST_COUNT=25
SEED=42

echo "📋 Configuration:"
echo "   Output Directory: $OUTPUT_DIR"
echo "   Training Documents: $TRAIN_COUNT"
echo "   Test Documents: $TEST_COUNT"
echo "   Random Seed: $SEED"
echo ""

# Check if source data exists
if [ ! -d "data/no-reservs" ] || [ ! -d "data/reservs" ]; then
    echo "❌ Error: Source data directories not found!"
    echo "   Please ensure data/no-reservs and data/reservs directories exist"
    exit 1
fi

# Generate the dataset
echo "🔄 Generating dataset..."
python scripts/generate_synthetic_dataset.py \
    --output_dir "$OUTPUT_DIR" \
    --num_train "$TRAIN_COUNT" \
    --num_test "$TEST_COUNT" \
    --seed "$SEED"

if [ $? -ne 0 ]; then
    echo "❌ Dataset generation failed!"
    exit 1
fi

# Validate the dataset
echo ""
echo "🔍 Validating dataset..."
python scripts/validate_synthetic_dataset.py --dataset_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "⚠️  Dataset validation failed, but generation completed"
    echo "   Please review the generated dataset manually"
fi

# Generate summary
echo ""
echo "📊 Generating dataset summary..."
python scripts/dataset_summary.py --dataset_dir "$OUTPUT_DIR"

# Display final results
echo ""
echo "✅ Production dataset generation complete!"
echo ""
echo "📁 Dataset Location: $OUTPUT_DIR"
echo "📊 Generated $TRAIN_COUNT training documents and $TEST_COUNT test documents"
echo ""
echo "📋 Directory Structure:"
echo "   $OUTPUT_DIR/"
echo "   ├── train/"
echo "   │   ├── pdfs/          # Training PDF files"
echo "   │   └── labels/        # Training JSON labels"
echo "   ├── test/"
echo "   │   ├── pdfs/          # Test PDF files"
echo "   │   └── labels/        # Test JSON labels"
echo "   ├── dataset_summary.json"
echo "   └── detailed_summary.json"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review the generated PDFs and labels"
echo "   2. Upload to Google Cloud Document AI:"
echo "      - Create a new dataset in Document AI"
echo "      - Upload PDFs from train/pdfs/"
echo "      - Upload corresponding labels from train/labels/"
echo "   3. Train your custom model"
echo "   4. Evaluate on test set"
echo "   5. Deploy for production use"
echo ""
echo "📖 For detailed instructions, see: SYNTHETIC_DATASET_README.md"
