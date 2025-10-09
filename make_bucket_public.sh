#!/bin/bash
# Make GCS bucket public for quick fix

set -e

echo "🔧 Making GCS bucket public for quick fix..."

# Get the bucket name
BUCKET_NAME="mineral-rights-pdfs-1759435410"
echo "📦 Bucket: $BUCKET_NAME"

# Make bucket public
echo "🔧 Making bucket public..."
gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME

echo "✅ Bucket is now public!"
echo ""
echo "🧪 Test the fix by running:"
echo "   python3 test_production_gcs.py"
