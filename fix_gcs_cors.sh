#!/bin/bash
# Fix GCS CORS configuration for browser uploads

set -e

echo "🔧 Fixing GCS CORS configuration for browser uploads..."

# Create CORS configuration file
cat > cors-config.json << EOF
[
  {
    "origin": ["*"],
    "method": ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
    "responseHeader": ["Content-Type", "Access-Control-Allow-Origin", "Access-Control-Allow-Methods", "Access-Control-Allow-Headers"],
    "maxAgeSeconds": 3600
  }
]
EOF

echo "📄 Created CORS configuration file"

# Apply CORS configuration to the bucket
BUCKET_NAME="mineral-rights-pdfs-1759435410"
echo "📦 Applying CORS to bucket: $BUCKET_NAME"

gsutil cors set cors-config.json gs://$BUCKET_NAME

echo "✅ CORS configuration applied successfully!"
echo ""
echo "🧪 Test the fix by uploading a file through the web interface"
