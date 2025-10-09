#!/bin/bash
# Fix GCS permissions for Cloud Run service account

set -e

echo "🔧 Fixing GCS permissions for Cloud Run service account..."

# Get the project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No project ID found. Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "📋 Project ID: $PROJECT_ID"

# Get the Cloud Run service account
SERVICE_ACCOUNT=$(gcloud run services describe mineral-rights-processor --region=us-central1 --format="value(spec.template.spec.serviceAccountName)" 2>/dev/null || echo "")
if [ -z "$SERVICE_ACCOUNT" ]; then
    echo "⚠️ Could not get Cloud Run service account, using default compute service account"
    SERVICE_ACCOUNT="${PROJECT_ID}-compute@developer.gserviceaccount.com"
fi

echo "🔑 Service Account: $SERVICE_ACCOUNT"

# Get the GCS bucket name
BUCKET_NAME="mineral-rights-pdfs-1759435410"
echo "📦 Bucket: $BUCKET_NAME"

# Grant Storage Object Viewer role to the service account
echo "🔧 Granting Storage Object Viewer role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/storage.objectViewer"

echo "✅ Storage Object Viewer role granted"

# Also grant Storage Object Admin for the specific bucket (more permissive)
echo "🔧 Granting Storage Object Admin role for bucket..."
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectAdmin gs://$BUCKET_NAME

echo "✅ Storage Object Admin role granted for bucket"

# Verify the permissions
echo "🔍 Verifying permissions..."
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:$SERVICE_ACCOUNT" | grep storage || echo "⚠️ Storage roles not found in project-level IAM"

echo "✅ Permission fix completed!"
echo ""
echo "🧪 Test the fix by running:"
echo "   python3 test_production_gcs.py"
