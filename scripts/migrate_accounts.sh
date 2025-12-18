#!/bin/bash
# Migration Script for Moving to New Vercel/Google Cloud/Anthropic Accounts
# Usage: ./scripts/migrate_accounts.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Account Migration Script${NC}"
echo -e "${BLUE}===========================${NC}"
echo ""

# Check if migration config exists
if [ ! -f "migration_config.env" ]; then
    echo -e "${YELLOW}⚠️  migration_config.env not found. Creating template...${NC}"
    cat > migration_config.env << 'EOF'
# Migration Configuration Template
# Fill in the values below for your new accounts

# Google Cloud Platform
NEW_GCP_PROJECT_ID=your-new-project-id
NEW_GCP_PROJECT_NUMBER=your-project-number
NEW_GCS_BUCKET_NAME=your-new-bucket-name
NEW_DOCUMENT_AI_ENDPOINT=projects/YOUR_PROJECT_NUMBER/locations/us/processors/YOUR_PROCESSOR_ID

# Vercel
NEW_VERCEL_API_URL=https://mineral-rights-api-YOUR_PROJECT_NUMBER.us-central1.run.app

# Anthropic (will be stored in Secret Manager, not here)
# NEW_ANTHROPIC_API_KEY=sk-ant-api03-...

# Service Account (base64 encoded JSON - generate after creating service account)
# NEW_GOOGLE_CREDENTIALS_BASE64=...
EOF
    echo -e "${GREEN}✅ Created migration_config.env template${NC}"
    echo -e "${YELLOW}Please edit migration_config.env with your new account details, then run this script again.${NC}"
    exit 0
fi

# Load configuration
source migration_config.env

# Validate required variables
if [ -z "$NEW_GCP_PROJECT_ID" ] || [ "$NEW_GCP_PROJECT_ID" = "your-new-project-id" ]; then
    echo -e "${RED}❌ NEW_GCP_PROJECT_ID not set in migration_config.env${NC}"
    exit 1
fi

if [ -z "$NEW_GCS_BUCKET_NAME" ] || [ "$NEW_GCS_BUCKET_NAME" = "your-new-bucket-name" ]; then
    echo -e "${RED}❌ NEW_GCS_BUCKET_NAME not set in migration_config.env${NC}"
    exit 1
fi

if [ -z "$NEW_VERCEL_API_URL" ] || [ "$NEW_VERCEL_API_URL" = "https://mineral-rights-api-YOUR_PROJECT_NUMBER.us-central1.run.app" ]; then
    echo -e "${RED}❌ NEW_VERCEL_API_URL not set in migration_config.env${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Configuration loaded${NC}"
echo ""

# Backup original files
echo -e "${BLUE}📦 Creating backups...${NC}"
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
cp cloudbuild.yaml "$BACKUP_DIR/"
cp web/vercel.json "$BACKUP_DIR/"
cp web/src/components/ModelManager.tsx "$BACKUP_DIR/"
cp web/src/lib/api_async.ts "$BACKUP_DIR/"
cp web/src/app/page.tsx "$BACKUP_DIR/"
cp web/src/lib/direct_gcs_upload.ts "$BACKUP_DIR/"
echo -e "${GREEN}✅ Backups created in $BACKUP_DIR${NC}"
echo ""

# Update cloudbuild.yaml
echo -e "${BLUE}🔧 Updating cloudbuild.yaml...${NC}"
if [ -n "$NEW_GOOGLE_CREDENTIALS_BASE64" ]; then
    # This is complex - we'll need to manually update the base64 credentials
    echo -e "${YELLOW}⚠️  Note: You'll need to manually update GOOGLE_CREDENTIALS_BASE64 in cloudbuild.yaml${NC}"
    echo -e "${YELLOW}   Replace the long base64 string on lines 30 and 59${NC}"
fi

# Update bucket name in cloudbuild.yaml (line 30 and 59)
sed -i.bak "s/GCS_BUCKET_NAME=mineral-rights-pdfs-1759435410/GCS_BUCKET_NAME=$NEW_GCS_BUCKET_NAME/g" cloudbuild.yaml

# Update Document AI endpoint if provided (OPTIONAL)
if [ -n "$NEW_DOCUMENT_AI_ENDPOINT" ] && [ "$NEW_DOCUMENT_AI_ENDPOINT" != "projects/YOUR_PROJECT_NUMBER/locations/us/processors/YOUR_PROCESSOR_ID" ] && [ "$NEW_DOCUMENT_AI_ENDPOINT" != "" ]; then
    # Extract project number from endpoint for the format used in cloudbuild.yaml
    PROJECT_NUM=$(echo "$NEW_DOCUMENT_AI_ENDPOINT" | sed -n 's|projects/\([^/]*\).*|\1|p')
    PROCESSOR_ID=$(echo "$NEW_DOCUMENT_AI_ENDPOINT" | sed -n 's|.*processors/\([^/]*\).*|\1|p')
    if [ -n "$PROJECT_NUM" ] && [ -n "$PROCESSOR_ID" ]; then
        sed -i.bak "s|DOCUMENT_AI_ENDPOINT=projects/381937358877/locations/us/processors/895767ed7f252878|DOCUMENT_AI_ENDPOINT=projects/$PROJECT_NUM/locations/us/processors/$PROCESSOR_ID|g" cloudbuild.yaml
        echo -e "${GREEN}✅ Updated Document AI endpoint${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Document AI endpoint not provided - app will use fallback splitting methods${NC}"
    echo -e "${YELLOW}   This is fine - Document AI is optional!${NC}"
fi

echo -e "${GREEN}✅ Updated cloudbuild.yaml${NC}"

# Update web/vercel.json
echo -e "${BLUE}🔧 Updating web/vercel.json...${NC}"
sed -i.bak "s|https://mineral-rights-api-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/vercel.json
echo -e "${GREEN}✅ Updated web/vercel.json${NC}"

# Update frontend files with API URL fallbacks
echo -e "${BLUE}🔧 Updating frontend API URL fallbacks...${NC}"

# ModelManager.tsx
sed -i.bak "s|https://mineral-rights-api-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/src/components/ModelManager.tsx

# api_async.ts
sed -i.bak "s|https://mineral-rights-api-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/src/lib/api_async.ts

# page.tsx
sed -i.bak "s|https://mineral-rights-api-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/src/app/page.tsx

# direct_gcs_upload.ts
sed -i.bak "s|https://mineral-rights-api-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/src/lib/direct_gcs_upload.ts

# API route files
sed -i.bak "s|https://mineral-rights-api-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/src/app/api/get-signed-upload-url/route.ts
sed -i.bak "s|https://mineral-rights-api-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/src/app/api/process-large-pdf-pages/route.ts
sed -i.bak "s|https://mineral-rights-api-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/src/app/api/process-large-pdf/route.ts

# gcs_upload.ts (uses different URL pattern)
sed -i.bak "s|https://mineral-rights-processor-1081023230228.us-central1.run.app|$NEW_VERCEL_API_URL|g" web/src/lib/gcs_upload.ts

echo -e "${GREEN}✅ Updated frontend files${NC}"

# Clean up backup files
rm -f cloudbuild.yaml.bak web/vercel.json.bak web/src/components/ModelManager.tsx.bak web/src/lib/api_async.ts.bak web/src/app/page.tsx.bak web/src/lib/direct_gcs_upload.ts.bak web/src/app/api/get-signed-upload-url/route.ts.bak web/src/app/api/process-large-pdf-pages/route.ts.bak web/src/app/api/process-large-pdf/route.ts.bak web/src/lib/gcs_upload.ts.bak

echo ""
echo -e "${GREEN}✅ Migration script completed!${NC}"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Manual steps remaining:${NC}"
echo "1. Update GOOGLE_CREDENTIALS_BASE64 in cloudbuild.yaml (lines 30 and 59)"
echo "2. Create Anthropic API key secret in Google Secret Manager"
echo "3. Deploy backend: gcloud builds submit --config cloudbuild.yaml ."
echo "4. Update Vercel environment variables"
echo "5. Deploy frontend to Vercel"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "1. Review the changes: git diff"
echo "2. Commit: git add . && git commit -m 'Migrate to new accounts'"
echo "3. Follow Phase 1-7 from the migration plan"

