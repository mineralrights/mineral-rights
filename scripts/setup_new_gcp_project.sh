#!/bin/bash
# Setup Script for New Google Cloud Project
# This script automates Phase 1 of the migration plan
# Usage: ./scripts/setup_new_gcp_project.sh NEW_PROJECT_ID NEW_BUCKET_NAME

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

if [ $# -lt 2 ]; then
    echo -e "${RED}❌ Usage: $0 NEW_PROJECT_ID NEW_BUCKET_NAME${NC}"
    echo "Example: $0 my-mineral-rights-app my-bucket-name"
    exit 1
fi

NEW_PROJECT_ID=$1
NEW_BUCKET_NAME=$2
REGION=${3:-us-central1}

echo -e "${BLUE}🚀 Setting up new GCP project: $NEW_PROJECT_ID${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed. Please install it first:${NC}"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  Not authenticated with gcloud. Please run:${NC}"
    echo "   gcloud auth login"
    exit 1
fi

# Step 1: Create new project
echo -e "${BLUE}📋 Step 1: Creating new GCP project...${NC}"
if gcloud projects describe "$NEW_PROJECT_ID" &>/dev/null; then
    echo -e "${YELLOW}⚠️  Project $NEW_PROJECT_ID already exists${NC}"
else
    gcloud projects create "$NEW_PROJECT_ID" --name="Mineral Rights App"
    echo -e "${GREEN}✅ Project created${NC}"
fi

# Set as active project
gcloud config set project "$NEW_PROJECT_ID"
echo -e "${GREEN}✅ Project set as active${NC}"
echo ""

# Step 2: Enable billing (manual step required)
echo -e "${YELLOW}⚠️  Step 2: Please enable billing for project $NEW_PROJECT_ID${NC}"
echo "   Go to: https://console.cloud.google.com/billing/linkedaccount?project=$NEW_PROJECT_ID"
read -p "Press Enter after billing is enabled..."

# Step 3: Enable required APIs
echo -e "${BLUE}🔧 Step 3: Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable documentai.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable cloudtasks.googleapis.com
gcloud services enable firestore.googleapis.com
echo -e "${GREEN}✅ APIs enabled${NC}"
echo ""

# Step 4: Create service account
echo -e "${BLUE}🔑 Step 4: Creating service account...${NC}"
SERVICE_ACCOUNT_NAME="mineral-rights-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${NEW_PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &>/dev/null; then
    echo -e "${YELLOW}⚠️  Service account already exists${NC}"
else
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="Mineral Rights Service Account"
    echo -e "${GREEN}✅ Service account created${NC}"
fi

# Grant roles
echo -e "${BLUE}🔐 Granting roles to service account...${NC}"
gcloud projects add-iam-policy-binding "$NEW_PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding "$NEW_PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/documentai.apiUser"

gcloud projects add-iam-policy-binding "$NEW_PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/run.invoker"

gcloud projects add-iam-policy-binding "$NEW_PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

echo -e "${GREEN}✅ Roles granted${NC}"
echo ""

# Step 5: Create and download service account key
echo -e "${BLUE}📥 Step 5: Creating service account key...${NC}"
KEY_FILE="service-account-${NEW_PROJECT_ID}.json"
if [ -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}⚠️  Key file already exists: $KEY_FILE${NC}"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping key creation"
    else
        gcloud iam service-accounts keys create "$KEY_FILE" \
            --iam-account="$SERVICE_ACCOUNT_EMAIL"
        echo -e "${GREEN}✅ Key file created: $KEY_FILE${NC}"
    fi
else
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SERVICE_ACCOUNT_EMAIL"
    echo -e "${GREEN}✅ Key file created: $KEY_FILE${NC}"
fi

# Base64 encode the key
if command -v base64 &> /dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        BASE64_KEY=$(base64 -i "$KEY_FILE" | tr -d '\n')
    else
        # Linux
        BASE64_KEY=$(base64 -w 0 "$KEY_FILE")
    fi
    echo ""
    echo -e "${GREEN}✅ Base64 encoded credentials:${NC}"
    echo "$BASE64_KEY" > "service-account-${NEW_PROJECT_ID}.base64.txt"
    echo -e "${BLUE}📝 Saved to: service-account-${NEW_PROJECT_ID}.base64.txt${NC}"
    echo -e "${YELLOW}⚠️  Keep this file secure! Add it to migration_config.env as NEW_GOOGLE_CREDENTIALS_BASE64${NC}"
fi
echo ""

# Step 5: Create GCS bucket
echo -e "${BLUE}🪣 Step 6: Creating GCS bucket...${NC}"
if gsutil ls -b "gs://${NEW_BUCKET_NAME}" &>/dev/null; then
    echo -e "${YELLOW}⚠️  Bucket already exists${NC}"
else
    gsutil mb -l "$REGION" "gs://${NEW_BUCKET_NAME}"
    echo -e "${GREEN}✅ Bucket created: gs://${NEW_BUCKET_NAME}${NC}"
fi

# Grant permissions to service account
gsutil iam ch "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:objectAdmin" "gs://${NEW_BUCKET_NAME}"
echo -e "${GREEN}✅ Permissions granted${NC}"
echo ""

# Step 6: Get project number (needed for Document AI endpoint - OPTIONAL)
PROJECT_NUMBER=$(gcloud projects describe "$NEW_PROJECT_ID" --format="value(projectNumber)")
echo -e "${BLUE}📊 Project Number: $PROJECT_NUMBER${NC}"
echo ""

# Step 7: Document AI Processor (OPTIONAL)
echo -e "${YELLOW}⚠️  Step 7: Document AI Processor Setup (OPTIONAL)${NC}"
echo "Document AI is optional - the app will use fallback splitting if not configured."
read -p "Do you want to set up Document AI processor now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}📄 To create Document AI processor:${NC}"
    echo "1. Go to: https://console.cloud.google.com/ai/document-ai/processors?project=$NEW_PROJECT_ID"
    echo "2. Create a new processor"
    echo "3. Note the processor ID"
    echo "4. Format: projects/$PROJECT_NUMBER/locations/us/processors/PROCESSOR_ID"
    read -p "Enter Document AI endpoint (or press Enter to skip): " DOC_AI_ENDPOINT
    if [ -n "$DOC_AI_ENDPOINT" ]; then
        echo "DOCUMENT_AI_ENDPOINT=$DOC_AI_ENDPOINT" >> migration_config.env
        echo -e "${GREEN}✅ Document AI endpoint saved${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Skipping Document AI setup. App will use fallback splitting methods.${NC}"
fi
echo ""

# Step 8: Create Secret Manager secret placeholder
echo -e "${BLUE}🔐 Step 8: Creating Secret Manager secret...${NC}"
SECRET_NAME="anthropic-api-key"
if gcloud secrets describe "$SECRET_NAME" &>/dev/null; then
    echo -e "${YELLOW}⚠️  Secret already exists${NC}"
else
    echo "Please enter your Anthropic API key (or press Enter to skip):"
    read -s ANTHROPIC_KEY
    if [ -n "$ANTHROPIC_KEY" ]; then
        echo -n "$ANTHROPIC_KEY" | gcloud secrets create "$SECRET_NAME" --data-file=-
        echo -e "${GREEN}✅ Secret created${NC}"
        
        # Grant access to service account
        gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
            --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
            --role="roles/secretmanager.secretAccessor"
        echo -e "${GREEN}✅ Service account granted access to secret${NC}"
    else
        echo -e "${YELLOW}⚠️  Skipped secret creation. Create it manually later:${NC}"
        echo "   gcloud secrets create $SECRET_NAME --data-file=-"
    fi
fi
echo ""

# Summary
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo -e "${BLUE}📋 Summary:${NC}"
echo "  Project ID: $NEW_PROJECT_ID"
echo "  Project Number: $PROJECT_NUMBER"
echo "  Bucket: gs://${NEW_BUCKET_NAME}"
echo "  Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "  Key File: $KEY_FILE"
if [ -f "service-account-${NEW_PROJECT_ID}.base64.txt" ]; then
    echo "  Base64 Key: service-account-${NEW_PROJECT_ID}.base64.txt"
fi
echo ""
echo -e "${YELLOW}⚠️  Next Steps:${NC}"
echo "1. (OPTIONAL) Create Document AI processor in Cloud Console if you want to use it"
echo "   - The app works fine without it (uses fallback splitting)"
echo "   - If creating: Note the processor ID (format: projects/$PROJECT_NUMBER/locations/us/processors/PROCESSOR_ID)"
echo "2. Update migration_config.env with:"
echo "   - NEW_GCP_PROJECT_ID=$NEW_PROJECT_ID"
echo "   - NEW_GCP_PROJECT_NUMBER=$PROJECT_NUMBER"
echo "   - NEW_GCS_BUCKET_NAME=$NEW_BUCKET_NAME"
echo "   - NEW_DOCUMENT_AI_ENDPOINT=projects/$PROJECT_NUMBER/locations/us/processors/YOUR_PROCESSOR_ID"
echo "   - NEW_GOOGLE_CREDENTIALS_BASE64=<from base64 file>"
echo "   - NEW_VERCEL_API_URL=https://mineral-rights-api-$PROJECT_NUMBER.us-central1.run.app"
echo "4. Run: ./scripts/migrate_accounts.sh"

