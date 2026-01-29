#!/bin/bash

# Google Cloud Deployment Script for Mineral Rights App
# This script sets up and deploys the app to Google Cloud Run

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${PROJECT_ID:-"mineral-rights"}
REGION=${REGION:-"us-central1"}
BUCKET_NAME="mineral-rights-storage-${PROJECT_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/scripts/setup_mineral_rights_static_vars.sh"

echo -e "${BLUE}🚀 Google Cloud Deployment Script${NC}"
echo -e "${BLUE}================================${NC}"

# Run setup script first (buckets, SA, Secret Manager credentials) if it exists
if [ -f "${SETUP_SCRIPT}" ]; then
    echo -e "${BLUE}📋 Running setup for static variables (buckets, service account, secrets)...${NC}"
    PROJECT_ID="${PROJECT_ID}" REGION="${REGION}" bash "${SETUP_SCRIPT}"
    echo ""
else
    echo -e "${YELLOW}⚠️  Setup script not found: ${SETUP_SCRIPT}${NC}"
    echo -e "${YELLOW}   Run once: PROJECT_ID=${PROJECT_ID} ./scripts/setup_mineral_rights_static_vars.sh${NC}"
    echo -e "${YELLOW}   Then re-run this deploy script.${NC}"
    read -p "Continue without setup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    echo ""
fi

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

# Set the project
echo -e "${BLUE}📋 Setting project to: ${PROJECT_ID}${NC}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "${BLUE}🔧 Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable documentai.googleapis.com
gcloud services enable cloudtasks.googleapis.com
gcloud services enable firestore.googleapis.com

# Create Cloud Storage bucket
echo -e "${BLUE}🪣 Creating Cloud Storage bucket: ${BUCKET_NAME}${NC}"
if ! gsutil ls -b gs://${BUCKET_NAME} &> /dev/null; then
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
    echo -e "${GREEN}✅ Bucket created successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Bucket already exists${NC}"
fi

# Create Firestore database
echo -e "${BLUE}🔥 Setting up Firestore database...${NC}"
if ! gcloud firestore databases list --format="value(name)" | grep -q "projects/${PROJECT_ID}/databases/(default)"; then
    gcloud firestore databases create --location=${REGION}
    echo -e "${GREEN}✅ Firestore database created${NC}"
else
    echo -e "${YELLOW}⚠️  Firestore database already exists${NC}"
fi

# Build and deploy using Cloud Build
echo -e "${BLUE}🏗️  Building and deploying with Cloud Build...${NC}"
gcloud builds submit --config cloudbuild.yaml .

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${BLUE}📋 Next steps:${NC}"
echo "1. Set up environment variables in Cloud Run:"
echo "   - ANTHROPIC_API_KEY"
echo "   - DOCUMENT_AI_ENDPOINT"
echo "   - GOOGLE_CREDENTIALS_BASE64"
echo "   - GCS_BUCKET_NAME=${BUCKET_NAME}"
echo ""
echo "2. Update your frontend to use the new API URL:"
echo "   https://mineral-rights-api-<hash>-uc.a.run.app"
echo ""
echo "3. Test the deployment:"
echo "   curl https://mineral-rights-api-<hash>-uc.a.run.app/health"
echo ""
echo -e "${GREEN}🎉 Your app is now running on Google Cloud Run!${NC}"
