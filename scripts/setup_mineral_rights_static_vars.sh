#!/bin/bash
# Setup static variables and resources for Mineral Rights GCP deployment
# Run this once per project before deploy_gcp.sh. It creates:
#   - Cloud Build bucket + permissions (so gcloud builds submit works)
#   - App storage bucket for PDFs
#   - Default compute SA granted access to app bucket (no keys; app uses ADC)
# Usage: PROJECT_ID=mineral-rights ./scripts/setup_mineral_rights_static_vars.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ID=${PROJECT_ID:-"mineral-rights"}
REGION=${REGION:-"us-central1"}
APP_BUCKET="mineral-rights-storage-${PROJECT_ID}"
CLOUDBUILD_BUCKET="${PROJECT_ID}_cloudbuild"

echo -e "${BLUE}🚀 Setting up static variables for Mineral Rights (project: ${PROJECT_ID})${NC}"
echo -e "${BLUE}================================================================${NC}"

if ! command -v gcloud &>/dev/null; then
    echo -e "${RED}❌ gcloud CLI not installed. See https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  Not authenticated. Run: gcloud auth login${NC}"
    exit 1
fi

gcloud config set project "${PROJECT_ID}"
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo -e "${BLUE}📋 Project Number: ${PROJECT_NUMBER}${NC}"
echo ""

# Enable APIs
echo -e "${BLUE}🔧 Enabling APIs...${NC}"
gcloud services enable secretmanager.googleapis.com --quiet
gcloud services enable storage.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable artifactregistry.googleapis.com --quiet
echo -e "${GREEN}✅ APIs enabled${NC}"
echo ""

# Grant default compute SA + Cloud Build SA permission to push images and write logs
echo -e "${BLUE}🔐 Granting Artifact Registry Writer + Logging to build SAs...${NC}"
for SA in "${COMPUTE_SA}" "${CLOUDBUILD_SA}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA}" \
    --role="roles/artifactregistry.writer" \
    --quiet
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA}" \
    --role="roles/logging.logWriter" \
    --quiet
done
echo -e "${GREEN}✅ Build SAs can push images and write logs${NC}"
echo ""

# Artifact Registry Docker repo (required for Cloud Build push; gcr.io is deprecated for new projects)
AR_REPO="mineral-rights"
echo -e "${BLUE}📦 Creating Artifact Registry Docker repo: ${AR_REPO} (region: ${REGION})${NC}"
if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud artifacts repositories create "${AR_REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --project="${PROJECT_ID}" \
        --description="Mineral Rights API and worker images"
    echo -e "${GREEN}✅ Artifact Registry repo created${NC}"
else
    echo -e "${YELLOW}⚠️  Artifact Registry repo already exists${NC}"
fi
echo ""

# 1) Cloud Build bucket (so gcloud builds submit can upload and Cloud Build can read)
echo -e "${BLUE}🪣 Creating Cloud Build bucket: gs://${CLOUDBUILD_BUCKET}${NC}"
if ! gsutil ls -b "gs://${CLOUDBUILD_BUCKET}" &>/dev/null; then
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${CLOUDBUILD_BUCKET}"
    echo -e "${GREEN}✅ Cloud Build bucket created${NC}"
else
    echo -e "${YELLOW}⚠️  Bucket already exists${NC}"
fi
echo -e "${BLUE}   Granting Storage Object Admin to compute + Cloud Build SAs...${NC}"
gsutil iam ch "serviceAccount:${COMPUTE_SA}:objectAdmin" "gs://${CLOUDBUILD_BUCKET}" 2>/dev/null || true
gsutil iam ch "serviceAccount:${CLOUDBUILD_SA}:objectAdmin" "gs://${CLOUDBUILD_BUCKET}" 2>/dev/null || true
echo -e "${GREEN}✅ Cloud Build bucket ready${NC}"
echo ""

# 2) App storage bucket (PDFs)
echo -e "${BLUE}🪣 Creating app storage bucket: gs://${APP_BUCKET}${NC}"
if ! gsutil ls -b "gs://${APP_BUCKET}" &>/dev/null; then
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${APP_BUCKET}"
    echo -e "${GREEN}✅ App bucket created${NC}"
else
    echo -e "${YELLOW}⚠️  Bucket already exists${NC}"
fi
echo ""

# 3) Grant default Cloud Run SA access to the app bucket (no key creation - org policy may block it)
#    Cloud Run uses the default compute SA; app will use Application Default Credentials (ADC).
echo -e "${BLUE}🔑 Granting default Cloud Run SA access to app bucket (no keys; using ADC)${NC}"
gsutil iam ch "serviceAccount:${COMPUTE_SA}:objectAdmin" "gs://${APP_BUCKET}"
echo -e "${GREEN}✅ Default compute SA (${COMPUTE_SA}) can access gs://${APP_BUCKET}${NC}"
echo -e "${YELLOW}   (Key creation skipped - app uses Application Default Credentials.)${NC}"
echo ""

# Summary
echo -e "${GREEN}✅ Static variables setup complete${NC}"
echo ""
echo -e "${BLUE}📋 Summary${NC}"
echo "  PROJECT_ID:              ${PROJECT_ID}"
echo "  PROJECT_NUMBER:          ${PROJECT_NUMBER}"
echo "  APP_BUCKET:              gs://${APP_BUCKET}"
echo "  CLOUDBUILD_BUCKET:       gs://${CLOUDBUILD_BUCKET}"
echo "  ARTIFACT_REGISTRY:       ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"
echo "  CREDENTIALS:             Application Default Credentials (default compute SA)"
echo ""
echo -e "${GREEN}Next: run deploy with PROJECT_ID=${PROJECT_ID}${NC}"
echo "  PROJECT_ID=${PROJECT_ID} ./deploy_gcp.sh"
echo ""
