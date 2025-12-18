# Google Cloud Migration Guide

This guide will help you migrate your Mineral Rights app from Railway to Google Cloud Run with an async job pattern.

## Prerequisites

1. **Google Cloud CLI** installed and authenticated
2. **Google Cloud Project** created
3. **Environment variables** ready (API keys, credentials)

## Step 1: Set up Google Cloud Project

```bash
# Install Google Cloud CLI (if not already installed)
# On macOS:
brew install google-cloud-sdk

# Create a new project
gcloud projects create mineral-rights-app --name="Mineral Rights App"
gcloud config set project mineral-rights-app

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable documentai.googleapis.com
gcloud services enable cloudtasks.googleapis.com
gcloud services enable firestore.googleapis.com

# Authenticate
gcloud auth login
gcloud auth configure-docker
```

## Step 2: Deploy to Google Cloud

```bash
# Make the deployment script executable
chmod +x deploy_gcp.sh

# Run the deployment script
./deploy_gcp.sh
```

This script will:
- Create a Cloud Storage bucket for files
- Set up Firestore database
- Build and deploy the API service to Cloud Run
- Deploy the job worker to Cloud Run Jobs

## Step 3: Set Environment Variables

After deployment, set the required environment variables in Cloud Run:

```bash
# Set environment variables for the API service
gcloud run services update mineral-rights-api \
  --region=us-central1 \
  --set-env-vars="ANTHROPIC_API_KEY=your_api_key_here" \
  --set-env-vars="DOCUMENT_AI_ENDPOINT=your_endpoint_here" \
  --set-env-vars="GOOGLE_CREDENTIALS_BASE64=your_base64_credentials_here" \
  --set-env-vars="GCS_BUCKET_NAME=mineral-rights-storage-mineral-rights-app"

# Set environment variables for the job worker
gcloud run jobs update mineral-rights-worker \
  --region=us-central1 \
  --set-env-vars="ANTHROPIC_API_KEY=your_api_key_here" \
  --set-env-vars="DOCUMENT_AI_ENDPOINT=your_endpoint_here" \
  --set-env-vars="GOOGLE_CREDENTIALS_BASE64=your_base64_credentials_here" \
  --set-env-vars="GCS_BUCKET_NAME=mineral-rights-storage-mineral-rights-app"
```

## Step 4: Update Frontend

1. **Update API URL** in your frontend:
   ```typescript
   // In web/src/lib/api_async.ts
   const API_CONFIG = {
     baseUrl: 'https://mineral-rights-api-<your-hash>-uc.a.run.app',
   };
   ```

2. **Use the new async components**:
   ```typescript
   import AsyncJobProcessor from '@/components/AsyncJobProcessor';
   ```

## Step 5: Test the Deployment

```bash
# Test the health endpoint
curl https://mineral-rights-api-<your-hash>-uc.a.run.app/health

# Test job creation
curl -X POST https://mineral-rights-api-<your-hash>-uc.a.run.app/jobs \
  -F "file=@test.pdf" \
  -F "processing_mode=single_deed" \
  -F "splitting_strategy=document_ai"
```

## Architecture Overview

### New Async Job Pattern

1. **API Service** (Cloud Run):
   - Handles file uploads
   - Stores files in Cloud Storage
   - Creates job records in Firestore
   - Returns job ID immediately

2. **Job Worker** (Cloud Run Jobs):
   - Processes documents in the background
   - Updates job status in Firestore
   - Stores results in Cloud Storage

3. **Frontend**:
   - Creates jobs via API
   - Polls job status
   - Displays progress and results

### Benefits

- **No timeout issues**: Jobs run independently of HTTP requests
- **Scalable**: Multiple workers can process jobs in parallel
- **Reliable**: Jobs can be retried if they fail
- **Progress tracking**: Real-time status updates
- **Cost effective**: Only pay for actual processing time

## Troubleshooting

### Common Issues

1. **Authentication errors**:
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   ```

2. **API not enabled**:
   ```bash
   gcloud services enable <service-name>
   ```

3. **Environment variables not set**:
   Check Cloud Run service configuration in the console

4. **Job worker not starting**:
   Check Cloud Run Jobs logs in the console

### Logs and Monitoring

- **API Service logs**: `gcloud run services logs read mineral-rights-api --region=us-central1`
- **Job Worker logs**: `gcloud run jobs executions list --job=mineral-rights-worker --region=us-central1`
- **Firestore data**: Check in the Firebase console
- **Cloud Storage**: Check in the Cloud Storage console

## Cost Optimization

- **Cloud Run**: Pay per request and CPU time
- **Cloud Run Jobs**: Pay per execution time
- **Cloud Storage**: Pay per GB stored and transferred
- **Firestore**: Pay per document read/write

## Security

- **Service accounts**: Use least privilege principle
- **Environment variables**: Store sensitive data securely
- **CORS**: Configured for your frontend domains
- **Authentication**: Use Google Cloud IAM

## Next Steps

1. **Monitor performance** and adjust resource allocation
2. **Set up alerts** for failed jobs
3. **Implement retry logic** for transient failures
4. **Add more job types** as needed
5. **Optimize costs** based on usage patterns
