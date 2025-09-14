# Deployment Checklist

## Pre-Deployment Testing

### 1. Local Environment Test
```bash
# Set your API key
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

# Run comprehensive local test
python test_local_production.py
```

### 2. Required Environment Variables
- [ ] `ANTHROPIC_API_KEY` - Anthropic API key for LLM classification
- [ ] `GOOGLE_APPLICATION_CREDENTIALS` - Path to Google Cloud service account (if using Document AI)
- [ ] `PORT` - Port number (defaults to 8080 for Cloud Run)

### 3. Dependencies Check
- [ ] All Python packages in `api/requirements.txt` are installed
- [ ] Google Cloud credentials are properly configured
- [ ] Anthropic API key is valid and has sufficient credits

## Deployment Steps

### 1. Set Environment Variables in Cloud Run
```bash
# Set the Anthropic API key
gcloud run services update mineral-rights-api \
  --region=us-central1 \
  --project=mineral-rights-app \
  --set-env-vars="ANTHROPIC_API_KEY=your-anthropic-api-key-here"
```

### 2. Deploy the Application
```bash
# Build and deploy
gcloud builds submit --config cloudbuild.yaml . --project=mineral-rights-app
```

### 3. Verify Deployment
```bash
# Check health endpoint
curl https://mineral-rights-api-1081023230228.us-central1.run.app/health

# Check logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=mineral-rights-api" --limit=10 --project=mineral-rights-app
```

## Post-Deployment Testing

### 1. Health Check
- [ ] `/health` endpoint returns 200 OK
- [ ] No startup errors in logs

### 2. Functional Test
- [ ] Upload a test PDF via the web interface
- [ ] Verify classification results are returned
- [ ] Check that processing completes without errors

### 3. Error Monitoring
- [ ] Monitor Cloud Run logs for any errors
- [ ] Check that API key is properly loaded
- [ ] Verify DocumentProcessor initializes correctly

## Common Issues and Solutions

### Issue: "ANTHROPIC_API_KEY not found in environment"
**Solution**: Set the environment variable in Cloud Run:
```bash
gcloud run services update mineral-rights-api \
  --region=us-central1 \
  --project=mineral-rights-app \
  --set-env-vars="ANTHROPIC_API_KEY=your-key-here"
```

### Issue: "Model not initialized"
**Solution**: Check that the API key is valid and the DocumentProcessor can initialize

### Issue: Port 8000 vs 8080
**Solution**: The app now uses `PORT` environment variable (defaults to 8080)

### Issue: Import errors in production
**Solution**: Ensure all dependencies are in `requirements.txt` and properly installed

## Testing Commands

### Local Testing
```bash
# Full test suite
python test_local_production.py

# Test specific components
python -c "from app import app; print('API OK')"
python -c "from mineral_rights.document_classifier import DocumentProcessor; print('Processor OK')"
```

### Production Testing
```bash
# Health check
curl https://mineral-rights-api-1081023230228.us-central1.run.app/health

# Test with curl (replace with your actual API key)
curl -X POST "https://mineral-rights-api-1081023230228.us-central1.run.app/predict" \
  -F "file=@test.pdf" \
  -F "processing_mode=single_deed" \
  -F "splitting_strategy=document_ai"
```
