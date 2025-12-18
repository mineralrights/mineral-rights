# 🚀 Document AI Deployment Fix Guide

## 🚨 Issue Identified

The error `Unknown splitting strategy: document_ai` occurs because the deployed version doesn't have the updated code with Document AI integration.

## ✅ Immediate Fix Applied

I've temporarily reverted the frontend to use `smart_detection` as the default strategy to fix the immediate deployment issue:

### Changes Made:
1. **Frontend**: Changed default strategy from `document_ai` to `smart_detection`
2. **API**: Updated default strategy in both endpoints
3. **UI**: Disabled Document AI option with "Coming soon" label

## 🔧 Full Deployment Steps

To properly deploy the Document AI integration:

### 1. **Update Dependencies**

Make sure your deployment includes the new Google Cloud dependencies:

```bash
# In your deployment environment
pip install google-cloud-documentai==2.32.0 google-auth==2.35.0
```

### 2. **Set Environment Variables**

Add these environment variables to your Vercel deployment:

```bash
# Required: Your Document AI processor endpoint
DOCUMENT_AI_ENDPOINT=https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process

# Optional: Path to Google Cloud service account JSON file
DOCUMENT_AI_CREDENTIALS_PATH=/path/to/service-account.json

# Required: Anthropic API key for classification
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### 3. **Deploy Updated Code**

Deploy the complete updated codebase that includes:

- ✅ `src/mineral_rights/document_ai_service.py`
- ✅ `src/mineral_rights/deed_tracker.py`
- ✅ Updated `src/mineral_rights/document_classifier.py`
- ✅ Updated `api/app.py`
- ✅ Updated `requirements.txt`

### 4. **Re-enable Document AI**

After successful deployment, update the frontend to use Document AI:

```typescript
// In web/src/app/page.tsx
const [splittingStrategy, setSplittingStrategy] = useState<SplittingStrategy>("document_ai");
```

```python
# In api/app.py
splitting_strategy: str = Form("document_ai")
```

And remove the `disabled` attribute from the Document AI option:

```typescript
// In web/src/components/ProcessingModeSelector.tsx
<option value="document_ai">
  🤖 Document AI - Google Cloud custom trained model (Best accuracy)
</option>
```

## 🧪 Testing Deployment

### 1. **Health Check**
```bash
curl https://your-app.vercel.app/health
```

### 2. **Debug Info**
```bash
curl https://your-app.vercel.app/debug
```

### 3. **Test Document AI**
```bash
curl -X POST https://your-app.vercel.app/predict \
  -F "file=@test.pdf" \
  -F "processing_mode=multi_deed" \
  -F "splitting_strategy=document_ai"
```

## 🔍 Troubleshooting

### If Document AI Still Fails:

1. **Check Credentials**: Verify Google Cloud service account has Document AI permissions
2. **Check Endpoint**: Ensure the processor endpoint is correct and active
3. **Check Logs**: Look for authentication or permission errors
4. **Fallback**: The system will automatically fall back to `smart_detection`

### Common Issues:

1. **Authentication Error**: Set up Google Cloud credentials properly
2. **Permission Error**: Ensure service account has Document AI access
3. **Network Error**: Check if Vercel can reach Google Cloud APIs
4. **Timeout Error**: Document AI processing may take longer than expected

## 📋 Deployment Checklist

- [ ] Updated code deployed to Vercel
- [ ] Google Cloud dependencies installed
- [ ] Environment variables set
- [ ] Health check passes
- [ ] Document AI endpoint accessible
- [ ] Fallback system working
- [ ] Frontend updated to use Document AI

## 🎯 Next Steps

1. **Deploy the complete updated codebase**
2. **Set up Google Cloud credentials**
3. **Test with a real multi-deed PDF**
4. **Monitor performance and accuracy**
5. **Re-enable Document AI as default**

The system is designed to be robust - even if Document AI fails, it will automatically fall back to the existing smart detection method, so your app will continue working while you set up the full integration.