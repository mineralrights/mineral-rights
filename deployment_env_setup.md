# 🚀 Deployment Environment Setup

## ✅ Document AI Credentials Fixed

Your Document AI credentials are now working! The multi-deed processing will use your trained Document AI model instead of falling back to simple page-based splitting.

## 🔧 Environment Variables for Deployment

### For Render (Backend)

Set these environment variables in your Render dashboard:

```bash
# Required: Anthropic API key for classification
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Required: Document AI processor endpoint
DOCUMENT_AI_ENDPOINT=https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process

# Optional: Google Cloud credentials (if using service account)
# GOOGLE_CREDENTIALS_BASE64=base64-encoded-service-account-json
# OR
# DOCUMENT_AI_CREDENTIALS_PATH=/path/to/service-account.json
```

### For Vercel (Frontend)

Update your `web/.env.local`:

```bash
NEXT_PUBLIC_API_URL=https://your-render-app.onrender.com
```

## 🧪 Testing the Fix

### 1. Test Locally First

```bash
# Set your API key
export ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Test the complete pipeline
python test_document_ai_fixed.py
```

### 2. Test with Real PDF

```bash
# Test with a multi-deed PDF
python test_document_ai_local.py
```

### 3. Deploy and Test

1. **Deploy backend** with environment variables
2. **Deploy frontend** with updated API URL
3. **Test with large PDF** (>50 pages)

## 🎯 Expected Results

After this fix:

- ✅ **Document AI splitting** - Uses your trained model for deed boundary detection
- ✅ **Faster processing** - More accurate splitting reduces processing time
- ✅ **Better accuracy** - Custom trained model vs simple page-based splitting
- ✅ **No more fallback** - No more "Using Document AI fallback service" messages

## 🚨 Important Notes

1. **Your trained Document AI model** is now being used for deed boundary detection
2. **Processing should be faster** because the splitting is more accurate
3. **The timeout issues** should be reduced because processing is more efficient
4. **You still need to address platform timeouts** for very large files (300+ pages)

## 🔄 If You Still Have Timeout Issues

The Document AI fix will help, but for very large files (300+ pages), you may still need to:

1. **Migrate to Fly.io** (recommended for no timeout limits)
2. **Use Render Jobs** instead of Web Service
3. **Implement chunking** for very large documents

But first, test with the Document AI fix - it should significantly improve performance!
