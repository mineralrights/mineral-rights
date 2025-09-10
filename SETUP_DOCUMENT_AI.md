# 🔧 Document AI Setup Guide

## 🎯 Goal
Get your Document AI checkpoint working locally so we can test it before integrating into the app.

## 📋 Step-by-Step Setup

### Step 1: Install Google Cloud CLI (if not already installed)

```bash
# On macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### Step 2: Authenticate with Google Cloud

```bash
# Login to your Google Cloud account
gcloud auth login

# Set up Application Default Credentials
gcloud auth application-default login
```

### Step 3: Set your project

```bash
# Set the project ID (replace with your actual project ID)
gcloud config set project 381937358877
```

### Step 4: Enable Document AI API

```bash
# Enable the Document AI API
gcloud services enable documentai.googleapis.com
```

### Step 5: Test the connection

```bash
# Run our test script
python test_document_ai_local.py
```

## 🔑 Alternative: Service Account Method

If you prefer to use a service account JSON file:

### Step 1: Create Service Account
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to IAM & Admin > Service Accounts
3. Create a new service account
4. Grant it "Document AI API User" role
5. Create and download the JSON key file

### Step 2: Test with JSON file
```bash
python test_document_ai_local.py
# When prompted, enter the path to your JSON file
```

## 🧪 Testing Your Checkpoint

Once credentials are set up, the test script will:

1. ✅ **Test Connection**: Verify you can connect to Document AI
2. 📄 **Find PDFs**: Look for test PDFs in your data folders
3. 🔍 **Process Document**: Send a PDF to your custom processor
4. 📊 **Show Results**: Display detected deed boundaries and confidence scores

## 🎯 What to Look For

When testing, you should see:

```
✅ Document AI processing completed!
📊 Document info:
   - Pages: X
   - Text length: XXXX
   - Entities: X

🔍 Found X deed-related entities:
   - DEED_BOUNDARY: 'some text' (confidence: 0.95, pages: [0])
   - DEED_START: 'other text' (confidence: 0.88, pages: [3])
```

## 🚨 Troubleshooting

### Common Issues:

1. **"Credentials not found"**
   - Run: `gcloud auth application-default login`

2. **"API not enabled"**
   - Run: `gcloud services enable documentai.googleapis.com`

3. **"Permission denied"**
   - Ensure your account has Document AI permissions
   - Check if the processor endpoint is correct

4. **"Processor not found"**
   - Verify the processor ID: `895767ed7f252878`
   - Check if the processor is deployed and active

## 📞 Getting Help

If you encounter issues:

1. **Check Google Cloud Console**: Verify your processor is active
2. **Check Permissions**: Ensure your account can access Document AI
3. **Check API Status**: Verify Document AI API is enabled
4. **Check Processor**: Confirm the processor endpoint is correct

## 🎉 Success Indicators

You'll know it's working when you see:
- ✅ Document AI connection established
- ✅ PDF processed successfully  
- ✅ Deed boundaries detected (if present in the PDF)
- ✅ Confidence scores displayed

Once this is working, we can integrate it into your app and remove the non-working splitting methods!



