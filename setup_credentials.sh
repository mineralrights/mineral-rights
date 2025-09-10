#!/bin/bash

echo "🔧 Setting up Google Cloud credentials for Document AI"
echo "=================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI not found"
    echo "📥 Please install it first:"
    echo "   - macOS: brew install google-cloud-sdk"
    echo "   - Or download from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ Google Cloud CLI found"

# Check if already authenticated
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "✅ Already authenticated with Google Cloud"
    ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    echo "   Account: $ACTIVE_ACCOUNT"
else
    echo "🔐 Authenticating with Google Cloud..."
    gcloud auth login
fi

# Set up application default credentials
echo "🔑 Setting up Application Default Credentials..."
gcloud auth application-default login

# Set the project
echo "📁 Setting project..."
gcloud config set project 381937358877

# Enable Document AI API
echo "🚀 Enabling Document AI API..."
gcloud services enable documentai.googleapis.com

echo ""
echo "🎉 Setup complete!"
echo "🧪 Now run: python test_document_ai_local.py"
echo ""
echo "📋 What was set up:"
echo "   ✅ Google Cloud authentication"
echo "   ✅ Application Default Credentials"
echo "   ✅ Project set to 381937358877"
echo "   ✅ Document AI API enabled"
