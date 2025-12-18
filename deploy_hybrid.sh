#!/bin/bash

# Deploy Hybrid Mineral Rights API with Cloud Run Jobs
set -e

echo "🚀 Deploying Hybrid Mineral Rights API..."

# Set project ID
PROJECT_ID="mineral-rights-app"
LOCATION="us-central1"
QUEUE_NAME="mineral-rights-queue"

echo "📋 Project: $PROJECT_ID"
echo "📍 Location: $LOCATION"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudtasks.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable storage.googleapis.com

# Create Firestore database if it doesn't exist
echo "🗄️ Setting up Firestore..."
gcloud firestore databases create --location=$LOCATION --quiet || echo "Firestore database already exists"

# Create Cloud Storage bucket if it doesn't exist
echo "🪣 Setting up Cloud Storage..."
BUCKET_NAME="mineral-rights-storage-$PROJECT_ID"
gsutil mb gs://$BUCKET_NAME || echo "Bucket already exists"

# Create Cloud Tasks queue
echo "📋 Setting up Cloud Tasks queue..."
gcloud tasks queues create $QUEUE_NAME --location=$LOCATION || echo "Queue already exists"

# Build and deploy using Cloud Build
echo "🏗️ Building and deploying with Cloud Build..."
gcloud builds submit --config cloudbuild.yaml .

echo "✅ Hybrid deployment completed!"
echo ""
echo "🔗 API URL: https://mineral-rights-api-1081023230228.us-central1.run.app"
echo "📊 Firestore: https://console.cloud.google.com/firestore"
echo "🪣 Storage: https://console.cloud.google.com/storage"
echo "📋 Tasks: https://console.cloud.google.com/cloudtasks"
echo ""
echo "🧪 Test the API:"
echo "curl https://mineral-rights-api-1081023230228.us-central1.run.app/health"
