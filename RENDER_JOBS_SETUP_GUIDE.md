# 🚀 Render Jobs Setup Guide

## Overview

This guide will help you set up Render Jobs for long-running processing (8+ hours) without the timeout constraints of web services.

## 📋 Prerequisites

1. **Render Account**: You already have this
2. **GitHub Repository**: Your code should be in a GitHub repo
3. **API Key**: Your Anthropic API key

## 🛠️ Step 1: Prepare Your Repository

### 1.1 Update Requirements
Your `requirements-clean.txt` is ready for deployment. Make sure it's in your repo root.

### 1.2 Add Job Files
The following files should be in your repository:
- `render.yaml` - Blueprint configuration
- `render_jobs_solution.py` - Job processing script
- `job_manager.py` - Job management utilities
- `job_api_endpoints.py` - API endpoints for job management

### 1.3 Update Your Main API
Add the job endpoints to your `api/app.py`:

```python
# Add this import at the top
from job_api_endpoints import job_router

# Add this after creating your FastAPI app
app.include_router(job_router)
```

## 🚀 Step 2: Deploy to Render

### 2.1 Using Render Blueprint (Recommended)

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +"** → **"Blueprint"**
3. **Connect your GitHub repository**
4. **Select your repository** and branch
5. **Render will detect `render.yaml`** and show you the services to deploy
6. **Review the configuration**:
   - Web Service: `mineral-rights-api`
   - Job Service: `mineral-rights-processor`
7. **Click "Apply"** to deploy both services

### 2.2 Manual Deployment (Alternative)

If you prefer to deploy manually:

#### Deploy Web Service:
1. **New +** → **"Web Service"**
2. **Connect GitHub** and select your repo
3. **Configure**:
   - **Name**: `mineral-rights-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-clean.txt`
   - **Start Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Starter` (or higher)

#### Deploy Job Service:
1. **New +** → **"Background Worker"**
2. **Connect GitHub** and select your repo
3. **Configure**:
   - **Name**: `mineral-rights-processor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-clean.txt`
   - **Start Command**: `python render_jobs_solution.py "$INPUT_PDF_PATH" --mode "$PROCESSING_MODE" --strategy "$SPLITTING_STRATEGY"`
   - **Plan**: `Starter` (or higher)

## 🔧 Step 3: Configure Environment Variables

### 3.1 For Web Service:
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `PYTHON_VERSION`: `3.11`

### 3.2 For Job Service:
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `INPUT_PDF_PATH`: `/tmp/input.pdf` (default)
- `PROCESSING_MODE`: `multi_deed` (default)
- `SPLITTING_STRATEGY`: `smart_detection` (default)
- `PYTHON_VERSION`: `3.11`

## 🧪 Step 4: Test the Setup

### 4.1 Test Web Service
```bash
curl https://your-web-service-url.onrender.com/health
```

### 4.2 Test Job Creation
```bash
curl -X POST https://your-web-service-url.onrender.com/jobs/create \
  -F "file=@test-document.pdf" \
  -F "processing_mode=multi_deed" \
  -F "splitting_strategy=smart_detection"
```

### 4.3 Monitor Job Status
```bash
curl https://your-web-service-url.onrender.com/jobs/{job_id}/status
```

## 📊 Step 5: Using the Job System

### 5.1 Create a Long-Running Job

Instead of using `/predict` for long documents, use `/jobs/create`:

```javascript
// Frontend code example
const formData = new FormData();
formData.append('file', pdfFile);
formData.append('processing_mode', 'multi_deed');
formData.append('splitting_strategy', 'smart_detection');

const response = await fetch('/jobs/create', {
  method: 'POST',
  body: formData
});

const { job_id } = await response.json();
console.log('Job created:', job_id);
```

### 5.2 Monitor Job Progress

```javascript
// Poll for job status
const checkJobStatus = async (jobId) => {
  const response = await fetch(`/jobs/${jobId}/status`);
  const status = await response.json();
  
  if (status.status === 'completed') {
    // Get results
    const resultResponse = await fetch(`/jobs/${jobId}/result`);
    const result = await resultResponse.json();
    return result;
  } else if (status.status === 'failed') {
    throw new Error('Job failed');
  } else {
    // Still processing, check again later
    setTimeout(() => checkJobStatus(jobId), 5000);
  }
};
```

### 5.3 Stream Job Logs (Real-time Updates)

```javascript
// Stream job logs for real-time updates
const streamJobLogs = (jobId) => {
  const eventSource = new EventSource(`/jobs/${jobId}/logs/stream`);
  
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.final) {
      eventSource.close();
      // Job completed
    } else {
      // Update UI with log message
      console.log('Job log:', data.log);
    }
  };
};
```

## 🔄 Step 6: Triggering Render Jobs

### 6.1 Manual Job Triggering

For testing, you can manually trigger jobs through the Render dashboard:

1. **Go to your Job Service** in Render dashboard
2. **Click "Manual Deploy"**
3. **Set environment variables**:
   - `INPUT_PDF_PATH`: Path to your test PDF
   - `PROCESSING_MODE`: `multi_deed`
   - `SPLITTING_STRATEGY`: `smart_detection`
4. **Deploy** to start the job

### 6.2 Programmatic Job Triggering

The job manager provides a simple interface for triggering jobs:

```python
from job_manager import create_processing_job

# Create a job
job_id = create_processing_job(
    "/path/to/document.pdf",
    "multi_deed",
    "smart_detection"
)

# Monitor progress
from job_manager import get_job_status, get_job_result

while True:
    status = get_job_status(job_id)
    if status['status'] == 'completed':
        result = get_job_result(job_id)
        break
    time.sleep(10)
```

## 📈 Step 7: Monitoring and Scaling

### 7.1 Monitor Job Performance

- **Render Dashboard**: Check job logs and resource usage
- **API Endpoints**: Use `/jobs/{job_id}/status` for programmatic monitoring
- **Logs**: Stream logs via `/jobs/{job_id}/logs/stream`

### 7.2 Scale Resources

If jobs are taking too long or failing due to resource constraints:

1. **Upgrade Plan**: Go to Render dashboard → Your Job Service → Settings → Plan
2. **Recommended Plans**:
   - **Starter**: 512MB RAM, 0.1 CPU (good for small documents)
   - **Standard**: 1GB RAM, 0.5 CPU (good for medium documents)
   - **Pro**: 2GB RAM, 1 CPU (good for large documents)

### 7.3 Optimize Processing

- **Adjust chunk size** in your processing code
- **Use memory-efficient processing** methods
- **Monitor memory usage** via the API endpoints

## 🚨 Troubleshooting

### Common Issues:

1. **Job Not Starting**:
   - Check environment variables are set correctly
   - Verify the start command in Render dashboard
   - Check build logs for errors

2. **Job Failing**:
   - Check job logs in Render dashboard
   - Verify PDF file path is correct
   - Ensure API key is set properly

3. **Job Taking Too Long**:
   - Consider upgrading to a higher plan
   - Optimize your processing code
   - Use smaller chunk sizes

4. **API Endpoints Not Working**:
   - Ensure job router is included in your FastAPI app
   - Check that all dependencies are installed
   - Verify the web service is running

### Getting Help:

1. **Render Logs**: Check the logs in your Render dashboard
2. **API Health**: Use `/jobs/health` endpoint to check system status
3. **Render Support**: Contact Render support for platform issues

## 🎯 Benefits of This Setup

✅ **No Timeout Limits**: Jobs can run for 8+ hours without disconnection
✅ **Better Resource Allocation**: Dedicated resources for long-running tasks
✅ **Real-time Monitoring**: Stream logs and monitor progress
✅ **Scalable**: Easy to upgrade resources as needed
✅ **Reliable**: Jobs run to completion without web service constraints
✅ **Cost Effective**: Pay only for the resources you use

## 🚀 Next Steps

1. **Deploy the setup** using the steps above
2. **Test with a small document** first
3. **Monitor performance** and adjust resources as needed
4. **Update your frontend** to use the new job endpoints
5. **Scale up** for production workloads

This setup will solve your 8-hour processing requirement while keeping you on the Render platform!
