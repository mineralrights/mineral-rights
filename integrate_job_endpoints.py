#!/usr/bin/env python3
"""
Integration Script for Job Endpoints
====================================

This script shows how to integrate the job endpoints into your existing API.
Run this to add the job functionality to your current api/app.py
"""

def integrate_job_endpoints():
    """
    Instructions for integrating job endpoints into your existing API
    """
    
    integration_instructions = """
# 🚀 How to Integrate Job Endpoints into Your Existing API

## Step 1: Add Import to api/app.py

Add this import at the top of your api/app.py file (after the existing imports):

```python
# Add this import for job endpoints
try:
    from job_api_endpoints import job_router
    JOB_ENDPOINTS_AVAILABLE = True
except ImportError:
    JOB_ENDPOINTS_AVAILABLE = False
    print("⚠️ Job endpoints not available - job_manager.py and job_api_endpoints.py required")
```

## Step 2: Include Job Router

Add this after creating your FastAPI app (around line 18):

```python
# Include job endpoints if available
if JOB_ENDPOINTS_AVAILABLE:
    app.include_router(job_router)
    print("✅ Job endpoints integrated successfully")
else:
    print("⚠️ Job endpoints not integrated - missing dependencies")
```

## Step 3: Update Your Frontend

### For Quick Processing (< 10 minutes):
Continue using your existing `/predict` endpoint.

### For Long Processing (8+ hours):
Use the new job endpoints:

```javascript
// Create a long-running job
const createLongJob = async (file, processingMode = 'multi_deed', strategy = 'smart_detection') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('processing_mode', processingMode);
  formData.append('splitting_strategy', strategy);
  
  const response = await fetch('/jobs/create', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};

// Monitor job progress
const monitorJob = async (jobId) => {
  const response = await fetch(`/jobs/${jobId}/status`);
  return await response.json();
};

// Get job result when completed
const getJobResult = async (jobId) => {
  const response = await fetch(`/jobs/${jobId}/result`);
  return await response.json();
};

// Stream job logs for real-time updates
const streamJobLogs = (jobId, onLog) => {
  const eventSource = new EventSource(`/jobs/${jobId}/logs/stream`);
  
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.final) {
      eventSource.close();
    } else {
      onLog(data.log);
    }
  };
  
  return eventSource;
};
```

## Step 4: Test the Integration

1. **Start your API**: `uvicorn api.app:app --reload`
2. **Test job endpoints**: `python test_render_jobs.py`
3. **Check health**: `curl http://localhost:8000/jobs/health`

## Step 5: Deploy to Render

1. **Commit all files** to your GitHub repository
2. **Deploy using Render Blueprint** (recommended) or manual deployment
3. **Set environment variables** in Render dashboard
4. **Test with actual documents**

## 🎯 Usage Examples

### Example 1: Quick Processing (Existing)
```javascript
// For documents that should complete in < 10 minutes
const response = await fetch('/predict', {
  method: 'POST',
  body: formData
});
```

### Example 2: Long Processing (New)
```javascript
// For documents that might take 8+ hours
const { job_id } = await createLongJob(file, 'multi_deed', 'smart_detection');

// Monitor progress
const checkProgress = async () => {
  const status = await monitorJob(job_id);
  
  if (status.status === 'completed') {
    const result = await getJobResult(job_id);
    console.log('Processing completed:', result);
  } else if (status.status === 'failed') {
    console.error('Processing failed');
  } else {
    // Still processing, check again in 5 seconds
    setTimeout(checkProgress, 5000);
  }
};

checkProgress();
```

### Example 3: Real-time Log Streaming
```javascript
// Stream logs for real-time updates
const logContainer = document.getElementById('logs');

streamJobLogs(jobId, (logMessage) => {
  const logElement = document.createElement('div');
  logElement.textContent = logMessage;
  logContainer.appendChild(logElement);
  logContainer.scrollTop = logContainer.scrollHeight;
});
```

## 🔧 Configuration Options

### Environment Variables for Jobs:
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `RENDER_API_KEY`: (Optional) For programmatic job management
- `RENDER_SERVICE_ID`: (Optional) Your job service ID

### Job Parameters:
- `processing_mode`: "single_deed" or "multi_deed"
- `splitting_strategy`: "smart_detection" or "ai_assisted"

## 📊 Monitoring and Debugging

### Check Job System Health:
```bash
curl https://your-api-url.onrender.com/jobs/health
```

### List All Jobs:
```bash
curl https://your-api-url.onrender.com/jobs/
```

### Get Job Status:
```bash
curl https://your-api-url.onrender.com/jobs/{job_id}/status
```

### Stream Job Logs:
```bash
curl -N https://your-api-url.onrender.com/jobs/{job_id}/logs/stream
```

## 🚨 Troubleshooting

### Common Issues:

1. **Job endpoints not available**:
   - Ensure `job_manager.py` and `job_api_endpoints.py` are in your repo
   - Check that imports are working correctly

2. **Jobs not starting**:
   - Verify environment variables are set in Render
   - Check job service logs in Render dashboard

3. **API errors**:
   - Check that job router is properly included
   - Verify all dependencies are installed

### Getting Help:

1. **Check logs**: Use `/jobs/{job_id}/logs` endpoint
2. **Monitor health**: Use `/jobs/health` endpoint  
3. **Render dashboard**: Check service logs and status
4. **Test locally**: Use `python test_render_jobs.py`

## 🎉 Benefits

✅ **No timeout limits** for long-running jobs
✅ **Real-time monitoring** with log streaming
✅ **Better resource allocation** for intensive tasks
✅ **Scalable** - easy to upgrade resources
✅ **Reliable** - jobs run to completion
✅ **Cost effective** - pay only for what you use

This integration gives you the best of both worlds: quick processing for small documents and unlimited processing time for large documents!
"""
    
    print(integration_instructions)

if __name__ == "__main__":
    integrate_job_endpoints()
