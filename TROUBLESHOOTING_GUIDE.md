# 🔧 Troubleshooting Guide - SSL Errors and Processing Freezes

## 🚨 Current Issue

**Problem**: Multi-deed processing still freezes with SSL errors:
- `net::ERR_SSL_BAD_RECORD_MAC_ALERT`
- `TypeError: Failed to fetch`
- Processing stuck at "processing" status

## 🔍 Diagnostic Steps

### Step 1: Check API Health
Visit these endpoints to diagnose the issue:

```bash
# Health check
curl https://your-app.onrender.com/health

# Debug information
curl https://your-app.onrender.com/debug

# Memory status
curl https://your-app.onrender.com/memory-status
```

### Step 2: Check Render Logs
1. Go to your Render dashboard
2. Click on your service
3. Go to "Logs" tab
4. Look for:
   - ✅ "DocumentProcessor initialized successfully"
   - ❌ Import errors (fitz, psutil, anthropic)
   - ❌ API key issues
   - ❌ Memory errors

### Step 3: Test API Endpoints
Run the test script:
```bash
python test_api_endpoints.py https://your-app.onrender.com
```

## 🚨 Common Issues and Solutions

### Issue 1: DocumentProcessor Not Initializing
**Symptoms**: 
- `/health` shows `processor_initialized: false`
- `/debug` shows import errors

**Solutions**:
1. **Check API Key**: Ensure `ANTHROPIC_API_KEY` is set in Render environment variables
2. **Check Dependencies**: Verify all packages are installed correctly
3. **Check Logs**: Look for specific import errors

### Issue 2: Import Errors
**Symptoms**:
- `/debug` shows import failures
- Server crashes on startup

**Common Missing Packages**:
```bash
# If fitz fails:
pip install PyMuPDF

# If psutil fails:
pip install psutil

# If anthropic fails:
pip install anthropic
```

### Issue 3: Memory Issues
**Symptoms**:
- Server restarts frequently
- Memory usage grows continuously
- Processing freezes

**Solutions**:
1. **Check Memory Usage**: Use `/memory-status` endpoint
2. **Reduce Chunk Size**: Modify memory-efficient processing parameters
3. **Monitor Logs**: Look for memory-related errors

### Issue 4: SSL Connection Issues
**Symptoms**:
- `net::ERR_SSL_BAD_RECORD_MAC_ALERT`
- `TypeError: Failed to fetch`
- Connection timeouts

**Possible Causes**:
1. **Server Crashes**: API server is crashing during processing
2. **Memory Exhaustion**: Server runs out of memory and restarts
3. **Long Processing**: Request times out before completion
4. **Network Issues**: Connection drops during long operations

## 🔧 Enhanced Debugging

### Added Debug Endpoints
The updated API now includes:

1. **`/health`**: Basic health check
2. **`/debug`**: Detailed system information
3. **`/memory-status`**: Memory usage monitoring
4. **Enhanced error handling**: Better error messages and logging

### Enhanced Error Handling
- **Processor initialization**: Retry mechanism if initialization fails
- **Import validation**: Check all required packages on startup
- **Better error messages**: More detailed error information
- **Graceful degradation**: Continue processing even if some components fail

## 🚀 Deployment Checklist

### Before Deployment
- [ ] All dependencies are in `api/requirements.txt`
- [ ] API key is set in Render environment variables
- [ ] Dockerfile uses only API requirements
- [ ] No platform-specific packages

### After Deployment
- [ ] Check `/health` endpoint
- [ ] Verify `/debug` shows all imports working
- [ ] Test with small PDF first
- [ ] Monitor memory usage
- [ ] Check Render logs for errors

## 📊 Expected Behavior

### Successful Startup Logs
```
🔧 Initializing DocumentProcessor...
API Key present: Yes
✅ PyMuPDF (fitz) imported successfully
✅ psutil imported successfully
✅ anthropic imported successfully
✅ DocumentProcessor initialized successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Successful Processing Logs
```
💾 Starting job - Memory usage: 45.2 MB
🎯 Processing mode: 'multi_deed'
📁 File: test.pdf
⏰ Started at: 2024-01-15 10:30:00
🔧 Starting multi-deed processing with strategy: smart_detection
📄 Split into 2 deed files
Processing deed 1/2...
✅ Deed 1 completed: 0 (confidence: 0.750)
Processing deed 2/2...
✅ Deed 2 completed: 1 (confidence: 0.850)
🎯 Multi-deed processing completed: 2 deeds processed
🧹 Cleaning up temporary deed files...
✅ Multi-deed processing completed successfully: 2 deeds
```

## 🛠️ Troubleshooting Commands

### Check API Status
```bash
# Health check
curl -s https://your-app.onrender.com/health | jq

# Debug info
curl -s https://your-app.onrender.com/debug | jq

# Memory status
curl -s https://your-app.onrender.com/memory-status | jq
```

### Test Processing
```bash
# Test with small file
curl -X POST https://your-app.onrender.com/predict \
  -F "file=@small_test.pdf" \
  -F "processing_mode=single_deed"
```

### Monitor Logs
```bash
# Watch Render logs in real-time
# (Use Render dashboard or CLI)
```

## 🎯 Next Steps

1. **Deploy the updated API** with enhanced debugging
2. **Check the debug endpoints** to identify the specific issue
3. **Review Render logs** for detailed error information
4. **Test with small files** before trying large documents
5. **Monitor memory usage** during processing

## 📞 If Issues Persist

If the problem continues after following this guide:

1. **Share the debug output** from `/debug` endpoint
2. **Share relevant Render logs** showing the error
3. **Test with a minimal PDF** to isolate the issue
4. **Check if single-deed processing works** (to isolate multi-deed issues)

---

**Goal**: Identify the root cause of the SSL errors and processing freezes using the enhanced debugging capabilities.
