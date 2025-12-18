# 🚀 Deployment Guide: Connection Timeout Fixes

## Overview

This guide explains how to deploy the fixes for the connection timeout issues in your mineral rights classification app. The fixes address the "connection lost" errors that occur when processing large PDF files.

## 🔧 Issues Fixed

### 1. **Missing `split_pdf_by_deeds` Method**
- **Problem**: The `process_multi_deed_document` method was calling a non-existent `split_pdf_by_deeds` method
- **Solution**: Implemented the missing method with multiple splitting strategies

### 2. **Connection Timeouts During Long Processing**
- **Problem**: No heartbeat mechanism to keep connections alive during long operations
- **Solution**: Added heartbeat system with 30-second intervals

### 3. **No Retry Logic**
- **Problem**: Connection failures resulted in immediate errors
- **Solution**: Added retry logic with exponential backoff (up to 3 retries)

### 4. **Server Timeout Configuration**
- **Problem**: Server timeouts were too short for long document processing
- **Solution**: Increased timeout configurations in Dockerfile

### 5. **No Progress Updates**
- **Problem**: Users couldn't see progress during long processing
- **Solution**: Added progress updates every 3 pages

## 📋 Files Modified

### Backend (API)
- `src/mineral_rights/document_classifier.py` - Added missing methods and progress updates
- `api/app.py` - Added heartbeat mechanism and better error handling
- `Dockerfile` - Increased timeout configurations

### Frontend (Web App)
- `web/src/lib/api.ts` - Added retry logic and heartbeat handling

## 🚀 Deployment Steps

### Step 1: Test Locally
```bash
# Run the test script to verify fixes
python test_connection_fix.py
```

### Step 2: Build and Deploy Backend
```bash
# Build the Docker image with new timeout settings
docker build -t mineral-rights-api .

# Deploy to your hosting platform (Render, etc.)
# The Dockerfile now includes proper timeout configurations
```

### Step 3: Deploy Frontend
```bash
# Deploy the updated web app to Vercel
cd web
vercel --prod
```

### Step 4: Verify Deployment
1. Upload a large PDF file (>50 pages)
2. Monitor the processing progress
3. Check that heartbeats are being sent (check browser console)
4. Verify that the connection doesn't timeout

## 🔍 Monitoring and Debugging

### Check Heartbeats
Look for these messages in the browser console:
```
💓 Heartbeat received
```

### Check Progress Updates
Look for progress messages like:
```
📊 Progress: 3/15 pages processed (20.0%)
```

### Check Retry Logic
Look for retry messages like:
```
🔄 Retrying connection (1/3)...
```

## ⚙️ Configuration Options

### Heartbeat Interval
- **Current**: 30 seconds
- **Location**: `api/app.py` line ~130
- **Adjustment**: Change `heartbeat_interval = 30`

### Retry Attempts
- **Current**: 3 attempts
- **Location**: `web/src/lib/api.ts` line ~45
- **Adjustment**: Change `maxRetries = 3`

### Server Timeouts
- **Current**: 300 seconds (5 minutes)
- **Location**: `Dockerfile` line ~28
- **Adjustment**: Modify `--timeout-keep-alive` and `--timeout-graceful-shutdown`

## 🧪 Testing Large Files

### Test Scenarios
1. **Small PDF** (<10 pages) - Should work as before
2. **Medium PDF** (10-50 pages) - Should show progress updates
3. **Large PDF** (>50 pages) - Should maintain connection with heartbeats
4. **Very Large PDF** (>100 pages) - Should process without timeouts

### Expected Behavior
- ✅ Progress updates every 3 pages
- ✅ Heartbeat messages every 30 seconds
- ✅ Automatic retry on connection failures
- ✅ Proper cleanup of temporary files
- ✅ No "connection lost" errors

## 🚨 Troubleshooting

### If Connection Still Times Out
1. Check server logs for errors
2. Verify API key is valid
3. Check network connectivity
4. Increase timeout values if needed

### If Processing Fails
1. Check that all dependencies are installed
2. Verify PDF file is valid
3. Check API rate limits
4. Review error logs

### If Frontend Shows Errors
1. Check browser console for JavaScript errors
2. Verify API endpoint is accessible
3. Check CORS configuration
4. Test with smaller files first

## 📊 Performance Improvements

### Before Fixes
- ❌ Connection timeouts on large files
- ❌ No progress feedback
- ❌ Immediate failure on network issues
- ❌ Missing method errors

### After Fixes
- ✅ Stable connections for hours-long processing
- ✅ Real-time progress updates
- ✅ Automatic retry and recovery
- ✅ Proper error handling
- ✅ Memory cleanup

## 🔄 Future Enhancements

### Potential Improvements
1. **Parallel Processing**: Process multiple deeds simultaneously
2. **Resume Capability**: Resume interrupted processing
3. **Progress Persistence**: Save progress to database
4. **WebSocket Support**: Real-time bidirectional communication
5. **Queue System**: Handle multiple concurrent requests

### Monitoring
1. **Metrics Dashboard**: Track processing times and success rates
2. **Alert System**: Notify on failures or timeouts
3. **Logging**: Enhanced logging for debugging
4. **Health Checks**: Regular system health monitoring

## 📞 Support

If you encounter issues after deployment:
1. Check the logs for error messages
2. Run the test script to verify functionality
3. Review the configuration settings
4. Test with smaller files first

The fixes should resolve the connection timeout issues and allow your app to process very large PDF files reliably.
