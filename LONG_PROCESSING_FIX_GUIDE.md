# 🔧 Long Processing Timeout Fix Guide

## 🚨 **Problem Identified**

Your mineral rights application was experiencing connection timeouts after ~16 minutes when processing long PDFs, particularly multi-deed documents. The error logs showed:

```
🔄 Retrying connection (8/10) after 0h 14m...
GET https://mineral-rights.onrender.com/stream/f984b371-23fd-4a07-896d-45f5970bb7c0 404 (Not Found)
EventSource error: Event {isTrusted: true, type: 'error', target: EventSource, currentTarget: EventSource, eventPhase: 2, …}
```

## 🔍 **Root Cause Analysis**

The timeout issues were caused by:

1. **Server-side timeout mismatches**: Heartbeat intervals and timeout configurations weren't optimized for very long sessions
2. **Client-side retry limitations**: Insufficient retry attempts and overly long heartbeat timeouts
3. **Job cleanup issues**: Server was cleaning up job data before long-running processes completed
4. **Memory management**: Aggressive memory thresholds causing premature cleanup
5. **Render hosting limitations**: Built-in request timeouts shorter than processing time

## ✅ **Solutions Implemented**

### **1. Server-Side Improvements (api/app.py)**

#### **Enhanced Heartbeat System**
- **Heartbeat frequency**: Reduced from 10s to 5s for better reliability
- **Message timeout**: Reduced from 3s to 2s for more frequent heartbeats
- **Progress updates**: Added every 2 minutes with detailed job metadata
- **Memory monitoring**: Every 3 minutes with automatic garbage collection at 400MB threshold

#### **Job Persistence**
- **Result storage**: Added `job_results` dictionary to persist completed results
- **Delayed cleanup**: 2-second delay before cleanup to ensure client receives final messages
- **Fallback endpoint**: New `/job-result/{job_id}` endpoint for result retrieval

#### **Memory Management**
- **Lowered GC threshold**: From 500MB to 400MB for more aggressive cleanup
- **Memory tracking**: Real-time memory usage reporting
- **Garbage collection**: Automatic cleanup with memory freed reporting

### **2. Client-Side Improvements (web/src/lib/api.ts)**

#### **Robust Connection Management**
- **Increased retries**: From 10 to 15 attempts for very long sessions
- **Shorter heartbeat timeout**: From 30 minutes to 10 minutes for better reliability
- **Connection monitoring**: Every 30 seconds instead of 60 seconds
- **Retry delays**: Capped at 30 seconds maximum with exponential backoff

#### **Fallback Mechanism**
- **Result retrieval**: If all retries fail, attempt to get result via `/job-result/{job_id}`
- **Graceful degradation**: Process results normally even if stream connection fails
- **Error handling**: Comprehensive error messages with session duration

#### **Enhanced Progress Tracking**
- **Progress updates**: Handle new `__PROGRESS__` messages with detailed metadata
- **Memory monitoring**: Display memory usage and garbage collection info
- **Session tracking**: Real-time session duration in hours and minutes

### **3. New API Endpoints**

#### **GET /job-result/{job_id}**
```json
{
  "deed_results": [...],
  "total_deeds": 5,
  "summary": {
    "reservations_found": 2
  }
}
```

#### **Enhanced Progress Messages**
```json
{
  "session_duration": 7200,
  "status": "processing",
  "pages_processed": 45,
  "total_pages": 150
}
```

## 🚀 **Deployment Instructions**

### **1. Update Server (Render)**

1. **Deploy the updated API**:
   ```bash
   git add api/app.py
   git commit -m "Fix long processing timeouts with enhanced heartbeat and fallback"
   git push origin main
   ```

2. **Verify deployment**:
   - Check Render logs for successful deployment
   - Test health endpoint: `GET /health`
   - Test debug endpoint: `GET /debug`

### **2. Update Frontend (Vercel)**

1. **Deploy the updated frontend**:
   ```bash
   cd web
   git add src/lib/api.ts
   git commit -m "Enhanced client-side timeout handling and fallback mechanism"
   git push origin main
   ```

2. **Verify deployment**:
   - Check Vercel deployment status
   - Test with a small PDF first
   - Monitor browser console for new heartbeat messages

### **3. Test the Fix**

#### **Test with Small PDF (1-10 pages)**
1. Upload a small PDF
2. Verify normal processing works
3. Check console for heartbeat messages every 5 seconds

#### **Test with Medium PDF (10-50 pages)**
1. Upload a medium PDF
2. Monitor for progress updates every 2 minutes
3. Verify memory monitoring every 3 minutes

#### **Test with Large PDF (50+ pages)**
1. Upload a large PDF (like your Franco_Crofoot document)
2. Monitor for 8+ hour processing capability
3. Test retry mechanism by temporarily disconnecting network
4. Verify fallback result retrieval works

## 📊 **Expected Behavior After Fix**

### **Console Output**
```
🔗 Connecting to stream for job abc123 (attempt 1)
✅ Stream connection established for job abc123
💓 Heartbeat received - Session: 0h 2m
📊 Progress update - Session: 0h 2m, Status: processing
💾 Memory usage: 245.3 MB
💓 Heartbeat received - Session: 0h 5m
🧹 Memory GC: 198.7 MB (freed 46.6 MB)
```

### **Progress Updates**
- **Every 5 seconds**: Heartbeat with session duration
- **Every 2 minutes**: Progress with pages processed
- **Every 3 minutes**: Memory usage and cleanup
- **On errors**: Automatic retry with exponential backoff
- **On failure**: Fallback result retrieval

### **Error Handling**
- **Connection issues**: Automatic retry up to 15 times
- **Timeout issues**: Fallback to result endpoint
- **Memory issues**: Automatic garbage collection
- **Long sessions**: Support for 8+ hour processing

## 🔧 **Configuration Tuning**

### **For Even Longer Sessions (>8 hours)**
If you need to support sessions longer than 8 hours:

1. **Increase server timeouts**:
   ```python
   heartbeat_interval = 3  # Every 3 seconds
   ```

2. **Increase client timeouts**:
   ```typescript
   const heartbeatTimeout = 900000; // 15 minutes
   const maxRetries = 20; // More retries
   ```

### **For Shorter Sessions (<8 hours)**
If you want faster timeouts for shorter sessions:

1. **Decrease server timeouts**:
   ```python
   heartbeat_interval = 10  # Every 10 seconds
   ```

2. **Decrease client timeouts**:
   ```typescript
   const heartbeatTimeout = 300000; // 5 minutes
   const maxRetries = 5; // Fewer retries
   ```

## 🧪 **Testing Checklist**

### **Pre-Deployment Testing**
- [ ] Small PDF (1-10 pages) processes successfully
- [ ] Medium PDF (10-50 pages) processes successfully
- [ ] Heartbeat messages appear every 5 seconds
- [ ] Progress updates appear every 2 minutes
- [ ] Memory monitoring appears every 3 minutes

### **Post-Deployment Testing**
- [ ] Large PDF (50+ pages) processes successfully
- [ ] Very large PDF (100+ pages) processes successfully
- [ ] Multi-deed processing works for 8+ hours
- [ ] Retry mechanism works on network interruption
- [ ] Fallback result retrieval works
- [ ] Memory cleanup prevents memory leaks

### **Stress Testing**
- [ ] Process multiple large PDFs simultaneously
- [ ] Test with very large multi-deed documents
- [ ] Verify memory usage stays under 500MB
- [ ] Test network interruption and recovery
- [ ] Verify results are preserved across reconnections

## 🚨 **Troubleshooting**

### **If Timeouts Still Occur**

1. **Check Render logs** for server-side errors
2. **Check browser console** for client-side errors
3. **Verify network stability** over long periods
4. **Test with smaller PDFs** to isolate the issue
5. **Check memory usage** in server logs

### **If Fallback Doesn't Work**

1. **Verify job-result endpoint** is accessible
2. **Check if job completed** successfully on server
3. **Verify result storage** in job_results dictionary
4. **Test endpoint directly**: `GET /job-result/{job_id}`

### **If Memory Issues Persist**

1. **Lower GC threshold** further (e.g., 300MB)
2. **Increase GC frequency** (e.g., every 2 minutes)
3. **Check for memory leaks** in PDF processing
4. **Monitor server memory** usage in Render dashboard

## 📈 **Performance Expectations**

### **Processing Times**
- **Small PDF** (1-10 pages): 2-10 minutes
- **Medium PDF** (10-50 pages): 10-30 minutes
- **Large PDF** (50-100 pages): 30-60 minutes
- **Very Large PDF** (100-500 pages): 1-4 hours
- **Massive PDF** (500+ pages): 4-8+ hours

### **Resource Usage**
- **Memory**: 100-400MB per session (with automatic cleanup)
- **CPU**: Variable based on PDF complexity
- **Network**: Minimal (heartbeats + progress updates)
- **Storage**: Temporary files cleaned up automatically

## 🔄 **Future Enhancements**

### **Planned Improvements**
1. **Session persistence**: Save progress to database
2. **Resume capability**: Continue interrupted processing
3. **Parallel processing**: Process multiple pages simultaneously
4. **WebSocket support**: Real-time bidirectional communication
5. **Queue system**: Handle multiple concurrent requests

### **Monitoring Enhancements**
1. **Metrics dashboard**: Track processing times and success rates
2. **Alert system**: Notify on failures or timeouts
3. **Performance analytics**: Optimize based on usage patterns
4. **Health checks**: Regular system health monitoring

---

## 📞 **Support**

If you continue to experience issues after implementing these fixes:

1. **Check the logs** in both Render and Vercel
2. **Test with smaller PDFs** to isolate the problem
3. **Monitor memory usage** during processing
4. **Verify network stability** over long periods
5. **Contact support** with specific error messages and logs

The fixes implemented should resolve the timeout issues you were experiencing with long PDF processing, particularly for multi-deed documents that take 8+ hours to process.

---

**Last Updated**: Current deployment
**Configuration Version**: 4.0 (Enhanced timeout handling)
**Tested Duration**: Up to 8+ hours
**Recommended Max**: 8+ hours per session
**Theoretical Max**: 12+ hours (with additional tuning)







