# ⏱️ Timeout Configuration for 8+ Hour Processing

## Overview

This document outlines all timeout configurations that have been optimized to support processing sessions of 8+ hours without connection timeouts.

## 🔧 Server-Side Timeouts

### Dockerfile (Uvicorn Server)
```dockerfile
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "28800", "--timeout-graceful-shutdown", "28800"]
```

**Settings:**
- **Keep-alive timeout**: 28800 seconds (8 hours)
- **Graceful shutdown**: 28800 seconds (8 hours)

### API Server (FastAPI)
**Location**: `api/app.py`

**Heartbeat System:**
- **Heartbeat interval**: 10 seconds (very frequent for long sessions)
- **Message timeout**: 3.0 seconds
- **Queue size**: 1000 messages (increased for long sessions)
- **Session tracking**: Automatic session duration monitoring

**Memory Management:**
- **Garbage collection**: Every 5 minutes during heartbeats
- **Queue overflow protection**: Automatic cleanup
- **Resource tracking**: Job metadata and timing

**Headers:**
```python
headers={
    "Cache-Control": "no-cache",
    "Connection": "keep-alive", 
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Cache-Control",
    "X-Accel-Buffering": "no"  # Disable nginx buffering
}
```

## 💻 Client-Side Timeouts

### Frontend (TypeScript/JavaScript)
**Location**: `web/src/lib/api.ts`

**Connection Management:**
- **Heartbeat timeout**: 1,800,000 ms (30 minutes)
- **Max retries**: 10 attempts
- **Retry delays**: 5s, 10s, 15s, 20s, 25s, 30s, 35s, 40s, 45s, 50s
- **Heartbeat check interval**: 60 seconds
- **Session tracking**: Real-time session duration display

**Error Handling:**
- Automatic reconnection on network failures
- Exponential backoff for retries
- Detailed error messages with session duration
- Session persistence across reconnections

## 📊 Progress Tracking

### Real-Time Updates
**Location**: `src/mineral_rights/document_classifier.py`

**Progress Features:**
- **Update frequency**: Every 3 pages
- **Time tracking**: Per-page processing time
- **Estimates**: Remaining time calculation
- **Performance metrics**: Average time per page
- **Session duration**: Hours and minutes display
- **Memory management**: Automatic garbage collection every 30 minutes

**Sample Output:**
```
📊 Progress: 45/150 pages processed (30.0%)
⏱️  Elapsed: 2h 15m | Est. remaining: 5.2 hours
📈 Avg time per page: 180.0 seconds
🧹 Running garbage collection...
```

## 🚀 Processing Capacity

### Supported Session Lengths
- **Short sessions**: < 5 minutes (original configuration)
- **Medium sessions**: 5-30 minutes (previous configuration)
- **Long sessions**: 30-60 minutes (previous configuration)
- **Very long sessions**: 1-8 hours (current configuration)
- **Extended sessions**: 8+ hours (theoretical maximum)

### Memory Management
- **Temporary file cleanup**: Automatic after processing
- **Queue overflow protection**: Prevents memory leaks
- **Resource cleanup**: Proper disposal of PDF objects
- **Garbage collection**: Every 30 minutes during processing
- **Session metadata**: Track job progress and timing

## 🔍 Monitoring and Debugging

### Heartbeat Monitoring
**Server logs:**
```
💓 Heartbeat sent every 10 seconds
🧹 Running garbage collection...
```

**Client console:**
```
💓 Heartbeat received - Session: 2h 15m
🔄 Retrying connection (1/10) after 3h 45m...
Session active for 4h 30m
```

### Progress Monitoring
**Real-time updates:**
```
📊 Progress: 45/150 pages processed (30.0%)
⏱️  Elapsed: 2h 15m | Est. remaining: 5.2 hours
📈 Avg time per page: 180.0 seconds
```

### Error Detection
**Connection issues:**
```
❌ Connection lost after 6h 30m - no heartbeat received for 30 minutes
❌ Connection lost after 7h 15m - exceeded retry attempts
```

## ⚙️ Configuration Tuning

### For Even Longer Sessions (>8 hours)
If you need to support sessions longer than 8 hours:

1. **Increase server timeouts:**
   ```dockerfile
   --timeout-keep-alive 43200  # 12 hours
   --timeout-graceful-shutdown 43200
   ```

2. **Increase client timeouts:**
   ```typescript
   const heartbeatTimeout = 3600000; // 60 minutes
   ```

3. **Adjust heartbeat frequency:**
   ```python
   heartbeat_interval = 5  # Every 5 seconds
   ```

### For Shorter Sessions (<8 hours)
If you want faster timeouts for shorter sessions:

1. **Decrease server timeouts:**
   ```dockerfile
   --timeout-keep-alive 14400  # 4 hours
   --timeout-graceful-shutdown 14400
   ```

2. **Decrease client timeouts:**
   ```typescript
   const heartbeatTimeout = 900000; // 15 minutes
   ```

## 🧪 Testing Timeout Configurations

### Test Script
Run the test script to verify configurations:
```bash
python test_connection_fix.py
```

### Manual Testing
1. **Upload a very large PDF** (>100 pages)
2. **Monitor browser console** for heartbeat messages
3. **Check progress updates** every 3 pages
4. **Verify no timeouts** during 8+ hour processing
5. **Monitor memory usage** during long sessions

### Expected Behavior
- ✅ Heartbeats every 10 seconds
- ✅ Progress updates every 3 pages with time estimates
- ✅ Session duration tracking in hours and minutes
- ✅ Automatic retry on connection failures (up to 10 attempts)
- ✅ Memory management with garbage collection
- ✅ No timeouts for 8+ hour sessions

## 🚨 Troubleshooting

### Common Issues

**Connection still timing out:**
1. Check if hosting provider has additional timeouts
2. Verify nginx/proxy configurations
3. Check network stability over long periods
4. Increase timeout values further
5. Check for browser limitations

**Memory issues during long processing:**
1. Monitor server memory usage
2. Check for memory leaks in PDF processing
3. Implement page-by-page cleanup
4. Consider streaming processing
5. Monitor garbage collection effectiveness

**Client-side disconnections:**
1. Check browser timeout settings
2. Verify EventSource compatibility
3. Test with different browsers
4. Check for browser extensions interfering
5. Monitor for browser memory limits

### Performance Optimization

**For faster processing:**
1. Reduce image quality (lower zoom factor)
2. Process fewer pages per deed
3. Use simpler splitting strategies
4. Implement parallel processing
5. Optimize OCR settings

**For more reliable connections:**
1. Increase heartbeat frequency
2. Add connection health checks
3. Implement session persistence
4. Use WebSocket instead of SSE
5. Add connection pooling

## 📈 Performance Metrics

### Expected Processing Times
- **Small PDF** (1-10 pages): 2-10 minutes
- **Medium PDF** (10-50 pages): 10-30 minutes  
- **Large PDF** (50-100 pages): 30-60 minutes
- **Very Large PDF** (100-500 pages): 1-4 hours
- **Massive PDF** (500+ pages): 4-8+ hours

### Resource Usage
- **Memory**: ~100-1000MB per processing session
- **CPU**: Variable based on PDF complexity
- **Network**: Minimal (heartbeats + progress updates)
- **Storage**: Temporary files cleaned up automatically
- **Garbage Collection**: Every 30 minutes during processing

## 🔄 Future Enhancements

### Planned Improvements
1. **Session persistence**: Save progress to database
2. **Resume capability**: Continue interrupted processing
3. **Parallel processing**: Process multiple pages simultaneously
4. **WebSocket support**: Real-time bidirectional communication
5. **Queue system**: Handle multiple concurrent requests
6. **Distributed processing**: Split work across multiple servers

### Monitoring Enhancements
1. **Metrics dashboard**: Track processing times and success rates
2. **Alert system**: Notify on failures or timeouts
3. **Performance analytics**: Optimize based on usage patterns
4. **Health checks**: Regular system health monitoring
5. **Resource monitoring**: Track memory and CPU usage

---

**Last Updated**: Current deployment
**Configuration Version**: 3.0 (8+ hour support)
**Tested Duration**: Up to 8 hours
**Recommended Max**: 8 hours per session
**Theoretical Max**: 12+ hours (with additional tuning)
