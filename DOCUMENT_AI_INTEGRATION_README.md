# 🤖 Google Cloud Document AI Integration

This document describes the integration of Google Cloud Document AI custom processor for precise deed boundary detection in multi-deed PDF processing.

## 🎯 Overview

The integration provides a 2-step process:
1. **Deed Boundary Detection**: Uses your custom trained Google Cloud Document AI model to identify deed boundaries
2. **Classification**: Processes each detected deed individually for mineral rights reservations

## 🏗️ Architecture

### Components

1. **DocumentAIService** (`src/mineral_rights/document_ai_service.py`)
   - Handles communication with Google Cloud Document AI
   - Processes PDFs using your custom trained model
   - Extracts deed boundaries and confidence scores
   - Provides fallback to simple page-based splitting

2. **DeedTracker** (`src/mineral_rights/deed_tracker.py`)
   - Tracks and saves deed boundary information
   - Records classification results for each deed
   - Provides session management and summary statistics
   - Enables analysis of splitting accuracy

3. **Updated DocumentProcessor** (`src/mineral_rights/document_classifier.py`)
   - Integrates Document AI service
   - Supports new `document_ai` splitting strategy
   - Tracks processing sessions
   - Provides comprehensive results with boundary information

## 🚀 Setup

### 1. Environment Variables

Set the following environment variables:

```bash
# Required: Your Document AI processor endpoint
export DOCUMENT_AI_ENDPOINT="https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"

# Optional: Path to Google Cloud service account JSON file
export DOCUMENT_AI_CREDENTIALS_PATH="/path/to/service-account.json"

# Required: Anthropic API key for classification
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 2. Dependencies

The following packages have been added to `requirements.txt`:

```
google-cloud-documentai==2.32.0
google-auth==2.35.0
```

Install with:
```bash
pip install -r requirements.txt
```

### 3. Google Cloud Authentication

You have two options for authentication:

#### Option A: Service Account File
1. Download your service account JSON file from Google Cloud Console
2. Set `DOCUMENT_AI_CREDENTIALS_PATH` to the file path
3. Ensure the service account has Document AI permissions

#### Option B: Default Credentials
1. Set up Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```
2. Or set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

## 🎮 Usage

### Frontend

The web interface now includes a new splitting strategy:

```typescript
// New splitting strategy available
type SplittingStrategy = "document_ai" | "smart_detection" | "page_based" | "ai_assisted";
```

**Default Strategy**: `document_ai` (uses your custom trained model)

### API

The API automatically uses Document AI when available:

```python
# Process multi-deed document with Document AI
response = await predict(
    file=pdf_file,
    processing_mode="multi_deed",
    splitting_strategy="document_ai"  # Uses your custom model
)
```

### Programmatic Usage

```python
from src.mineral_rights.document_classifier import DocumentProcessor

# Initialize with Document AI
processor = DocumentProcessor(
    api_key="your-anthropic-key",
    document_ai_endpoint="your-document-ai-endpoint",
    document_ai_credentials="path/to/service-account.json"  # Optional
)

# Process multi-deed document
results = processor.process_multi_deed_document(
    pdf_path="multi_deed.pdf",
    strategy="document_ai"
)

# Results include deed boundary information
for result in results:
    print(f"Deed {result['deed_number']}: {result['classification']}")
    print(f"Pages: {result['deed_boundary_info']['page_range']}")
    print(f"Splitting confidence: {result['splitting_confidence']}")
```

## 📊 Deed Tracking

### Session Management

Each multi-deed processing creates a tracking session:

```python
# Access deed tracker
tracker = processor.deed_tracker

# List all sessions
sessions = tracker.list_sessions()

# Get session summary
summary = tracker.get_session_summary(session_id)
```

### API Endpoints

New endpoints for accessing deed tracking data:

```bash
# List all processing sessions
GET /deed-tracking/sessions

# Get session summary
GET /deed-tracking/sessions/{session_id}

# Get deed boundaries for a session
GET /deed-tracking/sessions/{session_id}/boundaries

# Get classification results for a session
GET /deed-tracking/sessions/{session_id}/results
```

### Tracking Data Structure

```json
{
  "session_id": "session_1234567890_1234",
  "original_filename": "multi_deed.pdf",
  "total_pages": 15,
  "splitting_strategy": "document_ai",
  "document_ai_used": true,
  "deed_boundaries": [
    {
      "deed_number": 1,
      "pages": [0, 1, 2],
      "confidence": 0.95,
      "page_range": "1-3",
      "detected_at": 1234567890.123,
      "splitting_strategy": "document_ai",
      "document_ai_used": true
    }
  ],
  "classification_results": [
    {
      "deed_number": 1,
      "classification": 1,
      "confidence": 0.92,
      "pages_in_deed": 3,
      "processing_time": 2.5,
      "deed_boundary_info": {...}
    }
  ],
  "summary": {
    "total_deeds": 3,
    "deeds_with_reservations": 2,
    "deeds_without_reservations": 1,
    "average_confidence": 0.89,
    "splitting_confidence": 0.92
  }
}
```

## 🔧 Configuration

### Document AI Endpoint

Your custom processor endpoint:
```
https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process
```

### Fallback Behavior

If Document AI is unavailable, the system automatically falls back to:
1. Smart detection (text pattern analysis)
2. Page-based splitting (every 3 pages)
3. Error handling with graceful degradation

### Performance Tuning

The system is optimized for:
- **Memory efficiency**: Processes deeds individually
- **Error handling**: Continues processing even if individual deeds fail
- **Tracking**: Comprehensive logging of all processing steps

## 🧪 Testing

Run the integration test:

```bash
python test_document_ai_integration.py
```

This will test:
- Document AI service initialization
- DocumentProcessor integration
- Deed tracker functionality
- Splitting strategies

## 📈 Benefits

### Accuracy
- **Custom trained model**: Uses your specific deed boundary detection model
- **High confidence scores**: Provides confidence metrics for each boundary
- **Fallback options**: Multiple strategies ensure processing continues

### Tracking
- **Complete audit trail**: Every step is recorded
- **Performance metrics**: Processing times and confidence scores
- **Error analysis**: Detailed error tracking for debugging

### Integration
- **Seamless workflow**: Drop-in replacement for existing splitting methods
- **API compatibility**: Works with existing frontend and API
- **Backward compatibility**: Falls back to existing methods if needed

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```
   ❌ Document AI service initialization failed: 403 Forbidden
   ```
   - Check service account permissions
   - Verify credentials file path
   - Ensure Document AI API is enabled

2. **Endpoint Errors**
   ```
   ❌ Document AI processing failed: 404 Not Found
   ```
   - Verify processor endpoint URL
   - Check processor status in Google Cloud Console
   - Ensure processor is deployed and active

3. **Fallback Activation**
   ```
   ⚠️ Document AI service initialization failed: [error]
   🔄 Will use fallback splitting methods
   ```
   - System automatically falls back to smart detection
   - Check logs for specific error details
   - Verify network connectivity to Google Cloud

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

Check system status:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/debug
```

## 🔮 Future Enhancements

1. **Batch Processing**: Process multiple documents simultaneously
2. **Custom Confidence Thresholds**: Adjustable confidence levels
3. **Advanced Analytics**: Detailed performance metrics and trends
4. **Model Versioning**: Support for multiple Document AI model versions
5. **Real-time Monitoring**: Live processing status and metrics

## 📞 Support

For issues with:
- **Document AI**: Check Google Cloud Console and documentation
- **Integration**: Review logs and test script output
- **Authentication**: Verify service account setup
- **Performance**: Check memory usage and processing times

The system is designed to be robust and provide clear error messages to help diagnose any issues.



