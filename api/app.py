from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile
import os
import time
import json
import threading
import asyncio
import uuid
from contextlib import redirect_stdout
from io import StringIO
import psutil
import gc

# Import your pipeline
from src.mineral_rights.document_classifier import DocumentProcessor

# GCS imports for large file handling
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
    JOBS_AVAILABLE = False  # We don't need Cloud Jobs anymore
except ImportError:
    GCS_AVAILABLE = False
    JOBS_AVAILABLE = False
    print("⚠️ Google Cloud Storage not available. Install: pip install google-cloud-storage")

# Initialize FastAPI app
app = FastAPI(title="Mineral-Rights API - Simple SSE Version")

# Global job results storage (in production, use Redis or database)
job_results = {}

# CORS configuration - Explicit and robust
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "Cache-Control",
        "Pragma",
        "Expires",
        "User-Agent",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["*"],
    max_age=3600  # Cache preflight for 1 hour
)

# Add cache-busting middleware
@app.middleware("http")
async def add_cache_busting_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Explicit CORS handler for all requests
@app.middleware("http")
async def cors_handler(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# Global processor
processor = None

def initialize_processor():
    global processor
    try:
        print("🔧 Initializing DocumentProcessor...")
        
        # Get API key from environment (REQUIRED)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found in environment")
            return False
            
        # Get Document AI endpoint (OPTIONAL - only needed for multi-deed mode)
        document_ai_endpoint = os.getenv("DOCUMENT_AI_ENDPOINT")
        if not document_ai_endpoint:
            print("⚠️ DOCUMENT_AI_ENDPOINT not found in environment - will use fallback for multi-deed mode")
            # This is OK - Document AI is only needed for multi-deed splitting
            
        print(f"API Key present: {'Yes' if api_key else 'No'}")
        print(f"Document AI Endpoint: {document_ai_endpoint or 'Not set (will use fallback)'}")
        
        # Handle Google credentials
        credentials_path = None
        google_credentials_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        
        if google_credentials_base64:
            try:
                import base64
                import tempfile
                # Decode base64 credentials
                credentials_json = base64.b64decode(google_credentials_base64).decode('utf-8')
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                temp_file.write(credentials_json)
                temp_file.close()
                credentials_path = temp_file.name
                print(f"✅ Created temporary credentials file from base64")
            except Exception as e:
                print(f"⚠️ Failed to decode base64 credentials: {e}")
        
        # Initialize processor with explicit parameters including credentials
        processor = DocumentProcessor(
            api_key=api_key,
            document_ai_endpoint=document_ai_endpoint,
            document_ai_credentials=credentials_path
        )
        
        print("✅ DocumentProcessor initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize DocumentProcessor: {e}")
        import traceback
        traceback.print_exc()
        return False

# Processor will be initialized on first request, not during startup

# Simple in-memory job registry for SSE streaming
jobs: dict[str, asyncio.Queue[str]] = {}
job_metadata: dict[str, dict] = {}
job_start_times: dict[str, float] = {}
job_results: dict[str, dict] = {}

class QueueWriter:
    def __init__(self, queue):
        self.queue = queue
    
    def write(self, text):
        if text.strip():
            try:
                self.queue.put_nowait(text.strip())
            except asyncio.QueueFull:
                pass  # Ignore if queue is full
    
    def flush(self):
        pass

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    processing_mode: str = Form("single_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """Process document directly and return results"""
    print(f"🔍 Processing document - Mode: {processing_mode}, File: {file.filename}")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Save file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            if processor is None:
                try:
                    if not initialize_processor():
                        raise HTTPException(status_code=500, detail="Model not initialized")
                except Exception as e:
                    print(f"❌ Processor initialization failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
            
            print("🤖 Initializing document processor...")
            print(f"🔧 Processing with mode: {processing_mode}")
            
            # Process the document based on mode
            if processing_mode == "single_deed":
                print(f"🔍 About to call process_document for: {file.filename}")
                # Enable high_recall_mode and use appropriate parameters for better detection
                result = processor.process_document(
                    tmp_file_path,
                    max_samples=6,
                    confidence_threshold=0.7,
                    page_strategy="first_few",
                    high_recall_mode=True  # Enable high recall mode to catch reservations
                )
                
                # Debug: Log what we got back
                print(f"🔍 Result keys: {list(result.keys())}")
                print(f"🔍 samples_used: {result.get('samples_used', 'NOT FOUND')}")
                print(f"🔍 classification: {result.get('classification', 'NOT FOUND')}")
                print(f"🔍 confidence: {result.get('confidence', 'NOT FOUND')}")
                
                # Check if LLM processing actually succeeded
                detailed_samples = result.get("detailed_samples", []) or []
                samples_used = result.get("samples_used", 0)
                
                # Handle None case
                if detailed_samples is None:
                    detailed_samples = []
                
                print(f"🔍 detailed_samples length: {len(detailed_samples)}")
                print(f"🔍 samples_used: {samples_used}")
                
                # VALIDATION: Check if LLM calls actually succeeded
                if samples_used == 0 or len(detailed_samples) == 0:
                    error_msg = (
                        f"LLM processing failed - no samples were generated. "
                        f"This may indicate an API key issue, API error, or OCR failure. "
                        f"Please check server logs for details. "
                        f"(samples_used: {samples_used}, detailed_samples: {len(detailed_samples)})"
                    )
                    print(f"❌ {error_msg}")
                    raise HTTPException(
                        status_code=500,
                        detail=error_msg
                    )
                
                if detailed_samples:
                    print(f"🔍 First sample keys: {list(detailed_samples[0].keys()) if detailed_samples[0] else 'EMPTY'}")
                    first_reasoning = detailed_samples[0].get("reasoning", "") if detailed_samples[0] else ""
                    print(f"🔍 First sample reasoning length: {len(str(first_reasoning))}")
                    print(f"🔍 First sample reasoning preview: {str(first_reasoning)[:100] if first_reasoning else 'EMPTY'}")
                
                # Get reasoning from first sample if available
                reasoning = "No reasoning provided"
                if detailed_samples and len(detailed_samples) > 0:
                    reasoning = detailed_samples[0].get("reasoning", "No reasoning provided")
                
                # Convert result to expected format
                response_data = {
                    "has_reservation": result.get("classification", 0) == 1,
                    "confidence": result.get("confidence", 0.0),
                    "reasoning": reasoning,
                    "processing_mode": processing_mode,
                    "filename": file.filename
                }
                
                print(f"🔍 FINAL RESPONSE DATA: {response_data}")
                print(f"🔍 Response reasoning length: {len(reasoning)}")
                print(f"🔍 Response reasoning content: {reasoning[:200] if len(reasoning) > 200 else reasoning}")
                
                return response_data
                
            elif processing_mode == "multi_deed":
                # Use the proper multi-deed processing with Document AI splitting
                result = processor.process_multi_deed_document(tmp_file_path, strategy=splitting_strategy)
                
                # Convert to expected format
                deed_results = []
                for deed_result in result:
                    deed_results.append({
                        "deed_number": deed_result.get("deed_number", 0),
                        "has_reservations": deed_result.get("classification", 0) == 1,
                        "confidence": deed_result.get("confidence", 0.0),
                        "reasoning": deed_result.get("detailed_samples", [{}])[0].get("reasoning", "No reasoning provided") if deed_result.get("detailed_samples") else "No reasoning provided",
                        "pages": deed_result.get("deed_boundary_info", {}).get("pages", []) if deed_result.get("deed_boundary_info") else [],
                        "deed_boundary_info": deed_result.get("deed_boundary_info", {}),
                        "pages_in_deed": deed_result.get("pages_in_deed", 0)
                    })
                
                return {
                    "deed_results": deed_results,
                    "total_deeds": len(deed_results),
                    "processing_mode": processing_mode,
                    "filename": file.filename
                }
                
            elif processing_mode == "page_by_page":
                result = processor.process_document(tmp_file_path, page_strategy="all_pages")
                
                # Convert to page-by-page format
                page_results = []
                if "page_results" in result:
                    for i, page_result in enumerate(result["page_results"]):
                        page_results.append({
                            "page_number": i + 1,
                            "has_reservations": page_result.get("classification", 0) == 1,
                            "confidence": page_result.get("confidence", 0.0),
                            "explanation": page_result.get("detailed_samples", [{}])[0].get("reasoning", "No reasoning provided") if page_result.get("detailed_samples") else "No reasoning provided"
                        })
                
                return {
                    "page_results": page_results,
                    "total_pages": len(page_results),
                    "processing_mode": processing_mode,
                    "filename": file.filename
                }
                
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except HTTPException:
        # Re-raise HTTPExceptions (like our LLM failure errors)
        raise
    except Exception as e:
        print(f"❌ Failed to process document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/stream/{job_id}")
async def stream(job_id: str):
    """Stream processing logs and results via Server-Sent Events"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    queue = jobs[job_id]

    async def event_generator():
        last_heartbeat = time.time()
        heartbeat_interval = 5  # Send heartbeat every 5 seconds
        session_start = time.time()
        
        while True:
            try:
                # Wait for message with timeout for heartbeats
                line = await asyncio.wait_for(queue.get(), timeout=2.0)
                
                # Send the message
                yield f"data: {line}\n\n"
                
                # Check if we're done
                if line == "__END__":
                    break
                    
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                current_time = time.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    session_duration = current_time - session_start
                    yield f"data: __HEARTBEAT__{current_time}|{session_duration}\n\n"
                    last_heartbeat = current_time

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "processor_initialized": processor is not None,
            "timestamp": time.time(),
            "api_key_present": bool(os.getenv("ANTHROPIC_API_KEY")),
            "document_ai_endpoint_present": bool(os.getenv("DOCUMENT_AI_ENDPOINT")),
            "google_credentials_present": bool(os.getenv("GOOGLE_CREDENTIALS_BASE64"))
        }
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/")
async def root():
    """Root endpoint for basic connectivity test"""
    return {"message": "Mineral Rights API is running", "timestamp": time.time()}

@app.get("/heartbeat")
async def heartbeat():
    """Simple heartbeat endpoint for Railway health checks"""
    return {"status": "alive"}

@app.get("/test")
async def test():
    """Simple test endpoint that doesn't require processor initialization"""
    return {"message": "Test endpoint working", "timestamp": time.time()}

@app.get("/test-anthropic")
async def test_anthropic():
    """Test Anthropic API connectivity with detailed diagnostics"""
    import anthropic
    import os
    import httpx
    import socket
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    diagnostics = {
        "api_key_present": bool(api_key),
        "api_key_length": len(api_key) if api_key else 0,
        "api_key_prefix": api_key[:10] + "..." if api_key else None,
        "timestamp": time.time()
    }
    
    if not api_key:
        return {
            "status": "error",
            "message": "ANTHROPIC_API_KEY not found in environment",
            **diagnostics
        }
    
    # Test 1: Basic network connectivity
    try:
        import urllib.request
        import urllib.error
        test_url = "https://api.anthropic.com/v1/messages"
        req = urllib.request.Request(test_url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            diagnostics["network_test"] = {
                "status": "success",
                "http_code": response.getcode(),
                "url": test_url
            }
    except Exception as e:
        diagnostics["network_test"] = {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        }
    
    # Test 2: DNS resolution
    try:
        hostname = "api.anthropic.com"
        ip = socket.gethostbyname(hostname)
        diagnostics["dns_test"] = {
            "status": "success",
            "hostname": hostname,
            "ip": ip
        }
    except Exception as e:
        diagnostics["dns_test"] = {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        }
    
    # Test 3: Anthropic client with timeout
    try:
        timeout = httpx.Timeout(60.0, connect=30.0)
        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=0  # Disable retries for testing
        )
        diagnostics["client_init"] = {"status": "success"}
        
        # Try a simple API call
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'test'"}]
        )
        
        return {
            "status": "success",
            "message": "Anthropic API connection successful",
            "response": response.content[0].text,
            "model": "claude-3-5-haiku-20241022",
            **diagnostics
        }
    except anthropic.APIConnectionError as e:
        import traceback
        return {
            "status": "connection_error",
            "error_type": "APIConnectionError",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            **diagnostics
        }
    except anthropic.APIError as e:
        return {
            "status": "api_error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            **diagnostics
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            **diagnostics
        }

@app.post("/get-signed-upload-url")
async def get_signed_upload_url(request: dict):
    """Get a signed URL for direct GCS upload (bypasses Cloud Run size limits)"""
    print(f"🔍 GCS_AVAILABLE: {GCS_AVAILABLE}")
    if not GCS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Google Cloud Storage not available")
    
    try:
        filename = request.get("filename", "document.pdf")
        content_type = request.get("content_type", "application/pdf")
        
        print(f"🔑 Generating signed URL for: {filename}")
        
        # Initialize GCS client with service account
        try:
            # Try to use base64 encoded credentials first
            credentials_b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
            print(f"🔍 GOOGLE_CREDENTIALS_BASE64 present: {bool(credentials_b64)}")
            if credentials_b64:
                import base64
                import json
                from google.oauth2 import service_account
                
                # Decode the base64 credentials
                credentials_json = base64.b64decode(credentials_b64).decode('utf-8')
                credentials_info = json.loads(credentials_json)
                
                # Create credentials object
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                client = storage.Client(credentials=credentials)
                print("✅ Using base64 encoded service account credentials")
            else:
                # Fallback to default credentials
                client = storage.Client()
                print("✅ Using default service account credentials")
        except Exception as e:
            print(f"❌ GCS client initialization failed: {e}")
            raise HTTPException(status_code=500, detail="GCS client initialization failed")
        
        bucket_name = os.getenv("GCS_BUCKET_NAME", "mineral-rights-pdfs-1759435410")
        bucket = client.bucket(bucket_name)
        
        # Generate unique blob name
        file_id = str(uuid.uuid4())
        blob_name = f"uploads/{file_id}/{filename}"
        blob = bucket.blob(blob_name)
        
        # Generate signed URL (valid for 1 hour) using custom method
        import time
        from datetime import datetime, timedelta
        
        current_time = int(time.time())
        expiration_time = current_time + 3600
        print(f"🕐 Current time: {current_time}, Expiration: {expiration_time}")
        
        # Try alternative signing method
        try:
            # Method 1: Use datetime instead of seconds
            expiration_dt = datetime.utcnow() + timedelta(hours=1)
            signed_url = blob.generate_signed_url(
                expiration=expiration_dt,
                method="PUT",
                content_type=content_type
            )
            print(f"✅ Signed URL generated with datetime method")
        except Exception as e:
            print(f"❌ Datetime method failed: {e}")
            # Fallback to original method
            signed_url = blob.generate_signed_url(
                expiration=3600,  # 1 hour
                method="PUT",
                content_type=content_type
            )
            print(f"✅ Signed URL generated with seconds method")
        
        print(f"🔗 Generated signed URL: {signed_url[:100]}...")
        
        print(f"✅ Signed URL generated: {blob_name}")
        
        return {
            "signed_url": signed_url,
            "bucket_name": bucket_name,
            "blob_name": blob_name,
            "expiration": "1 hour",
            "gcs_url": f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        }
        
    except Exception as e:
        print(f"❌ Signed URL generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Signed URL generation failed: {str(e)}")

@app.post("/upload-gcs")
async def upload_to_gcs(
    file: UploadFile = File(...),
    processing_mode: str = Form("multi_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """Upload large files to Google Cloud Storage (handles up to 5TB)"""
    if not GCS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Google Cloud Storage not available")
    
    print(f"🔍 Uploading large file to GCS: {file.filename}")
    
    try:
        # Initialize GCS client
        client = storage.Client()
        bucket_name = os.getenv("GCS_BUCKET_NAME", "mineral-rights-pdfs-1759435410")
        bucket = client.bucket(bucket_name)
        
        # Generate unique blob name
        file_id = str(uuid.uuid4())
        blob_name = f"uploads/{file_id}/{file.filename}"
        blob = bucket.blob(blob_name)
        
        # Upload file content to GCS
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        print(f"📁 File size: {file_size_mb:.1f}MB")
        
        # Upload with resumable upload for large files
        blob.upload_from_string(file_content, content_type=file.content_type)
        
        # Make it publicly accessible
        blob.make_public()
        
        # Get public URL
        public_url = blob.public_url
        
        print(f"✅ GCS upload successful: {public_url}")
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "file_size_mb": file_size_mb,
            "gcs_url": public_url,
            "blob_name": blob_name,
            "processing_mode": processing_mode,
            "splitting_strategy": splitting_strategy,
            "message": f"File uploaded to GCS successfully. Size: {file_size_mb:.1f}MB"
        }
        
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"GCS upload failed: {str(e)}")

@app.post("/process-gcs")
async def process_from_gcs(
    gcs_url: str = Form(...),
    processing_mode: str = Form("multi_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """Process a file from Google Cloud Storage URL"""
    print(f"🔍 Processing file from GCS: {gcs_url}")
    
    try:
        # Download file from GCS using same credential logic as signed URL
        credentials_b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        if credentials_b64:
            import base64
            import json
            from google.oauth2 import service_account
            
            # Decode the base64 credentials
            credentials_json = base64.b64decode(credentials_b64).decode('utf-8')
            credentials_info = json.loads(credentials_json)
            
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            client = storage.Client(credentials=credentials)
            print("✅ Using base64 encoded service account credentials for GCS download")
        else:
            # Fallback to default credentials
            client = storage.Client()
            print("✅ Using default service account credentials for GCS download")
        
        # Extract blob name from URL
        # URL format: https://storage.googleapis.com/bucket-name/path/to/file
        url_parts = gcs_url.split('/')
        bucket_name = url_parts[3]
        blob_name = '/'.join(url_parts[4:])
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            blob.download_to_filename(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        print(f"📁 Downloaded file to: {tmp_file_path}")
        
        # Process the file using existing logic
        if processor is None:
            try:
                if not initialize_processor():
                    raise HTTPException(status_code=500, detail="Model not initialized")
            except Exception as e:
                print(f"❌ Processor initialization failed: {e}")
                raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
        
        print("🤖 Processing file from GCS...")
        
        if processing_mode == "multi_deed":
            print("📄 Processing as multi-deed document...")
            results = processor.process_multi_deed_document(
                tmp_file_path, 
                strategy=splitting_strategy
            )
            
            # Convert results to the expected format
            deed_results = []
            for i, result in enumerate(results):
                deed_results.append({
                    "deed_number": result.get('deed_number', i + 1),
                    "has_reservations": result.get('classification', 0) == 1,
                    "confidence": result.get('confidence', 0.0),
                    "reasoning": result.get('reasoning', 'No reasoning provided'),
                    "pages": result.get('pages', []),
                    "pages_in_deed": result.get('pages_in_deed', 0),
                    "deed_boundary_info": result.get('deed_boundary_info', {}),
                    "deed_file": result.get('deed_file', '')
                })
            
            return {
                "deed_results": deed_results,
                "total_deeds": len(deed_results),
                "processing_mode": "multi_deed",
                "filename": blob_name.split('/')[-1],
                "gcs_url": gcs_url,
                "success": True
            }
        else:
            # Single deed processing
            result = processor.process_document(
                tmp_file_path,
                max_samples=6,
                confidence_threshold=0.7,
                page_strategy="first_few",
                high_recall_mode=True
            )
            
            # Check if LLM processing actually succeeded
            detailed_samples = result.get("detailed_samples", [])
            samples_used = result.get("samples_used", 0)
            
            # If no samples were successfully generated, this indicates an API error
            if samples_used == 0 or len(detailed_samples) == 0:
                error_msg = "LLM processing failed - no samples were generated. This may indicate an API key issue or API error."
                print(f"❌ {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )
            
            # Get reasoning from first sample if available
            reasoning = result.get('reasoning', 'No reasoning provided')
            if detailed_samples and len(detailed_samples) > 0:
                reasoning = detailed_samples[0].get("reasoning", reasoning)
            
            return {
                "has_reservation": result.get('classification', 0) == 1,
                "confidence": result.get('confidence', 0.0),
                "reasoning": reasoning,
                "processing_mode": "single_deed",
                "filename": blob_name.split('/')[-1],
                "gcs_url": gcs_url,
                "success": True
            }
            
    except Exception as e:
        print(f"❌ GCS processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"GCS processing failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass

@app.post("/process-large-pdf")
async def process_large_pdf_chunked(
    gcs_url: str = Form(...),
    processing_mode: str = Form("single_deed"),
    splitting_strategy: str = Form("document_ai")
):
    """Process large PDFs by splitting into chunks to avoid memory limits"""
    if not GCS_AVAILABLE:
        raise HTTPException(status_code=500, detail="GCS not available")
    
    try:
        print(f"🚀 Processing large PDF with chunked approach...")
        print(f"🔧 GCS URL: {gcs_url}")
        print(f"🔧 Processing mode: {processing_mode}")
        print(f"🔧 Splitting strategy: {splitting_strategy}")
        
        # Handle page-by-page processing mode
        if processing_mode == "page_by_page":
            # Use the original working page-by-page processor
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
            from mineral_rights.large_pdf_processor import LargePDFProcessor
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found")
            
            page_processor = LargePDFProcessor(api_key=api_key)
            
            # Check if it's a local file path (for testing) or GCS URL
            if gcs_url.startswith('file://'):
                # Extract local file path
                local_path = gcs_url[7:]  # Remove 'file://' prefix
                result = page_processor.process_large_pdf_local(local_path)
            else:
                # Process from GCS using the original working logic
                result = page_processor.process_large_pdf_from_gcs(gcs_url)
        else:
            # Initialize processor for non-page-by-page modes
            if not initialize_processor():
                raise HTTPException(status_code=500, detail="Failed to initialize processor")
            
            # Process the file with chunked approach for large PDFs
            result = processor.process_large_document_chunked(
                gcs_url=gcs_url,
                processing_mode=processing_mode,
                splitting_strategy=splitting_strategy
            )
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing large PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing large PDF: {str(e)}")

@app.options("/process-large-pdf-pages")
async def process_large_pdf_pages_options():
    """Handle CORS preflight requests"""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

@app.post("/process-large-pdf-pages")
async def process_large_pdf_pages(
    gcs_url: str = Form(...)
):
    """Process large PDFs page by page for mineral rights detection - returns immediately with job ID"""
    if not GCS_AVAILABLE:
        raise HTTPException(status_code=500, detail="GCS not available")
    
    try:
        print(f"🔍 Starting async processing for PDF...")
        print(f"🔧 GCS URL: {gcs_url}")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Start processing in background thread
        def process_pdf_background():
            global job_results
            try:
                from src.mineral_rights.large_pdf_processor import LargePDFProcessor
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    print(f"❌ Job {job_id}: ANTHROPIC_API_KEY not found")
                    return
                
                # Initialize progress tracking
                job_results[job_id] = {
                    "status": "processing",
                    "progress": {
                        "current_page": 0,
                        "total_pages": 0,
                        "pages_with_reservations": [],
                        "processing_time": 0,
                        "estimated_remaining": 0,
                        "current_page_result": None,
                        "progress_percentage": 0
                    },
                    "timestamp": time.time()
                }
                print(f"✅ Job {job_id}: Initialized with progress tracking")
                print(f"🔧 Job results before processing: {job_results[job_id]}")
                
                processor = LargePDFProcessor(api_key=api_key)
                print(f"🔧 Calling process_large_pdf_from_gcs_with_progress...")
                result = processor.process_large_pdf_from_gcs_with_progress(gcs_url, job_id, job_results)
                print(f"🔧 Processing method returned, checking job results...")
                print(f"🔧 Job results after processing: {job_results.get(job_id, 'NOT FOUND')}")
                
                # Store final result
                job_results[job_id] = {
                    "status": "completed",
                    "result": result,
                    "progress": job_results[job_id].get("progress", {}),
                    "timestamp": time.time()
                }
                print(f"✅ Job {job_id}: Processing completed")
                
            except Exception as e:
                print(f"❌ Job {job_id}: Error processing PDF: {e}")
                job_results[job_id] = {
                    "status": "error",
                    "error": str(e),
                    "progress": job_results[job_id].get("progress", {}),
                    "timestamp": time.time()
                }
        
        # Start background processing
        thread = threading.Thread(target=process_pdf_background)
        thread.daemon = True
        thread.start()
        
        # Initialize job status
        global job_results
        job_results[job_id] = {
            "status": "processing",
            "timestamp": time.time()
        }
        
        # Return immediately with job ID
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "PDF processing started. Use /process-status/{job_id} to check progress."
        }
        
    except Exception as e:
        print(f"❌ Error starting PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting PDF processing: {str(e)}")

@app.get("/process-status/{job_id}")
async def get_process_status(job_id: str):
    """Get the status of a PDF processing job"""
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_results[job_id]

@app.options("/process-status/{job_id}")
async def get_process_status_options(job_id: str):
    """Handle CORS preflight requests for status endpoint"""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

@app.get("/resume-processing/{job_id}")
async def resume_processing(job_id: str):
    """Check if processing can be resumed for a job"""
    try:
        # Check if job exists in memory
        if job_id in job_results:
            return {
                "can_resume": True,
                "status": job_results[job_id].get("status", "unknown"),
                "progress": job_results[job_id].get("progress", {}),
                "message": "Job found in memory"
            }
        
        # Check if job exists in GCS (saved progress)
        import os
        import base64
        import json
        from google.cloud import storage
        
        # Initialize GCS client
        credentials_json = base64.b64decode(os.getenv("GOOGLE_CREDENTIALS_BASE64", "")).decode('utf-8')
        credentials = json.loads(credentials_json)
        client = storage.Client.from_service_account_info(credentials)
        bucket_name = os.getenv("GCS_BUCKET_NAME", "mineral-rights-pdfs-1759435410")
        bucket = client.bucket(bucket_name)
        
        # Check for saved progress file
        progress_blob_name = f"progress/{job_id}.json"
        progress_blob = bucket.blob(progress_blob_name)
        
        if progress_blob.exists():
            # Load saved progress
            progress_data = json.loads(progress_blob.download_as_text())
            return {
                "can_resume": True,
                "status": "resumable",
                "progress": progress_data.get("progress", {}),
                "message": "Found saved progress in cloud storage"
            }
        
        return {
            "can_resume": False,
            "message": "No saved progress found"
        }
        
    except Exception as e:
        print(f"❌ Error checking resume status: {e}")
        return {
            "can_resume": False,
            "message": f"Error checking resume status: {str(e)}"
        }

@app.options("/resume-processing/{job_id}")
async def resume_processing_options(job_id: str):
    """Handle CORS preflight requests for resume endpoint"""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400"
        }
    )

@app.post("/update-api-key")
async def update_api_key(api_key: str = Form(...)):
    """Update the Anthropic API key for the service"""
    try:
        # Validate the API key format
        if not api_key.startswith("sk-ant-"):
            raise HTTPException(status_code=400, detail="Invalid API key format. Must start with 'sk-ant-'")
        
        if len(api_key) < 50:
            raise HTTPException(status_code=400, detail="Invalid API key format. Key appears too short.")
        
        # Test the API key by making a simple request to Anthropic
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Make a simple test request to validate the key
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            print(f"✅ API key validation successful: {response.id}")
        except Exception as e:
            print(f"❌ API key validation failed: {e}")
            raise HTTPException(status_code=400, detail=f"API key validation failed: {str(e)}")
        
        # Update the global processor with the new API key
        global processor
        if processor:
            processor.api_key = api_key
            print("✅ Updated existing processor with new API key")
        
        # Update environment variable (this will persist for the current instance)
        os.environ["ANTHROPIC_API_KEY"] = api_key
        print("✅ Updated environment variable with new API key")
        
        return {
            "status": "success",
            "message": "API key updated successfully",
            "api_key_validated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating API key: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating API key: {str(e)}")

@app.options("/update-api-key")
async def update_api_key_options():
    """Handle CORS preflight requests for API key update endpoint"""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
