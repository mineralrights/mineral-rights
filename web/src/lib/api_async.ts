import { PredictionRow, DeedResult, PageResult, ProcessingMode, SplittingStrategy } from './types';

// API configuration
const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-processor-ms7ew6g6zq-uc.a.run.app',
  // Add fallback URLs for SSL issues
  fallbackUrls: [
    'https://mineral-rights-processor-ms7ew6g6zq-uc.a.run.app',
    'https://mineral-rights-api-1081023230228.us-central1.run.app', // Old URL fallback
    'https://mineral-rights-production.up.railway.app', // Railway fallback
  ]
};

// Simple fetch with retry logic
async function robustFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const maxRetries = 3;
  const retryDelay = 1000; // 1 second
  
  // Only use HTTPS - no HTTP fallback to avoid mixed content errors
  const urlVariants = [url];
  
  for (const currentUrl of urlVariants) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        // Add cache busting parameters
        const separator = currentUrl.includes('?') ? '&' : '?';
        const cacheBustUrl = `${currentUrl}${separator}t=${Date.now()}&r=${Math.random()}`;
        
        // Add cache busting headers and SSL-friendly headers
        const headers = {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0',
          'Connection': 'keep-alive',
          'User-Agent': 'Mozilla/5.0 (compatible; MineralRightsApp/1.0)',
          ...options.headers,
        };
        
        // Create fetch options with SSL-friendly settings
        const fetchOptions: RequestInit = {
          ...options,
          headers,
          signal: AbortSignal.timeout(1800000), // 30 minute timeout for large PDFs
          // Add mode and credentials for CORS
          mode: 'cors',
          credentials: 'omit', // Don't send credentials to avoid SSL issues
        };
        
        console.log(`🔄 Attempting fetch to: ${cacheBustUrl} (attempt ${attempt})`);
        const response = await fetch(cacheBustUrl, fetchOptions);
        
        if (response.ok) {
          console.log(`✅ Fetch successful: ${response.status}`);
          return response;
        } else {
          console.warn(`⚠️ Fetch failed with status ${response.status}: ${response.statusText}`);
          if (attempt === maxRetries) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
        }
      } catch (error) {
        console.warn(`❌ Fetch attempt ${attempt} failed:`, error);
        if (attempt === maxRetries) {
          throw error;
        }
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, retryDelay * attempt));
      }
    }
  }
  
  throw new Error('All fetch attempts failed');
}

// Job status types
export interface JobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  logs: string[];
  result?: any;
  error?: string;
  created_at: number;
  updated_at: number;
  completed_at?: number;
}

// Process document directly and return results
export async function processDocument(
  file: File,
  processingMode: ProcessingMode = 'single_deed',
  splittingStrategy: SplittingStrategy = 'document_ai'
): Promise<any> {
  // Check file size and choose appropriate endpoint
  const fileSizeMB = file.size / (1024 * 1024);
  const isLargeFile = fileSizeMB > 30; // Use GCS for files > 30MB (Cloud Run limit is 32MB)
  const isVeryLargeFile = fileSizeMB > 40; // Use chunked processing for files > 40MB (memory efficient)
  
  console.log(`📁 File size: ${fileSizeMB.toFixed(1)}MB, using ${isVeryLargeFile ? 'Chunked Processing' : isLargeFile ? 'GCS upload' : 'direct upload'}`);
  
  if (isVeryLargeFile) {
    // Use chunked processing for very large files (memory efficient)
    return await processVeryLargeFileChunked(file, processingMode, splittingStrategy);
  } else if (isLargeFile) {
    // Use GCS upload for large files (up to 5TB)
    return await processLargeFileWithGCS(file, processingMode, splittingStrategy);
  } else {
    // Use direct upload for smaller files
    const formData = new FormData();
    formData.append('file', file);
    formData.append('processing_mode', processingMode);
    formData.append('splitting_strategy', splittingStrategy);

    const response = await robustFetch(`${API_CONFIG.baseUrl}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to process document: ${response.status} ${errorText}`);
    }

    return await response.json();
  }
}

// Process very large files using chunked approach (memory efficient)
async function processVeryLargeFileChunked(
  file: File,
  processingMode: ProcessingMode,
  splittingStrategy: SplittingStrategy
): Promise<any> {
  console.log(`🚀 Starting chunked processing for very large file: ${file.name}`);
  
  try {
    // Step 1: Get signed upload URL
    console.log(`🔑 Step 1: Getting signed upload URL...`);
    const uploadResponse = await robustFetch(`${API_CONFIG.baseUrl}/get-signed-upload-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: file.name,
        content_type: file.type || 'application/pdf'
      })
    });

    if (!uploadResponse.ok) {
      throw new Error(`Failed to get signed URL: ${uploadResponse.status}`);
    }

    const { signed_url, gcs_url } = await uploadResponse.json();
    console.log(`✅ Signed URL obtained: ${gcs_url}`);

    // Step 2: Upload directly to GCS
    console.log(`📤 Step 2: Uploading directly to GCS...`);
    const uploadResult = await fetch(signed_url, {
      method: 'PUT',
      body: file,
      headers: {
        'Content-Type': file.type || 'application/pdf',
      }
    });

    if (!uploadResult.ok) {
      throw new Error(`Failed to upload to GCS: ${uploadResult.status}`);
    }
    console.log(`✅ File uploaded to GCS successfully`);

    // Step 3: Process with chunked approach
    console.log(`🔧 Step 3: Processing with chunked approach...`);
    const processFormData = new FormData();
    processFormData.append('gcs_url', gcs_url);
    processFormData.append('processing_mode', processingMode);
    processFormData.append('splitting_strategy', splittingStrategy);

    const processResponse = await robustFetch(`${API_CONFIG.baseUrl}/process-large-pdf`, {
      method: 'POST',
      body: processFormData,
    });

    if (!processResponse.ok) {
      const errorText = await processResponse.text();
      throw new Error(`Failed to process large PDF: ${processResponse.status} ${errorText}`);
    }

    const result = await processResponse.json();
    console.log(`✅ Large PDF processed successfully`);
    console.log(`📊 Processing result:`, result);

    // Return the actual result from the backend without overriding
    return result;

  } catch (error) {
    console.error(`❌ Chunked processing workflow failed:`, error);
    throw error;
  }
}

// Process large files using direct GCS upload (bypasses Cloud Run limits)
async function processLargeFileWithGCS(
  file: File,
  processingMode: ProcessingMode,
  splittingStrategy: SplittingStrategy
): Promise<any> {
  console.log(`🚀 Starting direct GCS upload for large file: ${file.name}`);
  
  try {
    // Step 1: Get signed URL for direct GCS upload
    console.log('🔑 Step 1: Getting signed upload URL...');
    const signedUrlResponse = await robustFetch(`${API_CONFIG.baseUrl}/get-signed-upload-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        filename: file.name,
        content_type: file.type
      }),
    });

    if (!signedUrlResponse.ok) {
      const errorText = await signedUrlResponse.text();
      throw new Error(`Signed URL request failed: ${signedUrlResponse.status} ${errorText}`);
    }

    const signedUrlData = await signedUrlResponse.json();
    console.log(`✅ Signed URL obtained: ${signedUrlData.blob_name}`);
    
    // Step 2: Upload directly to GCS (bypasses Cloud Run)
    console.log('📤 Step 2: Uploading directly to GCS...');
    const gcsUploadResponse = await fetch(signedUrlData.signed_url, {
      method: 'PUT',
      body: file,
      headers: {
        'Content-Type': file.type,
      },
    });

    if (!gcsUploadResponse.ok) {
      throw new Error(`Direct GCS upload failed: ${gcsUploadResponse.status} ${gcsUploadResponse.statusText}`);
    }

    console.log(`✅ Direct GCS upload successful: ${signedUrlData.gcs_url}`);
    console.log(`📊 File size: ${(file.size / 1024 / 1024).toFixed(1)}MB`);
    
    // Step 3: Process from GCS
    console.log('🔍 Step 3: Processing from GCS...');
    const processFormData = new FormData();
    processFormData.append('gcs_url', signedUrlData.gcs_url);
    processFormData.append('processing_mode', processingMode);
    processFormData.append('splitting_strategy', splittingStrategy);

    const processResponse = await robustFetch(`${API_CONFIG.baseUrl}/process-gcs`, {
      method: 'POST',
      body: processFormData,
    });

    if (!processResponse.ok) {
      const errorText = await processResponse.text();
      throw new Error(`GCS processing failed: ${processResponse.status} ${errorText}`);
    }

    const processResult = await processResponse.json();
    console.log(`✅ Processing successful`);
    
    return {
      ...processResult,
      upload_info: {
        gcs_url: signedUrlData.gcs_url,
        blob_name: signedUrlData.blob_name,
        file_size_mb: file.size / (1024 * 1024)
      },
      success: true
    };
    
  } catch (error) {
    console.error('❌ Direct GCS workflow failed:', error);
    throw error;
  }
}

// Legacy function for backward compatibility - now just calls processDocument
export async function createJob(
  file: File,
  processingMode: ProcessingMode = 'single_deed',
  splittingStrategy: SplittingStrategy = 'document_ai'
): Promise<{ job_id: string; status: string; message: string; result?: any }> {
  // For backward compatibility, we'll simulate the old job-based API
  const result = await processDocument(file, processingMode, splittingStrategy);
  
  // Generate a fake job ID for compatibility
  const jobId = `direct-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  
  return {
    job_id: jobId,
    status: 'completed',
    message: 'Document processed successfully',
    result: result
  };
}

// Get job status
export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await robustFetch(`${API_CONFIG.baseUrl}/jobs/${jobId}`);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to get job status: ${response.status} ${errorText}`);
  }

  return await response.json();
}

// List recent jobs
export async function listJobs(limit: number = 10): Promise<{ jobs: JobStatus[]; count: number }> {
  const response = await robustFetch(`${API_CONFIG.baseUrl}/jobs?limit=${limit}`);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to list jobs: ${response.status} ${errorText}`);
  }

  return await response.json();
}

// Health check
export async function healthCheck(): Promise<{ status: string; timestamp: number }> {
  const response = await robustFetch(`${API_CONFIG.baseUrl}/health`);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Health check failed: ${response.status} ${errorText}`);
  }

  return await response.json();
}

// Heartbeat check
export async function heartbeat(): Promise<{ status: string; timestamp: number }> {
  const response = await robustFetch(`${API_CONFIG.baseUrl}/heartbeat`);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Heartbeat failed: ${response.status} ${errorText}`);
  }

  return await response.json();
}

// Poll job status until completion - now handles direct processing
export async function pollJobUntilComplete(
  jobId: string,
  onProgress?: (status: JobStatus) => void,
  pollInterval: number = 2000
): Promise<JobStatus> {
  // If this is a direct processing job (starts with "direct-"), return immediately
  if (jobId.startsWith('direct-')) {
    // For direct processing, we don't need to poll - the result is already available
    // This is a simplified approach for the new direct processing API
    return new Promise((resolve) => {
      // Simulate a completed job status
      const completedStatus: JobStatus = {
        job_id: jobId,
        status: 'completed',
        progress: 100,
        logs: ['Document processed successfully'],
        created_at: Date.now() - 1000,
        updated_at: Date.now(),
        completed_at: Date.now()
      };
      
      if (onProgress) {
        onProgress(completedStatus);
      }
      
      resolve(completedStatus);
    });
  }
  
  // Legacy polling for actual job-based processing
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getJobStatus(jobId);
        
        if (onProgress) {
          onProgress(status);
        }
        
        if (status.status === 'completed') {
          resolve(status);
        } else if (status.status === 'failed') {
          reject(new Error(status.error || 'Job failed'));
        } else {
          // Continue polling
          setTimeout(poll, pollInterval);
        }
      } catch (error) {
        reject(error);
      }
    };
    
    poll();
  });
}

// Legacy function for backward compatibility
export async function predictDocument(
  file: File,
  processingMode: ProcessingMode = 'single_deed',
  splittingStrategy: SplittingStrategy = 'document_ai'
): Promise<{ job_id: string; status: string; message: string }> {
  return createJob(file, processingMode, splittingStrategy);
}

// Test connection to backend
export async function testConnection(): Promise<{success: boolean, message: string, details?: any}> {
  try {
    console.log("🔍 Testing connection to backend...");
    const response = await robustFetch(`${API_CONFIG.baseUrl}/health`);
    
    if (response.ok) {
      const data = await response.json();
      console.log("✅ Backend connection successful:", data);
      return {
        success: true,
        message: "Backend connection successful",
        details: data
      };
    } else {
      console.error("❌ Backend health check failed:", response.status);
      return {
        success: false,
        message: `Backend health check failed (${response.status})`,
        details: { status: response.status, statusText: response.statusText }
      };
    }
  } catch (error) {
    console.error("❌ Backend connection test failed:", error);
    return {
      success: false,
      message: `Connection test failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      details: { error: error instanceof Error ? error.message : String(error) }
    };
  }
}