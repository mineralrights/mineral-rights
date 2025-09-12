import { PredictionRow, DeedResult, PageResult, ProcessingMode, SplittingStrategy } from './types';

// API configuration for Google Cloud Run
const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-api-<hash>-uc.a.run.app',
  fallbackUrls: [
    'https://mineral-rights-api-<hash>-uc.a.run.app',
  ]
};

// Job status types
export interface JobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  filename: string;
  processing_mode: string;
  splitting_strategy: string;
  created_at: number;
  updated_at: number;
  error?: string;
  result_path?: string;
  logs?: string[];
}

export interface JobResult {
  job_id: string;
  status: 'completed';
  result_path: string;
  result: any; // The actual processing result
}

// Simple fetch with retry logic
async function robustFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const maxRetries = 3;
  const retryDelay = 1000; // 1 second
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      // Add cache busting parameters
      const separator = url.includes('?') ? '&' : '?';
      const cacheBustUrl = `${url}${separator}t=${Date.now()}&r=${Math.random()}`;
      
      // Add cache busting headers
      const headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (compatible; MineralRightsApp/1.0)',
        ...options.headers,
      };
      
      // Create fetch options
      const fetchOptions: RequestInit = {
        ...options,
        headers,
        signal: AbortSignal.timeout(30000), // 30 second timeout
        mode: 'cors',
        credentials: 'include', // Include credentials for authenticated requests
      };
      
      console.log(`🔄 Attempting fetch to: ${cacheBustUrl} (attempt ${attempt})`);
      const response = await fetch(cacheBustUrl, fetchOptions);
      
      if (response.ok) {
        console.log(`✅ Fetch successful: ${response.status} ${response.statusText}`);
        return response;
      } else {
        console.log(`⚠️ Fetch failed: ${response.status} ${response.statusText}`);
        if (attempt === maxRetries) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
      }
    } catch (error) {
      console.log(`❌ Fetch error (attempt ${attempt}):`, error);
      if (attempt === maxRetries) {
        throw error;
      }
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, retryDelay * attempt));
    }
  }
  
  throw new Error('All fetch attempts failed');
}

// Health check
export async function checkHealth(): Promise<{ status: string; timestamp: number }> {
  const response = await robustFetch(`${API_CONFIG.baseUrl}/health`);
  return await response.json();
}

// Create a new processing job
export async function createJob(
  file: File,
  processingMode: ProcessingMode = 'single_deed',
  splittingStrategy: SplittingStrategy = 'document_ai'
): Promise<{ job_id: string; status: string; message: string }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('processing_mode', processingMode);
  formData.append('splitting_strategy', splittingStrategy);

  const response = await robustFetch(`${API_CONFIG.baseUrl}/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to create job: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return await response.json();
}

// Get job status
export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await robustFetch(`${API_CONFIG.baseUrl}/jobs/${jobId}`);
  
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('Job not found');
    }
    const errorText = await response.text();
    throw new Error(`Failed to get job status: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return await response.json();
}

// Get job result (download from Cloud Storage)
export async function getJobResult(jobId: string): Promise<any> {
  // First get the job status to get the result path
  const jobStatus = await getJobStatus(jobId);
  
  if (jobStatus.status !== 'completed') {
    throw new Error(`Job is not completed yet. Status: ${jobStatus.status}`);
  }
  
  if (!jobStatus.result_path) {
    throw new Error('No result path found for completed job');
  }
  
  // For now, we'll return the job status with a note about the result path
  // In a full implementation, you would download the result from Cloud Storage
  return {
    job_id: jobId,
    status: 'completed',
    result_path: jobStatus.result_path,
    message: 'Result is available at the specified path in Cloud Storage'
  };
}

// List recent jobs
export async function listJobs(limit: number = 10, status?: string): Promise<{ jobs: JobStatus[]; count: number }> {
  const params = new URLSearchParams();
  params.append('limit', limit.toString());
  if (status) {
    params.append('status', status);
  }
  
  const response = await robustFetch(`${API_CONFIG.baseUrl}/jobs?${params}`);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to list jobs: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return await response.json();
}

// Poll job status until completion
export async function pollJobUntilComplete(
  jobId: string,
  onProgress?: (status: JobStatus) => void,
  pollInterval: number = 2000
): Promise<JobStatus> {
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
          reject(new Error(`Job failed: ${status.error || 'Unknown error'}`));
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

// Legacy function for backward compatibility (returns a promise that never resolves)
export async function predictDocument(
  file: File,
  processingMode: ProcessingMode = 'single_deed',
  splittingStrategy: SplittingStrategy = 'document_ai'
): Promise<{ job_id: string }> {
  console.warn('predictDocument is deprecated. Use createJob instead.');
  const result = await createJob(file, processingMode, splittingStrategy);
  return { job_id: result.job_id };
}

// Legacy streaming function (returns a promise that never resolves)
export async function streamJobProgress(jobId: string): Promise<AsyncGenerator<string, void, unknown>> {
  console.warn('streamJobProgress is deprecated. Use pollJobUntilComplete instead.');
  
  // Return an async generator that never yields (for backward compatibility)
  return (async function* () {
    while (true) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      // Never yield anything to maintain compatibility
    }
  })();
}
