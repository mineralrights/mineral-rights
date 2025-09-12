import { PredictionRow, DeedResult, PageResult, ProcessingMode, SplittingStrategy } from './types';

// API configuration
const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-api-1081023230228.us-central1.run.app',
  // Add fallback URLs for SSL issues
  fallbackUrls: [
    'https://mineral-rights-api-1081023230228.us-central1.run.app',
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
          signal: AbortSignal.timeout(30000), // 30 second timeout
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

  const response = await robustFetch(`${API_CONFIG.baseUrl}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to create job: ${response.status} ${errorText}`);
  }

  return await response.json();
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