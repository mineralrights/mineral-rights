import { PredictionRow, ProcessingMode, SplittingStrategy, DeedResult, PageResult } from "./types";

// Robust API configuration with multiple fallbacks
const API_BASE = process.env.NEXT_PUBLIC_API_URL!;   // Railway backend: https://mineral-rights-production.up.railway.app

// Client-proof API configuration
const API_CONFIG = {
  baseUrl: API_BASE,
  timeout: 120000, // 2 minutes
  retries: 3,
  cacheBust: true,
  fallbackUrls: [
    API_BASE,
    // Add backup URLs if needed
  ]
};

// Robust fetch wrapper with automatic retries and fallbacks
async function robustFetch(url: string, options: RequestInit = {}, retryCount = 0): Promise<Response> {
  const cacheBustUrl = API_CONFIG.cacheBust ? `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}&r=${Math.random()}` : url;
  
  const fetchOptions: RequestInit = {
    ...options,
    headers: {
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0',
      'X-Requested-With': 'XMLHttpRequest',
      ...options.headers
    }
  };

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);
    
    const response = await fetch(cacheBustUrl, {
      ...fetchOptions,
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    return response;
    
  } catch (error) {
    console.warn(`Fetch attempt ${retryCount + 1} failed:`, error);
    
    // Retry with exponential backoff
    if (retryCount < API_CONFIG.retries) {
      const delay = Math.pow(2, retryCount) * 1000; // 1s, 2s, 4s
      await new Promise(resolve => setTimeout(resolve, delay));
      return robustFetch(url, options, retryCount + 1);
    }
    
    throw error;
  }
}

// Client-proof connection test
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
    console.error("❌ Backend connection failed:", error);
    return {
      success: false,
      message: `Connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      details: error
    };
  }
}

// Enhanced error handling for client experience
function handleClientError(error: any, context: string): string {
  console.error(`❌ ${context}:`, error);
  
  if (error.name === 'AbortError') {
    return "Request timed out. Please try again or contact support if the issue persists.";
  }
  
  if (error.message?.includes('Failed to fetch')) {
    return "Unable to connect to the server. Please check your internet connection and try again.";
  }
  
  if (error.message?.includes('SSL') || error.message?.includes('TLS')) {
    return "Connection security issue detected. Please try refreshing the page or using a different browser.";
  }
  
  if (error.message?.includes('CORS')) {
    return "Cross-origin request blocked. Please contact support.";
  }
  
  return `An error occurred: ${error.message || 'Unknown error'}. Please try again or contact support.`;
}

export async function predictBatch(
  files: File[],
  processingMode: ProcessingMode = "single_deed",
  splittingStrategy: SplittingStrategy = "document_ai",
  onChange: (rows: PredictionRow[]) => void = () => {}
): Promise<PredictionRow[]> {
  // create initial rows
  const rows: PredictionRow[] = files.map(f => ({
    filename: f.name,
    status: "waiting",
    steps: [],
    processingMode
  }));

  const emit = () => onChange(rows.map(r => ({ ...r })));   // clone for React
  const rowFor = (name: string) => rows.find(r => r.filename === name)!;

  // process files one after another (simple; parallel is possible too)
  for (const file of files) {
    const row = rowFor(file.name);
    row.status = "processing";
    emit();

    // 1️⃣  upload PDF → get job_id (use job system for long-running tasks)
    const form = new FormData();
    form.append("file", file);
    form.append("processing_mode", processingMode);
    if (processingMode === "multi_deed") {
      form.append("splitting_strategy", "document_ai"); // Always use Document AI Smart Chunking
    }
    
  // Use job system for long-running processing (8+ hours support)
  const res = await robustFetch(`${API_CONFIG.baseUrl}/jobs/create`, { 
    method: "POST", 
    body: form
  });
    if (!res.ok) {
      row.status = "error";
      try {
        const errorText = await res.text();
        row.explanation = handleClientError(new Error(`Server error (${res.status}): ${errorText}`), "Job creation");
      } catch (e) {
        row.explanation = handleClientError(new Error(`Network error: ${res.status} ${res.statusText}`), "Job creation");
      }
      emit();
      continue;
    }
    const { job_id } = await res.json();

    // 2️⃣  monitor job progress with polling (no timeout limits)
    await new Promise<void>((resolve, reject) => {
      let sessionStartTime = Date.now();
      const pollInterval = 5000; // Poll every 5 seconds
      const maxPollTime = 8 * 60 * 60 * 1000; // 8 hours max
      
      const pollJob = async () => {
        try {
          const sessionDuration = Date.now() - sessionStartTime;
          const hours = Math.floor(sessionDuration / 3600000);
          const minutes = Math.floor((sessionDuration % 3600000) / 60000);
          
          // Check if we've exceeded max polling time
          if (sessionDuration > maxPollTime) {
            row.status = "error";
            row.explanation = `Job exceeded maximum processing time (8 hours)`;
            emit();
            reject(new Error("Job timeout"));
            return;
          }
          
          // Poll job status with retry logic
          const statusResponse = await robustFetch(`${API_CONFIG.baseUrl}/jobs/${job_id}/status`);
          if (!statusResponse.ok) {
            if (statusResponse.status === 404) {
              console.warn(`⚠️ Job ${job_id} not found - may have been lost due to service restart`);
              row.status = "error";
              row.explanation = `Job not found: Service may have restarted. Please try again.`;
              emit();
              reject(new Error("Job not found"));
              return;
            }
            throw new Error(`Failed to get job status: ${statusResponse.status}`);
          }
          
          const jobStatus = await statusResponse.json();
          console.log(`📊 Job ${job_id} status: ${jobStatus.status} (${hours}h ${minutes}m)`);
          
          // Update progress
          const progressMsg = `Processing... (${hours}h ${minutes}m) - Status: ${jobStatus.status}`;
          const lastStep = row.steps![row.steps!.length - 1];
          if (!lastStep || !lastStep.includes(progressMsg)) {
            row.steps!.push(progressMsg);
            emit();
          }
          
          if (jobStatus.status === "completed") {
            // Get the result
    const resultResponse = await robustFetch(`${API_CONFIG.baseUrl}/jobs/${job_id}/result`);
            if (!resultResponse.ok) {
              throw new Error(`Failed to get job result: ${resultResponse.status}`);
            }
            
            const result = await resultResponse.json();
            console.log(`✅ Job ${job_id} completed successfully`);
            
            // Process the result
            if (processingMode === "single_deed") {
              row.prediction = result.classification === 1 ? "has_reservation" : "no_reservation";
              row.confidence = result.confidence;
              row.explanation = result.detailed_samples?.[0]?.reasoning || "";
            } else if (processingMode === "multi_deed") {
              row.totalDeeds = result.total_deeds;
              row.deedResults = result.deed_results.map((deedResult: any): DeedResult => ({
                deed_number: deedResult.deed_number,
                classification: deedResult.classification,
                confidence: deedResult.confidence,
                prediction: deedResult.classification === 1 ? "has_reservation" : "no_reservation",
                explanation: deedResult.detailed_samples?.[0]?.reasoning || "",
                deed_file: deedResult.deed_file,
                pages_in_deed: deedResult.pages_in_deed
              }));
              
              const reservationsFound = result.summary?.reservations_found || 0;
              row.prediction = reservationsFound > 0 ? "has_reservation" : "no_reservation";
              row.explanation = `${reservationsFound}/${result.total_deeds} deeds have reservations`;
            } else if (processingMode === "page_by_page") {
              row.totalPages = result.total_pages;
              row.pagesWithReservations = result.pages_with_reservations || [];
              row.pageResults = result.page_results?.map((pageResult: any): PageResult => ({
                page_number: pageResult.page_number,
                classification: pageResult.classification,
                confidence: pageResult.confidence,
                prediction: pageResult.classification === 1 ? "has_reservation" : "no_reservation",
                explanation: pageResult.reasoning || "",
                text_length: pageResult.text_length,
                processing_time: pageResult.processing_time,
                has_reservations: pageResult.has_reservations
              })) || [];
              
              const reservationsFound = result.total_pages_with_reservations || 0;
              row.prediction = reservationsFound > 0 ? "has_reservation" : "no_reservation";
              row.confidence = result.confidence;
              row.explanation = `${reservationsFound}/${result.total_pages} pages have reservations`;
            }
            
            row.status = "done";
            row.steps!.push(`✅ Processing completed successfully (${hours}h ${minutes}m)`);
            emit();
            resolve();
            
          } else if (jobStatus.status === "failed") {
            row.status = "error";
            row.explanation = jobStatus.error || "Job failed for unknown reason";
            emit();
            reject(new Error(row.explanation));
            
          } else if (jobStatus.status === "running") {
            // Continue polling
            setTimeout(pollJob, pollInterval);
            
          } else {
            // Unknown status, continue polling
            setTimeout(pollJob, pollInterval);
          }
          
        } catch (error) {
          console.error(`Error polling job ${job_id}:`, error);
          
          // Check if it's a network error that we can retry
          const isNetworkError = error instanceof Error && (
            error.message.includes('fetch') || 
            error.message.includes('network') ||
            error.message.includes('timeout') ||
            error.message.includes('SSL') ||
            error.message.includes('ERR_SSL') ||
            error.message.includes('Failed to fetch')
          );
          
          if (isNetworkError) {
            console.log(`🔄 Network error detected, retrying in 10 seconds...`);
            setTimeout(pollJob, 10000); // Retry after 10 seconds
            return;
          }
          
          row.status = "error";
          const errorMessage = error instanceof Error ? error.message : String(error);
          row.explanation = `Error monitoring job: ${errorMessage}`;
          emit();
          reject(error);
        }
      };
      
      // Start polling
      pollJob();
    });
  }

  return rows;
}
