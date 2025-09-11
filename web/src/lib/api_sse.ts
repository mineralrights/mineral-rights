import { PredictionRow, DeedResult, PageResult, ProcessingMode, SplittingStrategy } from '../types';

// API configuration
const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-production.up.railway.app',
};

// Robust fetch with retry logic and cache busting
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
        ...options.headers,
      };
      
      const response = await fetch(cacheBustUrl, {
        ...options,
        headers,
        signal: AbortSignal.timeout(30000), // 30 second timeout
      });
      
      return response;
    } catch (error) {
      console.warn(`Fetch attempt ${attempt} failed:`, error);
      
      if (attempt === maxRetries) {
        throw error;
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, retryDelay * attempt));
    }
  }
  
  throw new Error('All fetch attempts failed');
}

// Client-side error handling
function handleClientError(error: Error, context: string): string {
  if (error.message.includes('ERR_SSL_BAD_RECORD_MAC_ALERT')) {
    return `SSL connection error. Please try refreshing the page or using incognito mode.`;
  }
  if (error.message.includes('Failed to fetch')) {
    return `Network error. Please check your internet connection and try again.`;
  }
  if (error.message.includes('timeout')) {
    return `Request timed out. The server may be busy, please try again.`;
  }
  return `${context} failed: ${error.message}`;
}

// Heartbeat function to keep Railway alive
export async function sendHeartbeat(): Promise<boolean> {
  try {
    const response = await robustFetch(`${API_CONFIG.baseUrl}/heartbeat`);
    return response.ok;
  } catch (error) {
    console.warn("Heartbeat failed:", error);
    return false;
  }
}

// Main prediction function using SSE streaming
export async function predictBatch(
  files: File[],
  processingMode: ProcessingMode = "single_deed",
  splittingStrategy: SplittingStrategy = "document_ai",
  onChange: (rows: PredictionRow[]) => void = () => {}
): Promise<PredictionRow[]> {
  // Create initial rows
  const rows: PredictionRow[] = files.map(f => ({
    filename: f.name,
    status: "waiting",
    steps: [],
    processingMode
  }));

  const emit = () => onChange(rows.map(r => ({ ...r })));   // Clone for React
  const rowFor = (name: string) => rows.find(r => r.filename === name)!;

  // Process files one after another
  for (const file of files) {
    const row = rowFor(file.name);
    row.status = "processing";
    emit();

    try {
      // 1️⃣ Upload PDF and get job_id
      const form = new FormData();
      form.append("file", file);
      form.append("processing_mode", processingMode);
      form.append("splitting_strategy", "document_ai");
      
      const res = await robustFetch(`${API_CONFIG.baseUrl}/predict`, { 
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

      // 2️⃣ Stream processing logs via Server-Sent Events
      await new Promise<void>((resolve, reject) => {
        const eventSource = new EventSource(`${API_CONFIG.baseUrl}/stream/${job_id}`);
        let result: any = null;
        
        eventSource.onmessage = (event) => {
          const data = event.data;
          
          if (data.startsWith("__RESULT__")) {
            // Parse the result
            try {
              result = JSON.parse(data.substring("__RESULT__".length));
              console.log(`✅ Job ${job_id} completed successfully`);
              
              // Process the result based on processing mode
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
                  explanation: deedResult.reasoning,
                  deed_file: deedResult.deed_file,
                  smart_deed_name: deedResult.smart_deed_name,
                  pages_in_deed: deedResult.pages_in_deed
                }));
              } else if (processingMode === "page_by_page") {
                row.totalPages = result.total_pages;
                row.pagesWithReservations = result.pages_with_reservations;
                row.pageResults = result.page_results.map((pageResult: any): PageResult => ({
                  page_number: pageResult.page_number,
                  classification: pageResult.classification,
                  confidence: pageResult.confidence,
                  prediction: pageResult.classification === 1 ? "has_reservation" : "no_reservation",
                  explanation: pageResult.reasoning,
                  text_length: pageResult.text_length,
                  processing_time: pageResult.processing_time,
                  has_reservations: pageResult.has_reservations
                }));
              }
              
              row.status = "done";
              row.steps!.push("✅ Processing completed successfully");
              emit();
              
            } catch (parseError) {
              console.error("Error parsing result:", parseError);
              row.status = "error";
              row.explanation = "Error parsing processing result";
              emit();
              reject(parseError);
            }
          } else if (data === "__END__") {
            // Processing completed
            eventSource.close();
            if (result) {
              resolve();
            } else {
              reject(new Error("Processing ended without result"));
            }
          } else if (data.startsWith("❌ Error:")) {
            // Processing failed
            eventSource.close();
            row.status = "error";
            row.explanation = data.substring("❌ Error: ".length);
            emit();
            reject(new Error(data));
          } else if (data.startsWith("__HEARTBEAT__")) {
            // Heartbeat - just log it
            console.log(`💓 Heartbeat received for job ${job_id}`);
          } else {
            // Regular log message - add to steps
            row.steps!.push(data);
            emit();
          }
        };
        
        eventSource.onerror = (error) => {
          console.error(`SSE error for job ${job_id}:`, error);
          eventSource.close();
          row.status = "error";
          row.explanation = "Connection lost during processing";
          emit();
          reject(new Error("SSE connection error"));
        };
        
        // Set a timeout for very long jobs (8 hours)
        setTimeout(() => {
          eventSource.close();
          row.status = "error";
          row.explanation = "Processing timeout (8 hours)";
          emit();
          reject(new Error("Processing timeout"));
        }, 8 * 60 * 60 * 1000);
      });

    } catch (error) {
      console.error(`Error processing ${file.name}:`, error);
      row.status = "error";
      row.explanation = handleClientError(error as Error, "Processing");
      emit();
    }
  }

  return rows;
}
