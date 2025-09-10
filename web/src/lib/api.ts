import { PredictionRow, ProcessingMode, SplittingStrategy, DeedResult } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL!;   // already defined in .env.local

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
    const res = await fetch(`${API}/jobs/create`, { 
      method: "POST", 
      body: form,
      // Add timeout and retry configuration
      signal: AbortSignal.timeout(30000) // 30 second timeout for job creation
    });
    if (!res.ok) {
      row.status = "error";
      row.explanation = await res.text();
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
          const statusResponse = await fetch(`${API}/jobs/${job_id}/status`, {
            signal: AbortSignal.timeout(10000) // 10 second timeout for status checks
          });
          if (!statusResponse.ok) {
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
            const resultResponse = await fetch(`${API}/jobs/${job_id}/result`, {
              signal: AbortSignal.timeout(15000) // 15 second timeout for result retrieval
            });
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
            } else {
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
            error.message.includes('SSL')
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
