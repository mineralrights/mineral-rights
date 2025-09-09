import { PredictionRow, ProcessingMode, SplittingStrategy, DeedResult } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL!;   // already defined in .env.local

export async function predictBatch(
  files: File[],
  processingMode: ProcessingMode = "single_deed",
  splittingStrategy: SplittingStrategy = "smart_detection",
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

    // 1️⃣  upload PDF → get job_id
    const form = new FormData();
    form.append("file", file);
    form.append("processing_mode", processingMode);
    if (processingMode === "multi_deed") {
      form.append("splitting_strategy", splittingStrategy);
    }
    
    const res = await fetch(`${API}/predict`, { method: "POST", body: form });
    if (!res.ok) {
      row.status = "error";
      row.explanation = await res.text();
      emit();
      continue;
    }
    const { job_id } = await res.json();

    // 2️⃣  open SSE stream with retry logic for 8+ hour sessions
    await new Promise<void>((resolve, reject) => {
      let retryCount = 0;
      const maxRetries = 15;  // Increased retries for very long sessions
      let lastHeartbeat = Date.now();
      const heartbeatTimeout = 600000; // 10 minutes (600,000 ms) - shorter for better reliability
      let sessionStartTime = Date.now();
      let connectionEstablished = false;
      
      const connectStream = () => {
        console.log(`🔗 Connecting to stream for job ${job_id} (attempt ${retryCount + 1})`);
        const es = new EventSource(`${API}/stream/${job_id}`);
        
        es.onopen = () => {
          console.log(`✅ Stream connection established for job ${job_id}`);
          connectionEstablished = true;
          lastHeartbeat = Date.now();
        };
        
        es.onmessage = e => {
          const msg = e.data as string;
          lastHeartbeat = Date.now();

          if (msg.startsWith("__HEARTBEAT__")) {
            // Handle heartbeat with session duration
            const parts = msg.replace("__HEARTBEAT__", "").split("|");
            const sessionDuration = parts.length > 1 ? parseInt(parts[1]) : 0;
            const hours = Math.floor(sessionDuration / 3600);
            const minutes = Math.floor((sessionDuration % 3600) / 60);
            
            console.log(`💓 Heartbeat received - Session: ${hours}h ${minutes}m`);
            
            // Update progress with session duration (only if not already shown)
            const lastStep = row.steps![row.steps!.length - 1];
            if (!lastStep || !lastStep.includes(`Session active for ${hours}h ${minutes}m`)) {
              row.steps!.push(`Session active for ${hours}h ${minutes}m`);
              emit();
            }
            return;
          }

          if (msg.startsWith("__PROGRESS__")) {
            // Handle progress updates
            try {
              const progressData = JSON.parse(msg.replace("__PROGRESS__", ""));
              const hours = Math.floor(progressData.session_duration / 3600);
              const minutes = Math.floor((progressData.session_duration % 3600) / 60);
              
              console.log(`📊 Progress update - Session: ${hours}h ${minutes}m, Status: ${progressData.status}`);
              
              // Update progress with detailed info
              const progressMsg = `Progress: ${progressData.pages_processed}/${progressData.total_pages} pages (${hours}h ${minutes}m)`;
              row.steps!.push(progressMsg);
              emit();
            } catch (e) {
              console.error("Error parsing progress data:", e);
            }
            return;
          }

          if (msg.startsWith("__MEMORY__")) {
            // Handle memory status updates
            const memoryMB = msg.replace("__MEMORY__", "");
            console.log(`💾 Memory usage: ${memoryMB} MB`);
            
            // Update progress with memory info (only if significant change)
            const lastStep = row.steps![row.steps!.length - 1];
            if (!lastStep || !lastStep.includes(`Memory: ${memoryMB} MB`)) {
              row.steps!.push(`Memory: ${memoryMB} MB`);
              emit();
            }
            return;
          }

          if (msg.startsWith("__MEMORY_GC__")) {
            // Handle garbage collection updates
            const parts = msg.replace("__MEMORY_GC__", "").split("|");
            const afterGC = parts[0];
            const freed = parts[1];
            console.log(`🧹 Memory GC: ${afterGC} MB (freed ${freed} MB)`);
            
            row.steps!.push(`Memory cleanup: ${afterGC} MB (freed ${freed} MB)`);
            emit();
            return;
          }

          if (msg.startsWith("__RESULT__")) {
            const result = JSON.parse(msg.replace("__RESULT__", ""));
            
            if (processingMode === "single_deed") {
              // Handle single deed result - SEPARATE confidence from explanation
              row.prediction = result.classification === 1 ? "has_reservation" : "no_reservation";
              row.confidence = result.confidence; // NEW: Store confidence separately
              row.explanation = result.detailed_samples?.[0]?.reasoning || ""; // Only reasoning, no confidence
            } else {
              // Handle multi-deed result
              row.totalDeeds = result.total_deeds;
              row.deedResults = result.deed_results.map((deedResult: any): DeedResult => ({
                deed_number: deedResult.deed_number,
                classification: deedResult.classification,
                confidence: deedResult.confidence,
                prediction: deedResult.classification === 1 ? "has_reservation" : "no_reservation",
                explanation: deedResult.detailed_samples?.[0]?.reasoning || "", // Only reasoning
                deed_file: deedResult.deed_file,
                pages_in_deed: deedResult.pages_in_deed
              }));
              
              // Set overall prediction based on summary
              const reservationsFound = result.summary?.reservations_found || 0;
              row.prediction = reservationsFound > 0 ? "has_reservation" : "no_reservation";
              row.explanation = `${reservationsFound}/${result.total_deeds} deeds have reservations`;
            }
            
            row.status = "done";
            emit();
            es.close();
            resolve();
          } else if (msg.startsWith("__ERROR__")) {
            row.status = "error";
            row.explanation = msg.replace("__ERROR__", "");
            emit();
            es.close();
            reject(new Error(row.explanation));
          } else if (msg === "__END__") {
            es.close();
            resolve();
          } else {
            // normal progress line
            row.steps!.push(msg);
            emit();
          }
        };

        es.onerror = err => {
          console.error("EventSource error:", err);
          es.close();
          
          const sessionDuration = Date.now() - sessionStartTime;
          const hours = Math.floor(sessionDuration / 3600000);
          const minutes = Math.floor((sessionDuration % 3600000) / 60000);
          
          // Check if it's been too long since last heartbeat
          if (Date.now() - lastHeartbeat > heartbeatTimeout) {
            row.status = "error";
            row.explanation = `connection lost after ${hours}h ${minutes}m - no heartbeat received for 10 minutes`;
            emit();
            reject(new Error("Connection timeout"));
            return;
          }
          
          // Try to reconnect if we haven't exceeded max retries
          if (retryCount < maxRetries) {
            retryCount++;
            const retryDelay = Math.min(5000 * retryCount, 30000); // Cap at 30 seconds
            console.log(`🔄 Retrying connection (${retryCount}/${maxRetries}) after ${hours}h ${minutes}m... (delay: ${retryDelay}ms)`);
            
            // Add retry info to steps
            row.steps!.push(`🔄 Retrying connection (${retryCount}/${maxRetries}) - delay: ${retryDelay/1000}s`);
            emit();
            
            setTimeout(connectStream, retryDelay);
          } else {
            // Try to get result via fallback endpoint before giving up
            console.log(`🔄 All retries exhausted, attempting to retrieve result for job ${job_id}`);
            
            // Use async function to handle the fallback
            (async () => {
              try {
                const resultResponse = await fetch(`${API}/job-result/${job_id}`);
                if (resultResponse.ok) {
                  const result = await resultResponse.json();
                  console.log(`✅ Retrieved result via fallback endpoint`);
                  
                  // Process the result as if it came through the stream
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
                  row.steps!.push(`✅ Result retrieved via fallback after connection issues`);
                  emit();
                  resolve();
                  return;
                }
              } catch (fallbackError) {
                console.error("Fallback result retrieval failed:", fallbackError);
              }
              
              row.status = "error";
              row.explanation = `connection lost after ${hours}h ${minutes}m - exceeded retry attempts (${maxRetries})`;
              emit();
              reject(err);
            })();
          }
        };

        // Set up heartbeat monitoring with shorter intervals for better reliability
        const heartbeatCheck = setInterval(() => {
          const sessionDuration = Date.now() - sessionStartTime;
          const hours = Math.floor(sessionDuration / 3600000);
          const minutes = Math.floor((sessionDuration % 3600000) / 60000);
          
          if (Date.now() - lastHeartbeat > heartbeatTimeout) {
            clearInterval(heartbeatCheck);
            es.close();
            row.status = "error";
            row.explanation = `connection lost after ${hours}h ${minutes}m - heartbeat timeout (10 minutes)`;
            emit();
            reject(new Error("Heartbeat timeout"));
          }
        }, 30000); // Check every 30 seconds for better reliability

        // Clean up interval when connection closes
        es.addEventListener('close', () => clearInterval(heartbeatCheck));
      };
      
      connectStream();
    });
  }

  return rows;
}
