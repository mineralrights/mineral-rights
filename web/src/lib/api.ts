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
      const maxRetries = 10;  // Increased retries for very long sessions
      let lastHeartbeat = Date.now();
      const heartbeatTimeout = 1800000; // 30 minutes (1,800,000 ms) - much longer for 8+ hour sessions
      let sessionStartTime = Date.now();
      
      const connectStream = () => {
        const es = new EventSource(`${API}/stream/${job_id}`);
        
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
            
            // Update progress with session duration
            row.steps!.push(`Session active for ${hours}h ${minutes}m`);
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
            row.explanation = `connection lost after ${hours}h ${minutes}m - no heartbeat received for 30 minutes`;
            emit();
            reject(new Error("Connection timeout"));
            return;
          }
          
          // Try to reconnect if we haven't exceeded max retries
          if (retryCount < maxRetries) {
            retryCount++;
            console.log(`🔄 Retrying connection (${retryCount}/${maxRetries}) after ${hours}h ${minutes}m...`);
            setTimeout(connectStream, 5000 * retryCount); // Longer delays for very long sessions
          } else {
            row.status = "error";
            row.explanation = `connection lost after ${hours}h ${minutes}m - exceeded retry attempts`;
            emit();
            reject(err);
          }
        };

        // Set up heartbeat monitoring with longer intervals for 8+ hour sessions
        const heartbeatCheck = setInterval(() => {
          const sessionDuration = Date.now() - sessionStartTime;
          const hours = Math.floor(sessionDuration / 3600000);
          const minutes = Math.floor((sessionDuration % 3600000) / 60000);
          
          if (Date.now() - lastHeartbeat > heartbeatTimeout) {
            clearInterval(heartbeatCheck);
            es.close();
            row.status = "error";
            row.explanation = `connection lost after ${hours}h ${minutes}m - heartbeat timeout (30 minutes)`;
            emit();
            reject(new Error("Heartbeat timeout"));
          }
        }, 60000); // Check every 60 seconds for very long sessions

        // Clean up interval when connection closes
        es.addEventListener('close', () => clearInterval(heartbeatCheck));
      };
      
      connectStream();
    });
  }

  return rows;
}
