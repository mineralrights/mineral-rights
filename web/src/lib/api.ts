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
      continue;
    }
    const { job_id } = await res.json();

    // 2️⃣  open SSE stream
    await new Promise<void>((resolve, reject) => {
      const es = new EventSource(`${API}/stream/${job_id}`);

      es.onmessage = e => {
        const msg = e.data as string;

        if (msg.startsWith("__RESULT__")) {
          const result = JSON.parse(msg.replace("__RESULT__", ""));
          
          if (processingMode === "single_deed") {
            // Handle single deed result
            row.prediction = result.classification === 1 ? "has_reservation" : "no_reservation";
            row.explanation = result.detailed_samples?.[0]?.reasoning
                              ?? `Confidence ${result.confidence.toFixed(2)}`;
          } else {
            // Handle multi-deed result
            row.totalDeeds = result.total_deeds;
            row.deedResults = result.deed_results.map((deedResult: any): DeedResult => ({
              deed_number: deedResult.deed_number,
              classification: deedResult.classification,
              confidence: deedResult.confidence,
              prediction: deedResult.classification === 1 ? "has_reservation" : "no_reservation",
              explanation: deedResult.detailed_samples?.[0]?.reasoning 
                          ?? `Confidence ${deedResult.confidence?.toFixed(2) ?? 'N/A'}`,
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
        } else if (msg.startsWith("__ERROR__")) {
          row.status = "error";
          row.explanation = msg.replace("__ERROR__", "");
          emit();
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
        es.close();
        row.status = "error";
        row.explanation = "connection lost";
        emit();
        reject(err);
      };
    });
  }

  return rows;
}
