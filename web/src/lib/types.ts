export interface PredictionRow {
  filename: string;
  status: "waiting" | "processing" | "done" | "error";
  prediction?: string;
  explanation?: string;
  steps?: string[];         // ⬅️ for live log bubbles
}
