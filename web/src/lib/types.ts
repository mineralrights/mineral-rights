export interface PredictionRow {
  filename: string;
  status: "waiting" | "processing" | "done" | "error";
  prediction?: string;
  explanation?: string;
}
