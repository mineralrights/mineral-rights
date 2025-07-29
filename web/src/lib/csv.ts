import Papa from "papaparse";
import { PredictionRow } from "./types";

export function rowsToCSV(rows: PredictionRow[]): string {
  const data = rows.map(r => ({
    File: r.filename,
    Status: r.status,
    Prediction: r.prediction ?? "",
    Explanation: r.explanation ?? ""
  }));
  return Papa.unparse(data);
}
