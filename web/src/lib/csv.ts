import Papa from "papaparse";
import { PredictionRow } from "./types";

export function rowsToCSV(rows: PredictionRow[]): string {
  const data: any[] = [];
  
  rows.forEach(row => {
    if (row.processingMode === "multi_deed" && row.deedResults) {
      // For multi-deed, create a row for each individual deed
      row.deedResults.forEach(deed => {
        data.push({
          File: row.filename,
          "Deed Number": deed.deed_number,
          Status: row.status,
          Prediction: deed.prediction,
          Confidence: (deed.confidence * 100).toFixed(1) + "%",
          "Pages in Deed": deed.pages_in_deed || "",
          Explanation: deed.explanation || ""
        });
      });
      
      // Also add a summary row
      data.push({
        File: row.filename + " (Summary)",
        "Deed Number": "ALL",
        Status: row.status,
        Prediction: row.prediction,
        Confidence: "",
        "Pages in Deed": row.totalDeeds + " deeds total",
        Explanation: row.explanation || ""
      });
    } else {
      // Single deed format
      data.push({
        File: row.filename,
        "Deed Number": 1,
        Status: row.status,
        Prediction: row.prediction ?? "",
        Confidence: "",
        "Pages in Deed": "",
        Explanation: row.explanation ?? ""
      });
    }
  });
  
  return Papa.unparse(data);
}
