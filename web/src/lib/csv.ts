import Papa from "papaparse";
import { PredictionRow } from "./types";

export function rowsToCSV(rows: PredictionRow[]): string {
  const data: any[] = [];
  
  rows.forEach(row => {
    if (row.processingMode === "multi_deed" && row.deedResults) {
      // For multi-deed, create a row for each individual deed with smart names
      row.deedResults.forEach(deed => {
        data.push({
          "Original File": row.filename,
          "Deed Name": deed.smart_deed_name || `deed_${deed.deed_number}`,
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
        "Original File": row.filename + " (Summary)",
        "Deed Name": "ALL_DEEDS_SUMMARY",
        "Deed Number": "SUMMARY",
        Status: row.status,
        Prediction: row.prediction,
        Confidence: "",
        "Pages in Deed": row.totalDeeds + " deeds total",
        Explanation: row.explanation || ""
      });
    } else {
      // Single deed format
      data.push({
        "Original File": row.filename,
        "Deed Name": row.filename.replace('.pdf', '_single_deed'),
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
