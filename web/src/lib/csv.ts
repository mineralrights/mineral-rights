import Papa from "papaparse";
import { PredictionRow } from "./types";

export function rowsToCSV(rows: PredictionRow[]): string {
  const data: any[] = [];
  
  rows.forEach(row => {
    if (row.processingMode === "multi_deed" && row.deedResults) {
      // For multi-deed, create a row for each individual deed
      row.deedResults.forEach(deed => {
        data.push({
          "Original File": row.filename,
          "Deed Name": deed.smart_deed_name || `deed_${deed.deed_number}`,
          "Deed Number": deed.deed_number,
          Status: row.status,
          Prediction: deed.prediction,
          Confidence: (deed.confidence * 100).toFixed(1) + "%",
          "Page Range": deed.page_range || "",
          "Pages in Deed": deed.pages_in_deed || "",
          "Boundary Confidence": deed.deed_boundary_info ? (deed.deed_boundary_info.confidence * 100).toFixed(1) + "%" : "",
          "Has Reservations": deed.prediction === "has_reservation" ? "YES" : "NO",
          Explanation: deed.explanation || ""
        });
      });
      
      // Also add a summary row
      const deedsWithReservations = row.deedResults.filter(d => d.prediction === "has_reservation").length;
      data.push({
        "Original File": row.filename + " (Summary)",
        "Deed Name": "ALL_DEEDS_SUMMARY",
        "Deed Number": "SUMMARY",
        Status: row.status,
        Prediction: `${deedsWithReservations}/${row.deedResults.length} deeds have reservations`,
        Confidence: "",
        "Page Range": "All pages",
        "Pages in Deed": row.totalDeeds + " deeds total",
        "Boundary Confidence": "",
        "Has Reservations": deedsWithReservations > 0 ? "YES" : "NO",
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
        Confidence: row.confidence ? (row.confidence * 100).toFixed(1) + "%" : "",
        "Page Range": "All pages",
        "Pages in Deed": "",
        "Boundary Confidence": "",
        "Has Reservations": row.prediction === "has_reservation" ? "YES" : "NO",
        Explanation: row.explanation ?? ""
      });
    }
  });
  
  return Papa.unparse(data);
}
