"use client";

import PDFUpload from "@/components/PDFUpload";
import ResultsTable from "@/components/ResultsTable";
import { predictBatch } from "@/lib/api";
import { rowsToCSV } from "@/lib/csv";
import { useState } from "react";
import { PredictionRow } from "@/lib/types";

export default function Home() {
  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const handleFiles = async (files: File[]) => {
    setIsRunning(true);
    const result = await predictBatch(files);
    setRows(result);
    setIsRunning(false);
  };

  const downloadCSV = () => {
    const csv = rowsToCSV(rows);
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="max-w-3xl mx-auto py-16 px-4">
      <h1 className="text-4xl font-semibold mb-10">Mineral-Rights Classifier</h1>

      <PDFUpload onSelect={handleFiles} />

      {isRunning && (
        <p className="mt-4 text-blue-600 animate-pulse">Processing…</p>
      )}

      <ResultsTable rows={rows} />

      {rows.length > 0 && (
        <button
          onClick={downloadCSV}
          className="mt-6 inline-block bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Download CSV
        </button>
      )}
    </main>
  );
}
