"use client";

import PDFUpload from "@/components/PDFUpload";
import ProcessingModeSelector from "@/components/ProcessingModeSelector";
import ResultsTable from "@/components/ResultsTable";
import { predictBatch } from "@/lib/api";
import { rowsToCSV } from "@/lib/csv";
import { useState } from "react";
import { PredictionRow, ProcessingMode, SplittingStrategy } from "@/lib/types";

export default function Home() {
  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>("single_deed");
  const [splittingStrategy, setSplittingStrategy] = useState<SplittingStrategy>("smart_detection");

  const handleFiles = async (files: File[]) => {
    setIsRunning(true);
    await predictBatch(files, processingMode, splittingStrategy, setRows);
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
    <main className="flex justify-center py-16 px-4">
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-lg p-10">
        <h1 className="text-4xl font-semibold mb-10 text-[color:var(--accent)]">
          Mineral-Rights&nbsp;Classifier
        </h1>

        <ProcessingModeSelector
          processingMode={processingMode}
          splittingStrategy={splittingStrategy}
          onProcessingModeChange={setProcessingMode}
          onSplittingStrategyChange={setSplittingStrategy}
        />

        <PDFUpload onSelect={handleFiles} />

        {isRunning && (
          <p className="mt-4 text-[color:var(--accent)] animate-pulse">
            Processing…
          </p>
        )}

        <ResultsTable rows={rows} />

        {rows.length > 0 && (
          <button
            onClick={downloadCSV}
            className="mt-6 inline-block bg-[color:var(--accent)] text-white px-4 py-2 rounded hover:brightness-110"
          >
            Download CSV
          </button>
        )}
      </div>
    </main>
  );
}
