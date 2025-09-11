import { PredictionRow, PageResult } from "@/lib/types";
import { useState } from "react";
import StepBubble from "./StepBubble";

type Props = { rows: PredictionRow[] };

export default function ResultsTable({ rows }: Props) {
  if (!rows.length) return null;

  return (
    <div className="overflow-x-auto mt-8">
      <table className="min-w-full border text-sm">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-4 py-2 text-left">File</th>
            <th className="px-4 py-2 text-left">Status</th>
            <th className="px-4 py-2 text-left">Prediction</th>
            <th className="px-4 py-2 text-left">Confidence</th>
            <th className="px-4 py-2 text-left">Summary</th>
            <th className="px-4 py-2 text-left">Details</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(row => <Row key={row.filename} row={row} />)}
        </tbody>
      </table>
    </div>
  );
}

function Row({ row }: { row: PredictionRow }) {
  const [open, setOpen] = useState(false);

  const formatConfidence = (confidence?: number) => {
    if (confidence === undefined) return "—";
    return (confidence * 100).toFixed(0) + "%";
  };

  return (
    <>
      <tr className="border-t odd:bg-gray-50">
        <td className="px-4 py-2">{row.filename}</td>
        <td className="px-4 py-2">
          <StatusBadge status={row.status} />
        </td>
        <td className="px-4 py-2">{row.prediction ?? "—"}</td>
        <td className="px-4 py-2">{formatConfidence(row.confidence)}</td>
        <td className="px-4 py-2 whitespace-pre-wrap">
          {row.processingMode === "page_by_page" ? (
            <div className="text-sm">
              <div className="font-medium">
                {row.pagesWithReservations?.length || 0} of {row.totalPages || 0} pages
              </div>
              {row.pagesWithReservations && row.pagesWithReservations.length > 0 && (
                <div className="text-xs text-gray-600 mt-1">
                  Pages: {row.pagesWithReservations.join(", ")}
                </div>
              )}
            </div>
          ) : (
            row.explanation ?? "—"
          )}
        </td>
        <td className="px-4 py-2">
          {row.steps && (
            <button
              onClick={() => setOpen(!open)}
              className="text-[color:var(--accent)] hover:underline"
            >
              {open ? "Hide" : "Show"}
            </button>
          )}
        </td>
      </tr>
      {open && (
        <tr>
          <td colSpan={6} className="bg-gray-50 px-6 py-4">
            {/* Show processing steps */}
            {row.steps && (
              <div className="mb-4">
                <h4 className="font-medium text-gray-800 mb-2">Processing Steps:</h4>
                {row.steps.map((s, i) => (
                  <StepBubble key={i} text={s} />
                ))}
              </div>
            )}
            
            {/* Show page-by-page results */}
            {row.processingMode === "page_by_page" && row.pageResults && (
              <div>
                <h4 className="font-medium text-gray-800 mb-3">Page-by-Page Results:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {row.pageResults.map((pageResult) => (
                    <PageResultCard key={pageResult.page_number} pageResult={pageResult} />
                  ))}
                </div>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function StatusBadge({ status }: { status: PredictionRow["status"] }) {
  const map: Record<string, string> = {
    waiting: "bg-gray-100 text-gray-600",
    processing: "bg-yellow-100 text-yellow-800 animate-pulse",
    done: "bg-green-100 text-green-800",
    error: "bg-red-100 text-red-800",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${map[status]}`}>
      {status}
    </span>
  );
}

function PageResultCard({ pageResult }: { pageResult: PageResult }) {
  const hasReservations = pageResult.has_reservations;
  
  return (
    <div className={`border rounded-lg p-3 ${
      hasReservations 
        ? "border-red-200 bg-red-50" 
        : "border-gray-200 bg-gray-50"
    }`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-sm">
          Page {pageResult.page_number}
        </span>
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
          hasReservations
            ? "bg-red-100 text-red-800"
            : "bg-green-100 text-green-800"
        }`}>
          {hasReservations ? "Has Reservations" : "No Reservations"}
        </span>
      </div>
      
      <div className="text-xs text-gray-600 space-y-1">
        <div>Confidence: {(pageResult.confidence * 100).toFixed(0)}%</div>
        {pageResult.text_length && (
          <div>Text: {pageResult.text_length} chars</div>
        )}
        {pageResult.processing_time && (
          <div>Time: {pageResult.processing_time.toFixed(1)}s</div>
        )}
      </div>
      
      {pageResult.explanation && (
        <div className="mt-2 text-xs text-gray-700 bg-white p-2 rounded border">
          <div className="font-medium mb-1">Reasoning:</div>
          <div className="whitespace-pre-wrap max-h-20 overflow-y-auto">
            {pageResult.explanation}
          </div>
        </div>
      )}
    </div>
  );
}
