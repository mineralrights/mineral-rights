import { PredictionRow, PageResult, DeedResult } from "@/lib/types";
import { downloadCSV } from "@/lib/csv";

type Props = { rows: PredictionRow[] };

export default function ResultsTable({ rows }: Props) {
  if (!rows.length) return null;

  const handleExportCSV = () => {
    downloadCSV(rows);
  };

  return (
    <div className="mt-8">
      {/* Export Button */}
      <div className="mb-4 flex justify-between items-center">
        <h3 className="text-lg font-semibold">Processing Results</h3>
        <button
          onClick={handleExportCSV}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
        >
          📊 Export CSV
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full border text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 text-left">File</th>
              <th className="px-4 py-2 text-left">Status</th>
              <th className="px-4 py-2 text-left">Prediction</th>
              <th className="px-4 py-2 text-left">Confidence</th>
              <th className="px-4 py-2 text-left">Summary</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => <Row key={row.filename} row={row} />)}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Row({ row }: { row: PredictionRow }) {

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
          {row.processingMode === "multi_deed" && row.deedResults ? (
            <div className="text-sm">
              <div className="font-medium">
                {row.deedResults.length} deeds processed
              </div>
              <div className="text-xs text-gray-600 mt-1">
                {row.deedResults.filter(d => d.prediction === "has_reservation").length} with reservations
              </div>
            </div>
          ) : row.processingMode === "page_by_page" ? (
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
      </tr>
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

function DeedResultCard({ deedResult }: { deedResult: DeedResult }) {
  const hasReservations = deedResult.prediction === "has_reservation";
  
  return (
    <div className={`border rounded-lg p-3 ${
      hasReservations 
        ? "border-red-200 bg-red-50" 
        : "border-gray-200 bg-gray-50"
    }`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-sm">
          Deed {deedResult.deed_number}
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
        <div>Confidence: {(deedResult.confidence * 100).toFixed(0)}%</div>
        {deedResult.page_range && (
          <div>Pages: {deedResult.page_range}</div>
        )}
        {deedResult.pages_in_deed && (
          <div>Pages in deed: {deedResult.pages_in_deed}</div>
        )}
        {deedResult.deed_boundary_info && (
          <div>Boundary confidence: {(deedResult.deed_boundary_info.confidence * 100).toFixed(0)}%</div>
        )}
      </div>
      
      {deedResult.explanation && (
        <div className="mt-2 text-xs text-gray-700 bg-white p-2 rounded border">
          <div className="font-medium mb-1">Reasoning:</div>
          <div className="whitespace-pre-wrap max-h-20 overflow-y-auto">
            {deedResult.explanation}
          </div>
        </div>
      )}
    </div>
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
