import { PredictionRow, PageResult, DeedResult } from "@/lib/types";
import { downloadCSV } from "@/lib/csv";

type Props = { rows: PredictionRow[] };

export default function ResultsTable({ rows }: Props) {
  if (!rows.length) return null;

  const handleExportCSV = () => {
    downloadCSV(rows);
  };

  return (
    <div className="mt-8 bg-white border border-gray-200 rounded-lg shadow-sm">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-indigo-700 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                </svg>
              </div>
            </div>
            <div className="ml-3">
              <h3 className="text-lg font-semibold text-gray-900">Processing Results</h3>
              <p className="text-sm text-gray-600">{rows.length} document{rows.length !== 1 ? 's' : ''} processed</p>
            </div>
          </div>
          <button
            onClick={handleExportCSV}
            className="bg-indigo-700 text-white px-6 py-2 rounded-lg hover:bg-indigo-800 transition-colors font-medium flex items-center"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Export CSV
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">File</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Prediction</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Summary</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {rows.map(row => <Row key={row.filename} row={row} />)}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Row({ row }: { row: PredictionRow }) {
  return (
    <>
      <tr className="hover:bg-gray-50 transition-colors">
        <td className="px-6 py-4 whitespace-nowrap">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                <svg className="w-4 h-4 text-gray-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                </svg>
              </div>
            </div>
            <div className="ml-3">
              <div className="text-sm font-medium text-gray-900">{row.filename}</div>
            </div>
          </div>
        </td>
        <td className="px-6 py-4 whitespace-nowrap">
          <StatusBadge status={row.status} />
        </td>
        <td className="px-6 py-4 whitespace-nowrap">
          <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
            row.prediction === "has_reservation" 
              ? "bg-red-100 text-red-800" 
              : row.prediction === "no_reservation"
              ? "bg-green-100 text-green-800"
              : "bg-gray-100 text-gray-800"
          }`}>
            {row.prediction === "has_reservation" ? "Has Reservations" : 
             row.prediction === "no_reservation" ? "No Reservations" : 
             row.prediction ?? "—"}
          </div>
        </td>
        <td className="px-6 py-4">
          {row.processingMode === "multi_deed" && row.deedResults ? (
            <div className="text-sm">
              <div className="font-medium text-gray-900">
                {row.deedResults.length} deeds processed
              </div>
              <div className="text-xs text-gray-600 mt-1">
                {row.deedResults.filter(d => d.prediction === "has_reservation").length} with reservations
              </div>
            </div>
          ) : row.processingMode === "page_by_page" ? (
            <div className="text-sm">
              <div className="font-medium text-gray-900">
                {row.pagesWithReservations?.length || 0} of {row.totalPages || 0} pages
              </div>
              {row.pagesWithReservations && row.pagesWithReservations.length > 0 && (
                <div className="text-xs text-gray-600 mt-1">
                  Pages: {row.pagesWithReservations.join(", ")}
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-gray-600">
              {row.explanation ?? "—"}
            </div>
          )}
        </td>
      </tr>
    </>
  );
}

function StatusBadge({ status }: { status: PredictionRow["status"] }) {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case "waiting":
        return {
          bg: "bg-gray-100",
          text: "text-gray-800",
          icon: "⏳",
          label: "Waiting"
        };
      case "processing":
        return {
          bg: "bg-blue-100",
          text: "text-blue-800",
          icon: "🔄",
          label: "Processing"
        };
      case "done":
        return {
          bg: "bg-green-100",
          text: "text-green-800",
          icon: "✅",
          label: "Complete"
        };
      case "error":
        return {
          bg: "bg-red-100",
          text: "text-red-800",
          icon: "❌",
          label: "Error"
        };
      default:
        return {
          bg: "bg-gray-100",
          text: "text-gray-800",
          icon: "❓",
          label: status
        };
    }
  };

  const config = getStatusConfig(status);

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.bg} ${config.text} ${
      status === "processing" ? "animate-pulse" : ""
    }`}>
      <span className="mr-1">{config.icon}</span>
      {config.label}
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
        {deedResult.page_range && (
          <div>Pages: {deedResult.page_range}</div>
        )}
        {deedResult.pages_in_deed && (
          <div>Pages in deed: {deedResult.pages_in_deed}</div>
        )}
      </div>
      
      {deedResult.explanation && (
        <div className="mt-2 text-xs text-gray-700 bg-white p-2 rounded border">
          <div className="font-medium mb-1">Reasoning:</div>
          <div className="whitespace-pre-wrap">
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
          <div className="whitespace-pre-wrap">
            {pageResult.explanation}
          </div>
        </div>
      )}
    </div>
  );
}
