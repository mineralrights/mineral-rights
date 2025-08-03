import { PredictionRow } from "@/lib/types";
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
            <th className="px-4 py-2 text-left">Explanation</th>
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

  return (
    <>
      <tr className="border-t odd:bg-gray-50">
        <td className="px-4 py-2">{row.filename}</td>
        <td className="px-4 py-2">
          <StatusBadge status={row.status} />
        </td>
        <td className="px-4 py-2">{row.prediction ?? "—"}</td>
        <td className="px-4 py-2 whitespace-pre-wrap">{row.explanation ?? "—"}</td>
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
      {open && row.steps && (            // keeps rendering while processing
        <tr>
          <td colSpan={5} className="bg-gray-50 px-6 py-4">
            {row.steps.map((s, i) => (
              <StepBubble key={i} text={s} />
            ))}
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
