import { PredictionRow } from "@/lib/types";

type Props = { rows: PredictionRow[] };

export default function ResultsTable({ rows }: Props) {
  if (rows.length === 0) return null;

  return (
    <div className="overflow-x-auto mt-8">
      <table className="min-w-full border text-sm">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-4 py-2 text-left">File</th>
            <th className="px-4 py-2 text-left">Status</th>
            <th className="px-4 py-2 text-left">Prediction</th>
            <th className="px-4 py-2 text-left">Explanation</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.filename} className="border-t">
              <td className="px-4 py-2">{row.filename}</td>
              <td className="px-4 py-2">{row.status}</td>
              <td className="px-4 py-2">{row.prediction ?? "—"}</td>
              <td className="px-4 py-2 whitespace-pre-wrap">{row.explanation ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
