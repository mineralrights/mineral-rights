import { PredictionRow } from "./types";

export async function predictBatch(files: File[]): Promise<PredictionRow[]> {
  const rows: PredictionRow[] = files.map(f => ({
    filename: f.name,
    status: "waiting"
  }));

  for (const row of rows) {
    row.status = "processing";

    const form = new FormData();
    form.append("file", files.find(f => f.name === row.filename)!);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
        method: "POST",
        body: form
      });
      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      row.prediction  = data.prediction;
      row.explanation = data.explanation;
      row.status      = "done";
    } catch (err: any) {
      row.status = "error";
      row.explanation = err.message ?? "Unknown error";
    }
  }

  return rows;
}
