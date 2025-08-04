export interface PredictionRow {
  filename: string;
  status: "waiting" | "processing" | "done" | "error";
  prediction?: string;
  explanation?: string;
  steps?: string[];         // for live log bubbles
  // New fields for multi-deed support
  deedResults?: DeedResult[];
  processingMode?: "single_deed" | "multi_deed";
  totalDeeds?: number;
}

export interface DeedResult {
  deed_number: number;
  classification: number;
  confidence: number;
  prediction: "has_reservation" | "no_reservation";
  explanation?: string;
  deed_file?: string;
  pages_in_deed?: number;
}

export type ProcessingMode = "single_deed" | "multi_deed";
export type SplittingStrategy = "smart_detection" | "page_based" | "ai_assisted";
