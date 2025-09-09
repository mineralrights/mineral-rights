export interface PredictionRow {
  filename: string;
  status: "waiting" | "processing" | "done" | "error";
  prediction?: string;
  confidence?: number;  // NEW: Add confidence field
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
  smart_deed_name?: string;  // NEW: Smart generated name
  pages_in_deed?: number;
}

export type ProcessingMode = "single_deed" | "multi_deed";
export type SplittingStrategy = "document_ai" | "smart_detection" | "page_based" | "ai_assisted";
