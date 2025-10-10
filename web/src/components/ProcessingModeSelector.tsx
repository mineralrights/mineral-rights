"use client";
import { ProcessingMode, SplittingStrategy } from "@/lib/types";

type Props = {
  processingMode: ProcessingMode;
  splittingStrategy: SplittingStrategy;
  onProcessingModeChange: (mode: ProcessingMode) => void;
  onSplittingStrategyChange: (strategy: SplittingStrategy) => void;
};

export default function ProcessingModeSelector({
  processingMode,
  splittingStrategy,
  onProcessingModeChange,
  onSplittingStrategyChange
}: Props) {
  return (
    <div className="mb-6 p-4 bg-gray-50 rounded-lg">
      <h3 className="text-lg font-medium mb-4 text-gray-800">Processing Options</h3>
      
      {/* Processing Mode Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Document Type
        </label>
        <div className="space-y-3">
          <label className="flex items-center">
            <input
              type="radio"
              name="processingMode"
              value="single_deed"
              checked={processingMode === "single_deed"}
              onChange={(e) => onProcessingModeChange(e.target.value as ProcessingMode)}
              className="mr-2"
            />
            <span className="text-sm">
              <strong>Single Deed</strong> - One deed per PDF
            </span>
          </label>
          {/* Multi-deed option temporarily disabled - not working properly */}
          {/* <label className="flex items-center">
            <input
              type="radio"
              name="processingMode"
              value="multi_deed"
              checked={processingMode === "multi_deed"}
              onChange={(e) => onProcessingModeChange(e.target.value as ProcessingMode)}
              className="mr-2"
            />
            <span className="text-sm">
              <strong>Multiple Deeds</strong> - Multiple deeds in one PDF (uses Document AI)
            </span>
          </label> */}
          <label className="flex items-center">
            <input
              type="radio"
              name="processingMode"
              value="page_by_page"
              checked={processingMode === "page_by_page"}
              onChange={(e) => onProcessingModeChange(e.target.value as ProcessingMode)}
              className="mr-2"
            />
            <span className="text-sm">
              <strong>Page-by-Page</strong> - Treat each page as a separate deed
            </span>
          </label>
        </div>
      </div>

      {/* Splitting Strategy (only for multi-deed) - temporarily disabled */}
      {/* {processingMode === "multi_deed" && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Splitting Method
          </label>
          <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
            <div className="flex items-center">
              <span className="text-blue-600 font-medium">🤖 Document AI Smart Chunking</span>
            </div>
            <p className="text-sm text-blue-700 mt-1">
              Uses your custom trained Google Cloud Document AI model with smart chunking for precise deed boundary detection. 
              Handles large PDFs (up to 300+ pages) by processing in optimized chunks while maintaining accuracy.
            </p>
            <div className="mt-2 text-xs text-blue-600">
              ✅ Validated on 278-page documents with 100% accuracy<br/>
              ✅ Memory-efficient processing<br/>
              ✅ Automatic offset correction<br/>
              ✅ Real-time progress logging
            </div>
          </div>
        </div>
      )} */}

      {/* Page-by-Page Description */}
      {processingMode === "page_by_page" && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Processing Method
          </label>
          <div className="bg-green-50 border border-green-200 rounded-md p-3">
            <div className="flex items-center">
              <span className="text-green-600 font-medium">📄 Page-by-Page Classification</span>
            </div>
            <p className="text-sm text-green-700 mt-1">
              Treats each page as a separate deed and classifies each page individually for mineral rights reservations. 
              Perfect for long PDFs where you want to know exactly which pages contain reservations.
            </p>
            <div className="mt-2 text-xs text-green-600">
              ✅ No Document AI required - works with any PDF<br/>
              ✅ Reports exact page numbers with reservations<br/>
              ✅ Memory-efficient processing<br/>
              ✅ Fast processing for large documents<br/>
              ✅ Ideal for 300+ page documents
            </div>
          </div>
        </div>
      )}
    </div>
  );
}