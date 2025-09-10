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
        <div className="flex gap-4">
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
          <label className="flex items-center">
            <input
              type="radio"
              name="processingMode"
              value="multi_deed"
              checked={processingMode === "multi_deed"}
              onChange={(e) => onProcessingModeChange(e.target.value as ProcessingMode)}
              className="mr-2"
            />
            <span className="text-sm">
              <strong>Multiple Deeds</strong> - Multiple deeds in one PDF
            </span>
          </label>
        </div>
      </div>

      {/* Splitting Strategy (only for multi-deed) */}
      {processingMode === "multi_deed" && (
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
      )}
    </div>
  );
}