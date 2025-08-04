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
          <select
            value={splittingStrategy}
            onChange={(e) => onSplittingStrategyChange(e.target.value as SplittingStrategy)}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-[color:var(--accent)]"
          >
            <option value="smart_detection">
              Smart Detection - Automatically detect deed boundaries (Recommended)
            </option>
            <option value="page_based">
              Page-Based - Split every 3 pages (Simple fallback)
            </option>
            <option value="ai_assisted">
              AI-Assisted - Use Claude for complex documents (Advanced)
            </option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Smart detection analyzes text patterns to find where each deed begins
          </p>
        </div>
      )}
    </div>
  );
}