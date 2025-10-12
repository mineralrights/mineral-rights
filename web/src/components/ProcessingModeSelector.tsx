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
    <div className="mb-8 p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
      <div className="flex items-center mb-6">
        <div className="flex-shrink-0">
          <div className="w-8 h-8 bg-indigo-700 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
            </svg>
          </div>
        </div>
        <div className="ml-3">
          <h3 className="text-lg font-semibold text-gray-900">Processing Configuration</h3>
          <p className="text-sm text-gray-600">Select the appropriate processing method for your documents</p>
        </div>
      </div>
      
      {/* Processing Mode Selection */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-900 mb-4">
          Document Processing Method
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className={`relative flex items-start p-4 border-2 rounded-lg cursor-pointer transition-all ${
            processingMode === "single_deed" 
              ? "border-indigo-600 bg-indigo-50" 
              : "border-gray-200 hover:border-gray-300"
          }`}>
            <input
              type="radio"
              name="processingMode"
              value="single_deed"
              checked={processingMode === "single_deed"}
              onChange={(e) => onProcessingModeChange(e.target.value as ProcessingMode)}
              className="sr-only"
            />
            <div className="flex items-center">
              <div className={`w-4 h-4 rounded-full border-2 mr-3 ${
                processingMode === "single_deed" 
                  ? "border-indigo-600 bg-indigo-600" 
                  : "border-gray-300"
              }`}>
                {processingMode === "single_deed" && (
                  <div className="w-2 h-2 bg-white rounded-full mx-auto mt-0.5"></div>
                )}
              </div>
              <div>
                <div className="text-sm font-medium text-gray-900">Single Deed</div>
                <div className="text-xs text-gray-600">One deed per PDF document</div>
              </div>
            </div>
          </label>

          <label className={`relative flex items-start p-4 border-2 rounded-lg cursor-pointer transition-all ${
            processingMode === "page_by_page" 
              ? "border-indigo-600 bg-indigo-50" 
              : "border-gray-200 hover:border-gray-300"
          }`}>
            <input
              type="radio"
              name="processingMode"
              value="page_by_page"
              checked={processingMode === "page_by_page"}
              onChange={(e) => onProcessingModeChange(e.target.value as ProcessingMode)}
              className="sr-only"
            />
            <div className="flex items-center">
              <div className={`w-4 h-4 rounded-full border-2 mr-3 ${
                processingMode === "page_by_page" 
                  ? "border-indigo-600 bg-indigo-600" 
                  : "border-gray-300"
              }`}>
                {processingMode === "page_by_page" && (
                  <div className="w-2 h-2 bg-white rounded-full mx-auto mt-0.5"></div>
                )}
              </div>
              <div>
                <div className="text-sm font-medium text-gray-900">Page-by-Page</div>
                <div className="text-xs text-gray-600">Treat each page as a separate deed</div>
              </div>
            </div>
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
        <div className="mt-6">
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-indigo-700 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                  </svg>
                </div>
              </div>
              <div className="ml-3">
                <h4 className="text-lg font-semibold text-indigo-900">Page-by-Page Classification</h4>
                <p className="text-sm text-indigo-700">Advanced processing for comprehensive document analysis</p>
              </div>
            </div>
            <p className="text-sm text-indigo-800 mb-4">
              Treats each page as a separate deed and classifies each page individually for mineral rights reservations. 
              Perfect for long PDFs where you want to know exactly which pages contain reservations.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="flex items-center text-sm text-indigo-700">
                <svg className="w-4 h-4 text-indigo-600 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                No Document AI required - works with any PDF
              </div>
              <div className="flex items-center text-sm text-indigo-700">
                <svg className="w-4 h-4 text-indigo-600 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Reports exact page numbers with reservations
              </div>
              <div className="flex items-center text-sm text-indigo-700">
                <svg className="w-4 h-4 text-indigo-600 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Memory-efficient processing
              </div>
              <div className="flex items-center text-sm text-indigo-700">
                <svg className="w-4 h-4 text-indigo-600 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Fast processing for large documents
              </div>
              <div className="flex items-center text-sm text-indigo-700">
                <svg className="w-4 h-4 text-indigo-600 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Ideal for 40+ page documents
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}