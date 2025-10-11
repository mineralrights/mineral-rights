import React from 'react';

interface ProgressInfo {
  current_page: number;
  total_pages: number;
  pages_with_reservations: number[];
  processing_time: number;
  estimated_remaining: number;
  current_page_result?: {
    page_number: number;
    has_reservations: boolean;
    confidence: number;
    reasoning: string;
  };
  progress_percentage: number;
}

interface ProgressDisplayProps {
  progress: ProgressInfo;
  isVisible: boolean;
  inline?: boolean; // New prop to control positioning
}

export default function ProgressDisplay({ progress, isVisible, inline = false }: ProgressDisplayProps) {
  if (!isVisible || !progress) return null;

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  // Choose positioning based on inline prop
  const containerClasses = inline 
    ? "bg-white border border-gray-200 rounded-lg shadow-sm p-6 max-w-md mx-auto my-6"
    : "fixed top-4 right-4 bg-white border border-gray-200 rounded-lg shadow-lg p-6 max-w-md z-50";

  return (
    <div className={containerClasses}>
      <div className="flex items-center mb-4">
        <div className="flex-shrink-0">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </div>
        </div>
        <div className="ml-3">
          <h3 className="text-lg font-semibold text-gray-900">Processing Progress</h3>
          <p className="text-sm text-gray-600">Document analysis in progress</p>
        </div>
        <div className="ml-auto text-right">
          <div className="text-2xl font-bold text-blue-600">
            {progress.progress_percentage.toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">Complete</div>
        </div>
      </div>
      
      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-3 mb-6">
        <div 
          className="bg-blue-600 h-3 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progress.progress_percentage}%` }}
        />
      </div>
      
      {/* Progress Details */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="text-xs text-gray-600 mb-1">Pages Processed</div>
          <div className="text-lg font-semibold text-gray-900">{progress.current_page}/{progress.total_pages}</div>
        </div>
        
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="text-xs text-gray-600 mb-1">Elapsed Time</div>
          <div className="text-lg font-semibold text-gray-900">{formatTime(progress.processing_time)}</div>
        </div>
        
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="text-xs text-gray-600 mb-1">Est. Remaining</div>
          <div className="text-lg font-semibold text-gray-900">{formatTime(progress.estimated_remaining)}</div>
        </div>
        
        <div className="bg-green-50 p-3 rounded-lg">
          <div className="text-xs text-green-600 mb-1">Reservations Found</div>
          <div className="text-lg font-semibold text-green-700">
            {progress.pages_with_reservations.length} pages
          </div>
        </div>
      </div>
      
      {/* Current Page Result */}
      {progress.current_page_result && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="font-semibold text-gray-900">
              Page {progress.current_page_result.page_number} Analysis
            </div>
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${
              progress.current_page_result.has_reservations 
                ? 'bg-green-100 text-green-800' 
                : 'bg-gray-100 text-gray-800'
            }`}>
              {progress.current_page_result.has_reservations ? 'HAS RESERVATIONS' : 'NO RESERVATIONS'}
            </div>
          </div>
          <div className="text-sm text-gray-600 mb-2">
            Confidence: <span className="font-semibold text-blue-600">{(progress.current_page_result.confidence * 100).toFixed(1)}%</span>
          </div>
          {progress.current_page_result.reasoning && (
            <div className="text-xs text-gray-500 bg-white p-2 rounded border">
              {progress.current_page_result.reasoning}
            </div>
          )}
        </div>
      )}
      
      {/* Reservations List */}
      {progress.pages_with_reservations.length > 0 && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center mb-2">
            <svg className="w-5 h-5 text-green-600 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <div className="font-semibold text-green-800">
              Reservations Found
            </div>
          </div>
          <div className="text-sm text-green-700">
            Pages with mineral rights reservations: <span className="font-semibold">{progress.pages_with_reservations.join(', ')}</span>
          </div>
        </div>
      )}
    </div>
  );
}
