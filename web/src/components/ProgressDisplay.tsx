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
    ? "bg-white border border-gray-300 rounded-lg shadow-lg p-4 max-w-sm mx-auto my-4"
    : "fixed top-4 right-4 bg-white border border-gray-300 rounded-lg shadow-lg p-4 max-w-sm z-50";

  return (
    <div className={containerClasses}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-800">Processing Progress</h3>
        <div className="text-xs text-gray-500">
          {progress.progress_percentage.toFixed(1)}%
        </div>
      </div>
      
      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
        <div 
          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
          style={{ width: `${progress.progress_percentage}%` }}
        />
      </div>
      
      {/* Progress Details */}
      <div className="space-y-1 text-xs text-gray-600">
        <div className="flex justify-between">
          <span>Pages:</span>
          <span className="font-medium">{progress.current_page}/{progress.total_pages}</span>
        </div>
        
        <div className="flex justify-between">
          <span>Elapsed:</span>
          <span>{formatTime(progress.processing_time)}</span>
        </div>
        
        <div className="flex justify-between">
          <span>Est. remaining:</span>
          <span>{formatTime(progress.estimated_remaining)}</span>
        </div>
        
        <div className="flex justify-between">
          <span>Reservations found:</span>
          <span className="font-medium text-green-600">
            {progress.pages_with_reservations.length} pages
          </span>
        </div>
      </div>
      
      {/* Current Page Result */}
      {progress.current_page_result && (
        <div className="mt-3 p-2 bg-gray-50 rounded text-xs">
          <div className="font-medium text-gray-800">
            Page {progress.current_page_result.page_number}:
          </div>
          <div className={`mt-1 ${progress.current_page_result.has_reservations ? 'text-green-600' : 'text-gray-600'}`}>
            {progress.current_page_result.has_reservations ? '🎯 HAS RESERVATIONS' : '📄 No reservations'}
          </div>
          <div className="text-gray-500 mt-1">
            Confidence: {(progress.current_page_result.confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}
      
      {/* Reservations List */}
      {progress.pages_with_reservations.length > 0 && (
        <div className="mt-3 p-2 bg-green-50 rounded text-xs">
          <div className="font-medium text-green-800 mb-1">
            Reservations found on pages:
          </div>
          <div className="text-green-700">
            {progress.pages_with_reservations.join(', ')}
          </div>
        </div>
      )}
    </div>
  );
}
