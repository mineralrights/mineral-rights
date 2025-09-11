import React, { useState, useEffect } from 'react';
import { testConnection } from '../lib/api';

interface ConnectionTestProps {
  onConnectionStatus?: (isConnected: boolean) => void;
}

export default function ConnectionTest({ onConnectionStatus }: ConnectionTestProps) {
  const [isTesting, setIsTesting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<{
    success: boolean;
    message: string;
    details?: any;
  } | null>(null);
  const [lastTested, setLastTested] = useState<Date | null>(null);

  const runConnectionTest = async () => {
    setIsTesting(true);
    setConnectionStatus(null);
    
    try {
      const result = await testConnection();
      setConnectionStatus(result);
      setLastTested(new Date());
      onConnectionStatus?.(result.success);
    } catch (error) {
      setConnectionStatus({
        success: false,
        message: `Test failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        details: error
      });
      onConnectionStatus?.(false);
    } finally {
      setIsTesting(false);
    }
  };

  // Auto-test on component mount
  useEffect(() => {
    runConnectionTest();
  }, []);

  return (
    <div className="bg-white p-4 rounded-lg shadow-md border">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-800">Connection Status</h3>
        <button
          onClick={runConnectionTest}
          disabled={isTesting}
          className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {isTesting ? 'Testing...' : 'Test Again'}
        </button>
      </div>
      
      {connectionStatus && (
        <div className={`p-3 rounded-md ${
          connectionStatus.success 
            ? 'bg-green-50 border border-green-200' 
            : 'bg-red-50 border border-red-200'
        }`}>
          <div className="flex items-center">
            <div className={`w-3 h-3 rounded-full mr-2 ${
              connectionStatus.success ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span className={`font-medium ${
              connectionStatus.success ? 'text-green-800' : 'text-red-800'
            }`}>
              {connectionStatus.message}
            </span>
          </div>
          
          {connectionStatus.details && (
            <details className="mt-2">
              <summary className="text-sm text-gray-600 cursor-pointer">
                Technical Details
              </summary>
              <pre className="mt-1 text-xs bg-gray-100 p-2 rounded overflow-auto">
                {JSON.stringify(connectionStatus.details, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
      
      {lastTested && (
        <p className="text-xs text-gray-500 mt-2">
          Last tested: {lastTested.toLocaleTimeString()}
        </p>
      )}
      
      {connectionStatus && !connectionStatus.success && (
        <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
          <h4 className="font-medium text-yellow-800 mb-2">Troubleshooting Tips:</h4>
          <ul className="text-sm text-yellow-700 space-y-1">
            <li>• Try refreshing the page</li>
            <li>• Check your internet connection</li>
            <li>• Try using a different browser</li>
            <li>• Clear your browser cache</li>
            <li>• Contact support if the issue persists</li>
          </ul>
        </div>
      )}
    </div>
  );
}
