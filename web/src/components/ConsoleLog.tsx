import React, { useState, useEffect } from 'react';

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
  data?: any;
}

interface ConsoleLogProps {
  isVisible: boolean;
  maxEntries?: number;
}

export default function ConsoleLog({ isVisible, maxEntries = 100 }: ConsoleLogProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (!isVisible) return;

    // Override console methods to capture logs
    const originalLog = console.log;
    const originalError = console.error;
    const originalWarn = console.warn;

    const addLog = (level: LogEntry['level'], ...args: any[]) => {
      const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
      ).join(' ');
      
      const logEntry: LogEntry = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        timestamp: new Date().toLocaleTimeString(),
        level,
        message,
        data: args.length > 1 ? args : undefined
      };

      setLogs(prev => {
        const newLogs = [...prev, logEntry];
        return newLogs.slice(-maxEntries);
      });
    };

    console.log = (...args) => {
      originalLog(...args);
      addLog('info', ...args);
    };

    console.error = (...args) => {
      originalError(...args);
      addLog('error', ...args);
    };

    console.warn = (...args) => {
      originalWarn(...args);
      addLog('warning', ...args);
    };

    // Cleanup
    return () => {
      console.log = originalLog;
      console.error = originalError;
      console.warn = originalWarn;
    };
  }, [isVisible, maxEntries]);

  const clearLogs = () => {
    setLogs([]);
  };

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'success': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'error': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getLevelIcon = (level: LogEntry['level']) => {
    switch (level) {
      case 'success': return '✅';
      case 'warning': return '⚠️';
      case 'error': return '❌';
      default: return 'ℹ️';
    }
  };

  if (!isVisible) return null;

  return (
    <div className="bg-gray-900 text-white rounded-lg shadow-lg">
      <div className="flex items-center justify-between p-3 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium">Console Log</span>
          <span className="text-xs text-gray-400">({logs.length} entries)</span>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
          <button
            onClick={clearLogs}
            className="text-xs px-2 py-1 bg-red-600 hover:bg-red-700 rounded"
          >
            Clear
          </button>
        </div>
      </div>
      
      <div className={`${isExpanded ? 'max-h-96' : 'max-h-32'} overflow-y-auto`}>
        {logs.length === 0 ? (
          <div className="p-4 text-center text-gray-400 text-sm">
            No logs yet. Start processing a PDF to see detailed logs here.
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {logs.map((log) => (
              <div key={log.id} className="text-xs font-mono">
                <div className="flex items-start space-x-2">
                  <span className="text-gray-400 flex-shrink-0">
                    {log.timestamp}
                  </span>
                  <span className="flex-shrink-0">
                    {getLevelIcon(log.level)}
                  </span>
                  <span className={`flex-1 ${getLevelColor(log.level)}`}>
                    {log.message}
                  </span>
                </div>
                {log.data && (
                  <div className="ml-8 mt-1 text-gray-300">
                    <pre className="text-xs overflow-x-auto">
                      {JSON.stringify(log.data, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
