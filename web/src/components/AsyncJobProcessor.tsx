'use client';

import React, { useState, useCallback } from 'react';
import { createJob, getJobStatus, pollJobUntilComplete, JobStatus } from '@/lib/api_async';
import { ProcessingMode, SplittingStrategy } from '@/lib/types';

interface AsyncJobProcessorProps {
  onResult?: (result: any) => void;
  onError?: (error: string) => void;
}

export default function AsyncJobProcessor({ onResult, onError }: AsyncJobProcessorProps) {
  const [file, setFile] = useState<File | null>(null);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('single_deed');
  const [splittingStrategy, setSplittingStrategy] = useState<SplittingStrategy>('document_ai');
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const addLog = useCallback((message: string) => {
    setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      addLog(`File selected: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (!file) {
      onError?.('Please select a file');
      return;
    }

    setIsProcessing(true);
    setJobId(null);
    setJobStatus(null);
    setLogs([]);

    try {
      addLog('Creating job...');
      const jobResult = await createJob(file, processingMode, splittingStrategy);
      
      setJobId(jobResult.job_id);
      addLog(`Job created with ID: ${jobResult.job_id}`);
      addLog('Job queued for processing...');

      // Poll for job completion
      const finalStatus = await pollJobUntilComplete(
        jobResult.job_id,
        (status) => {
          setJobStatus(status);
          addLog(`Status: ${status.status} (${status.progress}%)`);
        },
        2000 // Poll every 2 seconds
      );

      addLog('Job completed successfully!');
      onResult?.(finalStatus);
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      addLog(`Error: ${errorMessage}`);
      onError?.(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setJobId(null);
    setJobStatus(null);
    setLogs([]);
    setIsProcessing(false);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Mineral Rights Document Processor</h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload */}
        <div>
          <label htmlFor="file" className="block text-sm font-medium text-gray-700 mb-2">
            Upload PDF Document
          </label>
          <input
            type="file"
            id="file"
            accept=".pdf"
            onChange={handleFileChange}
            disabled={isProcessing}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          {file && (
            <p className="mt-2 text-sm text-gray-600">
              Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
        </div>

        {/* Processing Mode */}
        <div>
          <label htmlFor="processingMode" className="block text-sm font-medium text-gray-700 mb-2">
            Processing Mode
          </label>
          <select
            id="processingMode"
            value={processingMode}
            onChange={(e) => setProcessingMode(e.target.value as ProcessingMode)}
            disabled={isProcessing}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="single_deed">Single Deed</option>
            <option value="multi_deed">Multi-Deed</option>
            <option value="page_by_page">Page by Page</option>
          </select>
        </div>

        {/* Splitting Strategy */}
        <div>
          <label htmlFor="splittingStrategy" className="block text-sm font-medium text-gray-700 mb-2">
            Splitting Strategy
          </label>
          <select
            id="splittingStrategy"
            value={splittingStrategy}
            onChange={(e) => setSplittingStrategy(e.target.value as SplittingStrategy)}
            disabled={isProcessing}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="document_ai">Document AI</option>
            <option value="custom">Custom</option>
          </select>
        </div>

        {/* Submit Button */}
        <div className="flex space-x-4">
          <button
            type="submit"
            disabled={!file || isProcessing}
            className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? 'Processing...' : 'Start Processing'}
          </button>
          
          <button
            type="button"
            onClick={handleReset}
            disabled={isProcessing}
            className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Reset
          </button>
        </div>
      </form>

      {/* Job Status */}
      {jobStatus && (
        <div className="mt-6 p-4 bg-gray-50 rounded-md">
          <h3 className="text-lg font-semibold mb-2">Job Status</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium">Job ID:</span> {jobStatus.job_id}
            </div>
            <div>
              <span className="font-medium">Status:</span> 
              <span className={`ml-2 px-2 py-1 rounded text-xs ${
                jobStatus.status === 'completed' ? 'bg-green-100 text-green-800' :
                jobStatus.status === 'failed' ? 'bg-red-100 text-red-800' :
                jobStatus.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                'bg-yellow-100 text-yellow-800'
              }`}>
                {jobStatus.status}
              </span>
            </div>
            <div>
              <span className="font-medium">Progress:</span> {jobStatus.progress}%
            </div>
            <div>
              <span className="font-medium">Filename:</span> {jobStatus.filename}
            </div>
          </div>
          
          {jobStatus.progress > 0 && (
            <div className="mt-3">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${jobStatus.progress}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Logs */}
      {logs.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Processing Logs</h3>
          <div className="bg-black text-green-400 p-4 rounded-md font-mono text-sm max-h-64 overflow-y-auto">
            {logs.map((log, index) => (
              <div key={index}>{log}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
