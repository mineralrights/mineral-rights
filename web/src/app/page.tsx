"use client";

import PDFUpload from "@/components/PDFUpload";
import ProcessingModeSelector from "@/components/ProcessingModeSelector";
import ResultsTable from "@/components/ResultsTable";
import ProgressDisplay from "@/components/ProgressDisplay";
import ConsoleLog from "@/components/ConsoleLog";
import ApiKeyManager from "@/components/ApiKeyManager";
import ModelManager from "@/components/ModelManager";
import { processDocument, checkResumeCapability } from "@/lib/api_async";
import { rowsToCSV } from "@/lib/csv";
import { useState, useEffect } from "react";
import { PredictionRow, ProcessingMode, SplittingStrategy } from "@/lib/types";

export default function Home() {
  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>("single_deed");
  const [splittingStrategy, setSplittingStrategy] = useState<SplittingStrategy>("document_ai");
  const [progressInfo, setProgressInfo] = useState<any>(null);
  const [resumeJobId, setResumeJobId] = useState<string | null>(null);
  const [showResumeOption, setShowResumeOption] = useState(false);
  const [showConsoleLog, setShowConsoleLog] = useState(false);
  const [showApiKeyManager, setShowApiKeyManager] = useState(false);
  const [showModelManager, setShowModelManager] = useState(false);

  // Debug progressInfo state changes
  useEffect(() => {
    console.log('🔄 progressInfo state changed:', progressInfo);
  }, [progressInfo]);

  // Check for resume capability on page load
  useEffect(() => {
    const checkForResume = async () => {
      // Check localStorage for recent job IDs
      const recentJobIds = JSON.parse(localStorage.getItem('recentJobIds') || '[]');
      
      for (const jobId of recentJobIds.slice(-3)) { // Check last 3 jobs
        try {
          const resumeInfo = await checkResumeCapability(jobId);
          if (resumeInfo.can_resume) {
            setResumeJobId(jobId);
            setShowResumeOption(true);
            console.log('🔄 Found resumable job:', jobId, resumeInfo);
            break;
          }
        } catch (error) {
          console.log('No resume capability for job:', jobId);
        }
      }
    };

    checkForResume();
  }, []);

  const convertJobResultToPredictionRows = (
    result: any,
    processingMode: ProcessingMode
  ): PredictionRow[] => {
    const rows: PredictionRow[] = [];

    // Safety check: ensure result is not null/undefined
    if (!result) {
      console.error('⚠️ convertJobResultToPredictionRows received null/undefined result');
      return rows;
    }

    if (processingMode === 'single_deed') {
      // Single deed result
      const row: PredictionRow = {
        filename: result.filename || 'document.pdf',
        status: 'done',
        prediction: result.has_reservation ? 'has_reservation' : 'no_reservation',
        confidence: result.confidence || 0,
        explanation: result.reasoning || '',
        processingMode: 'single_deed'
      };
      rows.push(row);
    } else if (processingMode === 'multi_deed') {
      // Multi-deed result - create individual rows for each deed
      if (result.deed_results && Array.isArray(result.deed_results)) {
        result.deed_results.forEach((deed: any, index: number) => {
          const row: PredictionRow = {
            filename: result.filename || 'multi_deed_document.pdf',
            status: 'done',
            prediction: deed.has_reservations ? 'has_reservation' : 'no_reservation',
            confidence: deed.confidence || 0,
            explanation: deed.reasoning || 'No reasoning provided',
            processingMode: 'multi_deed',
            deedResults: [{
              deed_number: deed.deed_number,
              classification: deed.has_reservations ? 1 : 0,
              confidence: deed.confidence,
              prediction: deed.has_reservations ? 'has_reservation' : 'no_reservation',
              explanation: deed.reasoning || 'No reasoning provided',
              deed_file: deed.deed_file,
              pages_in_deed: deed.pages_in_deed,
              page_range: deed.deed_boundary_info?.page_range || (deed.pages && deed.pages.length > 0 ? `${Math.min(...deed.pages) + 1}-${Math.max(...deed.pages) + 1}` : 'Unknown'),
              pages: deed.pages || [],
              deed_boundary_info: deed.deed_boundary_info
            }]
          };
          rows.push(row);
        });
      }
    } else if (processingMode === 'page_by_page') {
      // Page-by-page result - create individual rows for each page
      console.log('🔍 DEBUG: Processing page_by_page result:', result);
      console.log('🔍 DEBUG: result.results:', result.results);
      console.log('🔍 DEBUG: result.page_results:', result.page_results);
      
      if ((result.results && Array.isArray(result.results)) || (result.page_results && Array.isArray(result.page_results))) {
        const pages = result.results || result.page_results;
        console.log('🔍 DEBUG: Pages array length:', pages.length);
        console.log('🔍 DEBUG: First page:', pages[0]);
        
        pages.forEach((page: any, index: number) => {
          console.log(`🔍 DEBUG: Creating row for page ${page.page_number} (index ${index})`);
          const row: PredictionRow = {
            filename: result.filename || 'document.pdf',
            status: 'done',
            prediction: page.has_reservations ? 'has_reservation' : 'no_reservation',
            confidence: page.confidence || 0,
            explanation: page.reasoning || page.explanation || '',
            processingMode: 'page_by_page',
            pageResults: [page], // Single page in each row
            totalPages: result.total_pages,
            pagesWithReservations: result.reservation_pages || []
          };
          rows.push(row);
          console.log(`🔍 DEBUG: Added row ${rows.length} for page ${page.page_number}`);
        });
        console.log('🔍 DEBUG: Total rows created:', rows.length);
      } else {
        // Fallback: create a summary row if no detailed page results
        const row: PredictionRow = {
          filename: result.filename || 'document.pdf',
          status: 'done',
          prediction: result.pages_with_reservations > 0 ? 'has_reservation' : 'no_reservation',
          confidence: result.pages_with_reservations > 0 ? 1.0 : 0.0,
          explanation: result.reasoning || `Found mineral rights reservations on ${result.pages_with_reservations} pages: ${(result.reservation_pages || []).join(', ')}`,
          processingMode: 'page_by_page',
          totalPages: result.total_pages,
          pagesWithReservations: result.reservation_pages || []
        };
        rows.push(row);
      }
    }

    return rows;
  };

  const handleFiles = async (files: File[]) => {
    setIsRunning(true);
    
    // Create initial rows for the new files and add them to existing rows
    const newFileRows: PredictionRow[] = files.map(f => ({
      filename: f.name,
      status: "waiting",
      steps: [],
      processingMode
    }));
    
    // Add the new files to the existing rows immediately
    setRows(prevRows => [...prevRows, ...newFileRows]);
    
    // Process each file individually using direct processing
    for (const file of files) {
      try {
        // Update status to processing
        setRows(prevRows => {
          return prevRows.map(row => {
            if (row.filename === file.name) {
              return { ...row, status: 'processing' };
            }
            return row;
          });
        });
        
        console.log(`🚀 Processing file: ${file.name} with mode: ${processingMode}`);
        
        // Process document directly with progress tracking
        const result = await processDocument(file, processingMode, splittingStrategy, (progress) => {
          console.log('🔄 Progress callback received:', progress);
          console.log('🔄 Setting progressInfo to:', progress);
          setProgressInfo(progress);
          console.log('🔄 ProgressInfo state should now be:', progress);
        });
        console.log('✅ Processing completed:', result);
        
        // Save job ID to localStorage for resume capability
        if (result && result.jobId) {
          const recentJobIds = JSON.parse(localStorage.getItem('recentJobIds') || '[]');
          recentJobIds.push(result.jobId);
          // Keep only last 5 job IDs
          const updatedJobIds = recentJobIds.slice(-5);
          localStorage.setItem('recentJobIds', JSON.stringify(updatedJobIds));
          console.log('💾 Saved job ID to localStorage:', result.jobId);
        }
        
        // Clear progress info
        setProgressInfo(null);
        
        // Convert results to prediction rows
        // Handle different response formats: some return {data: ...}, others return the data directly
        if (!result) {
          throw new Error('Processing completed but no result was returned');
        }
        
        const resultData = result.data || result;
        console.log('🔍 Processing result data:', resultData);
        const results = convertJobResultToPredictionRows(resultData, processingMode);
        console.log('🔍 DEBUG: Converted results length:', results.length);
        if (results.length > 0) {
          setRows(prevRows => {
            // For page_by_page, we need to handle multiple rows per file
            if (processingMode === 'page_by_page') {
              // Remove the old row for this file and add all new rows
              const filteredRows = prevRows.filter(row => row.filename !== file.name);
              return [...filteredRows, ...results];
            } else {
              // For other modes, replace the single row
              return prevRows.map(row => {
                if (row.filename === file.name) {
                  return results[0];
                }
                return row;
              });
            }
          });
        }
        
      } catch (error) {
        console.error('Error processing file:', file.name, error);
        // Clear progress info on error
        setProgressInfo(null);
        
        // Extract error message from response if available
        let errorMessage = String(error);
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        
        // Check if it's a model-related error
        const isModelError = errorMessage.includes('model') && (
          errorMessage.includes('not found') || 
          errorMessage.includes('404') || 
          errorMessage.includes('not_found') ||
          errorMessage.includes('Invalid model')
        );
        
        // Check if it's an API error response
        if (errorMessage.includes('LLM processing failed') || errorMessage.includes('API key')) {
          errorMessage = `API Error: ${errorMessage}. Please check your API key configuration.`;
        } else if (isModelError) {
          errorMessage = `Model Error: ${errorMessage}. The model may not be available. Try updating the model name.`;
        }
        
        // Update the row with error status
        setRows(prevRows => {
          return prevRows.map(row => {
            if (row.filename === file.name) {
              const updatedRow = { ...row, status: 'error', explanation: errorMessage };
              // If it's a model error, show the model manager
              if (isModelError) {
                setTimeout(() => setShowModelManager(true), 500);
              }
              return updatedRow;
            }
            return row;
          });
        });
      }
    }
    
    setIsRunning(false);
  };

  const handleProcessingModeChange = (mode: ProcessingMode) => {
    console.log(`🔄 Processing mode changed to: ${mode}`);
    setProcessingMode(mode);
  };

  const handleSplittingStrategyChange = (strategy: SplittingStrategy) => {
    console.log(`🔄 Splitting strategy changed to: ${strategy}`);
    setSplittingStrategy(strategy);
  };

  const downloadCSV = () => {
    const csv = rowsToCSV(rows);
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearAllResults = () => {
    setRows([]);
  };

  const handleResume = async () => {
    if (!resumeJobId) return;
    
    setIsRunning(true);
    setShowResumeOption(false);
    
    try {
      // Resume processing by polling the job status
      const maxAttempts = 720; // 60 minutes
      let attempts = 0;
      
      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;
        
        const statusResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-api-1081023230228.us-central1.run.app'}/process-status/${resumeJobId}`);
        
        if (!statusResponse.ok) {
          throw new Error(`Failed to check job status: ${statusResponse.status}`);
        }
        
        const status = await statusResponse.json();
        
        if (status.status === 'processing' && status.progress) {
          setProgressInfo(status.progress);
        }
        
        if (status.status === 'completed') {
          const result = status.result;
          const formattedResult = {
            filename: result.filename || 'resumed_document.pdf',
            total_pages: result.total_pages,
            pages_with_reservations: result.pages_with_reservations,
            reservation_pages: result.reservation_pages || [],
            page_results: result.results || result.page_results || [],
            processing_method: result.processing_method || 'page_by_page',
            has_reservation: result.pages_with_reservations > 0,
            confidence: result.pages_with_reservations > 0 ? 1.0 : 0.0,
            reasoning: `Found mineral rights reservations on ${result.pages_with_reservations} pages: ${(result.reservation_pages || []).join(', ')}`
          };
          
          const results = convertJobResultToPredictionRows(formattedResult, 'page_by_page');
          setRows(prevRows => [...prevRows, ...results]);
          setProgressInfo(null);
          break;
        } else if (status.status === 'error') {
          throw new Error(`Job failed: ${status.error}`);
        }
      }
    } catch (error) {
      console.error('Error resuming job:', error);
    } finally {
      setIsRunning(false);
    }
  };

  const dismissResume = () => {
    setShowResumeOption(false);
    setResumeJobId(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-indigo-900 border-b border-indigo-800 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold text-white">Mineral Rights Classifier</h1>
              </div>
            </div>
            <div className="text-sm text-indigo-200">
              Smart Document Analysis
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          {/* Hero Section */}
          <div className="px-8 py-8 border-b border-gray-200">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Mineral Rights Document Analysis
              </h2>
              <p className="text-lg text-gray-600 max-w-3xl mx-auto">
               AI-powered document intelligence to streamline the analysis of mineral rights documents, delivering enterprise-level accuracy, speed, and reliability.
              </p>
            </div>
          </div>

          <div className="p-8">
            {/* System Architecture */}
            <div className="mb-8 p-6 bg-indigo-50 border border-indigo-200 rounded-lg">
              <div className="flex items-center mb-4">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                </div>
                <div className="ml-3">
                  <h3 className="text-lg font-semibold text-gray-900">System Architecture</h3>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="text-center p-3 bg-white rounded border">
                  <div className="text-sm font-medium text-gray-900">Frontend</div>
                  <div className="text-xs text-gray-600">Next.js on Vercel</div>
                </div>
                <div className="text-center p-3 bg-white rounded border">
                  <div className="text-sm font-medium text-gray-900">Backend</div>
                  <div className="text-xs text-gray-600">Python FastAPI on Google Cloud</div>
                </div>
                <div className="text-center p-3 bg-white rounded border">
                  <div className="text-sm font-medium text-gray-900">AI Engine</div>
                  <div className="text-xs text-gray-600">Anthropic Claude API</div>
                </div>
              </div>
              <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-lg">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <svg className="w-5 h-5 text-amber-600 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h4 className="text-sm font-medium text-amber-800">Processing Guidelines</h4>
                    <div className="mt-2 text-sm text-amber-700">
                      <ul className="space-y-1">
                        <li>• Maintain stable internet connection during processing</li>
                        <li>• Keep your browser open and device powered on</li>
                        <li>• Large documents (100+ pages) may require 30-60 minutes</li>
                        <li>• Avoid refreshing the page to prevent processing interruption</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <ProcessingModeSelector
              processingMode={processingMode}
              splittingStrategy={splittingStrategy}
              onProcessingModeChange={setProcessingMode}
              onSplittingStrategyChange={setSplittingStrategy}
            />

            <PDFUpload onSelect={handleFiles} />

            {/* Resume Option */}
            {showResumeOption && resumeJobId && (
              <div className="mt-6 p-6 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-blue-800">Resume Processing</h3>
                    <p className="text-blue-600">Found a previous processing job that can be resumed.</p>
                    <p className="text-sm text-blue-500">Job ID: {resumeJobId}</p>
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={handleResume}
                      className="bg-indigo-700 text-white px-6 py-2 rounded-lg hover:bg-indigo-800 transition-colors font-medium"
                    >
                      Resume
                    </button>
                    <button
                      onClick={dismissResume}
                      className="bg-gray-500 text-white px-6 py-2 rounded-lg hover:bg-gray-600 transition-colors font-medium"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            )}

            {isRunning && (
              <div className="mt-6 p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-700 mr-3"></div>
                  <p className="text-indigo-800 font-medium">Processing documents...</p>
                </div>
              </div>
            )}

            {/* Progress Display */}
            <ProgressDisplay progress={progressInfo} isVisible={isRunning && processingMode === "page_by_page"} inline={true} />

            <ResultsTable rows={rows} />

            {/* Action Buttons */}
            <div className="mt-8 flex gap-4 items-center flex-wrap">
              <button
                onClick={() => setShowConsoleLog(!showConsoleLog)}
                className={`px-6 py-3 rounded-lg text-sm font-medium transition-colors ${
                  showConsoleLog 
                    ? 'bg-gray-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {showConsoleLog ? 'Hide' : 'Show'} Console Log
              </button>
              
              <button
                onClick={() => setShowApiKeyManager(true)}
                className="px-6 py-3 rounded-lg text-sm font-medium transition-colors bg-amber-100 text-amber-800 hover:bg-amber-200 border border-amber-300 flex items-center"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                </svg>
                Update API Key
              </button>
              
              <button
                onClick={() => setShowModelManager(true)}
                className="px-6 py-3 rounded-lg text-sm font-medium transition-colors bg-purple-100 text-purple-800 hover:bg-purple-200 border border-purple-300 flex items-center"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                Update Model
              </button>
              
              {rows.length > 0 && (
                <>
                  <button
                    onClick={downloadCSV}
                    className="bg-indigo-700 text-white px-6 py-3 rounded-lg hover:bg-indigo-800 transition-colors font-medium flex items-center"
                  >
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Download CSV ({rows.length} files)
                  </button>
                  <button
                    onClick={clearAllResults}
                    className="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600 transition-colors font-medium flex items-center"
                  >
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Clear All Results
                  </button>
                </>
              )}
            </div>

            {/* Console Log Display */}
            <div className="mt-6">
              <ConsoleLog isVisible={showConsoleLog} />
            </div>
          </div>
        </div>
      </main>
      
      {/* API Key Manager Modal */}
      <ApiKeyManager 
        isVisible={showApiKeyManager} 
        onClose={() => setShowApiKeyManager(false)} 
      />
      
      {/* Model Manager Modal */}
      <ModelManager 
        isVisible={showModelManager} 
        onClose={() => setShowModelManager(false)}
        onModelChange={(modelName) => {
          console.log('Model changed to:', modelName);
        }}
      />
    </div>
  );
}
