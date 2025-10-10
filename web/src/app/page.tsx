"use client";

import PDFUpload from "@/components/PDFUpload";
import ProcessingModeSelector from "@/components/ProcessingModeSelector";
import ResultsTable from "@/components/ResultsTable";
import ProgressDisplay from "@/components/ProgressDisplay";
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
        if (result.jobId) {
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
        const results = convertJobResultToPredictionRows(result.data, processingMode);
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
        // Update the row with error status
        setRows(prevRows => {
          return prevRows.map(row => {
            if (row.filename === file.name) {
              return { ...row, status: 'error', explanation: String(error) };
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
    <main className="flex justify-center py-16 px-4">
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-lg p-10">
        <h1 className="text-4xl font-semibold mb-10 text-[color:var(--accent)]">
          Mineral-Rights&nbsp;Classifier
        </h1>
        
        {/* Technical Architecture Note */}
        <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h2 className="text-lg font-semibold text-blue-800 mb-3">🔧 Technical Architecture</h2>
          <div className="text-sm text-blue-700 space-y-2">
            <p><strong>Frontend:</strong> Next.js on Vercel → <strong>Backend:</strong> Python FastAPI on Google Cloud Run → <strong>AI:</strong> Anthropic Claude API</p>
            <p><strong>Processing:</strong> Large PDFs are processed page-by-page with real-time progress tracking and background threading.</p>
            <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded">
              <p className="text-yellow-800 font-medium mb-2">⚠️ Important: Long PDF Processing Scenarios</p>
              <ul className="text-xs text-yellow-700 space-y-1">
                <li>• <strong>Page refresh/computer off:</strong> Job continues in background, but you lose connection</li>
                <li>• <strong>Internet disconnection:</strong> Job continues, but you can't see progress</li>
                <li>• <strong>Cloud Run restart (after 15+ min):</strong> Job stops completely, need to restart</li>
                <li>• <strong>Very large PDFs (100+ pages):</strong> May take 30-60+ minutes to process</li>
              </ul>
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
          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-blue-800">Resume Processing</h3>
                <p className="text-blue-600">Found a previous processing job that can be resumed.</p>
                <p className="text-sm text-blue-500">Job ID: {resumeJobId}</p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleResume}
                  className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                >
                  Resume
                </button>
                <button
                  onClick={dismissResume}
                  className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        )}

        {isRunning && (
          <p className="mt-4 text-[color:var(--accent)] animate-pulse">
            Processing…
          </p>
        )}

        {/* Progress Display - positioned in the middle of the screen below Processing text */}
        <ProgressDisplay progress={progressInfo} isVisible={isRunning && processingMode === "page_by_page"} inline={true} />

        <ResultsTable rows={rows} />

        {rows.length > 0 && (
          <div className="mt-6 flex gap-4">
            <button
              onClick={downloadCSV}
              className="bg-[color:var(--accent)] text-white px-4 py-2 rounded hover:brightness-110"
            >
              Download CSV ({rows.length} files)
            </button>
            <button
              onClick={clearAllResults}
              className="bg-gray-500 text-white px-4 py-2 rounded hover:brightness-110"
            >
              Clear All Results
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
