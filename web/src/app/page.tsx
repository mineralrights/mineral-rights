"use client";

import PDFUpload from "@/components/PDFUpload";
import ProcessingModeSelector from "@/components/ProcessingModeSelector";
import ResultsTable from "@/components/ResultsTable";
import ProgressDisplay from "@/components/ProgressDisplay";
import { processDocument } from "@/lib/api_async";
import { rowsToCSV } from "@/lib/csv";
import { useState, useEffect } from "react";
import { PredictionRow, ProcessingMode, SplittingStrategy } from "@/lib/types";

export default function Home() {
  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>("single_deed");
  const [splittingStrategy, setSplittingStrategy] = useState<SplittingStrategy>("document_ai");
  const [progressInfo, setProgressInfo] = useState<any>(null);

  // Debug progressInfo state changes
  useEffect(() => {
    console.log('🔄 progressInfo state changed:', progressInfo);
  }, [progressInfo]);

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
      // Page-by-page result
      if (result.results && Array.isArray(result.results)) {
        result.results.forEach((page: any, index: number) => {
          const row: PredictionRow = {
            filename: `page-${page.page_number || index + 1}.pdf`,
            status: 'done',
            prediction: page.has_reservations ? 'has_reservation' : 'no_reservation',
            confidence: page.confidence || 0,
            explanation: page.reasoning || page.explanation || '',
            processingMode: 'page_by_page',
            pageResults: [page]
          };
          rows.push(row);
        });
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
        
        // Clear progress info
        setProgressInfo(null);
        
        // Convert results to prediction rows
        const results = convertJobResultToPredictionRows(result, processingMode);
        if (results.length > 0) {
          setRows(prevRows => {
            return prevRows.map(row => {
              if (row.filename === file.name) {
                return results[0];
              }
              return row;
            });
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

  return (
    <main className="flex justify-center py-16 px-4">
      {/* Progress Display */}
      <ProgressDisplay progress={progressInfo} isVisible={isRunning && processingMode === "page_by_page"} />
      
      {/* Debug: Show when ProgressDisplay should be visible */}
      {isRunning && processingMode === "page_by_page" && (
        <div className="fixed top-4 right-4 bg-red-500 text-white p-2 rounded z-50">
          DEBUG: ProgressDisplay should be visible! progressInfo: {progressInfo ? 'SET' : 'NULL'}
        </div>
      )}
      
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-lg p-10">
        <h1 className="text-4xl font-semibold mb-10 text-[color:var(--accent)]">
          Mineral-Rights&nbsp;Classifier
        </h1>

        <ProcessingModeSelector
          processingMode={processingMode}
          splittingStrategy={splittingStrategy}
          onProcessingModeChange={setProcessingMode}
          onSplittingStrategyChange={setSplittingStrategy}
        />

        <PDFUpload onSelect={handleFiles} />

        {isRunning && (
          <p className="mt-4 text-[color:var(--accent)] animate-pulse">
            Processing…
          </p>
        )}

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
