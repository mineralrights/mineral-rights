'use client';

import React, { useState, useCallback } from 'react';
import { ProcessingMode, SplittingStrategy, PredictionRow, DeedResult, PageResult } from '../lib/types';
import { createJob, getJobStatus, pollJobUntilComplete, JobStatus } from '../lib/api_async';

interface AsyncJobProcessorProps {
  onResults: (results: PredictionRow[]) => void;
  onError: (error: string) => void;
}

export default function AsyncJobProcessor({ onResults, onError }: AsyncJobProcessorProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);

  const processDocument = useCallback(async (
    file: File,
    processingMode: ProcessingMode,
    splittingStrategy: SplittingStrategy
  ) => {
    if (isProcessing) return;

    setIsProcessing(true);
    setProgress(0);
    setLogs([]);
    setCurrentJob(null);

    try {
      console.log(`🚀 Starting document processing: ${file.name}`);
      
      // Show initial progress
      setLogs(['📤 Uploading PDF...', '⏳ Processing document (this may take several minutes for large PDFs)...']);
      setProgress(10);
      
      // Create the job (now uses direct processing)
      const jobResponse = await createJob(file, processingMode, splittingStrategy);
      console.log('📋 Processing started:', jobResponse);

      // Simulate progress updates for long-running requests
      const progressInterval = setInterval(() => {
        setLogs(prev => [...prev, '⏳ Still processing... (Large PDFs can take 10-30 minutes)']);
        setProgress(prev => Math.min(prev + 5, 90));
      }, 30000); // Update every 30 seconds

      // Start polling for job status
      const finalStatus = await pollJobUntilComplete(
        jobResponse.job_id,
        (status) => {
          console.log('📊 Processing progress:', status);
          setCurrentJob(status);
          setProgress(status.progress);
          setLogs(status.logs || []);
        },
        2000 // Poll every 2 seconds
      );

      clearInterval(progressInterval);
      setProgress(100);
      setLogs(prev => [...prev, '✅ Processing completed!']);

      console.log('✅ Processing completed:', finalStatus);

      // Process the results
      if (finalStatus.result) {
        const results = convertJobResultToPredictionRows(finalStatus.result || {}, processingMode);
        onResults(results);
      } else {
        throw new Error('No results returned from processing');
      }

    } catch (error) {
      console.error('❌ Processing failed:', error);
      onError(error instanceof Error ? error.message : 'Unknown error occurred');
    } finally {
      setIsProcessing(false);
      setCurrentJob(null);
    }
  }, [isProcessing, onResults, onError]);

  const convertJobResultToPredictionRows = (
    result: any,
    processingMode: ProcessingMode
  ): PredictionRow[] => {
    const rows: PredictionRow[] = [];

    if (processingMode === 'single_deed') {
      // Single deed result
      const row: PredictionRow = {
        filename: 'document.pdf',
        status: 'done',
        prediction: result.has_reservation ? 'has_reservation' : 'no_reservation',
        confidence: result.confidence || 0,
        explanation: result.reasoning || '',
        processingMode: 'single_deed'
      };
      rows.push(row);
    } else if (processingMode === 'multi_deed') {
      // Multi-deed result - create a single row with all deed results
      if (result.deed_results && Array.isArray(result.deed_results)) {
        const deedResults = result.deed_results.map((deed: any) => ({
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
        }));
        
        const deedsWithReservations = result.deed_results.filter((d: any) => d.has_reservations).length;
        const row: PredictionRow = {
          filename: result.filename || result.document_path || 'multi_deed_document.pdf',
          status: 'done',
          prediction: deedsWithReservations > 0 ? 'has_reservation' : 'no_reservation',
          confidence: result.deed_results.reduce((sum: number, d: any) => sum + (d.confidence || 0), 0) / result.deed_results.length,
          explanation: `Processed ${result.deed_results.length} deeds. ${deedsWithReservations} have reservations.`,
          processingMode: 'multi_deed',
          totalDeeds: result.deed_results.length,
          deedResults: deedResults
        };
        rows.push(row);
      }
    } else if (processingMode === 'page_by_page') {
      // Page-by-page result
      if (result.page_results && Array.isArray(result.page_results)) {
        result.page_results.forEach((page: any, index: number) => {
          const row: PredictionRow = {
            filename: `page-${page.page_number}.pdf`,
            status: 'done',
            prediction: page.has_reservations ? 'has_reservation' : 'no_reservation',
            confidence: page.confidence || 0,
            explanation: page.explanation || '',
            processingMode: 'page_by_page',
            pageResults: [page]
          };
          rows.push(row);
        });
      }
    }

    return rows;
  };

  return {
    processDocument,
    isProcessing,
    currentJob,
    progress,
    logs
  };
}

// Hook for using the async job processor
export function useAsyncJobProcessor(onResults: (results: PredictionRow[]) => void, onError: (error: string) => void) {
  return AsyncJobProcessor({ onResults, onError });
}