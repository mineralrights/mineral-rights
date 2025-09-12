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
      console.log(`🚀 Starting async job processing: ${file.name}`);
      
      // Create the job
      const jobResponse = await createJob(file, processingMode, splittingStrategy);
      console.log('📋 Job created:', jobResponse);

      // Start polling for job status
      const finalStatus = await pollJobUntilComplete(
        jobResponse.job_id,
        (status) => {
          console.log('📊 Job progress:', status);
          setCurrentJob(status);
          setProgress(status.progress);
          setLogs(status.logs || []);
        },
        2000 // Poll every 2 seconds
      );

      console.log('✅ Job completed:', finalStatus);

      // Process the results
      if (finalStatus.result) {
        const results = convertJobResultToPredictionRows(finalStatus.result || {}, processingMode);
        onResults(results);
      } else {
        throw new Error('No results returned from job');
      }

    } catch (error) {
      console.error('❌ Job processing failed:', error);
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
      // Multi-deed result
      if (result.deed_results && Array.isArray(result.deed_results)) {
        result.deed_results.forEach((deed: any, index: number) => {
          const row: PredictionRow = {
            filename: `deed-${deed.deed_number}.pdf`,
            status: 'done',
            prediction: deed.prediction,
            confidence: deed.confidence || 0,
            explanation: deed.explanation || '',
            processingMode: 'multi_deed',
            deedResults: [deed]
          };
          rows.push(row);
        });
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