/**
 * Frontend Integration for Long-Running Jobs
 * ==========================================
 * 
 * This file shows how to update your frontend to use the job system
 * for long-running processing (8+ hours) instead of the regular /predict endpoint.
 */

// Configuration
const API_BASE_URL = 'https://your-render-app.onrender.com'; // Update this with your actual URL
const JOB_POLL_INTERVAL = 5000; // Poll every 5 seconds
const MAX_POLL_ATTEMPTS = 100; // Maximum polling attempts (about 8 minutes)

/**
 * Create a long-running job for documents that might take 8+ hours
 */
async function createLongRunningJob(file, processingMode = 'multi_deed', strategy = 'smart_detection') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('processing_mode', processingMode);
    formData.append('splitting_strategy', strategy);
    
    try {
        const response = await fetch(`${API_BASE_URL}/jobs/create`, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json',
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error creating job:', error);
        throw error;
    }
}

/**
 * Monitor job progress with polling
 */
async function monitorJobProgress(jobId, onProgress, onComplete, onError) {
    let attempts = 0;
    
    const pollJob = async () => {
        try {
            attempts++;
            
            const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/status`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const status = await response.json();
            
            // Call progress callback
            if (onProgress) {
                onProgress(status);
            }
            
            // Check if job is complete
            if (status.status === 'completed') {
                // Get the result
                const resultResponse = await fetch(`${API_BASE_URL}/jobs/${jobId}/result`);
                if (!resultResponse.ok) {
                    throw new Error(`Failed to get result: ${resultResponse.status}`);
                }
                
                const result = await resultResponse.json();
                if (onComplete) {
                    onComplete(result);
                }
                return;
            }
            
            // Check if job failed
            if (status.status === 'failed') {
                if (onError) {
                    onError(new Error(`Job failed: ${status.error || 'Unknown error'}`));
                }
                return;
            }
            
            // Check if we've exceeded max attempts
            if (attempts >= MAX_POLL_ATTEMPTS) {
                if (onError) {
                    onError(new Error('Job monitoring timeout - job may still be running'));
                }
                return;
            }
            
            // Continue polling
            setTimeout(pollJob, JOB_POLL_INTERVAL);
            
        } catch (error) {
            console.error('Error monitoring job:', error);
            if (onError) {
                onError(error);
            }
        }
    };
    
    // Start polling
    pollJob();
}

/**
 * Stream job logs for real-time updates
 */
function streamJobLogs(jobId, onLog, onComplete, onError) {
    const eventSource = new EventSource(`${API_BASE_URL}/jobs/${jobId}/logs/stream`);
    
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            if (data.final) {
                eventSource.close();
                if (onComplete) {
                    onComplete(data);
                }
            } else if (data.log) {
                if (onLog) {
                    onLog(data.log);
                }
            }
        } catch (error) {
            console.error('Error parsing log data:', error);
        }
    };
    
    eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        eventSource.close();
        if (onError) {
            onError(error);
        }
    };
    
    return eventSource;
}

/**
 * Enhanced file processing function that automatically chooses between
 * quick processing and long-running jobs based on file size
 */
async function processDocument(file, options = {}) {
    const {
        processingMode = 'multi_deed',
        strategy = 'smart_detection',
        maxQuickSize = 10 * 1024 * 1024, // 10MB threshold
        onProgress = null,
        onLog = null,
        onComplete = null,
        onError = null
    } = options;
    
    // Determine if we should use quick processing or long-running job
    const useQuickProcessing = file.size <= maxQuickSize;
    
    if (useQuickProcessing) {
        // Use the existing /predict endpoint for quick processing
        return await processDocumentQuick(file, processingMode, strategy, onComplete, onError);
    } else {
        // Use the new job system for long-running processing
        return await processDocumentLong(file, processingMode, strategy, onProgress, onLog, onComplete, onError);
    }
}

/**
 * Quick processing using the existing /predict endpoint
 */
async function processDocumentQuick(file, processingMode, strategy, onComplete, onError) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('processing_mode', processingMode);
    formData.append('splitting_strategy', strategy);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        if (onComplete) {
            onComplete(result);
        }
        return result;
    } catch (error) {
        console.error('Error in quick processing:', error);
        if (onError) {
            onError(error);
        }
        throw error;
    }
}

/**
 * Long-running processing using the job system
 */
async function processDocumentLong(file, processingMode, strategy, onProgress, onLog, onComplete, onError) {
    try {
        // Create the job
        const { job_id } = await createLongRunningJob(file, processingMode, strategy);
        
        // Set up log streaming if callback provided
        let logStream = null;
        if (onLog) {
            logStream = streamJobLogs(job_id, onLog, null, onError);
        }
        
        // Monitor job progress
        await monitorJobProgress(
            job_id,
            onProgress,
            (result) => {
                if (logStream) {
                    logStream.close();
                }
                if (onComplete) {
                    onComplete(result);
                }
            },
            (error) => {
                if (logStream) {
                    logStream.close();
                }
                if (onError) {
                    onError(error);
                }
            }
        );
        
    } catch (error) {
        console.error('Error in long processing:', error);
        if (onError) {
            onError(error);
        }
        throw error;
    }
}

/**
 * Test the job system
 */
async function testJobSystem() {
    try {
        const response = await fetch(`${API_BASE_URL}/test-jobs`);
        const result = await response.json();
        
        console.log('Job system test result:', result);
        return result;
    } catch (error) {
        console.error('Job system test failed:', error);
        throw error;
    }
}

/**
 * Check API health including job system
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const result = await response.json();
        
        console.log('API health check:', result);
        return result;
    } catch (error) {
        console.error('API health check failed:', error);
        throw error;
    }
}

// Export functions for use in your application
window.MineralRightsJobs = {
    createLongRunningJob,
    monitorJobProgress,
    streamJobLogs,
    processDocument,
    processDocumentQuick,
    processDocumentLong,
    testJobSystem,
    checkAPIHealth
};

// Example usage:
/*
// Test the job system
MineralRightsJobs.testJobSystem().then(result => {
    console.log('Job system status:', result);
});

// Process a document with automatic job selection
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
        await MineralRightsJobs.processDocument(file, {
            processingMode: 'multi_deed',
            strategy: 'smart_detection',
            onProgress: (status) => {
                console.log('Job progress:', status);
                // Update UI with progress
            },
            onLog: (log) => {
                console.log('Job log:', log);
                // Add log to UI
            },
            onComplete: (result) => {
                console.log('Processing completed:', result);
                // Handle completion
            },
            onError: (error) => {
                console.error('Processing error:', error);
                // Handle error
            }
        });
    } catch (error) {
        console.error('Failed to process document:', error);
    }
});
*/
