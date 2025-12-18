/**
 * Google Cloud Storage Upload Handler
 * Handles large file uploads (up to 5TB) via GCS
 */

interface GCSUploadResponse {
  success: boolean;
  file_id: string;
  filename: string;
  file_size_mb: number;
  gcs_url: string;
  blob_name: string;
  processing_mode: string;
  splitting_strategy: string;
  message: string;
}

interface GCSProcessResponse {
  success: boolean;
  gcs_url: string;
  local_path: string;
  processing_mode: string;
  splitting_strategy: string;
  message: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-processor-1081023230228.us-central1.run.app';

export class GCSUploadHandler {
  private apiBaseUrl: string;

  constructor(apiBaseUrl: string = API_BASE_URL) {
    this.apiBaseUrl = apiBaseUrl;
  }

  /**
   * Upload a large file to Google Cloud Storage
   * Handles files up to 5TB
   */
  async uploadLargeFile(
    file: File,
    processingMode: 'single_deed' | 'multi_deed' | 'page_by_page' = 'multi_deed',
    splittingStrategy: 'document_ai' | 'simple' = 'document_ai'
  ): Promise<GCSUploadResponse> {
    console.log(`📁 Uploading large file: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('processing_mode', processingMode);
    formData.append('splitting_strategy', splittingStrategy);

    const response = await fetch(`${this.apiBaseUrl}/upload-large`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`GCS upload failed: ${response.status} ${errorText}`);
    }

    return await response.json();
  }

  /**
   * Process a file from Google Cloud Storage
   */
  async processFromGCS(
    gcsUrl: string,
    processingMode: 'single_deed' | 'multi_deed' | 'page_by_page' = 'multi_deed',
    splittingStrategy: 'document_ai' | 'simple' = 'document_ai'
  ): Promise<GCSProcessResponse> {
    console.log(`🔍 Processing file from GCS: ${gcsUrl}`);
    
    const formData = new FormData();
    formData.append('gcs_url', gcsUrl);
    formData.append('processing_mode', processingMode);
    formData.append('splitting_strategy', splittingStrategy);

    const response = await fetch(`${this.apiBaseUrl}/process-from-gcs`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`GCS processing failed: ${response.status} ${errorText}`);
    }

    return await response.json();
  }

  /**
   * Complete workflow: Upload to GCS then process
   */
  async uploadAndProcess(
    file: File,
    processingMode: 'single_deed' | 'multi_deed' | 'page_by_page' = 'multi_deed',
    splittingStrategy: 'document_ai' | 'simple' = 'document_ai'
  ): Promise<any> {
    console.log(`🚀 Starting GCS upload and process workflow for: ${file.name}`);
    
    try {
      // Step 1: Upload to GCS
      console.log('📤 Step 1: Uploading to Google Cloud Storage...');
      const uploadResult = await this.uploadLargeFile(file, processingMode, splittingStrategy);
      
      console.log(`✅ Upload successful: ${uploadResult.gcs_url}`);
      console.log(`📊 File size: ${uploadResult.file_size_mb.toFixed(1)}MB`);
      
      // Step 2: Process from GCS
      console.log('🔍 Step 2: Processing from GCS...');
      const processResult = await this.processFromGCS(uploadResult.gcs_url, processingMode, splittingStrategy);
      
      console.log(`✅ Processing successful`);
      
      return {
        ...processResult,
        upload_info: uploadResult,
        success: true
      };
      
    } catch (error) {
      console.error('❌ GCS workflow failed:', error);
      throw error;
    }
  }

  /**
   * Check if a file is too large for direct upload
   */
  static isLargeFile(file: File, thresholdMB: number = 50): boolean {
    const fileSizeMB = file.size / (1024 * 1024);
    return fileSizeMB > thresholdMB;
  }

  /**
   * Get file size in MB
   */
  static getFileSizeMB(file: File): number {
    return file.size / (1024 * 1024);
  }
}

// Export singleton instance
export const gcsUploadHandler = new GCSUploadHandler();

// Export types
export type { GCSUploadResponse, GCSProcessResponse };
