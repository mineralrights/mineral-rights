/**
 * Direct Google Cloud Storage Upload
 * Uploads large files directly to GCS without going through Cloud Run
 */

interface SignedURLResponse {
  signed_url: string;
  bucket_name: string;
  blob_name: string;
  expiration: string;
}

interface GCSUploadResult {
  success: boolean;
  gcs_url: string;
  blob_name: string;
  file_size_mb: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://mineral-rights-api-1081023230228.us-central1.run.app';

export class DirectGCSUploader {
  private apiBaseUrl: string;

  constructor(apiBaseUrl: string = API_BASE_URL) {
    this.apiBaseUrl = apiBaseUrl;
  }

  /**
   * Get a signed URL for direct GCS upload
   */
  async getSignedUploadURL(
    filename: string,
    contentType: string = 'application/pdf'
  ): Promise<SignedURLResponse> {
    const response = await fetch(`${this.apiBaseUrl}/get-signed-upload-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        filename,
        content_type: contentType
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to get signed URL: ${response.status} ${errorText}`);
    }

    return await response.json();
  }

  /**
   * Upload file directly to GCS using signed URL
   */
  async uploadToGCS(
    file: File,
    signedUrl: string
  ): Promise<GCSUploadResult> {
    console.log(`📤 Uploading ${file.name} directly to GCS...`);
    
    const response = await fetch(signedUrl, {
      method: 'PUT',
      body: file,
      headers: {
        'Content-Type': file.type,
      },
    });

    if (!response.ok) {
      throw new Error(`GCS upload failed: ${response.status} ${response.statusText}`);
    }

    // Extract GCS URL from signed URL
    const gcsUrl = signedUrl.split('?')[0];
    const fileSizeMB = file.size / (1024 * 1024);

    console.log(`✅ Upload successful: ${gcsUrl}`);
    console.log(`📊 File size: ${fileSizeMB.toFixed(1)}MB`);

    return {
      success: true,
      gcs_url: gcsUrl,
      blob_name: gcsUrl.split('/').slice(-2).join('/'), // Get bucket/path
      file_size_mb: fileSizeMB
    };
  }

  /**
   * Complete workflow: Get signed URL → Upload to GCS → Process
   */
  async uploadAndProcess(
    file: File,
    processingMode: 'single_deed' | 'multi_deed' | 'page_by_page' = 'multi_deed',
    splittingStrategy: 'document_ai' | 'simple' = 'document_ai'
  ): Promise<any> {
    console.log(`🚀 Starting direct GCS upload for: ${file.name}`);
    
    try {
      // Step 1: Get signed URL
      console.log('🔑 Step 1: Getting signed upload URL...');
      const signedUrlData = await this.getSignedUploadURL(file.name, file.type);
      
      // Step 2: Upload directly to GCS
      console.log('📤 Step 2: Uploading directly to GCS...');
      const uploadResult = await this.uploadToGCS(file, signedUrlData.signed_url);
      
      // Step 3: Process from GCS
      console.log('🔍 Step 3: Processing from GCS...');
      const processFormData = new FormData();
      processFormData.append('gcs_url', uploadResult.gcs_url);
      processFormData.append('processing_mode', processingMode);
      processFormData.append('splitting_strategy', splittingStrategy);

      const processResponse = await fetch(`${this.apiBaseUrl}/process-gcs`, {
        method: 'POST',
        body: processFormData,
      });

      if (!processResponse.ok) {
        const errorText = await processResponse.text();
        throw new Error(`GCS processing failed: ${processResponse.status} ${errorText}`);
      }

      const processResult = await processResponse.json();
      console.log(`✅ Processing successful`);
      
      return {
        ...processResult,
        upload_info: uploadResult,
        success: true
      };
      
    } catch (error) {
      console.error('❌ Direct GCS workflow failed:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const directGCSUploader = new DirectGCSUploader();

// Export types
export type { SignedURLResponse, GCSUploadResult };
