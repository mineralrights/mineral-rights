import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.BACKEND_API_URL || 'https://mineral-rights-api-1081023230228.us-central1.run.app';

// Validate environment
console.log('🔍 BACKEND_API_URL:', process.env.BACKEND_API_URL);
console.log('🔍 Using API_BASE:', API_BASE);
if (!process.env.BACKEND_API_URL) {
  console.warn('⚠️ BACKEND_API_URL not set, using default');
}

// Input validation
function validateFilename(filename: string): string {
  if (!filename || typeof filename !== 'string') {
    return 'document.pdf';
  }
  
  // Sanitize filename - remove dangerous characters
  const sanitized = filename.replace(/[^a-zA-Z0-9._-]/g, '_');
  
  // Ensure it has a valid extension
  if (!sanitized.includes('.')) {
    return `${sanitized}.pdf`;
  }
  
  return sanitized;
}

function validateContentType(contentType: string): string {
  const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/tiff'];
  if (allowedTypes.includes(contentType)) {
    return contentType;
  }
  return 'application/pdf';
}

export async function OPTIONS() {
  return new NextResponse('OK', {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    }
  });
}

async function proxySignedUrlRequest(payload: { filename: string; content_type: string }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300000); // 300s (5 minutes)
  try {
    const res = await fetch(`${API_BASE}/get-signed-upload-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
      cache: 'no-store'
    });
    
    if (!res.ok) {
      const errorText = await res.text();
      console.error(`Backend error: ${res.status} ${errorText}`);
      throw new Error(`Backend responded with ${res.status}: ${errorText}`);
    }
    
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: {
        'Content-Type': res.headers.get('content-type') || 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Expose-Headers': '*'
      },
    });
  } catch (error) {
    console.error('Proxy error:', error);
    
    // Check if this is a timeout (backend busy)
    if (error instanceof Error && error.name === 'AbortError') {
      console.error('⏰ Backend is busy (timeout) - likely processing a large PDF');
      throw new Error('Backend is busy processing a large PDF. Please try again in a few minutes.');
    }
    
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const filename = validateFilename(searchParams.get('filename') || 'document.pdf');
    const content_type = validateContentType(searchParams.get('content_type') || 'application/pdf');
    
    console.log(`🔑 GET request for signed URL: ${filename} (${content_type})`);
    return await proxySignedUrlRequest({ filename, content_type });
  } catch (e: any) {
    console.error('❌ GET /get-signed-upload-url failed:', e);
    return new NextResponse(JSON.stringify({ 
      error: e?.message || 'proxy error',
      details: e?.stack || 'No stack trace available'
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    });
  }
}

export async function POST(req: NextRequest) {
  try {
    let body: any = {};
    try {
      body = await req.json();
    } catch {
      // Fallback: attempt to read form or query params if JSON parse fails
      try {
        const form = await req.formData();
        body = {
          filename: (form.get('filename') as string) || 'document.pdf',
          content_type: (form.get('content_type') as string) || 'application/pdf'
        };
      } catch {
        const { searchParams } = new URL(req.url);
        body = {
          filename: searchParams.get('filename') || 'document.pdf',
          content_type: searchParams.get('content_type') || 'application/pdf'
        };
      }
    }
    
    const payload = {
      filename: validateFilename(body.filename || 'document.pdf'),
      content_type: validateContentType(body.content_type || 'application/pdf')
    };
    
    console.log(`🔑 Requesting signed URL for: ${payload.filename} (${payload.content_type})`);
    return await proxySignedUrlRequest(payload);
  } catch (e: any) {
    console.error('❌ POST /get-signed-upload-url failed:', e);
    return new NextResponse(JSON.stringify({ 
      error: e?.message || 'proxy error',
      details: e?.stack || 'No stack trace available'
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    });
  }
}

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

