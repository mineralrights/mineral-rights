import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.BACKEND_API_URL || 'https://mineral-rights-api-1081023230228.us-central1.run.app';

export async function POST(req: NextRequest) {
  try {
    console.log('🔧 Processing large PDF pages via proxy...');
    
    // Get the form data from the request
    const formData = await req.formData();
    const gcs_url = formData.get('gcs_url') as string;
    
    if (!gcs_url) {
      return new NextResponse(JSON.stringify({ 
        error: 'Missing gcs_url parameter'
      }), {
        status: 400,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      });
    }
    
    console.log(`🔧 Proxying to backend: ${gcs_url}`);
    
    // Proxy the request to the backend
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3600000); // 1 hour timeout for processing
    
    try {
      const res = await fetch(`${API_BASE}/process-large-pdf-pages`, {
        method: 'POST',
        body: formData,
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
  } catch (e: any) {
    console.error('❌ POST /process-large-pdf-pages failed:', e);
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

export async function OPTIONS(req: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    }
  });
}
