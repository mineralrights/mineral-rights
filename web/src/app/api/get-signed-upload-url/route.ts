import { NextRequest, NextResponse } from 'next/server';

const API_BASE = 'https://mineral-rights-processor-1081023230228.us-central1.run.app';

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
  const timeout = setTimeout(() => controller.abort(), 120000); // 120s
  try {
    const res = await fetch(`${API_BASE}/get-signed-upload-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
      cache: 'no-store'
    });
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: {
        'Content-Type': res.headers.get('content-type') || 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Expose-Headers': '*'
      },
    });
  } finally {
    clearTimeout(timeout);
  }
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const filename = searchParams.get('filename') || 'document.pdf';
    const content_type = searchParams.get('content_type') || 'application/pdf';
    return await proxySignedUrlRequest({ filename, content_type });
  } catch (e: any) {
    return new NextResponse(JSON.stringify({ error: e?.message || 'proxy error' }), {
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
      filename: body.filename || 'document.pdf',
      content_type: body.content_type || 'application/pdf'
    };
    return await proxySignedUrlRequest(payload);
  } catch (e: any) {
    return new NextResponse(JSON.stringify({ error: e?.message || 'proxy error' }), {
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

