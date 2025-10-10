import { NextRequest, NextResponse } from 'next/server';

const API_BASE = 'https://mineral-rights-api-1081023230228.us-central1.run.app';

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

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const controller = new AbortController();
    // Allow long processing but fail fast on initial connectivity issues
    const timeout = setTimeout(() => controller.abort(), 45000);
    const res = await fetch(`${API_BASE}/process-large-pdf`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      cache: 'no-store'
    });
    clearTimeout(timeout);
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: {
        'Content-Type': res.headers.get('content-type') || 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Expose-Headers': '*'
      },
    });
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

