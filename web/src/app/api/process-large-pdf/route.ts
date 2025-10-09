import { NextRequest, NextResponse } from 'next/server';

const API_BASE = 'https://mineral-rights-processor-1081023230228.us-central1.run.app';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const res = await fetch(`${API_BASE}/process-large-pdf`, {
      method: 'POST',
      body: formData,
    });
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: { 'Content-Type': res.headers.get('content-type') || 'application/json' },
    });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || 'proxy error' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';

