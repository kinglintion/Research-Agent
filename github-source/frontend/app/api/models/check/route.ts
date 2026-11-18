import { NextRequest, NextResponse } from "next/server";

const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8765";

export async function POST(request: NextRequest) {
  try {
    const payload = await request.json();
    const response = await fetch(`${backendUrl}/api/v1/diagnostics/models`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    const body = await response.json();
    return NextResponse.json(body, { status: response.status });
  } catch {
    return NextResponse.json(
      { status: "unavailable", components: {}, error: "无法连接后端服务" },
      { status: 503 },
    );
  }
}
