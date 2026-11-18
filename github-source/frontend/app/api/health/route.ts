import { NextResponse } from "next/server";

const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8765";

export async function GET() {
  try {
    const response = await fetch(`${backendUrl}/api/v1/health`, { cache: "no-store" });
    const body = await response.json();
    return NextResponse.json(body, { status: response.status });
  } catch {
    return NextResponse.json(
      { status: "unavailable", service: "Research Copilot API", components: {} },
      { status: 503 },
    );
  }
}
