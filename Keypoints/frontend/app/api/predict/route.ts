import { NextResponse } from "next/server"

export async function POST(req: Request) {
  try {
    const body = await req.json()

    // Map UI names to backend model codes
    const modelMap: Record<string, string> = {
      "neural-network": "nn",
      "random-forest": "rf",
      "xgboost": "xgb",
    }

    const payload = {
      material: body.material,   // "steel" or "aluminium"
      model: modelMap[body.model],
      data: body.data,           // composition + processing if any
    }

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })

    const json = await res.json()
    if (!res.ok) {
      return NextResponse.json({ error: json.detail || "Prediction failed" }, { status: res.status })
    }

    return NextResponse.json(json)
  } catch (err: any) {
    console.error("Proxy error:", err)
    return NextResponse.json({ error: err.message || "Server error" }, { status: 500 })
  }
}
