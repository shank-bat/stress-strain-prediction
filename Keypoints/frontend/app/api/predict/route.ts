import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { material, model, ...inputData } = body

    // Call your Python FastAPI backend
    const backendResponse = await fetch("http://localhost:8000/predict?material=" + material + "&model=" + model, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputData),
    })

    const result = await backendResponse.json()

    return NextResponse.json(result)
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
