"use client"

import { useState } from "react"
import { AnimatedBackground } from "@/components/animated-background"
import { MaterialSelection } from "@/components/material-selection"
import { ModelSelection } from "@/components/model-selection"
import { InputForm } from "@/components/input-form"
import { ResultsDisplay } from "@/components/results-display"

export default function Home() {
  const [step, setStep] = useState<"landing" | "material" | "model" | "input">("landing")
  const [material, setMaterial] = useState<"aluminium" | "steel" | null>(null)
  const [model, setModel] = useState<"neural-network" | "random-forest" | "xgboost" | null>(null)
  const [results, setResults] = useState<{
    yieldStrength: number
    tensileStrength: number
    elongation: number
  } | null>(null)

  const handleMaterialSelect = (selectedMaterial: "aluminium" | "steel") => {
    setMaterial(selectedMaterial)
    setStep("model")
  }

  const handleModelSelect = (selectedModel: "neural-network" | "random-forest" | "xgboost") => {
    setModel(selectedModel)
    setStep("input")
  }

  const handlePredict = async (formData: Record<string, any>) => {
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        material,  // "steel" | "aluminium"
        model,     // "neural-network" | "random-forest" | "xgboost"
        data: formData,
      }),
    })

    const data = await response.json()
    if (response.ok) {
      const preds = data.predictions
      setResults({
        yieldStrength: preds["Yield Strength"],
        tensileStrength: preds["Tensile Strength"],
        elongation: preds["Elongation"],
      })
    } else {
      alert(data.error || "Prediction failed.")
    }
  } catch (err) {
    console.error("Prediction error:", err)
  }
}



  const handleReset = () => {
    setStep("material")
    setMaterial(null)
    setModel(null)
    setResults(null)
  }

  return (
    <main className="min-h-screen relative">
      <AnimatedBackground />

      <div className="relative z-10 container mx-auto px-4 py-8">
        {step === "landing" && (
          <div className="min-h-screen flex flex-col items-center justify-center">
            <h1 className="text-6xl md:text-8xl font-bold text-center mb-6 text-balance">
              <span className="bg-gradient-to-r from-primary via-primary to-secondary bg-clip-text text-transparent">
                Stress Prediction Tool
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground text-center mb-12 max-w-2xl text-pretty">
              Advanced material strength prediction using machine learning
            </p>
            <button
              onClick={() => setStep("material")}
              className="px-8 py-4 bg-primary text-primary-foreground rounded-lg text-lg font-semibold hover:bg-primary/90 transition-all hover:shadow-lg hover:shadow-primary/50 hover:scale-105"
            >
              Get Started
            </button>
          </div>
        )}

        {step === "material" && <MaterialSelection onSelect={handleMaterialSelect} />}

        {step === "model" && material && <ModelSelection onSelect={handleModelSelect} />}

        {step === "input" && material && model && (
          <div className="max-w-4xl mx-auto">
            <InputForm material={material} model={model} onPredict={handlePredict} onReset={handleReset} />
            {results && <ResultsDisplay results={results} />}
          </div>
        )}
      </div>
    </main>
  )
}
