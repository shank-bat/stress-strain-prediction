"use client"

import { Brain, GitBranch, Zap } from "lucide-react"

interface ModelSelectionProps {
  onSelect: (model: "neural-network" | "random-forest" | "xgboost") => void
}

export function ModelSelection({ onSelect }: ModelSelectionProps) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center">
      <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 text-balance">Select Prediction Model</h2>
      <p className="text-lg text-muted-foreground text-center mb-12 max-w-xl text-pretty">
        Choose the machine learning algorithm for prediction
      </p>

      <div className="grid md:grid-cols-3 gap-6 max-w-6xl w-full px-4">
        <button
          onClick={() => onSelect("neural-network")}
          className="group relative p-6 bg-card border-2 border-border rounded-xl hover:border-primary transition-all duration-300 hover:shadow-xl hover:shadow-primary/20 hover:scale-105"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-primary/10 ring-1 ring-primary/25 shadow-lg shadow-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors animate-pulse-glow">
              <Brain className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-bold">Neural Network</h3>
            <p className="text-muted-foreground text-center text-sm">Deep learning model with multiple hidden layers</p>
          </div>
          <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-primary/0 to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>

        <button
          onClick={() => onSelect("random-forest")}
          className="group relative p-6 bg-card border-2 border-border rounded-xl hover:border-primary transition-all duration-300 hover:shadow-xl hover:shadow-primary/20 hover:scale-105"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-primary/10 ring-1 ring-primary/25 shadow-lg shadow-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors animate-pulse-glow">
              <GitBranch className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-bold">Random Forest</h3>
            <p className="text-muted-foreground text-center text-sm">
              Ensemble of branched learners for robust predictions
            </p>
          </div>
          <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-primary/0 to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>

        <button
          onClick={() => onSelect("xgboost")}
          className="group relative p-6 bg-card border-2 border-border rounded-xl hover:border-primary transition-all duration-300 hover:shadow-xl hover:shadow-primary/20 hover:scale-105"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-primary/10 ring-1 ring-primary/25 shadow-lg shadow-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors animate-pulse-glow">
              <Zap className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-bold">XGBoost</h3>
            <p className="text-muted-foreground text-center text-sm">
              Gradient boosting framework for optimal performance
            </p>
          </div>
          <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-primary/0 to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>
      </div>
    </div>
  )
}
