"use client"

import { useEffect, useState } from "react"
import { Activity, Gauge, TrendingUp } from "lucide-react"

interface ResultsDisplayProps {
  results: {
    yieldStrength: number
    tensileStrength: number
    elongation: number
  }
}

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    // Trigger animation after component mounts
    const timer = setTimeout(() => setIsVisible(true), 100)
    return () => clearTimeout(timer)
  }, [results])

  const resultCards = [
    {
      title: "Yield Strength",
      value: results.yieldStrength,
      unit: "MPa",
      icon: Gauge,
      color: "primary",
      description: "Stress at which material begins to deform plastically",
    },
    {
      title: "Tensile Strength",
      value: results.tensileStrength,
      unit: "MPa",
      icon: TrendingUp,
      color: "secondary",
      description: "Maximum stress material can withstand while being stretched",
    },
    {
      title: "Elongation",
      value: results.elongation,
      unit: "%",
      icon: Activity,
      color: "primary",
      description: "Measure of ductility - how much material can stretch",
    },
  ]

  return (
    <div className="mt-12">
      <h3 className="text-2xl font-bold mb-6 text-center">Prediction Results</h3>

      <div className="grid md:grid-cols-3 gap-6">
        {resultCards.map((card, index) => {
          const Icon = card.icon
          return (
            <div
              key={card.title}
              className={`bg-card border-2 border-border rounded-xl p-6 transition-all duration-500 ${
                isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-5"
              }`}
              style={{ transitionDelay: `${index * 150}ms` }}
            >
              <div className="flex items-start justify-between mb-4">
                <div
                  className={`w-12 h-12 rounded-full bg-${card.color}/10 flex items-center justify-center animate-pulse-glow`}
                >
                  <Icon className={`w-6 h-6 text-${card.color}`} />
                </div>
              </div>

              <h4 className="text-sm font-medium text-muted-foreground mb-2">{card.title}</h4>

              <div className="flex items-baseline gap-2 mb-3">
                <span className="text-4xl font-bold">{card.value?.toFixed(2)}</span>
                <span className="text-xl text-muted-foreground">{card.unit}</span>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">{card.description}</p>

              {/* Animated progress bar */}
              <div className="mt-4 h-1 bg-muted rounded-full overflow-hidden">
                <div
                  className={`h-full bg-${card.color} transition-all duration-1000 ease-out`}
                  style={{
                    width: isVisible
                      ? `${Math.min((card.value / (card.unit === "%" ? 100 : 2000)) * 100, 100)}%`
                      : "0%",
                    transitionDelay: `${index * 150 + 300}ms`,
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div className="mt-8 p-6 bg-muted/50 border border-border rounded-xl">
        <h4 className="font-semibold mb-2 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Analysis Summary
        </h4>
        <p className="text-sm text-muted-foreground leading-relaxed">
          The predicted material properties indicate{" "}
          {results.yieldStrength > 400 ? "high" : results.yieldStrength > 200 ? "moderate" : "low"} yield strength and{" "}
          {results.elongation > 20 ? "excellent" : results.elongation > 10 ? "good" : "limited"} ductility. These values
          suggest the material is suitable for{" "}
          {results.tensileStrength > 800
            ? "high-stress structural applications"
            : results.tensileStrength > 400
              ? "general engineering applications"
              : "lightweight component applications"}
          .
        </p>
      </div>
    </div>
  )
}
