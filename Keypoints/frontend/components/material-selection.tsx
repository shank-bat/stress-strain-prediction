"use client"

import { Atom, Cog } from "lucide-react"

interface MaterialSelectionProps {
  onSelect: (material: "aluminium" | "steel") => void
}

export function MaterialSelection({ onSelect }: MaterialSelectionProps) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center">
      <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 text-balance">Select Material</h2>
      <p className="text-lg text-muted-foreground text-center mb-12 max-w-xl text-pretty">
        Choose the material type for strength prediction
      </p>

      <div className="grid md:grid-cols-2 gap-8 max-w-4xl w-full px-4">
        <button
          onClick={() => onSelect("aluminium")}
          className="group relative p-8 bg-card border-2 border-border rounded-xl hover:border-primary transition-all duration-300 hover:shadow-xl hover:shadow-primary/20 hover:scale-105"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
              <Atom className="w-10 h-10 text-primary" />
            </div>
            <h3 className="text-2xl font-bold">Aluminium</h3>
            <p className="text-muted-foreground text-center text-sm">
              Lightweight metal alloy with excellent corrosion resistance
            </p>
          </div>
          <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-primary/0 to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>

        <button
          onClick={() => onSelect("steel")}
          className="group relative p-8 bg-card border-2 border-border rounded-xl hover:border-secondary transition-all duration-300 hover:shadow-xl hover:shadow-secondary/20 hover:scale-105"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="w-20 h-20 rounded-full bg-secondary/10 flex items-center justify-center group-hover:bg-secondary/20 transition-colors">
              <Cog className="w-10 h-10 text-secondary" />
            </div>
            <h3 className="text-2xl font-bold">Steel</h3>
            <p className="text-muted-foreground text-center text-sm">
              High-strength iron alloy with superior structural properties
            </p>
          </div>
          <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-secondary/0 to-secondary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>
      </div>
    </div>
  )
}
