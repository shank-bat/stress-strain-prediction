"use client"

import type React from "react"

import { useState } from "react"
import { ArrowLeft, Play } from "lucide-react"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"

interface InputFormProps {
  material: "aluminium" | "steel"
  model: string
  onPredict: (data: Record<string, any>) => void
  onReset: () => void
}

const aluminiumElements = [
  "Ag",
  "Al",
  "B",
  "Be",
  "Bi",
  "Cd",
  "Co",
  "Cr",
  "Cu",
  "Er",
  "Eu",
  "Fe",
  "Ga",
  "Li",
  "Mg",
  "Mn",
  "Ni",
  "Pb",
  "Sc",
  "Si",
  "Sn",
  "Ti",
  "V",
  "Zn",
  "Zr",
]

const steelElements = ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"]

export function InputForm({ material, model, onPredict, onReset }: InputFormProps) {
  const [formData, setFormData] = useState<Record<string, any>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    await onPredict(formData)
    setIsSubmitting(false)
  }

  const handleInputChange = (key: string, value: string) => {
    setFormData((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="py-8">
      <div className="flex items-center gap-4 mb-8">
        <Button variant="outline" onClick={onReset} className="gap-2 bg-transparent">
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
        <div>
          <h2 className="text-3xl font-bold capitalize">
            {material} - {model.replace("-", " ")}
          </h2>
          <p className="text-muted-foreground">Enter material composition values</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="bg-card border border-border rounded-xl p-6 shadow-lg">
          {material === "aluminium" && (
            <>
              <div className="mb-6">
                <Label htmlFor="processing" className="text-base font-semibold mb-2 block">
                  Processing Method
                </Label>
                <Select onValueChange={(value) => handleInputChange("processing", value)}>
                  <SelectTrigger id="processing" className="input-glow">
                    <SelectValue placeholder="Select processing method" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cast">Cast</SelectItem>
                    <SelectItem value="wrought">Wrought</SelectItem>
                    <SelectItem value="powder">Powder Metallurgy</SelectItem>
                    <SelectItem value="additive">Additive Manufacturing</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {aluminiumElements.map((element) => (
                  <div key={element}>
                    <Label htmlFor={element} className="text-sm font-medium mb-1 block">
                      {element} (%)
                    </Label>
                    <Input
                      id={element}
                      type="number"
                      step="0.001"
                      placeholder="0.000"
                      className="input-glow"
                      onChange={(e) => handleInputChange(element, e.target.value)}
                    />
                  </div>
                ))}
              </div>
            </>
          )}

          {material === "steel" && (
            <>
              <div className="mb-6">
                <Label htmlFor="formula" className="text-base font-semibold mb-2 block">
                  Steel Formula/Grade
                </Label>
                <Input
                  id="formula"
                  type="text"
                  placeholder="e.g., AISI 1045, 304 Stainless"
                  className="input-glow"
                  onChange={(e) => handleInputChange("formula", e.target.value)}
                />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {steelElements.map((element) => (
                  <div key={element}>
                    <Label htmlFor={element} className="text-sm font-medium mb-1 block uppercase">
                      {element} (%)
                    </Label>
                    <Input
                      id={element}
                      type="number"
                      step="0.001"
                      placeholder="0.000"
                      className="input-glow"
                      onChange={(e) => handleInputChange(element, e.target.value)}
                    />
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <Button
          type="submit"
          disabled={isSubmitting}
          className="w-full py-6 text-lg font-semibold bg-primary hover:bg-primary/90 transition-all hover:shadow-lg hover:shadow-primary/50 group"
        >
          <span className="flex items-center justify-center gap-3">
            <Play className="w-5 h-5 group-hover:animate-press" />
            {isSubmitting ? "Predicting..." : "Predict Strength"}
          </span>
        </Button>
      </form>
    </div>
  )
}
