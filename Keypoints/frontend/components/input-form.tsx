"use client"

import type React from "react"
import { useState } from "react"
import { ArrowLeft, Play } from "lucide-react"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

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

const commonAluminium = ["Al", "Mg", "Si", "Cu", "Zn", "Mn"]
const commonSteel = ["c", "mn", "si", "cr", "ni", "mo"]

export function InputForm({ material, model, onPredict, onReset }: InputFormProps) {
  const [formData, setFormData] = useState<Record<string, any>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [selectedElements, setSelectedElements] = useState<string[]>(
    material === "aluminium" ? commonAluminium : commonSteel,
  )

  const allElements = material === "aluminium" ? aluminiumElements : steelElements
  const commonSet = material === "aluminium" ? commonAluminium : commonSteel

  const toggleElement = (el: string) => {
    setSelectedElements((prev) => (prev.includes(el) ? prev.filter((e) => e !== el) : [...prev, el]))
  }

  const selectAll = () => setSelectedElements(allElements)
  const clearAll = () => setSelectedElements([])
  const selectCommon = () => setSelectedElements(commonSet)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    const payload: Record<string, number | string> = {}

    if (material === "aluminium") {
      payload.processing = processing
    }

    for (const el of allElements) {
      if (selectedElements.includes(el)) {
        const val = formData[el]
        payload[el] = val === undefined || val === "" ? 0 : Number(val)
      } else {
        payload[el] = 0
      }
    }

    await onPredict(payload)
    setIsSubmitting(false)
  }

  const handleInputChange = (key: string, value: string) => {
    setFormData((prev) => ({ ...prev, [key]: value }))
  }

  const processingOptions = [
    "No Processing",
    "Solutionised",
    "Solutionised + Artificially peak aged",
    "Solutionised + Artificially over aged",
    "Solutionised + Cold Worked + Naturally aged",
    "Solutionised + Naturally aged",
    "Strain hardened",
    "Strain Hardened (Hard)",
    "Naturally aged",
    "Artificial aged",
  ]
  const [processing, setProcessing] = useState<string>(processingOptions[0])

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
            <div className="mb-6">
              <h3 className="text-base font-semibold mb-3">Processing</h3>
              <Select value={processing} onValueChange={setProcessing}>
                <SelectTrigger className="w-full md:w-[480px] input-glow bg-background/50">
                  <SelectValue placeholder="Select processing" />
                </SelectTrigger>
                <SelectContent>
                  {processingOptions.map((opt) => (
                    <SelectItem key={opt} value={opt}>
                      {opt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="mb-6">
            <h3 className="text-base font-semibold mb-3">Select Elements Present</h3>
            <div className="flex flex-wrap gap-2 mb-4">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={selectCommon}
                className="input-glow bg-transparent"
              >
                Use Common
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={selectAll}
                className="input-glow bg-transparent"
              >
                Select All
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={clearAll}
                className="input-glow bg-transparent"
              >
                Clear
              </Button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {allElements.map((el) => {
                const id = `el-${el}`
                return (
                  <label
                    key={el}
                    htmlFor={id}
                    className="flex items-center gap-2 px-2 py-2 rounded-md border border-border bg-muted/20 hover:bg-muted/30 cursor-pointer"
                  >
                    <Checkbox
                      id={id}
                      checked={selectedElements.includes(el)}
                      onCheckedChange={() => toggleElement(el)}
                    />
                    <span className="text-sm font-medium uppercase">{el}</span>
                  </label>
                )
              })}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {selectedElements.map((element) => (
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
