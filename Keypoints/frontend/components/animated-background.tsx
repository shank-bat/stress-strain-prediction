"use client"

export function AnimatedBackground() {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-background via-background to-muted opacity-90" />

      {/* Rotating gears */}
      <div className="absolute top-20 left-20 w-32 h-32 opacity-10">
        <svg className="animate-rotate-slow" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50" cy="50" r="30" stroke="currentColor" strokeWidth="2" className="text-primary" />
          <circle cx="50" cy="50" r="20" stroke="currentColor" strokeWidth="2" className="text-primary" />
          {[0, 45, 90, 135, 180, 225, 270, 315].map((angle) => (
            <line
              key={angle}
              x1="50"
              y1="50"
              x2={50 + 30 * Math.cos((angle * Math.PI) / 180)}
              y2={50 + 30 * Math.sin((angle * Math.PI) / 180)}
              stroke="currentColor"
              strokeWidth="2"
              className="text-primary"
            />
          ))}
        </svg>
      </div>

      <div className="absolute bottom-40 right-32 w-40 h-40 opacity-10">
        <svg
          className="animate-rotate-slow"
          style={{ animationDirection: "reverse" }}
          viewBox="0 0 100 100"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <circle cx="50" cy="50" r="35" stroke="currentColor" strokeWidth="2" className="text-secondary" />
          <circle cx="50" cy="50" r="25" stroke="currentColor" strokeWidth="2" className="text-secondary" />
          {[0, 60, 120, 180, 240, 300].map((angle) => (
            <line
              key={angle}
              x1="50"
              y1="50"
              x2={50 + 35 * Math.cos((angle * Math.PI) / 180)}
              y2={50 + 35 * Math.sin((angle * Math.PI) / 180)}
              stroke="currentColor"
              strokeWidth="2"
              className="text-secondary"
            />
          ))}
        </svg>
      </div>

      {/* Molecular network */}
      <svg className="absolute inset-0 w-full h-full opacity-5" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
            <circle cx="50" cy="50" r="2" fill="currentColor" className="text-primary" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        {/* Connection lines */}
        <line
          x1="10%"
          y1="20%"
          x2="30%"
          y2="40%"
          stroke="currentColor"
          strokeWidth="1"
          className="text-primary opacity-30"
        />
        <line
          x1="30%"
          y1="40%"
          x2="50%"
          y2="30%"
          stroke="currentColor"
          strokeWidth="1"
          className="text-primary opacity-30"
        />
        <line
          x1="50%"
          y1="30%"
          x2="70%"
          y2="50%"
          stroke="currentColor"
          strokeWidth="1"
          className="text-secondary opacity-30"
        />
        <line
          x1="70%"
          y1="50%"
          x2="80%"
          y2="70%"
          stroke="currentColor"
          strokeWidth="1"
          className="text-secondary opacity-30"
        />
        <line
          x1="20%"
          y1="80%"
          x2="40%"
          y2="60%"
          stroke="currentColor"
          strokeWidth="1"
          className="text-primary opacity-30"
        />
        <line
          x1="40%"
          y1="60%"
          x2="60%"
          y2="70%"
          stroke="currentColor"
          strokeWidth="1"
          className="text-secondary opacity-30"
        />
      </svg>

      {/* Orbiting molecules */}
      <div className="absolute top-1/3 left-1/4 w-4 h-4">
        <div className="animate-orbit">
          <div className="w-2 h-2 rounded-full bg-primary opacity-40" />
        </div>
      </div>

      <div className="absolute top-2/3 right-1/3 w-4 h-4">
        <div className="animate-orbit" style={{ animationDuration: "25s", animationDirection: "reverse" }}>
          <div className="w-2 h-2 rounded-full bg-secondary opacity-40" />
        </div>
      </div>
    </div>
  )
}
