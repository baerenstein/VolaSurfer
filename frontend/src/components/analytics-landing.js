"use client"

import { useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"

const generateData = (time: number) => {
  const data = []
  for (let x = 0; x <= 50; x++) {
    for (let y = 0; y <= 50; y++) {
      // More dramatic wave effect
      const value = 
        Math.sin((x / 25) * Math.PI + time) * 
        Math.cos((y / 25) * Math.PI + time * 0.5) * 30 + 50
      data.push({ x, y, z: value })
    }
  }
  return data
}

const drawSurface = (ctx: CanvasRenderingContext2D, data: { x: number; y: number; z: number }[]) => {
  const width = ctx.canvas.width
  const height = ctx.canvas.height
  const maxZ = Math.max(...data.map(point => point.z))
  const minZ = Math.min(...data.map(point => point.z))

  // Clear with a slight fade effect for smoother animation
  ctx.fillStyle = 'rgba(255, 255, 255, 0.1)'
  ctx.fillRect(0, 0, width, height)

  data.forEach(point => {
    const x = (point.x / 50) * width
    const y = (point.y / 50) * height
    const intensity = (point.z - minZ) / (maxZ - minZ)
    
    // Enhanced color gradient
    ctx.fillStyle = `hsl(${200 + intensity * 160}, 80%, ${40 + intensity * 40}%)`
    ctx.fillRect(x, y, width / 50 + 1, height / 50 + 1) // Slightly larger rectangles to avoid gaps
  })
}

export default function AnalyticsLanding() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size with higher resolution for sharper rendering
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    let animationFrameId: number
    let time = 0

    const render = () => {
      time += 0.03 // Slower animation speed
      const data = generateData(time)
      drawSurface(ctx, data)
      animationFrameId = requestAnimationFrame(render)
    }

    render()

    return () => {
      cancelAnimationFrame(animationFrameId)
    }
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center justify-between">
          <div className="flex items-center space-x-4">
            <span className="text-xl font-bold">VolSurface Pro</span>
          </div>
          <div className="flex items-center space-x-4">
            <Button variant="outline">Sign Up</Button>
            <Button>Login</Button>
          </div>
        </div>
      </header>
      
      <main className="container mx-auto">
        <section className="grid items-center gap-8 pb-8 pt-6 md:py-10">
          <div className="flex max-w-[980px] flex-col items-start gap-2">
            <h1 className="text-3xl font-extrabold leading-tight tracking-tighter md:text-4xl lg:text-5xl">
              Real-time Volatility Surface Analytics
            </h1>
            <p className="max-w-[700px] text-lg text-muted-foreground">
              Professional-grade options analytics with live market data
            </p>
          </div>
          
          <Card className="w-full overflow-hidden">
            <CardHeader>
              <CardTitle>Animated Volatility Surface</CardTitle>
              <CardDescription>Dynamic visualization of implied volatility by maturity and delta</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-video w-full overflow-hidden rounded-lg border">
                <canvas 
                  ref={canvasRef} 
                  className="h-full w-full"
                  style={{ width: '100%', height: '100%' }}
                />
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Real-time Analysis</CardTitle>
                <CardDescription>
                  Monitor market volatility patterns as they develop
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Our platform provides instant updates and sophisticated analysis tools to help you make informed decisions.
                </p>
                <Button className="mt-4" variant="outline">
                  Learn more <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Advanced Metrics</CardTitle>
                <CardDescription>
                  Comprehensive data visualization suite
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Access detailed metrics, custom indicators, and advanced charting tools all in one place.
                </p>
                <Button className="mt-4" variant="outline">
                  View features <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>
    </div>
  )
}