import { useEffect, useRef } from 'react'

/** Ambient background: a slowly drifting pressure field drawn as isobars, the way a synoptic weather chart
 *  shows it. Two pressure centres wander over a low-frequency swell; contours are traced with marching squares. */
export default function IsobarField() {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current!
    const ctx = canvas.getContext('2d')!
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches
    const CELL = 30
    const LEVELS = [-1.5, -1.15, -0.8, -0.45, -0.1, 0.25, 0.6, 0.95, 1.3]
    let w = 0, h = 0, cols = 0, rows = 0, raf = 0, last = 0
    let field = new Float32Array(0)

    const resize = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2)
      w = innerWidth; h = innerHeight
      canvas.width = w * dpr; canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      cols = Math.ceil(w / CELL) + 1; rows = Math.ceil(h / CELL) + 1
      field = new Float32Array(cols * rows)
      draw(performance.now())
    }

    const draw = (now: number) => {
      const t = now / 26000
      const hx = w * (0.72 + 0.12 * Math.sin(t * 1.3)), hy = h * (0.3 + 0.1 * Math.cos(t * 0.9))
      const lx = w * (0.22 + 0.1 * Math.cos(t * 1.1)), ly = h * (0.7 + 0.12 * Math.sin(t * 0.7))
      const s = Math.max(w, h) * 0.34
      for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
          const x = i * CELL, y = j * CELL
          const hi = Math.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (s * s))
          const lo = Math.exp(-((x - lx) ** 2 + (y - ly) ** 2) / (s * s * 0.8))
          field[j * cols + i] = 1.6 * hi - 1.7 * lo + 0.35 * Math.sin(x / 260 + t * 2) * Math.cos(y / 210 - t * 1.4)
        }
      }
      ctx.clearRect(0, 0, w, h)
      ctx.lineWidth = 1
      LEVELS.forEach((L, k) => {
        ctx.strokeStyle = k % 3 === 1 ? 'rgba(150,185,240,0.17)' : 'rgba(150,185,240,0.08)'
        ctx.beginPath()
        for (let j = 0; j < rows - 1; j++) {
          for (let i = 0; i < cols - 1; i++) {
            const a = field[j * cols + i], b = field[j * cols + i + 1]
            const c = field[(j + 1) * cols + i + 1], d = field[(j + 1) * cols + i]
            const idx = (a > L ? 8 : 0) | (b > L ? 4 : 0) | (c > L ? 2 : 0) | (d > L ? 1 : 0)
            if (idx === 0 || idx === 15) continue
            const x = i * CELL, y = j * CELL
            const T = [x + (CELL * (L - a)) / (b - a), y]
            const R = [x + CELL, y + (CELL * (L - b)) / (c - b)]
            const B = [x + (CELL * (L - d)) / (c - d), y + CELL]
            const Lf = [x, y + (CELL * (L - a)) / (d - a)]
            const seg = (p: number[], q: number[]) => { ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]) }
            switch (idx) {
              case 1: case 14: seg(Lf, B); break
              case 2: case 13: seg(B, R); break
              case 3: case 12: seg(Lf, R); break
              case 4: case 11: seg(T, R); break
              case 6: case 9: seg(T, B); break
              case 7: case 8: seg(T, Lf); break
              case 5: seg(T, Lf); seg(B, R); break
              case 10: seg(T, R); seg(Lf, B); break
            }
          }
        }
        ctx.stroke()
      })
    }

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      if (now - last < 66 || document.hidden) return // ~15 fps is plenty for a drift this slow
      last = now
      draw(now)
    }
    resize()
    addEventListener('resize', resize)
    if (!still) raf = requestAnimationFrame(loop)
    return () => { cancelAnimationFrame(raf); removeEventListener('resize', resize) }
  }, [])

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10">
      <div className="absolute inset-0" style={{ background: 'radial-gradient(900px 600px at 78% 8%, rgba(57,135,229,0.13), transparent 70%), radial-gradient(800px 600px at 12% 92%, rgba(201,133,0,0.09), transparent 70%)' }} />
      <canvas ref={ref} className="absolute inset-0 h-full w-full" />
      <div className="absolute inset-0 opacity-[0.05] mix-blend-overlay" style={{ backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")" }} />
    </div>
  )
}
