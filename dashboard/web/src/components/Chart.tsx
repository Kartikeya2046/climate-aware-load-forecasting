import * as echarts from 'echarts' // ponytail: full build (~1 MB). Switch to echarts/core imports if bundle size ever matters.
import { useEffect, useRef } from 'react'

export default function Chart({ option, height = 320, label }: { option: Record<string, any>; height?: number; label: string }) {
  const el = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    const c = (chart.current = echarts.init(el.current!, null, { renderer: 'canvas' }))
    const ro = new ResizeObserver(() => c.resize())
    ro.observe(el.current!)
    return () => { ro.disconnect(); c.dispose() }
  }, [])
  useEffect(() => { chart.current?.setOption(option, true) }, [option])

  return <div ref={el} role="img" aria-label={label} style={{ height }} className="w-full" />
}
