// ECharts option builders. Every chart draws only values returned by the API.
// Colour follows the entity everywhere: amber = predicted load, white = observed load, slate = calendar baseline,
// coral/blue = weather pushing load up/down, aqua = an alternative model. (Trio validated for dark, all pairs.)
import type { Frame } from './api'
import { fmtHour, ms, n0, nf } from './format'

export const C = {
  amber: '#c98500', amberHi: '#f2b24a', blue: '#3987e5', aqua: '#199e70', red: '#e66767', slate: '#8a97ad',
  observed: '#e8edf4', ink: '#e8edf4', ink2: '#a3afc0', ink3: '#6b788d',
  grid: 'rgba(190,210,255,0.07)', axis: 'rgba(190,210,255,0.18)', surface: '#0d131c',
}
const SEQ_AMBER = ['#1a1710', '#6b4a08', '#c98500', '#ffd98a']
const SEQ_BLUE = ['#131b28', '#184f95', '#2a78d6', '#5598e7']
const DIVERGING = ['#3987e5', '#232a35', '#e66767']

type Opt = Record<string, any>
const label = { color: C.ink3, fontSize: 11 }
const timeAxis = (o: Opt = {}) => ({
  type: 'time', axisLine: { lineStyle: { color: C.axis } }, axisTick: { show: false },
  axisLabel: { ...label, hideOverlap: true }, splitLine: { show: false }, ...o,
})
const valueAxis = (name?: string, o: Opt = {}) => ({
  type: 'value', scale: true, name, nameGap: 12, nameTextStyle: { ...label, align: 'left' },
  axisLabel: label, splitLine: { lineStyle: { color: C.grid } }, ...o,
})
const catAxis = (data: (string | number)[], o: Opt = {}) => ({
  type: 'category', data, axisLine: { lineStyle: { color: C.axis } }, axisTick: { show: false }, axisLabel: label, ...o,
})
const tooltip = (o: Opt = {}) => ({
  trigger: 'axis', backgroundColor: 'rgba(10,15,23,0.94)', borderColor: 'rgba(190,210,255,0.16)', borderWidth: 1,
  padding: [10, 12], textStyle: { color: C.ink, fontSize: 12 },
  extraCssText: 'backdrop-filter:blur(12px);border-radius:12px;box-shadow:0 24px 40px -24px #000;',
  axisPointer: { type: 'line', lineStyle: { color: 'rgba(232,237,244,0.3)' } }, ...o,
})
const legend = (o: Opt = {}) => ({
  top: 0, right: 0, icon: 'roundRect', itemWidth: 14, itemHeight: 3, itemGap: 16,
  textStyle: { color: C.ink2, fontSize: 12 }, inactiveColor: C.ink3, ...o,
})
const base = (o: Opt): Opt => ({
  useUTC: true, backgroundColor: 'transparent', textStyle: { fontFamily: 'Geist Variable, system-ui, sans-serif' },
  animationDuration: 800, animationEasing: 'cubicOut',
  grid: { left: 4, right: 14, top: 40, bottom: 4, containLabel: true }, ...o,
})
const zoom = [
  { type: 'inside' },
  {
    type: 'slider', height: 20, bottom: 6, borderColor: 'transparent', backgroundColor: 'rgba(255,255,255,0.03)',
    fillerColor: 'rgba(242,178,74,0.12)', moveHandleSize: 0, handleSize: '120%', handleStyle: { color: C.amberHi, borderColor: C.amberHi },
    dataBackground: { lineStyle: { color: C.ink3, opacity: 0.6 }, areaStyle: { color: 'rgba(255,255,255,0.06)' } },
    selectedDataBackground: { lineStyle: { color: C.amber }, areaStyle: { color: 'rgba(201,133,0,0.2)' } }, textStyle: label, showDetail: false,
  },
]

const swatch = (c: string) => `<i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${c};margin-right:7px"></i>`
const tip = (title: string, rows: ([string, string, string?] | null)[]) =>
  `<div style="color:${C.ink2};margin-bottom:6px">${title}</div>` +
  rows.filter(Boolean).map(r => `<div style="display:flex;gap:20px;justify-content:space-between;line-height:1.7"><span>${r![2] ? swatch(r![2]) : ''}${r![0]}</span><b style="font-weight:500;font-variant-numeric:tabular-nums">${r![1]}</b></div>`).join('')

const pairs = (x: number[], y: (number | null)[]) => x.map((t, i) => [t, y[i]])
const line = (name: string, data: any[], color: string, o: Opt = {}) => ({
  name, type: 'line', data, symbol: 'none', lineStyle: { color, width: 2 }, itemStyle: { color }, emphasis: { disabled: true }, ...o,
})
const has = (col?: (number | null)[]) => !!col && col.some(v => v != null)

/** Predicted load over time with the empirical 90 % band, observed load and computed alternatives. */
export function forecast(df: Frame, o: { alts?: Record<string, Frame>; names?: Record<string, string>; value?: 'power' | 'energy'; zoom?: boolean } = {}) {
  const x = df.t.map(ms)
  const unit = o.value === 'energy' ? 'MWh' : 'MW'
  const band = has(df.upper_mw)
  const alts = Object.entries(o.alts ?? {})
  const series: any[] = []
  if (band) {
    series.push(line('lower', pairs(x, df.lower_mw), 'transparent', { stack: 'band', lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } }))
    series.push(line('90 % interval', x.map((t, i) => [t, df.upper_mw[i] == null ? null : df.upper_mw[i] - df.lower_mw[i]]), C.amber,
      { stack: 'band', lineStyle: { opacity: 0 }, itemStyle: { color: 'rgba(201,133,0,0.42)' }, areaStyle: { color: 'rgba(201,133,0,0.2)' }, silent: true }))
  }
  series.push(line('Predicted', pairs(x, df.pred_mw), C.amber, { z: 5, lineStyle: { color: C.amber, width: 2.2 } }))
  if (has(df.actual_mw)) series.push(line('Observed', pairs(x, df.actual_mw), C.observed, { z: 4, lineStyle: { color: C.observed, width: 1.4, opacity: 0.9 } }))
  alts.forEach(([id, a]) => series.push(line(o.names?.[id] ?? id, pairs(a.t.map(ms), a.pred_mw), C.aqua, { lineStyle: { color: C.aqua, width: 2, type: [5, 4] } })))
  return base({
    legend: legend({ data: series.filter(s => s.name !== 'lower').map(s => (s.name === '90 % interval' ? { name: s.name, icon: 'roundRect', itemStyle: { color: 'rgba(201,133,0,0.42)' } } : s.name)) }),
    grid: { left: 4, right: 14, top: 40, bottom: o.zoom ? 36 : 4, containLabel: true },
    xAxis: timeAxis(), yAxis: valueAxis(`${unit}, hourly average`),
    dataZoom: o.zoom ? zoom : undefined,
    tooltip: tooltip({
      formatter: (p: any[]) => {
        const i = p[0].dataIndex
        return tip(fmtHour(x[i]), [
          ['Predicted', `${n0(df.pred_mw[i])} ${unit}`, C.amber],
          band && df.lower_mw[i] != null ? ['90 % interval', `${n0(df.lower_mw[i])} – ${n0(df.upper_mw[i])}`, 'rgba(201,133,0,0.45)'] : null,
          has(df.actual_mw) ? ['Observed', `${n0(df.actual_mw[i])} ${unit}`, C.observed] : null,
          ...alts.map(([id, a]): [string, string, string] => [o.names?.[id] ?? id, `${n0(a.pred_mw[i])} ${unit}`, C.aqua]),
        ])
      },
    }),
    series,
  })
}

/** E = E_baseline + ΔE_weather: baseline and prediction on top, the signed weather adjustment below (shared time axis). */
export function decomposition(df: Frame) {
  const x = df.t.map(ms)
  const grids = [{ left: 58, right: 14, top: 40, height: '52%' }, { left: 58, right: 14, bottom: 28, height: '24%' }]
  return base({
    grid: grids, axisPointer: { link: [{ xAxisIndex: 'all' }] },
    legend: legend({ data: ['Calendar baseline', 'Predicted', 'Weather adjustment'] }),
    xAxis: [timeAxis({ gridIndex: 0, axisLabel: { show: false } }), timeAxis({ gridIndex: 1 })],
    yAxis: [valueAxis('MW', { gridIndex: 0 }), valueAxis('ΔE, MW', { gridIndex: 1, scale: false, splitNumber: 3 })],
    tooltip: tooltip({
      formatter: (p: any[]) => {
        const i = p[0].dataIndex
        const d = df.adjustment_mw[i]
        return tip(fmtHour(x[i]), [
          ['Calendar baseline', `${n0(df.baseline_mw[i])} MW`, C.slate],
          ['Weather adjustment', d == null ? '–' : `${d >= 0 ? '+' : '−'}${n0(Math.abs(d))} MW`, d >= 0 ? C.red : C.blue],
          ['Predicted', `${n0(df.pred_mw[i])} MW`, C.amber],
        ])
      },
    }),
    series: [
      line('Calendar baseline', pairs(x, df.baseline_mw), C.slate, { lineStyle: { color: C.slate, width: 1.6, type: [6, 4] } }),
      line('Predicted', pairs(x, df.pred_mw), C.amber),
      {
        name: 'Weather adjustment', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, barCategoryGap: '18%', itemStyle: { color: C.red },
        data: x.map((t, i) => ({ value: [t, df.adjustment_mw[i]], itemStyle: { color: (df.adjustment_mw[i] ?? 0) >= 0 ? C.red : C.blue, borderRadius: (df.adjustment_mw[i] ?? 0) >= 0 ? [2, 2, 0, 0] : [0, 0, 2, 2] } })),
      },
    ],
  })
}

export function cumulative(df: Frame) {
  const x = df.t.map(ms)
  let acc = 0
  const cum = df.energy_mwh.map(v => (acc += v ?? 0))
  const full = !!df.actual_mw && df.actual_mw.every(v => v != null)
  let acc2 = 0
  const series: any[] = [line('Predicted', pairs(x, cum), C.amber, {
    areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(201,133,0,0.28)' }, { offset: 1, color: 'rgba(201,133,0,0)' }] } },
  })]
  if (full) series.push(line('Observed', pairs(x, df.actual_mw.map(v => (acc2 += v))), C.observed, { lineStyle: { color: C.observed, width: 1.4 } }))
  return base({
    legend: full ? legend() : undefined, xAxis: timeAxis(), yAxis: valueAxis('MWh, cumulative', { scale: false }),
    tooltip: tooltip({ valueFormatter: (v: number) => `${n0(v)} MWh` }), series,
  })
}

export function hourlyProfile(df: Frame) {
  const sum = Array(24).fill(0), cnt = Array(24).fill(0)
  df.t.forEach((t, i) => { if (df.pred_mw[i] != null) { const h = +t.slice(11, 13); sum[h] += df.pred_mw[i]; cnt[h]++ } })
  return base({
    xAxis: catAxis([...Array(24).keys()].map(h => `${String(h).padStart(2, '0')}`), { name: 'Hour, local standard time', nameLocation: 'middle', nameGap: 26, nameTextStyle: label }),
    yAxis: valueAxis('MW, average'),
    grid: { left: 4, right: 14, top: 40, bottom: 22, containLabel: true },
    tooltip: tooltip({ axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(255,255,255,0.04)' } }, formatter: (p: any[]) => tip(`${p[0].name}:00`, [['Average predicted', `${n0(p[0].value)} MW`, C.amber]]) }),
    series: [{ type: 'bar', data: sum.map((s, h) => (cnt[h] ? s / cnt[h] : null)), barCategoryGap: '22%', itemStyle: { color: C.amber, borderRadius: [4, 4, 0, 0] } }],
  })
}

export function hourDow(df: Frame) {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  const sum: Record<string, [number, number]> = {}
  df.t.forEach((t, i) => {
    if (df.pred_mw[i] == null) return
    const k = `${+t.slice(11, 13)}|${(new Date(ms(t)).getUTCDay() + 6) % 7}`
    sum[k] = [(sum[k]?.[0] ?? 0) + df.pred_mw[i], (sum[k]?.[1] ?? 0) + 1]
  })
  const data = Object.entries(sum).map(([k, [s, n]]) => { const [h, d] = k.split('|').map(Number); return [h, d, s / n] })
  const vals = data.map(d => d[2])
  return base({
    grid: { left: 4, right: 70, top: 12, bottom: 4, containLabel: true },
    xAxis: catAxis([...Array(24).keys()], { splitArea: { show: false } }), yAxis: catAxis(days, { inverse: true, axisLine: { show: false } }),
    visualMap: { min: Math.min(...vals), max: Math.max(...vals), calculable: false, orient: 'vertical', right: 0, top: 'center', itemWidth: 10, itemHeight: 110,
      inRange: { color: SEQ_AMBER }, textStyle: label, formatter: (v: number) => n0(v) },
    tooltip: tooltip({ trigger: 'item', formatter: (p: any) => tip(`${days[p.value[1]]} ${String(p.value[0]).padStart(2, '0')}:00`, [['Average predicted', `${n0(p.value[2])} MW`]]) }),
    series: [{ type: 'heatmap', data, itemStyle: { borderColor: C.surface, borderWidth: 2, borderRadius: 4 }, emphasis: { itemStyle: { borderColor: C.ink } } }],
  })
}

export function weatherSeries(df: Frame, v: string, name: string, unit: string) {
  const x = df.t.map(ms)
  return base({
    xAxis: timeAxis(), yAxis: valueAxis(unit), tooltip: tooltip({ valueFormatter: (y: number) => `${nf(y, 2)} ${unit}` }),
    series: [line(name, pairs(x, df[v]), C.blue, { areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(57,135,229,0.22)' }, { offset: 1, color: 'rgba(57,135,229,0)' }] } } })],
  })
}

const dot = (color: string, size = 8) => ({ symbolSize: size, itemStyle: { color, opacity: 0.85, borderColor: C.surface, borderWidth: 1.5 } })
const itemTip = (f: (p: any) => string) => tooltip({ trigger: 'item', formatter: f })

export function scatterVs(df: Frame, v: string, name: string, unit: string) {
  const data = df.t.map((t, i) => [df[v][i], df.pred_mw[i], ms(t)]).filter(d => d[0] != null && d[1] != null)
  return base({
    grid: { left: 4, right: 14, top: 30, bottom: 22, containLabel: true },
    xAxis: valueAxis(undefined, { name: `${name}, ${unit}`, nameLocation: 'middle', nameGap: 26, nameTextStyle: label }), yAxis: valueAxis('Predicted load, MW'),
    tooltip: itemTip(p => tip(fmtHour(p.value[2]), [[name, `${nf(p.value[0], 2)} ${unit}`, C.blue], ['Predicted', `${n0(p.value[1])} MW`, C.amber]])),
    series: [{ type: 'scatter', data, ...dot(C.amber) }],
  })
}

export function correlation(names: string[], matrix: number[][]) {
  const data = matrix.flatMap((row, i) => row.map((v, j) => [j, i, v]))
  return base({
    grid: { left: 4, right: 14, top: 8, bottom: 4, containLabel: true },
    xAxis: catAxis(names, { axisLabel: { ...label, rotate: 35, interval: 0 }, axisLine: { show: false } }),
    yAxis: catAxis(names, { inverse: true, axisLine: { show: false }, axisLabel: { ...label, interval: 0 } }),
    visualMap: { show: false, min: -1, max: 1, inRange: { color: DIVERGING } },
    tooltip: itemTip(p => tip(`${names[p.value[1]]} × ${names[p.value[0]]}`, [['Pearson r', nf(p.value[2], 2)]])),
    series: [{ type: 'heatmap', data, label: { show: names.length <= 9, color: C.ink, fontSize: 11, formatter: (p: any) => nf(p.value[2], 2) }, itemStyle: { borderColor: C.surface, borderWidth: 2, borderRadius: 4 } }],
  })
}

export function observedVsPredicted(df: Frame) {
  const data = df.t.map((t, i) => [df.actual_mw[i], df.pred_mw[i], ms(t)]).filter(d => d[0] != null && d[1] != null)
  const all = data.flatMap(d => [d[0], d[1]])
  const lo = Math.min(...all), hi = Math.max(...all)
  return base({
    legend: legend(),
    grid: { left: 4, right: 14, top: 40, bottom: 22, containLabel: true },
    xAxis: valueAxis(undefined, { name: 'Observed, MW', nameLocation: 'middle', nameGap: 26, nameTextStyle: label }), yAxis: valueAxis('Predicted, MW'),
    tooltip: itemTip(p => (p.seriesType === 'line' ? '' : tip(fmtHour(p.value[2]), [['Observed', `${n0(p.value[0])} MW`, C.observed], ['Predicted', `${n0(p.value[1])} MW`, C.amber]]))),
    series: [
      line('Perfect prediction', [[lo, lo], [hi, hi]], C.ink3, { lineStyle: { color: C.ink3, width: 1, type: 'dashed' }, silent: true }),
      { name: 'Hour', type: 'scatter', data, ...dot(C.amber) },
    ],
  })
}

export function dailyBars(daily: Frame) {
  const days = daily.t.map(t => t.slice(0, 10))
  const obs = has(daily.actual_energy_mwh)
  return base({
    legend: obs ? legend({ icon: 'roundRect', itemHeight: 8, itemWidth: 8 }) : undefined,
    xAxis: catAxis(days), yAxis: valueAxis('MWh per day', { scale: false }),
    tooltip: tooltip({ axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(255,255,255,0.04)' } }, valueFormatter: (v: number) => `${n0(v)} MWh` }),
    series: [
      { name: 'Predicted', type: 'bar', data: daily.energy_mwh, barMaxWidth: 34, barGap: '8%', itemStyle: { color: C.amber, borderRadius: [4, 4, 0, 0] } },
      ...(obs ? [{ name: 'Observed', type: 'bar', data: daily.actual_energy_mwh, barMaxWidth: 34, itemStyle: { color: '#cfd6e2', borderRadius: [4, 4, 0, 0] } }] : []),
    ],
  })
}

export function distributions(h: { edges: number[]; input: number[]; reference: number[] }, name: string, unit: string, refName: string) {
  const mid = h.edges.slice(0, -1).map((e, i) => (e + h.edges[i + 1]) / 2)
  const area = (n: string, y: number[], c: string, fill: string) => line(n, mid.map((m, i) => [m, y[i]]), c, { step: 'middle', lineStyle: { color: c, width: 1.6 }, areaStyle: { color: fill } })
  return base({
    legend: legend(), grid: { left: 4, right: 14, top: 40, bottom: 22, containLabel: true },
    xAxis: valueAxis(undefined, { name: `${name}, ${unit}`, nameLocation: 'middle', nameGap: 26, nameTextStyle: label, splitLine: { show: false } }),
    yAxis: valueAxis('Density', { scale: false }),
    tooltip: tooltip({ valueFormatter: (v: number) => nf(v, 4) }),
    series: [area(refName, h.reference, C.slate, 'rgba(138,151,173,0.2)'), area('Input weather', h.input, C.amber, 'rgba(201,133,0,0.28)')],
  })
}

export function pca(points: Record<string, [number, number][]>, names: Record<string, string>, input: [number, number][] | null, explained: number[]) {
  const greys = ['#4d586a', '#5b6679', '#6a7589', '#798498', '#8893a6', '#97a2b4']
  const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]
  const ids = Object.keys(points)
  return base({
    legend: input ? legend({ data: ['Input weather'], icon: 'circle', itemHeight: 8, itemWidth: 8 }) : undefined,
    grid: { left: 4, right: 14, top: 36, bottom: 22, containLabel: true },
    xAxis: valueAxis(undefined, { name: `PC1, ${nf(explained[0] * 100, 0)} % of variance`, nameLocation: 'middle', nameGap: 26, nameTextStyle: label }),
    yAxis: valueAxis(`PC2, ${nf(explained[1] * 100, 0)} %`),
    tooltip: itemTip(p => p.seriesName),
    series: [
      ...ids.map((c, k) => ({ name: names[c], type: 'scatter', data: points[c], symbolSize: 3.5, itemStyle: { color: greys[k], opacity: 0.4 }, large: true, emphasis: { disabled: true } })),
      { name: 'City', type: 'scatter', symbolSize: 0, silent: true, data: ids.map(c => ({ value: [med(points[c].map(p => p[0])), med(points[c].map(p => p[1]))], name: names[c] })),
        label: { show: true, formatter: '{b}', color: C.ink, fontSize: 12, backgroundColor: 'rgba(10,15,23,0.78)', padding: [3, 7], borderRadius: 6 }, labelLayout: { moveOverlap: 'shiftY' } },
      ...(input ? [{ name: 'Input weather', type: 'scatter', data: input, z: 10, ...dot(C.amberHi, 9) }] : []),
    ],
  })
}

export function matrix(names: string[], z: number[][]) {
  const data = z.flatMap((row, i) => row.map((v, j) => [j, i, v]))
  return base({
    grid: { left: 4, right: 14, top: 8, bottom: 4, containLabel: true },
    xAxis: catAxis(names, { axisLine: { show: false }, axisLabel: { ...label, interval: 0 } }), yAxis: catAxis(names, { inverse: true, axisLine: { show: false } }),
    visualMap: { show: false, min: 0, max: 100, inRange: { color: SEQ_BLUE } },
    tooltip: itemTip(p => tip(`${names[p.value[1]]} ↔ ${names[p.value[0]]}`, [['Climate similarity', `${nf(p.value[2], 1)} / 100`]])),
    series: [{ type: 'heatmap', data, label: { show: true, color: C.ink, fontSize: 12, formatter: (p: any) => nf(p.value[2], 0) }, itemStyle: { borderColor: C.surface, borderWidth: 3, borderRadius: 6 } }],
  })
}

export const pearson = (a: (number | null)[], b: (number | null)[]) => {
  const p = a.map((x, i) => [x, b[i]]).filter(d => d[0] != null && d[1] != null) as number[][]
  if (p.length < 3) return null
  const mx = p.reduce((s, d) => s + d[0], 0) / p.length, my = p.reduce((s, d) => s + d[1], 0) / p.length
  let sxy = 0, sx = 0, sy = 0
  p.forEach(([x, y]) => { sxy += (x - mx) * (y - my); sx += (x - mx) ** 2; sy += (y - my) ** 2 })
  return sx > 1e-12 && sy > 1e-12 ? sxy / Math.sqrt(sx * sy) : null
}
