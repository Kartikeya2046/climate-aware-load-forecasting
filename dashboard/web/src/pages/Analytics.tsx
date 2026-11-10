import { useMemo, useState } from 'react'
import Chart from '../components/Chart'
import { Chip, Disclosure, Notice, PageHead, Panel, Scroll, Stat, Tabs } from '../components/ui'
import * as charts from '../lib/charts'
import { fmtEnergy, fmtShort, mean, n0, nf } from '../lib/format'
import { useNames, useStore } from '../lib/store'
import { Equation } from './Overview'
import { NeedsResult } from './shared'

type Tab = 'load' | 'parts' | 'patterns' | 'weather' | 'accuracy' | 'compare'
const sum = (xs: (number | null)[]) => xs.reduce((a: number, b) => a + (b ?? 0), 0)

export default function Analytics() {
  const { result, meta } = useStore()
  const names = useNames()
  const [tab, setTab] = useState<Tab>('load')
  const [daily, setDaily] = useState<'hourly' | 'daily'>('hourly')
  const [shown, setShown] = useState<string[]>([])
  const [wvar, setWvar] = useState('temperature_c')
  const df = result?.predictions
  const win = result?.scores.find(s => s.selected)

  const o = useMemo(() => {
    if (!df || !result) return null
    const alts = Object.fromEntries(Object.entries(result.alternatives).filter(([k]) => shown.includes(k)))
    const wcols = Object.keys(meta.variables).filter(v => v !== 'load_mw' && df[v]?.some(x => x != null))
    const v = wcols.includes(wvar) ? wvar : wcols[0]
    const varying = wcols.filter(c => new Set(df[c].filter(x => x != null)).size > 1)
    const cols = [...varying, 'pred_mw']
    return {
      wcols, v,
      load: charts.forecast(df, { alts, names: names.model, zoom: true }),
      daily: result.daily ? charts.dailyBars(result.daily) : null,
      cumulative: charts.cumulative(df),
      parts: df.baseline_mw?.some(x => x != null) ? charts.decomposition(df) : null,
      profile: charts.hourlyProfile(df), heat: charts.hourDow(df),
      weather: charts.weatherSeries(df, v, meta.variables[v].label, meta.variables[v].unit),
      scatter: charts.scatterVs(df, v, meta.variables[v].label, meta.variables[v].unit),
      r: charts.pearson(df[v], df.pred_mw),
      corr: varying.length >= 1 ? charts.correlation(cols.map(c => (c === 'pred_mw' ? 'Predicted load' : meta.variables[c].label)),
        cols.map(a => cols.map(b => (a === b ? 1 : charts.pearson(df[a], df[b]) ?? 0)))) : null,
      accuracy: df.actual_mw?.some(x => x != null) && !result.request.scenario ? charts.observedVsPredicted(df) : null,
      compare: charts.forecast(df, { alts: result.alternatives, names: names.model, zoom: true }),
    }
  }, [df, result, shown, wvar, meta, names])

  if (!result || result.status !== 'ok' || !df || !o || !win) return <NeedsResult title="Analytics" />
  const altIds = Object.keys(result.alternatives)
  const err = df.actual_mw.map((a, i) => (a == null || df.pred_mw[i] == null ? null : a - df.pred_mw[i]))
  const ape = err.map((e, i) => (e == null ? null : Math.abs(e) / df.actual_mw[i]))
  const adj = mean(df.adjustment_mw), up = df.adjustment_mw?.filter(x => x != null && x > 0).length ?? 0

  return (
    <div>
      <PageHead title="Analytics">Read the prediction from several angles. Drag the range bar under a time chart to zoom; hover any mark for its value.</PageHead>
      <div className="mb-5">
        <Tabs label="Analytics view" value={tab} onChange={setTab} items={[
          { id: 'load', label: 'Load' }, { id: 'parts', label: 'Baseline and weather' }, { id: 'patterns', label: 'Daily patterns' },
          { id: 'weather', label: 'Weather' }, { id: 'accuracy', label: 'Accuracy' }, { id: 'compare', label: 'Model comparison' }]} />
      </div>

      {tab === 'load' && (
        <div className="grid gap-5">
          <Panel title={daily === 'daily' ? 'Energy per day' : 'Predicted load, hour by hour'}
            hint={daily === 'daily' ? 'Sum of the hourly values. A partial day shows fewer hours.' : 'One hour at an average of 1 MW is 1 MWh, so the same curve reads as energy per hour.'}
            action={<div className="flex flex-wrap items-center gap-2">
              {daily === 'hourly' && altIds.map(id => (
                <button key={id} onClick={() => setShown(s => (s.includes(id) ? s.filter(x => x !== id) : [...s, id]))} aria-pressed={shown.includes(id)}>
                  <Chip tone={shown.includes(id) ? 'good' : 'neutral'}>{shown.includes(id) ? 'Hide' : 'Compare'} {names.model[id].split(' · ')[0]}</Chip></button>
              ))}
              <Tabs label="Aggregation" size="sm" value={daily} onChange={setDaily} items={[{ id: 'hourly', label: 'Hourly' }, { id: 'daily', label: 'Daily' }]} />
            </div>}>
            {daily === 'daily' && o.daily ? <Chart option={o.daily} height={380} label="Predicted energy per day" /> : <Chart option={o.load} height={400} label="Predicted hourly load" />}
          </Panel>
          <Panel title="Energy accumulated over the timeframe"><Chart option={o.cumulative} height={280} label="Cumulative predicted energy" /></Panel>
          <Disclosure title="Table of hourly values">
            <Scroll max={420}><table className="data">
              <thead><tr><th>Hour</th><th className="r">Predicted MW</th><th className="r">Interval low</th><th className="r">Interval high</th><th className="r">Baseline</th><th className="r">Weather adj.</th><th className="r">Observed</th></tr></thead>
              <tbody>{df.t.map((t, i) => <tr key={t}><td className="strong whitespace-nowrap">{fmtShort(t)}</td>{['pred_mw', 'lower_mw', 'upper_mw', 'baseline_mw', 'adjustment_mw', 'actual_mw'].map(c => <td key={c} className="r">{nf(df[c]?.[i], 1)}</td>)}</tr>)}</tbody>
            </table></Scroll>
          </Disclosure>
        </div>
      )}

      {tab === 'parts' && (o.parts ? (
        <div className="grid gap-5">
          <Panel title={<Equation className="text-[26px]" />}
            hint="The baseline is what the calendar alone predicts: hour, weekday, season and trend. The adjustment is what the temporal convolutional network adds after reading a week of weather.">
            <Chart option={o.parts} height={480} label="Calendar baseline, predicted load and the weather adjustment over time" />
          </Panel>
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="Average baseline" value={`${n0(mean(df.baseline_mw))} MW`} sub="Calendar and trend only" />
            <Stat label="Average weather adjustment" value={`${adj! >= 0 ? '+' : '−'}${n0(Math.abs(adj!))} MW`} sub={`${nf((adj! / mean(df.baseline_mw)!) * 100)} % of the baseline`} />
            <Stat label="Hours pushed above baseline" value={`${up} of ${df.t.length}`} sub="Coral bars in the lower chart" />
          </div>
          <Notice>The adjustment is the network's residual term. It is learned from weather and from the recent history of the residual itself, so it is not a pure weather effect.
            As trained, log E = log E<sub>baseline</sub> + r, which in megawatts is ΔE = E<sub>baseline</sub> · (e<sup>r</sup> − 1).</Notice>
        </div>
      ) : <Notice>{win.display_name} predicts load directly. It has no baseline and weather-adjustment split; only the two-branch models do.</Notice>)}

      {tab === 'patterns' && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Panel title="Average load by hour of day"><Chart option={o.profile} height={320} label="Average predicted load for each hour of the day" /></Panel>
          <Panel title="Load by hour and weekday" hint="Empty cells are hours outside the timeframe."><Chart option={o.heat} height={320} label="Heat map of predicted load by hour and day of week" /></Panel>
        </div>
      )}

      {tab === 'weather' && (
        <div className="grid gap-5">
          <Panel title="Weather the model read" action={
            <select aria-label="Weather variable" className="field !h-9 !w-auto min-w-[220px]" value={o.v} onChange={e => setWvar(e.target.value)}>
              {o.wcols.map(c => <option key={c} value={c}>{meta.variables[c].label}</option>)}</select>}>
            <Chart option={o.weather} height={280} label={`${meta.variables[o.v].label} over time`} />
          </Panel>
          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title={`${meta.variables[o.v].label} against predicted load`} hint={o.r == null ? 'No variation to correlate.' : `Pearson r = ${o.r >= 0 ? '+' : '−'}${nf(Math.abs(o.r), 2)} across these hours. A correlation in this window, not a causal effect.`}>
              <Chart option={o.scatter} height={340} label="Scatter of the weather variable against predicted load" />
            </Panel>
            <Panel title="How the variables move together" hint="Pearson correlation within the timeframe. Blue is negative, coral positive.">
              {o.corr ? <Chart option={o.corr} height={340} label="Correlation matrix of weather variables and predicted load" /> : <Notice>A constant scenario has no variation to correlate.</Notice>}
            </Panel>
          </div>
          {result.request.scenario && <Notice>Scenario: the weather is constant, so the variation in predicted load comes from the calendar and the model's own dynamics.</Notice>}
        </div>
      )}

      {tab === 'accuracy' && (o.accuracy ? (
        <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
          <Panel title="Observed against predicted" hint="Each point is one hour. Points on the dashed line were predicted exactly.">
            <Chart option={o.accuracy} height={400} label="Scatter of observed against predicted load" />
          </Panel>
          <div className="grid content-start gap-3">
            <Stat label="Mean absolute percentage error" value={`${nf(mean(ape)! * 100, 2)} %`} sub={`${err.filter(e => e != null).length} hours with observed load`} />
            <Stat label="Mean absolute error" value={`${nf(mean(err.map(e => (e == null ? null : Math.abs(e)))))} MW`} />
            <Stat label="Bias" value={`${mean(err)! >= 0 ? '+' : '−'}${nf(Math.abs(mean(err)!))} MW`} sub="Observed minus predicted. Positive means the model ran low." />
            <Stat label="Held-out error for this model here" value={win.metrics.MAPE_percent == null ? 'Not available' : `${nf(win.metrics.MAPE_percent, 2)} %`} sub={win.metrics.holdout} />
          </div>
        </div>
      ) : <Notice>There is no observed load inside this timeframe{result.request.scenario ? ', and a scenario has no ground truth' : ''}, so there is nothing to compare the prediction with.</Notice>)}

      {tab === 'compare' && (altIds.length ? (
        <div className="grid gap-5">
          <Panel title="Every model that could run" hint="Alternatives were computed on the same inputs. Only the selected model carries an interval."><Chart option={o.compare} height={400} label="Predicted load from every eligible model" /></Panel>
          <Panel pad={false}><Scroll><table className="data">
            <thead><tr><th>Model</th><th className="r">Fitness</th><th className="r">Total energy</th><th className="r">Average</th><th className="r">Peak</th></tr></thead>
            <tbody>{[[win.model_id, df] as const, ...Object.entries(result.alternatives)].map(([id, a]) => {
              const sc = result.scores.find(s => s.model_id === id)
              return <tr key={id}><td className="strong">{names.model[id]}{id === win.model_id && <span className="ml-2"><Chip tone="accent">selected</Chip></span>}</td><td className="r">{nf(sc?.fitness)}</td>
                <td className="r">{fmtEnergy(sum(a.pred_mw))}</td><td className="r">{n0(mean(a.pred_mw))} MW</td><td className="r">{n0(Math.max(...a.pred_mw.filter((x: number | null) => x != null)))} MW</td></tr>
            })}</tbody></table></Scroll></Panel>
        </div>
      ) : <Notice>No other model was eligible for this request, so there is nothing to compare. Model selection lists why each one was excluded.</Notice>)}
    </div>
  )
}
