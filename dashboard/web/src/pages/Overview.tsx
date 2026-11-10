import { ArrowRight } from 'lucide-react'
import { useMemo } from 'react'
import Chart from '../components/Chart'
import { Button, Chip, Liquid, Notice, Num, Panel, Stat } from '../components/ui'
import { forecast } from '../lib/charts'
import { cityShort, energy, fmtDay, fmtEnergy, fmtHour, mean, n0, nf } from '../lib/format'
import { useNames, useStore } from '../lib/store'

/** The research equation, set as the product's signature. Term colours are the legend used in every chart. */
export const Equation = ({ className }: { className?: string }) => (
  <span className={`display italic ${className ?? ''}`}>
    <span className="text-amber-hi">E</span> <span className="not-italic text-ink-3">=</span>{' '}
    <span className="text-slate">E<sub className="text-[0.42em] not-italic">baseline</sub></span>{' '}
    <span className="not-italic text-ink-3">+</span> <span className="text-ice-hi">ΔE<sub className="text-[0.42em] not-italic">weather</sub></span>
  </span>
)

/** How the two branches add up. Drawn from fixed curves to explain the idea; it is labelled as a schematic because it is not data. */
function Schematic() {
  const N = 72, W = 420, x = (i: number) => (i / (N - 1)) * W
  const b = (t: number) => 50 + 22 * Math.sin((2 * Math.PI * (t - 9)) / 24) + 6 * Math.sin((4 * Math.PI * (t - 3)) / 24)
  const w = (t: number) => 15 * Math.sin((2 * Math.PI * t) / 72 - 1.3) * (0.65 + 0.35 * Math.sin((2 * Math.PI * (t - 8)) / 24))
  const path = (f: (t: number) => number, y0: number) => [...Array(N).keys()].map(i => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${(y0 - f(i) * 0.72).toFixed(1)}`).join('')
  const rows = [['Calendar baseline', 18], ['+ Weather adjustment', 108], ['= Predicted load', 178]] as const
  return (
    <figure className="hidden lg:block">
      <svg viewBox="0 0 420 260" className="w-full" role="img" aria-label="Schematic: a daily baseline curve plus a signed weather adjustment gives the predicted load">
        {rows.map(([l, y]) => <text key={l} x="0" y={y} fill="#6b788d" fontSize="12">{l}</text>)}
        <path d={path(b, 92)} stroke="#8a97ad" strokeWidth="1.6" strokeDasharray="6 4" fill="none" />
        <line x1="0" x2={W} y1="136" y2="136" stroke="rgba(190,210,255,0.16)" />
        {[...Array(N).keys()].map(i => { const v = w(i) * 0.9; return <rect key={i} x={x(i) - 1.6} width="3.2" y={v >= 0 ? 136 - v : 136} height={Math.abs(v)} rx="1" fill={v >= 0 ? '#e66767' : '#3987e5'} /> })}
        <path d={path(t => b(t) + w(t), 256)} stroke="#f2b24a" strokeWidth="2.2" fill="none" strokeLinecap="round" />
      </svg>
      <figcaption className="mt-2 text-[12.5px] text-ink-3">Schematic of three days. Run a forecast to see this split for real data.</figcaption>
    </figure>
  )
}

function Welcome() {
  const { meta, go, runExample, busy } = useStore()
  const names = useNames()
  const rows = meta.models.flatMap(m => [...m.trained_on, ...m.evaluated_on].map(c => {
    const e = meta.evidence.models?.[m.id]?.[c]?.one_step
    return e ? { key: m.id + c, model: m.display_name, city: cityShort(names.city[c]), mape: e.MAPE_percent as number, naive: e.naive?.MAPE_percent as number } : null
  })).filter(Boolean) as { key: string; model: string; city: string; mape: number; naive: number }[]
  const scale = Math.max(...rows.map(r => r.naive)) * 1.08

  return (
    <div className="grid gap-5">
      <Liquid className="grid items-center gap-10 px-6 py-12 sm:px-12 sm:py-14 lg:grid-cols-[1.3fr_1fr]">
        <div>
        <h1 className="text-[clamp(36px,5.6vw,80px)] leading-[1.05]"><Equation /></h1>
        <p className="mt-8 max-w-[58ch] text-[17px] text-ink-2">
          Hourly electricity demand is a calendar rhythm plus what the weather adds or takes away. Give this platform weather for a place and a time;
          it checks every trained model against your data, picks the best fit, and predicts the load with an honest account of how far to trust it.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Button variant="primary" onClick={() => go('workspace')}>Start a forecast <ArrowRight size={16} /></Button>
          <Button onClick={runExample} disabled={!!busy}>Run the Atlanta example</Button>
        </div>
        <p className="mt-3 text-[13px] text-ink-3">The example predicts 10 to 12 July 2023 from recorded Atlanta weather. It takes about ten seconds.</p>
        </div>
        <Schematic />
      </Liquid>

      <div className="grid gap-5 lg:grid-cols-[1.5fr_1fr]">
        <Panel title="Accuracy one hour ahead, on data the models never saw"
          hint="Mean absolute percentage error on each held-out period. The tick marks the naive benchmark: the same hour on the previous day.">
          <div className="grid gap-3.5">
            {rows.map(r => (
              <div key={r.key} className="grid grid-cols-[minmax(0,1.1fr)_minmax(0,1.6fr)_auto] items-center gap-4 text-[13.5px]">
                <div className="min-w-0"><span className="text-ink">{r.city}</span><span className="ml-2 hidden truncate text-ink-3 sm:inline">{r.model.split(' · ')[0]}</span></div>
                <div className="relative h-2 rounded-full bg-white/[0.06]" title={`${nf(r.mape, 2)} % against a naive ${nf(r.naive, 2)} %`}>
                  <div className="h-full rounded-full bg-amber" style={{ width: `${(r.mape / scale) * 100}%` }} />
                  <div className="absolute top-[-3px] h-[14px] w-[2px] rounded bg-ink-2" style={{ left: `${(r.naive / scale) * 100}%` }} />
                </div>
                <div className="num w-[118px] text-right text-ink-2"><span className="text-ink">{nf(r.mape, 2)} %</span> <span className="text-ink-3">/ {nf(r.naive, 2)}</span></div>
              </div>
            ))}
          </div>
          <p className="mt-5 max-w-[72ch] text-[13px] text-ink-3">
            These are one-hour-ahead models. Left to run on their own predictions for a week, most of them do no better than repeating last week's load. The Research page shows those figures too.
          </p>
        </Panel>

        <Panel title="Loaded and ready">
          <ul className="grid gap-3">
            {meta.models.map(m => (
              <li key={m.id} className="flex items-start gap-3">
                <span className={`mt-[7px] h-2 w-2 shrink-0 rounded-full ${m.loaded ? 'bg-good' : 'bg-bad'}`} />
                <div className="min-w-0">
                  <div className="text-[14px] text-ink">{m.display_name}</div>
                  <div className="text-[13px] text-ink-3">{m.loaded ? `Reads ${m.lookback_hours} h of history, ${m.weather_inputs.length} weather inputs` : m.load_error}</div>
                </div>
              </li>
            ))}
          </ul>
          <div className="mt-5 border-t border-line pt-4">
            <div className="mb-2 text-[13px] text-ink-3">Trained cities</div>
            <div className="flex flex-wrap gap-1.5">{meta.cities.map(c => <Chip key={c.id}>{cityShort(c.name)}</Chip>)}</div>
            <p className="mt-3 text-[13px] text-ink-3">Any other city works too: upload its weather and it is matched to the closest trained climate.</p>
          </div>
        </Panel>
      </div>
    </div>
  )
}

const Ring = ({ value }: { value: number }) => {
  const r = 30, c = 2 * Math.PI * r
  return (
    <svg width="76" height="76" viewBox="0 0 76 76" aria-hidden>
      <circle cx="38" cy="38" r={r} stroke="rgba(190,210,255,0.12)" strokeWidth="5" fill="none" />
      <circle cx="38" cy="38" r={r} stroke="#f2b24a" strokeWidth="5" fill="none" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - value / 100)}
        transform="rotate(-90 38 38)" style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.2,0.7,0.2,1)' }} />
    </svg>
  )
}

export default function Overview() {
  const { result, go } = useStore()
  const names = useNames()
  const df = result?.predictions
  const chart = useMemo(() => (df ? forecast(df) : null), [df])
  if (!result) return <Welcome />

  const { request: req, summary: s } = result
  const place = result.analog ? `${result.analog.name}, on ${cityShort(names.city[req.location_id])}'s grid` : names.city[req.location_id]
  const head = (
    <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="display text-[34px] leading-tight sm:text-[40px]">{place}</h1>
        <p className="mt-1 text-ink-2">{fmtHour(req.start)} to {fmtHour(req.end)}, local standard time</p>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Chip>{req.prediction_mode === 'one_step' ? 'One hour ahead' : 'Free-running forecast'}</Chip>
        <Chip>{req.scenario ? 'Manual scenario' : result.source === 'recorded' ? 'Recorded weather' : 'Uploaded weather'}</Chip>
      </div>
    </header>
  )

  if (result.status !== 'ok' || !df) {
    return (
      <div>
        {head}
        <div className="grid gap-4">
          <Notice tone="bad"><b className="font-medium text-ink">{result.message}</b>
            <ul className="mt-2 grid gap-1">{result.selection_explanation.map((t, i) => <li key={i}>{t}</li>)}</ul></Notice>
          <div><Button onClick={() => go('workspace')}>Change the inputs</Button></div>
        </div>
      </div>
    )
  }

  const win = result.scores.find(x => x.selected)!
  const total = energy(s.total_energy_mwh)
  const iv = s.total_energy_interval_mwh as [number, number] | undefined
  const base = mean(df.baseline_mw), adj = mean(df.adjustment_mw)
  const eligible = result.scores.filter(x => x.eligibility !== 'ineligible').length
  const notices = result.issues.filter(i => i.severity !== 'fatal').length
  const bench = s.benchmarks?.calendar_baseline_only ?? s.benchmarks?.naive_168h
  const worse = req.prediction_mode !== 'one_step' && win.metrics.MAPE_percent && bench && win.metrics.MAPE_percent > bench.MAPE_percent

  return (
    <div>
      {head}
      <div className="grid gap-5">
        {result.analog && (
          <Notice>{result.analog.name} is not a trained city. Its weather is closest to <b className="font-medium text-ink">{names.city[result.analog.city]}</b> ({nf(result.analog.similarity)} of 100, {result.analog.classification.toLowerCase()}),
            so this is that weather on {cityShort(names.city[result.analog.city])}'s grid: its load scale, infrastructure and population are assumed.</Notice>
        )}
        <Liquid className="grid gap-8 p-6 sm:p-9 lg:grid-cols-[1.55fr_1fr]">
          <div>
            <div className="text-ink-2">Total predicted energy</div>
            <div className="display mt-1 flex items-baseline gap-3 text-[clamp(52px,8vw,104px)] leading-none">
              <Num value={total.value} digits={total.digits} /><span className="text-[0.32em] text-ink-2">{total.unit}</span>
            </div>
            <p className="mt-3 text-[14px] text-ink-3">
              {iv ? <>90 % interval {fmtEnergy(iv[0])} to {fmtEnergy(iv[1])}, from held-out errors of this model here</> : 'No interval for this horizon: held-out window totals exist only for 24 h and 168 h'}
            </p>
            {base != null && adj != null && (
              <div className="mt-8 flex flex-wrap items-end gap-x-5 gap-y-3 border-t border-line pt-6">
                {[
                  { v: n0(s.average_power_mw), l: 'average load, MW', c: 'text-amber-hi' }, { op: '=' },
                  { v: n0(base), l: 'calendar baseline', c: 'text-slate' }, { op: adj >= 0 ? '+' : '−' },
                  { v: n0(Math.abs(adj)), l: 'weather adjustment', c: adj >= 0 ? 'text-coral' : 'text-ice-hi' },
                ].map((p, i) => 'op' in p ? <span key={i} className="display pb-6 text-[28px] leading-none text-ink-2">{p.op}</span> : (
                  <div key={i}><div className={`display num text-[34px] leading-none ${p.c}`}>{p.v}</div><div className="mt-1.5 text-[12.5px] text-ink-3">{p.l}</div></div>
                ))}
              </div>
            )}
          </div>
          <div className="flex flex-col justify-between gap-6 lg:border-l lg:border-line lg:pl-8">
            <div>
              <div className="text-ink-2">Selected model</div>
              <div className="mt-1 text-[19px] leading-snug text-ink">{win.display_name}</div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                <Chip tone={win.eligibility === 'eligible' ? 'good' : 'warn'}>{win.eligibility}</Chip>
                <Chip>{eligible} of {result.scores.length} candidates usable</Chip>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="relative"><Ring value={win.fitness ?? 0} /><span className="num absolute inset-0 grid place-items-center text-[15px] text-ink">{nf(win.fitness)}</span></div>
              <div className="text-[13.5px] text-ink-3"><div className="text-ink-2">Model fitness score, out of 100</div>A ranking rubric for choosing between models. It is not a probability that the prediction is right.</div>
            </div>
            <Button onClick={() => go('models')} className="self-start">See why it was chosen <ArrowRight size={15} /></Button>
          </div>
        </Liquid>

        {worse && (
          <Notice tone="warn">On held-out data this model's free-running error was {nf(win.metrics.MAPE_percent)} %, worse than the {s.benchmarks?.calendar_baseline_only ? 'calendar baseline alone' : 'same-hour-last-week benchmark'} ({nf(bench.MAPE_percent)} %).
            The models are most reliable one hour ahead with observed load.</Notice>
        )}

        <Panel title="Predicted load" hint={s.interval_basis ? `Band: ${s.interval_basis}.` : undefined}
          action={<Button variant="quiet" onClick={() => go('analytics')}>Explore in analytics <ArrowRight size={15} /></Button>}>
          {chart && <Chart option={chart} height={360} label="Predicted hourly load with its 90 % interval and observed load" />}
        </Panel>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Peak load" value={`${n0(s.peak_power_mw)} MW`} sub={fmtHour(s.peak_timestamp)} />
          <Stat label="Lowest load" value={`${n0(s.min_power_mw)} MW`} sub={`${s.hours_predicted} of ${s.hours_requested} hours predicted`} />
          {s.observed_MAPE_percent != null
            ? <Stat label="Error against observed load" value={`${nf(s.observed_MAPE_percent, 2)} %`} sub={`MAE ${nf(s.observed_MAE_MW)} MW, observed total ${fmtEnergy(s.observed_total_energy_mwh)}`} />
            : <Stat label="Error against observed load" value="Not available" sub={req.scenario ? 'A scenario has no ground truth' : 'No observed load for every predicted hour'} />}
          <button onClick={() => go('data')} className="text-left"><Stat label="Data quality" value={notices ? `${notices} notice${notices > 1 ? 's' : ''}` : 'No notices'} sub={`Request ${result.id}, ${fmtDay(req.start)}`} /></button>
        </div>
      </div>
    </div>
  )
}
