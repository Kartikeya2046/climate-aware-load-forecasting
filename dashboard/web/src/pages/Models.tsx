import { ChevronDown } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { Chip, cx, Disclosure, Meter, Notice, PageHead, Panel, Scroll, type Tone } from '../components/ui'
import type { Score } from '../lib/api'
import { n0, nf } from '../lib/format'
import { useNames, useStore } from '../lib/store'
import { NeedsResult } from './shared'

const ELIG: Record<string, Tone> = { eligible: 'good', conditional: 'warn', ineligible: 'bad' }
const MODES: Record<string, string> = { one_step: 'One hour ahead', forecast: 'Free-running', forecast_no_history: 'Free-running, no load history' }

/** Fitness as the sum of its parts: each segment is effective weight × score, so the filled length is the fitness itself. */
function Contributions({ s, labels }: { s: Score; labels: Record<string, string> }) {
  const parts = s.components.filter(c => c.score != null)
  return (
    <div className="flex h-2.5 w-full gap-[2px] overflow-hidden rounded-full bg-white/[0.06]" role="img"
      aria-label={`Fitness ${nf(s.fitness)} of 100: ` + parts.map(c => `${labels[c.name]} ${nf(c.score! * c.effective_weight)}`).join(', ')}>
      {parts.map((c, i) => (
        <div key={c.name}
          title={`${labels[c.name]}: ${nf(c.score, 0)} × ${nf(c.effective_weight * 100, 0)} % = ${nf(c.score! * c.effective_weight)} points`}
          className={cx('grow h-full first:rounded-l-full', s.selected ? 'bg-amber' : 'bg-slate/70')} style={{ width: `${c.score! * c.effective_weight}%`, opacity: 1 - i * 0.1, animationDelay: `${i * 60}ms` }} />
      ))}
    </div>
  )
}

function Candidate({ s }: { s: Score }) {
  const { meta, result } = useStore()
  const names = useNames()
  const [open, setOpen] = useState(s.selected)
  const loc = result!.request.location_id
  const ev = meta.evidence.models?.[s.model_id]?.[loc]
  const reported = meta.evidence.research_reported?.[s.model_id]
  const info = meta.models.find(m => m.id === s.model_id)
  return (
    <div className={cx('well overflow-hidden', s.selected && '!border-amber-hi/45')}>
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-4 gap-y-3 px-4 py-4 text-left sm:px-5">
        <span className={cx('display num w-8 text-[30px] leading-none', s.selected ? 'text-amber-hi' : 'text-ink-3')}>{s.rank ?? '–'}</span>
        <span className="min-w-0">
          <span className="block truncate text-[16px] text-ink">{s.display_name}</span>
          <span className="mt-1.5 flex flex-wrap gap-1.5">
            <Chip tone={ELIG[s.eligibility]}>{s.eligibility}</Chip>
            {s.selected && <Chip tone="accent">selected</Chip>}
            {s.near_tie && !s.selected && <Chip tone="warn">near tie</Chip>}
          </span>
        </span>
        <span className="flex items-center gap-3">
          <span className="num text-right"><span className="text-[24px] text-ink">{s.fitness == null ? '–' : nf(s.fitness)}</span>{s.fitness != null && <span className="text-[13px] text-ink-3"> / 100</span>}</span>
          <ChevronDown size={16} className={cx('text-ink-3 transition-transform', open && 'rotate-180')} aria-hidden />
        </span>
        {s.fitness != null && <span className="col-span-3 sm:col-span-2 sm:col-start-2"><Contributions s={s} labels={meta.labels} /></span>}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.28 }} className="overflow-hidden">
            <div className="grid gap-5 border-t border-line px-4 py-5 sm:px-5">
              {s.reasons.length > 0 && <Notice tone={s.eligibility === 'ineligible' ? 'bad' : 'warn'}>{s.reasons.join('. ')}</Notice>}
              {s.fitness != null && (
                <Scroll max={520}>
                  <table className="data">
                    <thead><tr><th>Component</th><th className="w-[22%]">Score, 0 to 100</th><th className="r">Weight</th><th className="r">Points</th><th>Evidence</th></tr></thead>
                    <tbody>
                      {s.components.map(c => (
                        <tr key={c.name}>
                          <td className="strong whitespace-nowrap">{meta.labels[c.name]}{c.confidence === 'low' && <span className="ml-2 text-[12px] text-warn">low confidence</span>}</td>
                          <td>{c.score == null ? <span className="text-ink-3">Unavailable, weight redistributed</span>
                            : <div className="flex items-center gap-3"><Meter value={c.score} className="min-w-[70px] flex-1" color={s.selected ? 'var(--color-amber)' : 'var(--color-slate)'} /><span className="num w-8 text-right text-ink">{nf(c.score, 0)}</span></div>}</td>
                          <td className="r">{nf(c.effective_weight * 100, 0)} %</td>
                          <td className="r strong">{c.score == null ? '–' : nf(c.score * c.effective_weight)}</td>
                          <td className="min-w-[240px] text-[13px]">{c.detail}</td>
                        </tr>
                      ))}
                      <tr><td className="strong" colSpan={3}>Fitness, the sum of the points</td><td className="r strong">{nf(s.fitness, 2)}</td><td /></tr>
                    </tbody>
                  </table>
                </Scroll>
              )}
              {ev && (
                <div>
                  <h3 className="text-[14px] text-ink">Held-out evidence in {names.city[loc]}</h3>
                  <p className="mt-0.5 text-[13px] text-ink-3">{ev.holdout?.description}</p>
                  <Scroll><table className="data mt-2">
                    <thead><tr><th>Mode</th><th className="r">MAPE</th><th className="r">MAE</th><th className="r">R²</th><th className="r">Naive MAPE</th><th className="r">Interval coverage</th><th className="r">Hours</th></tr></thead>
                    <tbody>{Object.entries(ev).filter(([, v]: any) => v?.naive).map(([k, v]: any) => (
                      <tr key={k}><td className="strong">{MODES[k] ?? k}</td><td className="r">{nf(v.MAPE_percent, 2)} %</td><td className="r">{nf(v.MAE_MW)} MW</td><td className="r">{nf(v.R2, 3)}</td>
                        <td className="r" title={v.naive_definition}>{nf(v.naive.MAPE_percent, 2)} %</td><td className="r">{v.calibration?.coverage == null ? '–' : `${nf(v.calibration.coverage * 100, 0)} %`}</td><td className="r">{n0(v.n)}</td></tr>
                    ))}</tbody>
                  </table></Scroll>
                </div>
              )}
              <div className="grid gap-1 text-[13px] text-ink-3">
                {reported && <p>Reported by the research notebook ({reported.experiment}): MAPE {nf(reported.MAPE_percent, 2)} %, MAE {nf(reported.MAE_MW)} MW, R² {nf(reported.R2, 4)}.</p>}
                <p>Version {s.version}. Artifacts: {Object.values(info?.artifacts ?? {}).length} files under {Object.values(info?.artifacts ?? {})[0]?.split('/').slice(0, 2).join('/')}.</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function Models() {
  const { result, meta } = useStore()
  const names = useNames()
  if (!result) return <NeedsResult title="Model selection" />
  const bench = meta.evidence.benchmarks?.[result.request.location_id] ?? {}
  const BENCH: Record<string, string> = { naive_24h: 'Same hour, previous day', naive_168h: 'Same hour, previous week', calendar_baseline_only: 'Calendar baseline alone, no weather' }
  return (
    <div>
      <PageHead title="Model selection">
        Each model is first checked for eligibility: its files load, its inputs are present, it has a pathway for this location. Those that pass are scored on six
        weighted components. The score ranks models against each other; it is not a probability that a prediction is correct.
      </PageHead>
      <div className="grid gap-5">
        <Panel title={`Candidates for ${names.city[result.request.location_id] ?? result.request.location_id}`}
          hint="The bar under each model is its fitness built from its parts: every segment is one component's score times its weight. Open a model for the figures and the evidence behind them.">
          <div className="grid gap-3">{result.scores.map(s => <Candidate key={s.model_id} s={s} />)}</div>
        </Panel>

        <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr]">
          <Panel title="Why this result">
            <ul className="grid gap-3">
              {result.selection_explanation.map((t, i) => (
                <li key={i} className="flex gap-3 text-[14.5px] text-ink-2"><span className={cx('mt-[9px] h-1.5 w-1.5 shrink-0 rounded-full', i === 0 ? 'bg-amber-hi' : 'bg-ink-3')} />{t}</li>
              ))}
            </ul>
          </Panel>
          <Panel title="How the score is weighted" hint="A component with no evidence is left out and the rest are rescaled. It is never counted as perfect.">
            <div className="grid gap-2.5">
              {meta.components.map(c => (
                <div key={c} className="grid grid-cols-[minmax(0,1fr)_90px_40px] items-center gap-3 text-[13.5px]">
                  <span className="truncate text-ink-2">{meta.labels[c]}</span><Meter value={meta.weights[c] * 100 / 0.35} color="var(--color-slate)" /><span className="num text-right text-ink">{nf(meta.weights[c] * 100, 0)} %</span>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        {Object.keys(bench).length > 0 && (
          <Disclosure title="Reference benchmarks at this location, with no learned weather model">
            <table className="data"><thead><tr><th>Benchmark</th><th className="r">MAPE</th><th className="r">MAE</th><th className="r">R²</th><th className="r">Hours</th></tr></thead>
              <tbody>{Object.entries(bench).map(([k, v]: any) => <tr key={k}><td className="strong">{BENCH[k] ?? k}</td><td className="r">{nf(v.MAPE_percent, 2)} %</td><td className="r">{nf(v.MAE_MW)} MW</td><td className="r">{nf(v.R2, 3)}</td><td className="r">{n0(v.n)}</td></tr>)}</tbody></table>
          </Disclosure>
        )}
      </div>
    </div>
  )
}
