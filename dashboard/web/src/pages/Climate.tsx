import { useEffect, useMemo, useState } from 'react'
import { Trace } from '../App'
import Chart from '../components/Chart'
import { Chip, Meter, Notice, PageHead, Panel, Scroll } from '../components/ui'
import { api } from '../lib/api'
import * as charts from '../lib/charts'
import { cityShort, nf } from '../lib/format'
import { useNames, useStore } from '../lib/store'

type Space = Awaited<ReturnType<typeof api.climateSpace>>

function Candidates() {
  const { result, meta } = useStore()
  const names = useNames()
  const scored = result!.scores.filter(s => s.training_domain?.climate?.variables)
  const [pick, setPick] = useState(scored.find(s => s.selected)?.model_id ?? scored[0]?.model_id)
  const [v, setV] = useState<string>()
  const [detail, setDetail] = useState<any>(null)
  useEffect(() => { setDetail(null); if (pick) api.climateDetail(result!.id, pick).then(setDetail).catch(() => setDetail(false)) }, [pick, result])
  const s = scored.find(x => x.model_id === pick)
  const cl = s?.training_domain.climate
  const vars = cl ? Object.keys(cl.variables) : []
  const cur = v && vars.includes(v) ? v : vars[0]
  const o = useMemo(() => {
    if (!detail || !cur || !detail.hist[cur]) return null
    const lab = (c: string) => meta.variables[c].label
    return {
      dist: charts.distributions(detail.hist[cur], lab(cur), meta.variables[cur].unit, `${cityShort(names.city[detail.reference_city])} training weather`),
      a: detail.corr && charts.correlation(detail.corr.variables.map(lab), detail.corr.input),
      b: detail.corr && charts.correlation(detail.corr.variables.map(lab), detail.corr.reference),
    }
  }, [detail, cur, meta, names])

  return (
    <>
      <Panel title="Your weather against each model's training climate"
        hint="The input weather is compared with the weather each model learned from, for the same months give or take one. This carries 25 % of the fitness score.">
        <div className="grid gap-2.5">
          {result!.scores.map(x => {
            const c = x.training_domain?.climate
            const on = x.model_id === pick, can = !!c?.variables
            return (
              <button key={x.model_id} disabled={!can} onClick={() => setPick(x.model_id)} aria-pressed={on}
                className={`well grid grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3 text-left transition disabled:opacity-50 ${on ? '!border-amber-hi/50' : 'hover:!border-line-2'}`}>
                <div className="min-w-0"><div className="truncate text-ink">{x.display_name}</div>
                  <div className="truncate text-[13px] text-ink-3">{c?.reference ?? (x.eligibility === 'ineligible' ? 'Not scored: the model is ineligible' : c?.note ?? 'No comparison')}</div></div>
                <Meter value={c?.overall ?? null} color="var(--color-ice)" />
                <div className="num w-[150px] text-right"><span className="text-[18px] text-ink">{c?.overall == null ? '–' : nf(c.overall)}</span>
                  {c?.classification && <div className="text-[12.5px] text-ink-3">{c.classification}{c.confidence === 'low' ? ', low confidence' : ''}</div>}</div>
              </button>
            )
          })}
        </div>
      </Panel>

      {s && cl && (
        <>
          {cl.note && <Notice tone="warn">{cl.note}</Notice>}
          <div className="grid gap-5 lg:grid-cols-[1.1fr_1fr]">
            <Panel title="Variable by variable" hint={cl.method === 'distribution'
              ? `Distribution ${nf(cl.distribution_score)}, correlation structure ${cl.correlation_score == null ? 'unavailable' : nf(cl.correlation_score)}, multivariate ${nf(cl.multivariate_score)}. Weighted 60, 20 and 20 % into ${nf(cl.overall)}.`
              : 'A single observation cannot support distribution statistics. Each value is placed in the training distribution instead.'}>
              <Scroll max={380}><table className="data">
                <thead>{cl.method === 'distribution'
                  ? <tr><th>Variable</th><th className="r">Score</th><th className="r">Your mean</th><th className="r">Training mean</th><th className="r">KS</th></tr>
                  : <tr><th>Variable</th><th className="r">Score</th><th className="r">Your value</th><th className="r">Training percentile</th></tr>}</thead>
                <tbody>{vars.map(k => {
                  const x = cl.variables[k]
                  return (
                    <tr key={k} onClick={() => setV(k)} className={`cursor-pointer ${k === cur ? 'bg-white/[0.04]' : 'hover:bg-white/[0.025]'}`}>
                      <td className="strong">{meta.variables[k].label}</td><td className="r strong">{nf(x.score)}</td>
                      {cl.method === 'distribution' ? <><td className="r">{nf(x.input_mean, 2)}</td><td className="r">{nf(x.reference_mean, 2)}</td><td className="r">{nf(x.KS, 2)}</td></>
                        : <><td className="r">{nf(x.value, 2)}</td><td className="r">{nf(x.training_percentile, 0)}</td></>}
                    </tr>
                  )
                })}</tbody>
              </table></Scroll>
            </Panel>
            <Panel title={cur ? `${meta.variables[cur].label}: your weather and the training weather` : 'Distribution'} hint="Select a row on the left to change the variable.">
              {detail === false ? <Notice>No distribution is available for this model.</Notice> : o ? <Chart option={o.dist} height={330} label="Distribution of the input weather against the training weather" /> : <div className="grid h-[330px] place-items-center"><Trace /></div>}
            </Panel>
          </div>
          {o?.a && (
            <div className="grid gap-5 lg:grid-cols-2">
              <Panel title="How your weather variables move together"><Chart option={o.a} height={340} label="Correlation matrix of the input weather" /></Panel>
              <Panel title={`The same, in ${cityShort(names.city[detail.reference_city])}'s training weather`}><Chart option={o.b} height={340} label="Correlation matrix of the training weather" /></Panel>
            </div>
          )}
        </>
      )}
    </>
  )
}

export default function Climate() {
  const { result, meta } = useStore()
  const names = useNames()
  const [space, setSpace] = useState<Space | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => { api.climateSpace().then(setSpace).catch(() => setFailed(true)) }, [])
  const short = useMemo(() => Object.fromEntries(meta.cities.map(c => [c.id, cityShort(c.name)])), [meta])
  const o = useMemo(() => space && {
    pca: charts.pca(space.points, short, result?.projection ?? null, space.explained),
    matrix: charts.matrix(space.cities.map(c => short[c]), space.matrix),
  }, [space, short, result])
  const a = result?.analog
  const tried = Object.fromEntries((a?.tried ?? []).map((t: any) => [t.city, t]))

  return (
    <div>
      <PageHead title="Climate">
        A model transfers best to weather that resembles what it was trained on. Similarity blends how each variable is distributed (60 %),
        how the variables move together (20 %) and how far apart the climates sit overall (20 %), on a scale of 0 to 100.
      </PageHead>
      <div className="grid gap-5">
        {a && (
          <Panel title={`${a.name} against the trained cities`} hint="Cities are tried from most to least similar. The first one with an eligible model stands in.">
            <Scroll><table className="data">
              <thead><tr><th>City</th><th className="w-[30%]">Similarity</th><th>Class</th><th>Outcome</th></tr></thead>
              <tbody>{a.ranking.map((r: any) => (
                <tr key={r.city}><td className="strong">{names.city[r.city]}</td>
                  <td><div className="flex items-center gap-3"><Meter value={r.similarity} color="var(--color-ice)" className="flex-1" /><span className="num w-10 text-right text-ink">{nf(r.similarity)}</span></div></td>
                  <td>{r.classification}</td>
                  <td>{tried[r.city]?.status === 'ok' ? <Chip tone="good">Used with {names.model[tried[r.city].model]?.split(' · ')[0]}</Chip> : tried[r.city] ? <span className="text-[13px]">Skipped. {tried[r.city].reason}</span> : 'Not needed'}</td></tr>
              ))}</tbody>
            </table></Scroll>
          </Panel>
        )}
        {result?.scores.length ? <Candidates /> : <Notice>Run a prediction to see how its weather compares with each model's training climate. The maps below need no prediction.</Notice>}

        <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
          <Panel title="Six cities in one weather space" hint={`Eight standardised weather variables reduced to two principal components, fitted once on 3,000 recorded hours per city.${result?.projection ? ' Your input hours are the amber points.' : ''} Two components cannot show every difference between climates.`}>
            {o ? <Chart option={o.pca} height={440} label="Principal component map of recorded weather for the six cities" /> : <Loading failed={failed} h={440} />}
          </Panel>
          <Panel title="City to city similarity" hint="The same score computed between full records of every pair of cities.">
            {o ? <Chart option={o.matrix} height={440} label="Matrix of climate similarity between the six cities" /> : <Loading failed={failed} h={440} />}
          </Panel>
        </div>
      </div>
    </div>
  )
}

const Loading = ({ failed, h }: { failed: boolean; h: number }) => (
  <div className="grid place-items-center gap-3 text-[14px] text-ink-3" style={{ height: h }}>
    {failed ? 'The climate maps could not be computed.' : <div className="flex flex-col items-center gap-3"><Trace />Comparing the recorded climates. The first time takes a few seconds.</div>}
  </div>
)
