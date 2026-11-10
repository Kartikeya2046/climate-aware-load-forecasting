import { Chip, Disclosure, Liquid, Meter, Notice, PageHead, Panel, Scroll } from '../components/ui'
import { cityShort, n0, nf } from '../lib/format'
import { useNames, useStore } from '../lib/store'
import { Equation } from './Overview'

const DEFINITIONS: Record<string, string> = {
  validation_performance: 'Skill against the naive benchmark on held-out hours for the same location and mode. 100 is perfect, 50 is no better than naive.',
  climate_similarity: "How closely the input weather resembles the model's training weather for the same months.",
  feature_compatibility: 'Whether each required input was supplied directly, converted from another unit, or derived.',
  input_data_quality: 'Share of the input free of missing, filled, implausible and duplicate values.',
  temporal_compatibility: 'Observed load history, the length of the horizon, and how far the dates sit from the evaluated period.',
  calibration_reliability: 'How well the 90 % interval held on the second half of the holdout, and how stable the error was between halves.',
}

export default function Research() {
  const { meta } = useStore()
  const names = useNames()
  const ev = meta.evidence
  const rows = meta.models.flatMap(m => meta.cities.map(c => ({ m, c, e: ev.models?.[m.id]?.[c.id] })).filter(r => r.e?.one_step))
  const ws = meta.weights_study
  const SETS: Record<string, string> = { configured: 'Configured weights', validation_only: 'Validation only', climate_only: 'Climate only', equal: 'Equal weights' }

  return (
    <div>
      <PageHead title="Research">The models behind the platform, what they were measured on, and where they stop being reliable.</PageHead>
      <div className="grid gap-5">
        <Liquid className="grid gap-8 p-6 sm:p-10 lg:grid-cols-[1fr_1.15fr]">
          <div>
            <h2 className="text-[clamp(30px,4.4vw,54px)] leading-[1.1]"><Equation /></h2>
            <p className="mt-5 max-w-[52ch] text-ink-2">The two-branch models split demand in two. One branch learns the rhythm of the calendar. The other reads a week of weather and learns how far it moves the load off that rhythm.</p>
          </div>
          <dl className="grid content-center gap-4">
            {[['text-amber-hi', 'E', 'Predicted load', 'Hourly average, in megawatts. Over one hour, megawatts and megawatt-hours are the same number.'],
              ['text-slate', 'E baseline', 'Calendar branch', 'A small network fed only hour, weekday, season and trend. One is trained for each city, so it carries that grid’s scale.'],
              ['text-ice-hi', 'ΔE weather', 'Weather branch', 'A temporal convolutional network reading 168 hours of weather and residual history. It predicts the next hour’s log residual, which becomes megawatts above or below the baseline.']].map(([c, sym, t, d]) => (
              <div key={t} className="grid grid-cols-[92px_minmax(0,1fr)] gap-4">
                <dt className={`display italic text-[20px] leading-tight ${c}`}>{sym}</dt>
                <dd><div className="text-ink">{t}</div><div className="mt-0.5 text-[13.5px] text-ink-3">{d}</div></dd>
              </div>
            ))}
          </dl>
        </Liquid>

        <div className="grid gap-5 lg:grid-cols-3">
          {meta.models.map(m => (
            <Panel key={m.id} title={m.display_name} action={<Chip tone={m.loaded ? 'good' : 'bad'}>{m.loaded ? 'loaded' : 'not loaded'}</Chip>}>
              <dl className="grid gap-2.5 text-[14px]">
                {[['Family', m.family], ['Trained on', m.trained_on.map(c => cityShort(names.city[c])).join(', ')],
                  ['Also evaluated on', m.evaluated_on.map(c => cityShort(names.city[c])).join(', ') || 'Nowhere else'],
                  ['Window', `${m.lookback_hours} h of weather, ${m.history_hours} h of load history`], ['Weather inputs', `${m.weather_inputs.length} variables`],
                  ['Splits demand', m.decomposes ? 'Yes, baseline and weather' : 'No, predicts load directly']].map(([k, v]) => (
                  <div key={k} className="grid grid-cols-[124px_minmax(0,1fr)] gap-3"><dt className="text-ink-3">{k}</dt><dd className="text-ink-2">{v}</dd></div>
                ))}
              </dl>
              <p className="mt-4 text-[13px] text-ink-3">{m.weather_inputs.map(v => meta.variables[v]?.label).join(', ')}.</p>
              {m.load_error && <div className="mt-3"><Notice tone="bad">{m.load_error}</Notice></div>}
            </Panel>
          ))}
        </div>

        <Panel title="Measured on held-out data" hint="Error on periods no model saw in training: 2023 for the two-branch models, and the final chronological split, June 2014 to December 2015, for the Dayton LSTM.">
          <Scroll max={520}><table className="data">
            <thead><tr><th>Model</th><th>City</th><th>Relation</th><th className="r">One hour ahead</th><th className="r">Naive, 24 h</th><th className="r">Free-running, 168 h</th><th className="r">Naive, 168 h</th><th className="r">R², one hour</th></tr></thead>
            <tbody>{rows.map(({ m, c, e }) => {
              const rel = m.trained_on.includes(c.id) ? 'trained here' : m.evaluated_on.includes(c.id) ? 'transferred' : 'outside research domain'
              const fr = e.forecast
              return (
                <tr key={m.id + c.id}>
                  <td className="strong">{m.display_name.split(' · ')[0]} <span className="text-ink-3">{cityShort(names.city[m.source_city])}</span></td><td className="strong">{cityShort(c.name)}</td>
                  <td><Chip tone={rel === 'trained here' ? 'good' : rel === 'transferred' ? 'accent' : 'warn'}>{rel}</Chip></td>
                  <td className="r strong">{nf(e.one_step.MAPE_percent, 2)} %</td><td className="r">{nf(e.one_step.naive?.MAPE_percent, 2)} %</td>
                  <td className={`r ${fr && fr.MAPE_percent > fr.naive?.MAPE_percent ? 'text-warn' : ''}`}>{fr ? `${nf(fr.MAPE_percent, 2)} %` : '–'}</td><td className="r">{fr ? `${nf(fr.naive?.MAPE_percent, 2)} %` : '–'}</td>
                  <td className="r">{nf(e.one_step.R2, 3)}</td>
                </tr>
              )
            })}</tbody>
          </table></Scroll>
          <div className="mt-4"><Notice tone="warn">Every model here was trained to predict one hour ahead. Fed its own predictions for a week, most do worse than repeating last week's load; those cases are shown in amber above.
            The platform reports this instead of hiding it; forecasting several hours ahead needs models trained for that horizon.</Notice></div>
        </Panel>

        <div className="grid gap-5 lg:grid-cols-[1.25fr_1fr]">
          <Panel title="The model fitness score" hint="Fitness is the weighted average of the components that have evidence. It ranks models for one request and is not a probability.">
            <div className="grid gap-4">
              {meta.components.map(c => (
                <div key={c} className="grid grid-cols-[minmax(0,1fr)_48px] gap-x-4 gap-y-1.5">
                  <div className="text-ink">{meta.labels[c]}</div><div className="num text-right text-ink">{nf(meta.weights[c] * 100, 0)} %</div>
                  <div className="col-span-2"><Meter value={(meta.weights[c] / 0.35) * 100} color="var(--color-slate)" /></div>
                  <p className="col-span-2 text-[13px] text-ink-3">{DEFINITIONS[c]}</p>
                </div>
              ))}
            </div>
          </Panel>
          <div className="grid content-start gap-5">
            {ws && (
              <Panel title="Do the weights matter?" hint={`Tested on ${ws.n_cases} held-out weekly windows, using validation evidence from earlier windows only.`}>
                <table className="data"><thead><tr><th>Weight set</th><th className="r">Picked the best model</th><th className="r">Mean regret</th></tr></thead>
                  <tbody>{Object.entries(ws.results).map(([k, v]: any) => <tr key={k}><td className="strong">{SETS[k] ?? k}</td><td className="r">{nf(v.selection_accuracy * 100)} %</td><td className="r">{nf(v.mean_regret_mape_points, 2)} pts</td></tr>)}</tbody></table>
                <p className="mt-4 text-[13.5px] text-ink-2">All four sets choose identically. Each city has at most two scorable candidates and one family dominates, so this study supports the selections but cannot tell weight sets apart.
                  The weights remain hypotheses.</p>
              </Panel>
            )}
            <Panel title="Known limits">
              <ul className="grid gap-3">{meta.limitations.map((t, i) => <li key={i} className="flex gap-3 text-[13.5px] text-ink-2"><span className="mt-[8px] h-1.5 w-1.5 shrink-0 rounded-full bg-ink-3" />{t}</li>)}</ul>
            </Panel>
          </div>
        </div>

        <Disclosure title="Every model file found in the repository" right={<span className="num text-[13px] text-ink-3">{meta.artifacts.filter(a => a.registered).length} of {n0(meta.artifacts.length)} in use</span>}>
          <p className="mb-3 max-w-[80ch] text-[13.5px] text-ink-3">Files not in use lack something needed to run them faithfully, such as saved scalers or a layer that reloads. No XGBoost model was saved by the research, so none is offered here.</p>
          <Scroll max={360}><table className="data"><tbody>
            {meta.artifacts.map(a => <tr key={a.artifact}><td className={a.registered ? 'strong' : ''} style={{ wordBreak: 'break-all' }}>{a.artifact}</td><td className="r">{a.registered ? <Chip tone="good">in use</Chip> : <Chip>not registered</Chip>}</td></tr>)}
          </tbody></table></Scroll>
        </Disclosure>
        <p className="text-[13px] text-ink-3">Evidence generated {ev.generated_utc?.slice(0, 10)} by running each model over its holdout offline.</p>
      </div>
    </div>
  )
}
