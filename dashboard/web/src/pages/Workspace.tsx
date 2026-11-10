import { FileUp, MapPin, Upload as UploadIcon } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button, Chip, cx, Disclosure, Notice, PageHead, Panel, Scroll, Tabs } from '../components/ui'
import { api } from '../lib/api'
import { cityShort, fmtDay, ms } from '../lib/format'
import { useStore } from '../lib/store'

const HOUR = 3600_000
const iso = (t: number) => new Date(t).toISOString().slice(0, 16)
const onHour = (v: string) => (v ? v.slice(0, 13) + ':00' : v)

function Location() {
  const { meta, form, setForm } = useStore()
  const pick = (id: string) => {
    const c = meta.cities.find(x => x.id === id)
    const start = c ? `${c.default_start}T00:00` : form.start
    setForm({ city: id, start, end: iso(ms(start) + 71 * HOUR), source: id === 'auto' ? 'csv' : form.source })
  }
  const city = meta.cities.find(c => c.id === form.city)
  return (
    <Panel title="Where" hint="Six cities have trained models. For anywhere else, upload its weather and the closest trained climate stands in.">
      <div role="radiogroup" aria-label="Location" className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
        {[...meta.cities, null].map(c => {
          const id = c?.id ?? 'auto', on = form.city === id
          return (
            <button key={id} role="radio" aria-checked={on} onClick={() => pick(id)}
              className={cx('well relative px-3.5 py-3 text-left transition', on ? '!border-amber-hi/60 !bg-amber-hi/[0.07]' : 'hover:!border-line-2', !c && 'col-span-2 md:col-span-2')}>
              {c ? <>
                <div className="text-[15px] text-ink">{cityShort(c.name)}</div>
                <div className="mt-0.5 truncate text-[12.5px] text-ink-3">{c.utility}</div>
                <div className="num mt-2 text-[12px] text-ink-3">{c.span[0].slice(0, 4)} to {c.span[1].slice(0, 4)}</div>
              </> : <>
                <div className="flex items-center gap-2 text-[15px] text-ink"><MapPin size={15} className="text-ice-hi" /> Another city</div>
                <div className="mt-0.5 text-[12.5px] text-ink-3">Matched by climate to a trained city. Needs a weather CSV.</div>
              </>}
            </button>
          )
        })}
      </div>
      {city ? (
        <div className="mt-4">
          <Disclosure title={`Which models can run in ${cityShort(city.name)}`} right={<span className="text-[13px] text-ink-3">{city.models.filter(m => m.applicable).length} of {city.models.length}</span>}>
            <table className="data"><tbody>
              {city.models.map(m => (
                <tr key={m.id}>
                  <td className="strong">{meta.models.find(x => x.id === m.id)?.display_name}</td>
                  <td><Chip tone={!m.applicable ? 'bad' : m.domain === 'outside research domain' ? 'warn' : 'good'}>{m.applicable ? m.domain : 'not applicable'}</Chip></td>
                  <td>{m.reason}</td>
                </tr>
              ))}
            </tbody></table>
          </Disclosure>
        </div>
      ) : (
        <div className="mt-4 max-w-sm">
          <label className="lbl" htmlFor="cname">City name, used in labels (optional)</label>
          <input id="cname" className="field" placeholder="Kirkland" value={form.name} onChange={e => setForm({ name: e.target.value })} />
        </div>
      )}
    </Panel>
  )
}

function Timeframe() {
  const { meta, form, setForm } = useStore()
  const hours = Math.round((ms(form.end) - ms(form.start)) / HOUR) + 1
  const ok = hours > 0 && hours <= meta.limits.max_horizon_hours
  const span = (h: number) => setForm({ end: iso(ms(form.start) + (h - 1) * HOUR) })
  return (
    <Panel title="When and how" hint="Timestamps are the city's local standard time, with no daylight saving. The models are hourly.">
      <div className="grid gap-4 sm:grid-cols-[1fr_1fr_auto]">
        <div><label className="lbl" htmlFor="start">First predicted hour</label>
          <input id="start" type="datetime-local" step={3600} className="field" value={form.start} onChange={e => setForm({ start: onHour(e.target.value) })} /></div>
        <div><label className="lbl" htmlFor="end">Last predicted hour</label>
          <input id="end" type="datetime-local" step={3600} className="field" value={form.end} onChange={e => setForm({ end: onHour(e.target.value) })} /></div>
        <div><span className="lbl">Set the length</span>
          <div className="flex gap-1.5">{[[24, '1 day'], [72, '3 days'], [168, '1 week']].map(([h, l]) =>
            <Button key={h} variant={hours === h ? 'ghost' : 'quiet'} className={cx('!px-3', hours === h && '!border-amber-hi/50')} onClick={() => span(h as number)}>{l}</Button>)}</div></div>
      </div>
      <p className={cx('num mt-3 text-[13.5px]', ok ? 'text-ink-3' : 'text-bad')}>
        {ok ? `${hours} hourly predictions` : `The end must be after the start, and the timeframe at most ${meta.limits.max_horizon_hours} hours.`}
      </p>
      <div className="mt-5 grid gap-2.5 sm:grid-cols-2" role="radiogroup" aria-label="Prediction mode">
        {([['one_step', 'One hour ahead', 'Each hour is predicted from observed load up to the hour before. This is how the models were trained and evaluated. Needs observed load for every hour.'],
          ['forecast', 'Free-running forecast', 'Each prediction is fed back as history, so no observed load is needed inside the timeframe. Errors compound and accuracy is markedly lower.']] as const).map(([id, t, d]) => (
          <button key={id} role="radio" aria-checked={form.mode === id} onClick={() => setForm({ mode: id })}
            className={cx('well px-4 py-3 text-left transition', form.mode === id ? '!border-amber-hi/60 !bg-amber-hi/[0.07]' : 'hover:!border-line-2')}>
            <div className="text-[15px] text-ink">{t}</div><div className="mt-1 text-[13px] text-ink-3">{d}</div>
          </button>
        ))}
      </div>
    </Panel>
  )
}

function Manual() {
  const { meta, form, setForm } = useStore()
  const [fields, setFields] = useState<{ variable: string; default: number; needed_by: string[] }[] | null>(null)
  const month = +form.start.slice(5, 7)
  useEffect(() => {
    let live = true
    setFields(null)
    api.manualFields(form.city, month).then(f => {
      if (!live) return
      setFields(f)
      const key = `${form.city}|${month}` // new city or month: start again from that month's recorded medians
      if (form.valuesKey !== key) setForm({ valuesKey: key, units: {}, values: Object.fromEntries(f.map(x => [x.variable, String(x.default)])) })
    }).catch(() => live && setFields([]))
    return () => { live = false }
  }, [form.city, month]) // eslint-disable-line
  const city = meta.cities.find(c => c.id === form.city)!
  const canUse = ms(city.span[0]) + meta.limits.max_history_hours * HOUR <= ms(form.start) && ms(form.start) <= ms(city.span[1]) + HOUR

  /** Changing the unit converts the typed value, so the physical quantity stays the same. */
  const setUnit = (k: string, to: string) => {
    const f = meta.variables[k].factors, [s0, o0] = f[form.units[k] ?? meta.variables[k].unit], [s1, o1] = f[to]
    const x = Number(form.values[k])
    setForm({ units: { ...form.units, [k]: to }, values: { ...form.values, [k]: Number.isFinite(x) ? String(+((x * s0 + o0 - o1) / s1).toFixed(3)) : form.values[k] } })
  }
  if (!fields) return <p className="text-ink-3">Loading the inputs these models need…</p>
  return (
    <div className="grid gap-4">
      <Notice>One set of weather values is applied to every hour, so this is a what-if scenario and not a weather forecast.
        The fields start at the recorded median for this month in {cityShort(city.name)}. Radiation inputs are derived from solar geometry.</Notice>
      {form.mode === 'one_step' && <Notice tone="warn">One hour ahead needs observed load for every hour, and a scenario has none. Switch to a free-running forecast.</Notice>}
      <div className="grid gap-x-4 gap-y-3.5 sm:grid-cols-2 xl:grid-cols-3">
        {fields.map(f => {
          const v = meta.variables[f.variable]
          return (
            <div key={f.variable}>
              <label className="lbl" htmlFor={f.variable} title={`${v.help} Needed by: ${f.needed_by.join(', ')}.`}>{v.label}</label>
              <div className="flex gap-1.5">
                <input id={f.variable} type="number" step="any" className="field" value={form.values[f.variable] ?? ''} onChange={e => setForm({ values: { ...form.values, [f.variable]: e.target.value } })} />
                {v.units.length > 1
                  ? <select aria-label={`${v.label} unit`} className="field !w-[104px] shrink-0" value={form.units[f.variable] ?? v.unit} onChange={e => setUnit(f.variable, e.target.value)}>
                      {v.units.map(u => <option key={u}>{u}</option>)}</select>
                  : <span className="grid w-[64px] shrink-0 place-items-center text-[13px] text-ink-3">{v.unit}</span>}
              </div>
              <div className="num mt-1 text-[12px] text-ink-3">{v.lo} to {v.hi} {v.unit}</div>
            </div>
          )
        })}
      </div>
      <label className={cx('flex items-start gap-3 text-[14px]', canUse ? 'text-ink-2' : 'text-ink-3')}>
        <input type="checkbox" className="mt-1 accent-[#f2b24a]" disabled={!canUse} checked={canUse && form.useRecorded} onChange={e => setForm({ useRecorded: e.target.checked })} />
        <span>Use recorded {cityShort(city.name)} weather and load before the start as model history.
          {!canUse && ' Not available: the start is outside the recorded period, so load history is unknown.'}</span>
      </label>
    </div>
  )
}

function Csv() {
  const { meta, form, setForm, setError } = useStore()
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [busy, setBusy] = useState(false)
  const up = form.upload
  const take = async (file?: File) => {
    if (!file) return
    setBusy(true)
    try {
      const u = await api.upload(file)
      setForm({ upload: u, mapping: Object.fromEntries(u.columns.map(c => [c.name, c.variable])), csvUnits: Object.fromEntries(u.columns.filter(c => c.variable && c.unit).map(c => [c.variable!, c.unit!])) })
    } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  const fatal = up?.issues.some(i => i.severity === 'fatal')
  const NOTE: Record<string, string> = { ambiguous: 'Several columns claim this variable. Choose one.', ambiguous_timestamp: 'Several timestamp-like columns. Choose one.', unmapped: 'Not recognised. Ignored unless you map it.' }
  const tsCount = Object.values(form.mapping).filter(v => v === 'timestamp').length

  return (
    <div className="grid gap-4">
      <button onClick={() => input.current?.click()} onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
        onDrop={e => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]) }}
        className={cx('well flex flex-col items-center gap-2 border-dashed px-6 py-8 text-center transition', over ? '!border-amber-hi !bg-amber-hi/[0.08]' : 'hover:!border-line-2')}>
        {up && !busy ? <FileUp size={22} className="text-amber-hi" /> : <UploadIcon size={22} className="text-ink-2" />}
        <div className="text-ink">{busy ? 'Reading the file…' : up ? up.name : 'Drop a weather CSV here, or choose a file'}</div>
        <div className="num text-[13px] text-ink-3">{up && !busy ? `${up.rows.toLocaleString()} rows, ${up.columns.length || Object.keys(up.head[0] ?? {}).length} columns. Drop another file to replace it.`
          : `Hourly or finer, one timestamp column, up to ${meta.limits.max_upload_mb} MB`}</div>
        <input ref={input} type="file" accept=".csv,.txt" hidden onChange={e => { take(e.target.files?.[0]); e.target.value = '' }} />
      </button>
      {form.city !== 'auto' && (
        <p className="text-[13.5px] text-ink-3">No file to hand? <a className="text-ice-hi underline-offset-4 hover:underline" href={api.sampleUrl(form.city, form.start.replace('T', ' '), form.end.replace('T', ' '))}>
          Download a sample</a> built from the recorded data for this city and timeframe, including the history the models need.</p>
      )}
      {up?.issues.map((i, k) => <Notice key={k} tone={i.severity === 'fatal' ? 'bad' : 'warn'}>{i.message}</Notice>)}
      {up && !fatal && <>
        <div>
          <h3 className="text-[15px] text-ink">What each column means</h3>
          <p className="mt-1 text-[13.5px] text-ink-3">Only exact names and known aliases are matched for you. Nothing unknown is guessed.</p>
        </div>
        <Scroll max={420}>
          <table className="data">
            <thead><tr><th>Column</th><th>Maps to</th><th>Unit</th><th>First values</th></tr></thead>
            <tbody>
              {up.columns.map(c => {
                const v = form.mapping[c.name]
                const variable = v && v !== 'timestamp' ? meta.variables[v] : null
                return (
                  <tr key={c.name}>
                    <td className="strong">{c.name}</td>
                    <td><select aria-label={`Meaning of ${c.name}`} className="field !h-9 min-w-[190px]" value={v ?? ''} onChange={e => setForm({ mapping: { ...form.mapping, [c.name]: e.target.value || null } })}>
                      <option value="">Ignore</option><option value="timestamp">Timestamp</option>
                      {Object.entries(meta.variables).map(([k, x]) => <option key={k} value={k}>{x.label}</option>)}</select></td>
                    <td>{variable && <select aria-label={`Unit of ${c.name}`} className="field !h-9 min-w-[110px]" value={form.csvUnits[v!] ?? variable.unit} onChange={e => setForm({ csvUnits: { ...form.csvUnits, [v!]: e.target.value } })}>
                      {variable.units.map(u => <option key={u}>{u}</option>)}</select>}</td>
                    <td><span className="num">{c.sample.join(', ').slice(0, 44)}</span>
                      {(c.unit_warning || (!v && NOTE[c.status])) && <div className="mt-1 text-[12.5px] text-warn">{c.unit_warning ?? NOTE[c.status]}</div>}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Scroll>
        {tsCount !== 1 && <Notice tone="bad">Map exactly one column to the timestamp.</Notice>}
        <div className="grid gap-4 sm:grid-cols-2">
          <div><label className="lbl" htmlFor="tsk">Timestamps are in</label>
            <select id="tsk" className="field" value={form.tsKind} onChange={e => setForm({ tsKind: e.target.value })}>
              <option value="local_standard">Local standard time, no daylight saving</option><option value="utc">UTC</option></select></div>
          {form.city !== 'auto' && <div><label className="lbl" htmlFor="ctx">Load history for the models</label>
            <select id="ctx" className="field" value={form.context} onChange={e => setForm({ context: e.target.value })}>
              <option value="provided">Rows before the start in this file</option><option value="repository">Recorded project data</option><option value="none">None</option></select></div>}
        </div>
      </>}
    </div>
  )
}

function Weather() {
  const { meta, form, setForm } = useStore()
  const auto = form.city === 'auto'
  const city = meta.cities.find(c => c.id === form.city)
  return (
    <Panel title="Weather" action={
      <Tabs label="Weather source" size="sm" value={form.source} onChange={source => setForm({ source })}
        items={[{ id: 'recorded', label: 'Recorded data', disabled: auto }, { id: 'manual', label: 'Manual scenario', disabled: auto }, { id: 'csv', label: 'Upload CSV' }]} />}>
      {form.source === 'recorded' && city && (
        <div className="grid gap-3">
          <p className="max-w-[72ch] text-ink-2">Uses the weather and load recorded for {cityShort(city.name)} in the research dataset, {fmtDay(city.span[0])} to {fmtDay(city.span[1])}.
            It goes through the same validation as an uploaded file, and the observed load lets you check the prediction against what happened.</p>
        </div>
      )}
      {form.source === 'manual' && city && <Manual />}
      {form.source === 'csv' && <Csv />}
    </Panel>
  )
}

function Review() {
  const { meta, form, pending, check, run, busy } = useStore()
  const [confirmed, setConfirmed] = useState(false)
  useEffect(() => setConfirmed(false), [pending])
  const city = meta.cities.find(c => c.id === form.city)
  const hours = Math.round((ms(form.end) - ms(form.start)) / HOUR) + 1
  const by = (s: string) => pending?.issues.filter(i => i.severity === s) ?? []
  const fatal = by('fatal'), fix = by('recoverable'), warn = by('warning')
  const ready = form.source !== 'csv' || !!form.upload
  const rows = [
    ['Location', city ? cityShort(city.name) : form.name || 'Another city'],
    ['Timeframe', hours > 0 ? `${hours} hours from ${fmtDay(form.start)}` : 'Invalid'],
    ['Mode', form.mode === 'one_step' ? 'One hour ahead' : 'Free-running'],
    ['Weather', { recorded: 'Recorded data', manual: 'Manual scenario', csv: form.upload?.name ?? 'No file yet' }[form.source]],
  ]
  return (
    <Panel title="Review and run" className="lg:sticky lg:top-24">
      <dl className="grid gap-2 text-[14px]">
        {rows.map(([k, v]) => <div key={k} className="flex justify-between gap-4"><dt className="text-ink-3">{k}</dt><dd className="num truncate text-right text-ink">{v}</dd></div>)}
      </dl>
      <div className="mt-5 border-t border-line pt-5">
        {!pending ? <>
          <Button variant="primary" className="w-full" disabled={!ready || !!busy || hours <= 0} onClick={check}>Check inputs</Button>
          <p className="mt-3 text-[13px] text-ink-3">Validation runs first. You see every correction and assumption before anything is predicted.</p>
        </> : (
          <div className="grid gap-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              {([[fatal.length, 'errors', 'text-bad'], [fix.length, 'corrections', 'text-warn'], [warn.length, 'warnings', 'text-ink-2']] as const).map(([n, l, c]) =>
                <div key={l} className="well py-2.5"><div className={cx('num text-[22px] leading-none', n ? c : 'text-ink-3')}>{n}</div><div className="mt-1 text-[12px] text-ink-3">{l}</div></div>)}
            </div>
            {pending.history_hours_observed != null && (
              <p className="num text-[13px] text-ink-3">{pending.rows} hourly rows with look-back. Observed load: {pending.history_hours_observed} h before the start, {pending.ground_truth_hours} h inside the timeframe.</p>
            )}
            <Scroll max={280}>
              <div className="grid gap-2">
                {fatal.map((i, k) => <Notice key={'f' + k} tone="bad">{i.message}</Notice>)}
                {fix.map((i, k) => <Notice key={'r' + k} tone="warn">Correction: {i.message}</Notice>)}
                {warn.map((i, k) => <Notice key={'w' + k}>{i.message}</Notice>)}
              </div>
            </Scroll>
            <Disclosure title={`Processing log, ${pending.log.length} steps`}>
              <ol className="grid gap-2 text-[13px] text-ink-3">{pending.log.map((e, k) => <li key={k}><span className="text-ink-2">{e.step}: {e.action}.</span> {e.detail}</li>)}</ol>
            </Disclosure>
            {fix.length > 0 && !fatal.length && (
              <label className="flex items-start gap-3 text-[14px] text-ink-2"><input type="checkbox" className="mt-1 accent-[#f2b24a]" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />
                I have reviewed the corrections above and want to continue</label>
            )}
            <Button variant="primary" className="w-full" disabled={!!fatal.length || (fix.length > 0 && !confirmed) || !!busy || !pending.pending_id} onClick={() => run(confirmed)}>Run prediction</Button>
            {fatal.length > 0 && <p className="text-[13px] text-ink-3">Fix the errors above, then check the inputs again.</p>}
          </div>
        )}
      </div>
    </Panel>
  )
}

export default function Workspace() {
  return (
    <div>
      <PageHead title="Workspace">Describe the place, the hours and the weather. Every model is then checked against these inputs before one is chosen.</PageHead>
      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="grid min-w-0 gap-5"><Location /><Timeframe /><Weather /></div>
        <Review />
      </div>
    </div>
  )
}
