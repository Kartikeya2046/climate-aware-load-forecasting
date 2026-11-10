import { Download } from 'lucide-react'
import { Chip, cx, Notice, PageHead, Panel, Scroll, Stat, type Tone } from '../components/ui'
import { api } from '../lib/api'
import { n0 } from '../lib/format'
import { useStore } from '../lib/store'
import { NeedsResult } from './shared'

const SEV: Record<string, Tone> = { fatal: 'bad', recoverable: 'warn', warning: 'neutral' }
const SEV_LABEL: Record<string, string> = { fatal: 'error', recoverable: 'correction', warning: 'warning' }

const Get = ({ href, children, primary }: { href: string; children: string; primary?: boolean }) => (
  <a href={href} download className={cx('inline-flex h-10 items-center gap-2 rounded-xl px-4 text-[14px] transition active:scale-[0.98]',
    primary ? 'bg-amber-hi font-medium text-[#1a1203] hover:bg-[#ffc566]' : 'border border-line-2 bg-white/[0.04] text-ink hover:bg-white/[0.08]')}>
    <Download size={15} />{children}
  </a>
)

export default function DataReport() {
  const { result, meta } = useStore()
  if (!result) return <NeedsResult title="Data and report" />
  const q = result.quality
  const cols = Object.entries((q?.columns ?? {}) as Record<string, any>)
  return (
    <div>
      <PageHead title="Data and report">What went into this prediction, every change made to it on the way, and the files to take with you.</PageHead>
      <div className="grid gap-5">
        <Panel title="Take it with you" hint={`Request ${result.id}. The HTML report is self-contained and prints cleanly.`}>
          <div className="flex flex-wrap gap-2.5">
            {result.predictions && <Get primary href={api.exportUrl(result.id, 'csv')}>Hourly predictions, CSV</Get>}
            <Get href={api.exportUrl(result.id, 'html')}>Full report, HTML</Get>
            <Get href={api.exportUrl(result.id, 'json')}>Complete response, JSON</Get>
            {result.has_processed && <Get href={api.exportUrl(result.id, 'processed')}>Processed input, CSV</Get>}
            {result.has_original && <Get href={api.exportUrl(result.id, 'original')}>Original upload, unchanged</Get>}
          </div>
        </Panel>

        {result.request.input_mode === 'manual' && <Notice>Manual entry: one range-checked observation applied as a scenario. There was no file to process.</Notice>}
        {q && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Rows read" value={n0(q.rows_uploaded)} sub={result.source === 'recorded' ? 'From the recorded project data' : result.upload_name} />
            <Stat label="Hourly rows after processing" value={n0(q.rows_hourly)} sub={q.span ? `${q.span[0].slice(0, 10)} to ${q.span[1].slice(0, 10)}` : undefined} />
            <Stat label="Duplicate timestamps" value={n0(q.duplicate_timestamps ?? 0)} sub="The first occurrence is kept" />
            <Stat label="Missing hours" value={n0(q.missing_hours ?? 0)} sub="Weather gaps of 6 h or less are forward-filled" />
          </div>
        )}

        {cols.length > 0 && (
          <Panel title="Each variable" hint="Implausible values fall outside the physical ranges used in training and are treated as missing. Unusual values are flagged and kept: a heatwave is a valid condition.">
            <Scroll max={420}><table className="data">
              <thead><tr><th>Variable</th><th className="r">Missing</th><th className="r">Implausible</th><th className="r">Forward-filled</th><th className="r">Unusual, kept</th></tr></thead>
              <tbody>{cols.map(([k, c]) => (
                <tr key={k}><td className="strong">{meta.variables[k]?.label ?? k}</td>
                  {[c.missing, c.invalid, c.filled, c.unusual_flagged].map((n, i) => <td key={i} className={cx('r', n > 0 && 'strong')}>{n0(n)}</td>)}</tr>
              ))}</tbody>
            </table></Scroll>
          </Panel>
        )}

        <div className="grid gap-5 lg:grid-cols-2">
          <Panel title="Conditions found">
            {result.issues.length ? (
              <ul className="grid gap-3">{result.issues.map((i, k) => (
                <li key={k} className="grid grid-cols-[92px_minmax(0,1fr)] gap-3 text-[14px] text-ink-2"><span><Chip tone={SEV[i.severity]}>{SEV_LABEL[i.severity]}</Chip></span>{i.message}</li>
              ))}</ul>
            ) : <p className="text-ink-3">None. The inputs passed every check without a correction.</p>}
          </Panel>
          <Panel title="Processing log" hint="Unit conversions, imputation, derived features and the history the models were given, in order.">
            <ol className="grid gap-3">{result.processing_log.map((e, k) => (
              <li key={k} className="grid grid-cols-[22px_minmax(0,1fr)] gap-3 text-[14px]">
                <span className="num text-ink-3">{k + 1}</span>
                <span className="text-ink-2"><span className="text-ink">{e.step}: {e.action}</span><br /><span className="text-[13px] text-ink-3">{e.detail}{e.rows_affected ? `. ${n0(e.rows_affected)} rows` : ''}</span></span>
              </li>
            ))}</ol>
          </Panel>
        </div>

        {result.mapping && (
          <Panel title="How the columns were read">
            <div className="flex flex-wrap gap-2">{Object.entries(result.mapping).map(([c, v]) => (
              <span key={c} className="well px-3 py-1.5 text-[13px]"><span className="text-ink">{c}</span><span className="mx-2 text-ink-3">as</span>
                <span className={v ? 'text-ink-2' : 'text-ink-3'}>{v === 'timestamp' ? 'Timestamp' : v ? meta.variables[v]?.label ?? v : 'ignored'}</span></span>
            ))}</div>
          </Panel>
        )}
      </div>
    </div>
  )
}
