import { RotateCcw, X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useState } from 'react'
import IsobarField from './components/IsobarField'
import { cx, Notice } from './components/ui'
import { api, type Meta } from './lib/api'
import { cityShort } from './lib/format'
import { ROUTES, StoreProvider, useNames, useStore } from './lib/store'
import Analytics from './pages/Analytics'
import Climate from './pages/Climate'
import DataReport from './pages/DataReport'
import Models from './pages/Models'
import Overview from './pages/Overview'
import Research from './pages/Research'
import Workspace from './pages/Workspace'

const PAGES = { overview: Overview, workspace: Workspace, models: Models, analytics: Analytics, climate: Climate, data: DataReport, research: Research }

/** A load curve being traced: the one loading indicator used everywhere. */
export const Trace = ({ size = 44 }: { size?: number }) => (
  <svg width={size} height={size * 0.5} viewBox="0 0 88 44" fill="none" aria-hidden>
    <path d="M2 34c10 0 12-22 22-22s10 16 20 16 10-24 20-24 12 18 22 18" stroke="rgba(190,210,255,0.16)" strokeWidth="2" strokeLinecap="round" />
    <path className="trace" d="M2 34c10 0 12-22 22-22s10 16 20 16 10-24 20-24 12 18 22 18" stroke="#f2b24a" strokeWidth="2" strokeLinecap="round" />
  </svg>
)

const Mark = () => (
  <svg width="26" height="26" viewBox="0 0 32 32" fill="none" aria-hidden>
    <circle cx="16" cy="16" r="14.5" stroke="rgba(190,210,255,0.22)" />
    <path d="M5 21c4-9 9-9 12-4s6 3 9-5" stroke="#f2b24a" strokeWidth="2.2" strokeLinecap="round" />
  </svg>
)

function Nav() {
  const { route, go, result, reset, busy } = useStore()
  const names = useNames()
  return (
    <div className="sticky top-3 z-40 px-3 sm:px-6">
      <nav aria-label="Main" className="glass mx-auto flex max-w-[1360px] items-center gap-3 !rounded-[18px] px-3 py-2 sm:px-4">
        <button onClick={() => go('overview')} className="flex shrink-0 items-center gap-2.5 pr-1">
          <Mark />
          <span className="hidden text-[15px] font-medium tracking-[-0.01em] text-ink lg:block">Energy Intelligence</span>
        </button>
        <div className="flex min-w-0 flex-1 gap-0.5 overflow-x-auto [scrollbar-width:none] lg:justify-center">
          {ROUTES.map(r => (
            <button key={r.id} onClick={() => go(r.id)} aria-current={route === r.id ? 'page' : undefined}
              className={cx('relative shrink-0 rounded-[11px] px-3 py-1.5 text-[14px] whitespace-nowrap transition-colors', route === r.id ? 'text-ink' : 'text-ink-3 hover:text-ink-2')}>
              {route === r.id && <motion.span layoutId="nav-pill" transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                className="absolute inset-0 rounded-[11px] border border-line-2 bg-white/[0.07] shadow-[inset_0_1px_0_rgb(255_255_255/0.1)]" />}
              <span className="relative">{r.label}</span>
            </button>
          ))}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {result?.status === 'ok' && (
            <button onClick={() => go('models')} title="Open model selection" className="num hidden items-center gap-2 rounded-full border border-line px-3 py-1 text-[12.5px] text-ink-2 hover:text-ink xl:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-hi" />
              {cityShort(names.city[result.request.location_id] ?? result.request.location_id)}, fitness {result.selected_fitness?.toFixed(1)}
            </button>
          )}
          <button onClick={reset} disabled={!!busy} title="Reset workspace: clear inputs and the current prediction" aria-label="Reset workspace"
            className="grid h-8 w-8 place-items-center rounded-lg text-ink-3 hover:bg-white/[0.06] hover:text-ink">
            <RotateCcw size={15} />
          </button>
        </div>
      </nav>
    </div>
  )
}

function Shell() {
  const { route, busy, error, setError } = useStore()
  const Page = PAGES[route]
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-[1360px] px-3 pt-8 pb-24 sm:px-6">
        <AnimatePresence mode="wait">
          <motion.div key={route} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.22 }}>
            <Page />
          </motion.div>
        </AnimatePresence>
      </main>
      <div className="pointer-events-none fixed inset-x-0 bottom-5 z-50 flex flex-col items-center gap-2 px-4" aria-live="polite">
        <AnimatePresence>
          {busy && (
            <motion.div key="busy" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 16 }}
              className="liquid pointer-events-auto flex items-center gap-4 !rounded-2xl px-5 py-3 text-[14px] text-ink">
              <Trace />{busy}
            </motion.div>
          )}
          {error && (
            <motion.div key="error" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 16 }} className="pointer-events-auto w-full max-w-xl rounded-[14px] bg-night/90 backdrop-blur">
              <Notice tone="bad">
                <div className="flex items-start justify-between gap-4"><span className="text-ink">{error}</span>
                  <button onClick={() => setError(null)} aria-label="Dismiss"><X size={16} /></button></div>
              </Notice>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  )
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { api.meta().then(setMeta).catch(e => setError(e.message)) }, [])
  return (
    <>
      <IsobarField />
      {meta ? <StoreProvider meta={meta}><Shell /></StoreProvider> : (
        <div className="grid min-h-screen place-items-center px-6">
          <div className="flex max-w-md flex-col items-center gap-5 text-center">
            {error ? <Notice tone="bad"><span className="text-ink">{error}</span></Notice> : <>
              <Trace size={88} />
              <p className="text-ink-2">Loading and health-checking the trained models. The first start takes up to half a minute.</p>
            </>}
          </div>
        </div>
      )}
    </>
  )
}
