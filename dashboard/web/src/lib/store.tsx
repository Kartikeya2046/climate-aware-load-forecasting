import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api, type Meta, type Pending, type Result, type Upload } from './api'

export const ROUTES = [
  { id: 'overview', label: 'Overview' },
  { id: 'workspace', label: 'Workspace' },
  { id: 'models', label: 'Model selection' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'climate', label: 'Climate' },
  { id: 'data', label: 'Data and report' },
  { id: 'research', label: 'Research' },
] as const
export type Route = (typeof ROUTES)[number]['id']

export type Form = {
  city: string; name: string; start: string; end: string; mode: 'one_step' | 'forecast'; source: 'recorded' | 'manual' | 'csv'
  values: Record<string, string>; valuesKey: string; units: Record<string, string>; useRecorded: boolean
  upload: Upload | null; mapping: Record<string, string | null>; csvUnits: Record<string, string>; tsKind: string; context: string
}
const EXAMPLE = { city: 'atlanta', start: '2023-07-10T00:00', end: '2023-07-12T23:00', mode: 'one_step', source: 'recorded' } as const
const initialForm: Form = {
  ...EXAMPLE, name: '', values: {}, valuesKey: '', units: {}, useRecorded: true, upload: null, mapping: {}, csvUnits: {}, tsKind: 'local_standard', context: 'provided',
}

type Store = {
  meta: Meta; result: Result | null; route: Route; go: (r: Route) => void
  form: Form; setForm: (p: Partial<Form>) => void; pending: Pending | null; setPending: (p: Pending | null) => void
  busy: string | null; error: string | null; setError: (e: string | null) => void
  check: () => Promise<Pending | null>; run: (confirmed: boolean) => Promise<void>; runExample: () => Promise<void>; reset: () => Promise<void>
}
const Ctx = createContext<Store>(null as never)
export const useStore = () => useContext(Ctx)

const readRoute = (): Route => {
  const h = location.hash.replace(/^#\/?/, '')
  return ROUTES.some(r => r.id === h) ? (h as Route) : 'overview'
}
const validateBody = (f: Form) => ({
  city: f.city, start: f.start.replace('T', ' '), end: f.end.replace('T', ' '), mode: f.mode, source: f.source,
  values: Object.fromEntries(Object.entries(f.values).map(([k, v]) => [k, Number(v)])), units: f.source === 'csv' ? f.csvUnits : f.units,
  use_recorded: f.useRecorded, upload_id: f.upload?.upload_id, mapping: f.mapping, ts_kind: f.tsKind, context: f.context,
})

export function StoreProvider({ meta, children }: { meta: Meta; children: ReactNode }) {
  const [result, setResult] = useState<Result | null>(null)
  const [route, setRoute] = useState<Route>(readRoute)
  const [form, setFormState] = useState<Form>(initialForm)
  const [pending, setPending] = useState<Pending | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const on = () => { setRoute(readRoute()); scrollTo({ top: 0 }) }
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  useEffect(() => { // a page reload keeps the last prediction the server still holds
    if (meta.last_result) api.result(meta.last_result).then(setResult).catch(() => {})
  }, [meta])

  const go = useCallback((r: Route) => { location.hash = `/${r}` }, [])
  const setForm = useCallback((p: Partial<Form>) => { setFormState(f => ({ ...f, ...p })); setPending(null) }, [])

  const guard = async <T,>(label: string, fn: () => Promise<T>): Promise<T | null> => {
    setBusy(label); setError(null)
    try { return await fn() } catch (e) { setError((e as Error).message); return null } finally { setBusy(null) }
  }
  const finish = (r: Result) => { setResult(r); setPending(null); go('overview') }

  const value = useMemo<Store>(() => ({
    meta, result, route, go, form, setForm, pending, setPending, busy, error, setError,
    check: () => guard('Checking inputs', async () => { const p = await api.validate(validateBody(form)); setPending(p); return p }),
    run: async confirmed => {
      if (!pending?.pending_id) return
      await guard('Scoring the models and running inference', async () => finish(await api.run(pending.pending_id!, form.name, confirmed)))
    },
    runExample: async () => {
      await guard('Running the Atlanta example', async () => {
        const p = await api.validate(validateBody({ ...initialForm, ...EXAMPLE }))
        if (!p.pending_id) throw new Error('The example inputs failed validation.')
        finish(await api.run(p.pending_id, '', true))
      })
    },
    reset: async () => { await guard('Resetting', () => api.reset()); setResult(null); setPending(null); setFormState(initialForm); go('overview') },
  }), [meta, result, route, form, pending, busy, error, go, setForm])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

/** Convenience lookups shared by the pages. */
export function useNames() {
  const { meta } = useStore()
  return useMemo(() => ({
    city: Object.fromEntries(meta.cities.map(c => [c.id, c.name])) as Record<string, string>,
    model: Object.fromEntries(meta.models.map(m => [m.id, m.display_name])) as Record<string, string>,
  }), [meta])
}
