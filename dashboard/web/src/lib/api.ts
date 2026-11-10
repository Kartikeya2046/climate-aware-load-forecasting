// Typed access to dashboard/api.py. Shapes mirror what core/service returns; nothing is computed here.

export type Frame = Record<string, any[]> // columnar: t (local standard time, 'YYYY-MM-DDTHH:MM') + numeric columns
export type Issue = { severity: 'fatal' | 'recoverable' | 'warning'; code: string; message: string; column: string | null; rows: number }
export type LogEntry = { step: string; action: string; detail: string; rows_affected: number }
export type Component = { name: string; score: number | null; weight: number; effective_weight: number; detail: string; confidence: string }
export type Score = {
  model_id: string; display_name: string; family: string; version: string
  eligibility: 'eligible' | 'conditional' | 'ineligible'; reasons: string[]; components: Component[]
  fitness: number | null; rank: number | null; selected: boolean; near_tie: boolean
  metrics: Record<string, any>; training_domain: Record<string, any>
}
export type City = {
  id: string; name: string; utility: string; utc_offset: number; lat: number; lon: number; pair: string
  span: [string, string]; default_start: string
  models: { id: string; applicable: boolean; domain: string | null; reason: string }[]
}
export type ModelInfo = {
  id: string; family: string; display_name: string; version: string; source_city: string; trained_on: string[]; evaluated_on: string[]
  lookback_hours: number; history_hours: number; weather_inputs: string[]; target: string; artifacts: Record<string, string>
  loaded: boolean; load_error: string | null; decomposes: boolean
}
export type Variable = { label: string; unit: string; units: string[]; factors: Record<string, [number, number]>; lo: number; hi: number; help: string; derivable: boolean }
export type Meta = {
  cities: City[]; models: ModelInfo[]; variables: Record<string, Variable>
  weights: Record<string, number>; labels: Record<string, string>; components: string[]
  limits: { max_horizon_hours: number; max_upload_mb: number; max_history_hours: number; near_tie_points: number }
  evidence: { generated_utc: string; models: Record<string, Record<string, any>>; benchmarks: Record<string, Record<string, any>>; research_reported: Record<string, any> }
  weights_study: any; limitations: string[]; artifacts: { artifact: string; registered: boolean }[]; last_result: string | null
}
export type Result = {
  id: string; source: 'recorded' | 'manual' | 'csv'; upload_name: string | null
  request: { location_id: string; start: string; end: string; input_mode: string; prediction_mode: 'forecast' | 'one_step'; scenario: boolean; context_source: string; aggregation: string }
  status: string; message: string; selected_model: string | null; selected_fitness: number | null
  scores: Score[]; predictions: Frame | null; alternatives: Record<string, Frame>; daily: Frame | null
  summary: Record<string, any>; selection_explanation: string[]; processing_log: LogEntry[]; issues: Issue[]
  analog: null | { name: string; city: string; similarity: number; classification: string; ranking: any[]; tried: any[] }
  projection: [number, number][] | null; quality: Record<string, any> | null; mapping: Record<string, string | null> | null
  has_processed: boolean; has_original: boolean
}
export type Upload = {
  upload_id: string; name: string; size: number; rows: number; head: Record<string, string>[]; issues: Issue[]
  columns: { name: string; variable: string | null; status: string; sample: string[]; unit: string | null; unit_warning: string | null }[]
}
export type Pending = {
  pending_id: string | null; issues: Issue[]; log: LogEntry[]; preview: Frame | null; fatal: boolean; needs_confirmation: boolean
  history_hours_observed: number | null; ground_truth_hours: number | null; rows: number | null
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new Error('The prediction server is not reachable. Start it with: python -m uvicorn api:app --port 8000')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ? String(body.detail) : `Request failed (${res.status}).`)
  }
  return res.json()
}
const post = <T,>(path: string, body: unknown) =>
  call<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  meta: () => call<Meta>('/api/meta'),
  manualFields: (city: string, month: number) =>
    call<{ variable: string; default: number; needed_by: string[] }[]>(`/api/manual-fields?city=${city}&month=${month}`),
  upload: (file: File) => call<Upload>(`/api/upload?filename=${encodeURIComponent(file.name)}`, { method: 'POST', body: file }),
  validate: (body: Record<string, unknown>) => post<Pending>('/api/validate', body),
  run: (pending_id: string, name: string, confirmed: boolean) => post<Result>('/api/run', { pending_id, name, confirmed }),
  result: (id: string) => call<Result>(`/api/result/${id}`),
  climateDetail: (id: string, model: string) => call<any>(`/api/result/${id}/climate/${model}`),
  climateSpace: () => call<{ points: Record<string, [number, number][]>; explained: number[]; cities: string[]; matrix: number[][] }>('/api/climate/space'),
  reset: () => post('/api/reset', {}),
  exportUrl: (id: string, kind: string) => `/api/result/${id}/export/${kind}`,
  sampleUrl: (city: string, start: string, end: string) => `/api/sample?city=${city}&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
}
