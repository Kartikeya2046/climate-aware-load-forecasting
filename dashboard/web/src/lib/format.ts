// Timestamps from the API are naive local standard time. They are parsed as UTC and always formatted in UTC,
// so the wall-clock hour shown is the city's hour regardless of the browser's time zone.
export const ms = (t: string) => Date.parse(t.length === 16 ? t + ':00Z' : t.replace(' ', 'T') + 'Z')

const dt = (o: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat('en-GB', { timeZone: 'UTC', ...o })
const F = {
  day: dt({ day: 'numeric', month: 'short', year: 'numeric' }),
  hour: dt({ weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }),
  short: dt({ day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }),
}
export const fmtDay = (t: string | number) => F.day.format(typeof t === 'number' ? t : ms(t))
export const fmtHour = (t: string | number) => F.hour.format(typeof t === 'number' ? t : ms(t))
export const fmtShort = (t: string | number) => F.short.format(typeof t === 'number' ? t : ms(t))

export const n0 = (v: number | null | undefined) => (v == null ? '–' : Math.round(v).toLocaleString('en-US'))
export const nf = (v: number | null | undefined, d = 1) =>
  v == null ? '–' : v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })

/** MWh -> a readable magnitude; GWh from 10,000 MWh, as in the report. */
export const energy = (mwh: number) => (Math.abs(mwh) >= 10_000 ? { value: mwh / 1000, unit: 'GWh', digits: 2 } : { value: mwh, unit: 'MWh', digits: 0 })
export const fmtEnergy = (mwh: number) => {
  const e = energy(mwh)
  return `${nf(e.value, e.digits)} ${e.unit}`
}

export const mean = (xs: (number | null)[]) => {
  const v = xs.filter((x): x is number => x != null)
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}
export const cityShort = (name: string) => name.split(',')[0]
