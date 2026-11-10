import NumberFlow from '@number-flow/react'
import { AlertTriangle, CheckCircle2, ChevronDown, Info, OctagonAlert } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useId, useState, type ButtonHTMLAttributes, type PointerEvent, type ReactNode } from 'react'

export const cx = (...a: (string | false | null | undefined)[]) => a.filter(Boolean).join(' ')

/** Primary glass panel with an optional heading row. */
export function Panel({ title, hint, action, children, className, pad = true }: {
  title?: ReactNode; hint?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; pad?: boolean
}) {
  return (
    <section className={cx('glass', pad && 'p-5 sm:p-6', className)}>
      {(title || action) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-x-6 gap-y-2">
          <div className="min-w-0">
            <h2 className="text-[16px] text-ink">{title}</h2>
            {hint && <p className="mt-1 max-w-[70ch] text-[13.5px] text-ink-3">{hint}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  )
}

/** The hero surface. The light inside it follows the pointer (adapted from ui-layouts' spotlight card). */
export function Liquid({ children, className }: { children: ReactNode; className?: string }) {
  const move = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`)
    e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`)
  }
  return <div onPointerMove={move} className={cx('liquid', className)}>{children}</div>
}

/** Animated pill tabs: the active pill is one shared element that slides (ui-layouts' tabs technique). */
export function Tabs<T extends string>({ items, value, onChange, size = 'md', label }: {
  items: { id: T; label: ReactNode; disabled?: boolean }[]; value: T; onChange: (v: T) => void; size?: 'sm' | 'md'; label: string
}) {
  const id = useId()
  return (
    <div role="tablist" aria-label={label} className="well inline-flex max-w-full gap-0.5 overflow-x-auto p-1 [scrollbar-width:none]">
      {items.map(it => (
        <button key={it.id} role="tab" aria-selected={value === it.id} disabled={it.disabled} onClick={() => onChange(it.id)}
          className={cx('relative shrink-0 rounded-[10px] whitespace-nowrap transition-colors disabled:opacity-35', size === 'sm' ? 'px-3 py-1 text-[13px]' : 'px-3.5 py-1.5 text-[14px]',
            value === it.id ? 'text-ink' : 'text-ink-3 hover:text-ink-2')}>
          {value === it.id && <motion.span layoutId={id} transition={{ type: 'spring', stiffness: 420, damping: 34 }}
            className="absolute inset-0 rounded-[10px] border border-line-2 bg-white/[0.07] shadow-[inset_0_1px_0_rgb(255_255_255/0.08)]" />}
          <span className="relative">{it.label}</span>
        </button>
      ))}
    </div>
  )
}

export function Button({ variant = 'ghost', className, ...p }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'ghost' | 'quiet' }) {
  const v = {
    primary: 'bg-amber-hi text-[#1a1203] font-medium hover:bg-[#ffc566] shadow-[0_10px_30px_-12px_rgb(242_178_74/0.7),inset_0_1px_0_rgb(255_255_255/0.4)]',
    ghost: 'border border-line-2 bg-white/[0.04] text-ink hover:bg-white/[0.08]',
    quiet: 'text-ink-2 hover:text-ink hover:bg-white/[0.05]',
  }[variant]
  return <button {...p} className={cx('inline-flex h-10 items-center justify-center gap-2 rounded-xl px-4 text-[14px] transition active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40', v, className)} />
}

const TONES = {
  neutral: 'border-line text-ink-2 bg-white/[0.04]',
  good: 'border-good/30 text-good bg-good/10',
  warn: 'border-warn/30 text-warn bg-warn/10',
  bad: 'border-bad/30 text-bad bg-bad/10',
  accent: 'border-amber-hi/35 text-amber-hi bg-amber-hi/10',
}
export type Tone = keyof typeof TONES
export const Chip = ({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) => (
  <span className={cx('inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[12.5px] whitespace-nowrap', TONES[tone])}>{children}</span>
)

const NOTICE = {
  info: { icon: Info, c: 'text-ice-hi', b: 'border-ice/25 bg-ice/[0.07]' },
  good: { icon: CheckCircle2, c: 'text-good', b: 'border-good/25 bg-good/[0.07]' },
  warn: { icon: AlertTriangle, c: 'text-warn', b: 'border-warn/25 bg-warn/[0.06]' },
  bad: { icon: OctagonAlert, c: 'text-bad', b: 'border-bad/30 bg-bad/[0.07]' },
}
export function Notice({ tone = 'info', children }: { tone?: keyof typeof NOTICE; children: ReactNode }) {
  const n = NOTICE[tone]
  return (
    <div className={cx('flex gap-3 rounded-[14px] border px-4 py-3 text-[14px] text-ink-2', n.b)}>
      <n.icon size={17} className={cx('mt-0.5 shrink-0', n.c)} aria-hidden />
      <div className="min-w-0 max-w-[90ch]">{children}</div>
    </div>
  )
}

/** 0–100 score bar. */
export const Meter = ({ value, color = 'var(--color-amber)', className }: { value: number | null; color?: string; className?: string }) => (
  <div className={cx('h-1.5 overflow-hidden rounded-full bg-white/[0.07]', className)}>
    {value != null && <div className="grow h-full rounded-full" style={{ background: color, width: `${Math.max(0, Math.min(100, value))}%` }} />}
  </div>
)

export function Disclosure({ title, children, open: initial = false, right }: { title: ReactNode; children: ReactNode; open?: boolean; right?: ReactNode }) {
  const [open, setOpen] = useState(initial)
  return (
    <div className="well">
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left text-[14px] text-ink-2 hover:text-ink">
        <span className="min-w-0">{title}</span>
        <span className="flex shrink-0 items-center gap-3">{right}<ChevronDown size={16} className={cx('transition-transform', open && 'rotate-180')} aria-hidden /></span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.25 }} className="overflow-hidden">
            <div className="border-t border-line px-4 py-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** A value that rolls to its new figure when it changes (Number Flow, as in ui-layouts' motion number). */
export const Num = ({ value, digits = 0, suffix, prefix }: { value: number; digits?: number; suffix?: string; prefix?: string }) => (
  <NumberFlow value={value} prefix={prefix} suffix={suffix} format={{ minimumFractionDigits: digits, maximumFractionDigits: digits }} locales="en-US" />
)

export const Stat = ({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) => (
  <div className="well px-4 py-3.5">
    <div className="text-[13px] text-ink-3">{label}</div>
    <div className="num mt-1 text-[22px] leading-tight text-ink">{value}</div>
    {sub && <div className="mt-1 text-[12.5px] text-ink-3">{sub}</div>}
  </div>
)

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="glass flex flex-col items-start gap-3 p-8 sm:p-10">
      <h2 className="display text-[28px] text-ink">{title}</h2>
      {children && <p className="max-w-[60ch] text-ink-2">{children}</p>}
      {action}
    </div>
  )
}

export const PageHead = ({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) => (
  <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
    <div>
      <h1 className="display text-[34px] leading-tight text-ink sm:text-[40px]">{title}</h1>
      {children && <p className="mt-2 max-w-[72ch] text-ink-2">{children}</p>}
    </div>
    {action}
  </header>
)

export const Scroll = ({ children, max = 360 }: { children: ReactNode; max?: number }) => (
  <div className="overflow-auto" style={{ maxHeight: max }}>{children}</div>
)
