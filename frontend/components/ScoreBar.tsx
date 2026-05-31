interface ScoreBarProps {
  label: string
  value: number       // 0–1
  inverse?: boolean   // if true, high value = bad (red)
  showValue?: boolean
}

export default function ScoreBar({ label, value, inverse = false, showValue = true }: ScoreBarProps) {
  const pct = Math.round(value * 100)

  let barColor: string
  if (inverse) {
    barColor = pct >= 60 ? '#ef4444' : pct >= 35 ? '#f59e0b' : '#10b981'
  } else {
    barColor = pct >= 65 ? '#10b981' : pct >= 40 ? '#3b82f6' : '#f59e0b'
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center text-sm">
        <span className="text-slate-600 font-medium">{label}</span>
        {showValue && (
          <span
            className="font-semibold tabular-nums"
            style={{ color: barColor }}
          >
            {pct}%
          </span>
        )}
      </div>
      <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  )
}
