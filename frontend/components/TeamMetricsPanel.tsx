import ScoreBar from './ScoreBar'
import type { TeamScoreBreakdown } from '@/lib/types'

interface TeamMetricsPanelProps {
  breakdown: TeamScoreBreakdown
  compact?: boolean
}

export default function TeamMetricsPanel({ breakdown, compact = false }: TeamMetricsPanelProps) {
  const metrics = [
    { label: 'Skill Coverage',           value: breakdown.skill_coverage,       inverse: false },
    { label: 'Behavioral Compatibility', value: breakdown.behavioral_compat,    inverse: false },
    { label: 'Availability Overlap',     value: breakdown.availability_overlap, inverse: false },
    { label: 'Conflict Risk',            value: breakdown.conflict_risk,        inverse: true  },
    { label: 'Match Confidence',         value: breakdown.match_confidence,     inverse: false },
  ]

  return (
    <div className={`flex flex-col gap-${compact ? '2' : '3'}`}>
      {metrics.map((m) => (
        <ScoreBar
          key={m.label}
          label={m.label}
          value={m.value}
          inverse={m.inverse}
        />
      ))}
    </div>
  )
}
