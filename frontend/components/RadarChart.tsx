'use client'

import type { MemberRadarData, RadarData } from '@/lib/types'

interface RadarChartProps {
  data: RadarData
  size?: number
  teamColor?: string
}

function polarToXY(angle: number, radius: number, cx: number, cy: number) {
  return {
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  }
}

function buildPolygon(values: number[], radius: number, cx: number, cy: number, axes: number) {
  const points = values.map((v, i) => {
    const angle = (i / axes) * 2 * Math.PI - Math.PI / 2
    return polarToXY(angle, v * radius, cx, cy)
  })
  return points.map((p) => `${p.x},${p.y}`).join(' ')
}

export default function RadarChart({ data, size = 260, teamColor = '#1e3a8a' }: RadarChartProps) {
  const cx = size / 2
  const cy = size / 2
  const radius = size * 0.36
  const axes = data.labels.length
  const gridLevels = [0.25, 0.5, 0.75, 1.0]

  return (
    <svg width={size} height={size} viewBox={`-70 -10 ${size + 140} ${size + 20}`} aria-label="Behavioral radar chart">
      {/* Grid rings */}
      {gridLevels.map((level) => (
        <polygon
          key={level}
          points={buildPolygon(Array(axes).fill(level), radius, cx, cy, axes)}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="1"
        />
      ))}

      {/* Axis lines */}
      {data.labels.map((_, i) => {
        const angle = (i / axes) * 2 * Math.PI - Math.PI / 2
        const end = polarToXY(angle, radius, cx, cy)
        return (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={end.x}
            y2={end.y}
            stroke="#cbd5e1"
            strokeWidth="1"
          />
        )
      })}

      {/* Member radar overlays (thin, translucent) */}
      {data.members_radar.map((member: MemberRadarData, mi: number) => {
        const colors = ['#93c5fd', '#6ee7b7', '#fca5a5', '#fcd34d']
        const c = colors[mi % colors.length]
        return (
          <polygon
            key={member.student_id}
            points={buildPolygon(member.values, radius, cx, cy, axes)}
            fill={c}
            fillOpacity={0.12}
            stroke={c}
            strokeWidth="1.2"
            strokeDasharray="3 2"
          />
        )
      })}

      {/* Team average polygon */}
      <polygon
        points={buildPolygon(data.values, radius, cx, cy, axes)}
        fill={teamColor}
        fillOpacity={0.18}
        stroke={teamColor}
        strokeWidth="2"
      />

      {/* Axis labels */}
      {data.labels.map((label, i) => {
        const angle = (i / axes) * 2 * Math.PI - Math.PI / 2
        const labelRadius = radius + 28
        const pos = polarToXY(angle, labelRadius, cx, cy)
        const anchor =
          Math.abs(pos.x - cx) < 4 ? 'middle' : pos.x < cx ? 'end' : 'start'
        const shortLabels: Record<string, string> = {
          'Accountability': 'Account.',
          'Communication': 'Comm.',
          'Leadership': 'Leader.',
          'Conflict Handling': 'Conflict',
        }
        const displayLabel = shortLabels[label] ?? label
        return (
          <text
            key={i}
            x={pos.x}
            y={pos.y}
            textAnchor={anchor}
            dominantBaseline="middle"
            fontSize="10"
            fill="#475569"
            fontFamily="system-ui, sans-serif"
          >
            {label}
          </text>
        )
      })}

      {/* Value dots on team polygon */}
      {data.values.map((v, i) => {
        const angle = (i / axes) * 2 * Math.PI - Math.PI / 2
        const pos = polarToXY(angle, v * radius, cx, cy)
        return (
          <circle
            key={i}
            cx={pos.x}
            cy={pos.y}
            r="3"
            fill={teamColor}
          />
        )
      })}
    </svg>
  )
}
