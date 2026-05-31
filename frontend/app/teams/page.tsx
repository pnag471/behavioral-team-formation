'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import TeamMetricsPanel from '@/components/TeamMetricsPanel'
import { getTeams } from '@/lib/api'
import type { Team } from '@/lib/types'

const TEAM_NAMES = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta']

function confidenceBadge(score: number) {
  if (score >= 0.7) return { label: 'High Confidence', className: 'bg-emerald-100 text-emerald-800' }
  if (score >= 0.5) return { label: 'Moderate Confidence', className: 'bg-amber-100 text-amber-800' }
  return { label: 'Low Confidence', className: 'bg-red-100 text-red-800' }
}

function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = pct >= 70 ? '#10b981' : pct >= 50 ? '#3b82f6' : '#f59e0b'
  return (
    <div className="flex items-center gap-1.5">
      <svg width="36" height="36" viewBox="0 0 36 36" className="flex-shrink-0">
        <circle cx="18" cy="18" r="14" fill="none" stroke="#e2e8f0" strokeWidth="3" />
        <circle
          cx="18"
          cy="18"
          r="14"
          fill="none"
          stroke={color}
          strokeWidth="3"
          strokeDasharray={`${pct * 0.879} 100`}
          strokeLinecap="round"
          transform="rotate(-90 18 18)"
        />
        <text x="18" y="22" textAnchor="middle" fontSize="9" fontWeight="bold" fill={color}>
          {pct}
        </text>
      </svg>
    </div>
  )
}

export default function TeamsPage() {
  const [teams, setTeams] = useState<Team[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getTeams()
      .then((r) => setTeams(r.teams))
      .catch(() => setError('No teams found. Generate teams from the Dashboard first.'))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-800 mb-1">Generated Teams</h1>
          <p className="text-slate-500 text-sm">
            {teams.length > 0
              ? `${teams.length} teams formed via greedy compatibility optimization`
              : 'No teams generated yet'}
          </p>
        </div>
        <Link
          href="/dashboard"
          className="px-4 py-2 border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          ← Back to Dashboard
        </Link>
      </div>

      {error && (
        <div className="mb-6 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-sm text-amber-700 flex items-center justify-between">
          {error}
          <Link href="/dashboard" className="font-semibold underline">Generate Teams</Link>
        </div>
      )}

      {loading ? (
        <div className="text-center text-slate-400 py-16 text-sm">Loading teams…</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {teams.map((team, idx) => {
            const teamName = `Team ${TEAM_NAMES[idx] ?? idx + 1}`
            const badge = confidenceBadge(team.score_breakdown.match_confidence)
            return (
              <div
                key={team.id}
                className="bg-white rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden"
              >
                {/* Header */}
                <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
                  <div>
                    <h2 className="font-bold text-slate-800 text-lg">{teamName}</h2>
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${badge.className}`}>
                      {badge.label}
                    </span>
                  </div>
                  <ScoreBadge score={team.score_breakdown.total} />
                </div>

                {/* Members */}
                <div className="px-6 py-3 border-b border-slate-100">
                  <div className="flex flex-wrap gap-2">
                    {team.members.map((m) => (
                      <div
                        key={m.student_id}
                        className="flex items-center gap-1.5 bg-slate-50 rounded-lg px-3 py-1.5"
                      >
                        <span className="text-sm font-medium text-slate-700">{m.name}</span>
                        {m.roles[0] && (
                          <span className="text-xs text-slate-400 capitalize">· {m.roles[0].replace(/-/g, ' ')}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Metrics */}
                <div className="px-6 py-4 border-b border-slate-100">
                  <TeamMetricsPanel breakdown={team.score_breakdown} compact />
                </div>

                {/* Footer */}
                <div className="px-6 py-3">
                  <Link
                    href={`/team/${team.id}`}
                    className="text-sm font-semibold text-[#1e3a8a] hover:text-blue-700 transition-colors"
                  >
                    View Full Analysis →
                  </Link>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
