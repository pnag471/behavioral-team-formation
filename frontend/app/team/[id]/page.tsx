'use client'

import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import RadarChart from '@/components/RadarChart'
import TeamMetricsPanel from '@/components/TeamMetricsPanel'
import { getTeam, getTeamExplanation, getTeamRadar } from '@/lib/api'
import type { RadarData, Team, TeamExplanation } from '@/lib/types'

const TEAM_NAMES = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta']

const NORM_ICONS: Record<string, string> = {
  'Communication':       '💬',
  'Conflict Resolution': '🤝',
  'Decision Making':     '⚡',
  'Planning':            '📅',
  'Accountability':      '✅',
}

function teamDisplayName(id: string) {
  const match = id.match(/team-(\d+)/)
  if (!match) return id
  const idx = parseInt(match[1]) - 1
  return `Team ${TEAM_NAMES[idx] ?? match[1]}`
}

function ConfidencePill({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  let cls = 'bg-emerald-100 text-emerald-800'
  if (pct < 70) cls = 'bg-amber-100 text-amber-800'
  if (pct < 50) cls = 'bg-red-100 text-red-800'
  return (
    <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold ${cls}`}>
      Match confidence: {pct}%
    </span>
  )
}

export default function TeamDetailPage() {
  const params = useParams<{ id: string }>()
  const teamId = params.id

  const [explanation, setExplanation] = useState<TeamExplanation | null>(null)
  const [radar, setRadar] = useState<RadarData | null>(null)
  const [team, setTeam] = useState<Team | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([getTeamExplanation(teamId), getTeamRadar(teamId), getTeam(teamId)])
      .then(([exp, rad, t]) => {
        setExplanation(exp)
        setRadar(rad)
        setTeam(t)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load team data.'))
      .finally(() => setLoading(false))
  }, [teamId])

  if (loading) {
    return <div className="text-center text-slate-400 py-16 text-sm">Loading team analysis…</div>
  }

  if (error || !explanation) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-10">
        <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700 mb-4">
          {error || 'Team not found.'}
        </div>
        <Link href="/teams" className="text-sm font-medium text-[#1e3a8a] hover:underline">
          ← Back to Teams
        </Link>
      </div>
    )
  }

  // We need the team object for members — reconstruct from explanation
  // The explanation has team_id; we need a parent Team for member grid.
  // We'll derive what we need from explanation + radar members_radar.

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex items-center gap-3 mb-8">
        <Link href="/teams" className="text-slate-400 hover:text-slate-600 transition-colors text-sm">
          ← Teams
        </Link>
        <span className="text-slate-200">/</span>
        <h1 className="text-2xl font-bold text-slate-800">{teamDisplayName(teamId)}</h1>
        <div className="ml-2 flex items-center gap-2">
          <span className="px-3 py-1 bg-[#1e3a8a] text-white rounded-full text-xs font-semibold">
            {Math.round(explanation.compatibility_score * 100)}% overall
          </span>
          <ConfidencePill score={explanation.match_confidence} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column — metrics + radar */}
        <div className="lg:col-span-1 flex flex-col gap-5">
          {/* Team Profile Metrics */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <h2 className="font-semibold text-slate-800 mb-4 text-sm uppercase tracking-wide">Team Profile</h2>
            {team && (
              <TeamMetricsPanel breakdown={team.score_breakdown} />
            )}
          </div>

          {/* Radar chart */}
          {radar && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h2 className="font-semibold text-slate-800 mb-3 text-sm uppercase tracking-wide">
                Behavioral Radar
              </h2>
              <p className="text-xs text-slate-400 mb-3">
                Solid = team average · Dashed = individual members
              </p>
              <div className="flex justify-center">
                <RadarChart data={radar} size={230} />
              </div>
              {/* Member legend */}
              <div className="mt-3 flex flex-wrap gap-2">
                {radar.members_radar.map((m, i) => {
                  const colors = ['#93c5fd', '#6ee7b7', '#fca5a5', '#fcd34d']
                  return (
                    <div key={m.student_id} className="flex items-center gap-1 text-xs text-slate-500">
                      <span
                        className="inline-block w-3 h-0.5 rounded-full"
                        style={{ backgroundColor: colors[i % colors.length] }}
                      />
                      {m.name.split(' ')[0]}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right column */}
        <div className="lg:col-span-2 flex flex-col gap-5">
          {/* Members grid */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <h2 className="font-semibold text-slate-800 mb-4 text-sm uppercase tracking-wide">Members</h2>
            <div className="grid grid-cols-2 gap-3">
              {radar?.members_radar.map((m) => (
                <div key={m.student_id} className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                  <p className="font-semibold text-slate-800 text-sm">{m.name}</p>
                  <p className="text-xs text-slate-400 mt-0.5">ID: {m.student_id}</p>
                  <div className="mt-2 flex gap-1 flex-wrap">
                    {m.values.map((v, i) => (
                      <div key={i} className="flex flex-col items-center">
                        <div
                          className="w-1.5 rounded-full bg-[#1e3a8a] opacity-70"
                          style={{ height: `${Math.round(v * 24)}px`, minHeight: '4px' }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Team Norms */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <h2 className="font-semibold text-slate-800 mb-1 text-sm uppercase tracking-wide">
              Suggested Team Norms
            </h2>
            <p className="text-xs text-slate-400 mb-4">
              Auto-generated from behavioral signatures. Use as a starting point for your team charter.
            </p>
            <div className="flex flex-col gap-3">
              {explanation.team_norms.map((norm) => (
                <div
                  key={norm.category}
                  className="flex gap-3 p-3 rounded-xl bg-slate-50 border border-slate-100"
                >
                  <span className="text-lg flex-shrink-0">
                    {NORM_ICONS[norm.category] ?? '📌'}
                  </span>
                  <div>
                    <p className="text-xs font-semibold text-slate-500 mb-0.5">{norm.category}</p>
                    <p className="text-sm text-slate-700 leading-relaxed">{norm.norm}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Strengths */}
          <div className="bg-white rounded-2xl border border-emerald-100 shadow-sm p-5">
            <h2 className="font-semibold text-emerald-800 mb-3 text-sm uppercase tracking-wide">
              Strengths
            </h2>
            <ul className="flex flex-col gap-2">
              {explanation.strengths.map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                  <span className="text-emerald-500 mt-0.5 flex-shrink-0">✓</span>
                  {s.replace(/\*\*(.*?)\*\*/g, '$1')}
                </li>
              ))}
            </ul>
          </div>

          {/* Risks */}
          <div className="bg-white rounded-2xl border border-amber-100 shadow-sm p-5">
            <h2 className="font-semibold text-amber-800 mb-3 text-sm uppercase tracking-wide">
              Risk Factors
            </h2>
            <ul className="flex flex-col gap-2">
              {explanation.risks.map((r, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                  <span className="text-amber-500 mt-0.5 flex-shrink-0">⚠</span>
                  {r}
                </li>
              ))}
            </ul>
          </div>

          {/* Explanation prose */}
          <div className="bg-[#f0f6ff] rounded-2xl border border-blue-100 p-5">
            <h2 className="font-semibold text-[#1e3a8a] mb-3 text-sm uppercase tracking-wide">
              Formation Explanation
            </h2>
            <p className="text-sm text-slate-700 leading-relaxed">{explanation.explanation}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
