'use client'

import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import RadarChart from '@/components/RadarChart'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface StudentProfile {
  id: string
  name: string
  competence_signature: {
    skills: string[]
    roles: string[]
    experience_level: string
    skill_depth: string
    skill_no_gos: string[]
  }
  work_rhythm_signature: {
    planning_style: string
    communication_style: string
    execution_style: string
    unavailable_times: string[]
    timezone_notes: string
  }
  collaboration_signature: {
    conflict_style: string
    leadership_style: string
    accountability: string
    help_seeking: string
    safety_contribution: string
    stress_response: string
  }
  confidence_layer: {
    confidence_score: number
    confidence_scores: Record<string, number>
    consistency_flags: { dimension: string; note: string }[]
  }
}

function signatureToRadar(collab: StudentProfile['collaboration_signature'], work: StudentProfile['work_rhythm_signature']) {
  const commMap: Record<string, number> = { async: 0.35, mixed: 0.65, sync: 0.90 }
  const conflictMap: Record<string, number> = { avoidant: 0.20, confrontational: 0.60, collaborative: 0.85 }
  const leaderMap: Record<string, number> = { emergent: 0.30, facilitative: 0.70, directive: 0.92 }
  const accountMap: Record<string, number> = { low: 0.20, medium: 0.60, high: 0.90 }
  const planMap: Record<string, number> = { spontaneous: 0.20, adaptive: 0.60, planner: 0.90 }

  return {
    labels: ['Communication', 'Conflict Handling', 'Leadership', 'Accountability', 'Planning'],
    values: [
      commMap[work.communication_style] ?? 0.5,
      conflictMap[collab.conflict_style] ?? 0.5,
      leaderMap[collab.leadership_style] ?? 0.5,
      accountMap[collab.accountability] ?? 0.5,
      planMap[work.planning_style] ?? 0.5,
    ],
    members_radar: [],
  }
}

export default function ProfilePage() {
  const params = useParams<{ id: string }>()
  const [profile, setProfile] = useState<StudentProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${BASE_URL}/students/${params.id}`)
      .then(res => {
        if (!res.ok) throw new Error('Student not found')
        return res.json()
      })
      .then(setProfile)
      .catch(() => setError('Could not load profile.'))
      .finally(() => setLoading(false))
  }, [params.id])

  if (loading) return <div className="text-center text-slate-400 py-16 text-sm">Loading profile...</div>
  if (error || !profile) return (
    <div className="max-w-3xl mx-auto px-6 py-10">
      <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700 mb-4">{error}</div>
      <Link href="/interview" className="text-sm font-medium text-[#1e3a8a] hover:underline">← Back to Interview</Link>
    </div>
  )

  const radar = signatureToRadar(profile.collaboration_signature, profile.work_rhythm_signature)
  const overallConfidence = profile.confidence_layer?.confidence_score ?? 0
  const confidenceScores = profile.confidence_layer?.confidence_scores ?? {}
  const consistencyFlags = profile.confidence_layer?.consistency_flags ?? []

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-8">
        <Link href="/interview" className="text-sm text-slate-400 hover:text-slate-600 transition-colors">← New Interview</Link>
        <h1 className="text-3xl font-bold text-slate-800 mt-2">{profile.name}</h1>
        <div className="flex items-center gap-3 mt-2">
          <span className="text-sm text-slate-500">Behavioral Profile</span>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
            overallConfidence >= 0.75 ? 'bg-emerald-100 text-emerald-800' :
            overallConfidence >= 0.55 ? 'bg-amber-100 text-amber-800' :
            'bg-red-100 text-red-800'
          }`}>
            {Math.round(overallConfidence * 100)}% overall confidence
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column */}
        <div className="lg:col-span-1 flex flex-col gap-5">
          {/* Radar */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <h2 className="font-semibold text-slate-800 mb-3 text-sm uppercase tracking-wide">Behavioral Radar</h2>
            <div className="flex justify-center">
              <RadarChart data={radar} size={220} />
            </div>
          </div>

          {/* Skills */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <h2 className="font-semibold text-slate-800 mb-3 text-sm uppercase tracking-wide">Skills</h2>
            <div className="flex flex-wrap gap-2 mb-3">
              {profile.competence_signature.skills.map(skill => (
                <span key={skill} className="px-2 py-1 bg-blue-50 text-blue-700 rounded-lg text-xs font-medium">{skill}</span>
              ))}
            </div>
            <p className="text-xs text-slate-400 mb-1">Depth: <span className="text-slate-600 capitalize font-medium">{profile.competence_signature.skill_depth}</span></p>
            {profile.competence_signature.skill_no_gos?.length > 0 && (
              <>
                <p className="text-xs text-slate-400 mt-3 mb-1">Hard no's:</p>
                <div className="flex flex-wrap gap-2">
                  {profile.competence_signature.skill_no_gos.map(s => (
                    <span key={s} className="px-2 py-1 bg-red-50 text-red-600 rounded-lg text-xs font-medium">{s}</span>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Availability */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <h2 className="font-semibold text-slate-800 mb-3 text-sm uppercase tracking-wide">Availability</h2>
            {profile.work_rhythm_signature.unavailable_times?.length > 0 ? (
              <>
                <p className="text-xs text-slate-400 mb-2">Hard no's:</p>
                <div className="flex flex-col gap-1">
                  {profile.work_rhythm_signature.unavailable_times.map(t => (
                    <span key={t} className="text-sm text-slate-600">• {t}</span>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-slate-400">No hard constraints specified</p>
            )}
            {profile.work_rhythm_signature.timezone_notes && (
              <p className="text-xs text-slate-500 mt-3">🌍 {profile.work_rhythm_signature.timezone_notes}</p>
            )}
          </div>
        </div>

        {/* Right column */}
        <div className="lg:col-span-2 flex flex-col gap-5">
          {/* Behavioral dimensions */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <h2 className="font-semibold text-slate-800 mb-4 text-sm uppercase tracking-wide">Behavioral Dimensions</h2>
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'Planning Style', value: profile.work_rhythm_signature.planning_style, dim: 'planning_style' },
                { label: 'Communication', value: profile.work_rhythm_signature.communication_style, dim: 'communication_style' },
                { label: 'Execution Style', value: profile.work_rhythm_signature.execution_style, dim: 'execution_style' },
                { label: 'Conflict Style', value: profile.collaboration_signature.conflict_style, dim: 'conflict_style' },
                { label: 'Leadership', value: profile.collaboration_signature.leadership_style, dim: 'leadership_style' },
                { label: 'Accountability', value: profile.collaboration_signature.accountability, dim: 'accountability' },
                { label: 'Help Seeking', value: profile.collaboration_signature.help_seeking, dim: 'help_seeking' },
                { label: 'Safety Contribution', value: profile.collaboration_signature.safety_contribution, dim: 'safety_contribution' },
                { label: 'Stress Response', value: profile.collaboration_signature.stress_response, dim: 'stress_response' },
              ].map(({ label, value, dim }) => (
                <div key={dim} className="bg-slate-50 rounded-xl p-3 border border-slate-100">
                  <p className="text-xs text-slate-400 mb-1">{label}</p>
                  <p className="text-sm font-semibold text-slate-800 capitalize">{value}</p>
                  {confidenceScores[dim] !== undefined && (
                    <div className="mt-2">
                      <div className="w-full bg-slate-200 rounded-full h-1">
                        <div
                          className="bg-[#1e3a8a] h-1 rounded-full"
                          style={{ width: `${Math.round(confidenceScores[dim] * 100)}%` }}
                        />
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">{Math.round(confidenceScores[dim] * 100)}% confidence</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Consistency flags */}
          {consistencyFlags.length > 0 && (
            <div className="bg-amber-50 rounded-2xl border border-amber-100 shadow-sm p-5">
              <h2 className="font-semibold text-amber-800 mb-3 text-sm uppercase tracking-wide">⚠ Consistency Flags</h2>
              <div className="flex flex-col gap-3">
                {consistencyFlags.map((flag, i) => (
                  <div key={i} className="bg-white rounded-xl p-3 border border-amber-100">
                    <p className="text-xs font-semibold text-amber-600 mb-1 capitalize">{flag.dimension.replace(/_/g, ' ')}</p>
                    <p className="text-sm text-slate-600">{flag.note}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Next steps */}
          <div className="bg-[#f0f6ff] rounded-2xl border border-blue-100 p-5">
            <h2 className="font-semibold text-[#1e3a8a] mb-2 text-sm uppercase tracking-wide">What Happens Next</h2>
            <p className="text-sm text-slate-600 leading-relaxed">
              Your behavioral profile has been saved. Once your instructor generates teams, you'll be matched with teammates based on your behavioral signature, skills, and availability. You'll be able to view your team's full compatibility analysis including strengths, risk factors, and suggested working norms.
            </p>
            <Link href="/teams" className="inline-block mt-3 text-sm font-semibold text-[#1e3a8a] hover:underline">
              View Generated Teams →
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}