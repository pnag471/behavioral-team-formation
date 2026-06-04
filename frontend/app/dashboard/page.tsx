'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { generateTeams, getStudents } from '@/lib/api'
import type { Student, TeamGenerationRequest } from '@/lib/types'

type Weights = TeamGenerationRequest['weights']

function normalizeWeights(w: Weights): Weights {
  const total = w.skill_coverage + w.behavioral_compat + w.availability_overlap + w.shared_interests
  if (total === 0) return w
  return {
    skill_coverage:       parseFloat((w.skill_coverage / total).toFixed(3)),
    behavioral_compat:    parseFloat((w.behavioral_compat / total).toFixed(3)),
    availability_overlap: parseFloat((w.availability_overlap / total).toFixed(3)),
    shared_interests:     parseFloat((w.shared_interests / total).toFixed(3)),
  }
}

const WEIGHT_META: { key: keyof Weights; label: string; description: string }[] = [
  { key: 'skill_coverage',       label: 'Skill Coverage',           description: 'How many reference skills the team covers together' },
  { key: 'behavioral_compat',    label: 'Behavioral Compatibility',  description: 'Pairwise compatibility of working styles' },
  { key: 'availability_overlap', label: 'Availability Overlap',      description: 'Jaccard similarity of scheduling windows' },
  { key: 'shared_interests',     label: 'Interest Alignment',        description: 'Shared learning goals and topic interests' },
]

export default function DashboardPage() {
  const router = useRouter()
  const [students, setStudents] = useState<Student[]>([])
  const [loadingStudents, setLoadingStudents] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')
  const [teamSize, setTeamSize] = useState(4)
  const [formationMode, setFormationMode] = useState<'behavioral' | 'skill' | 'random'>('behavioral')
  const [comparisonResults, setComparisonResults] = useState<Record<string, { avg_behavioral_compat: number; avg_skill_coverage: number; avg_confidence: number; avg_conflict_risk: number }>>({})
  const [weights, setWeights] = useState<Weights>({
    skill_coverage: 0.4,
    behavioral_compat: 0.3,
    availability_overlap: 0.2,
    shared_interests: 0.1,
  })

  useEffect(() => {
    getStudents()
      .then(setStudents)
      .catch(() => setError('Could not reach backend. Is it running on port 8000?'))
      .finally(() => setLoadingStudents(false))
  }, [])

  const handleWeightChange = (key: keyof Weights, raw: number) => {
    const updated = { ...weights, [key]: raw }
    setWeights(normalizeWeights(updated))
  }

  const handleGenerate = async () => {
    setError('')
    setGenerating(true)
    try {
      const result = await generateTeams({ team_size: teamSize, weights, formation_mode: formationMode })

      const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length
      setComparisonResults(prev => ({
        ...prev,
        [formationMode]: {
          avg_behavioral_compat: avg(result.teams.map(t => t.score_breakdown.behavioral_compat)),
          avg_skill_coverage: avg(result.teams.map(t => t.score_breakdown.skill_coverage)),
          avg_confidence: avg(result.teams.map(t => t.score_breakdown.match_confidence)),
          avg_conflict_risk: avg(result.teams.map(t => t.score_breakdown.conflict_risk)),
        }
      }))
      router.push('/teams')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Team generation failed.')
      setGenerating(false)
    }
  }

  const teamCount = Math.floor(students.length / teamSize)

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800 mb-2">Instructor Dashboard</h1>
        <p className="text-slate-500 text-sm">
          Configure team formation parameters and generate optimal teams from the student pool.
        </p>
      </div>

      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}


      {/* Formation Mode Selector */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm mb-6">
        <h2 className="font-semibold text-slate-800 mb-1 text-base">Formation Mode</h2>
        <p className="text-xs text-slate-400 mb-4">Select how teams will be formed. Run all three to compare results.</p>
        <div className="grid grid-cols-3 gap-3">
          {([
            { mode: 'random', label: 'Random', desc: 'Baseline — no optimization' },
            { mode: 'skill', label: 'Skill-Based', desc: 'Traditional approach' },
            { mode: 'behavioral', label: 'Behavioral', desc: 'Research contribution' },
          ] as const).map(({ mode, label, desc }) => (
            <button
              key={mode}
              onClick={() => setFormationMode(mode)}
              className={`p-4 rounded-xl border-2 text-left transition-all ${
                formationMode === mode
                  ? 'border-[#1e3a8a] bg-blue-50'
                  : 'border-slate-200 hover:border-slate-300'
              }`}
            >
              <p className={`font-semibold text-sm ${formationMode === mode ? 'text-[#1e3a8a]' : 'text-slate-700'}`}>
                {label}
              </p>
              <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Comparison Results */}
      {Object.keys(comparisonResults).length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm mb-6">
          <h2 className="font-semibold text-slate-800 mb-1 text-base">Formation Comparison</h2>
          <p className="text-xs text-slate-400 mb-4">Run all three modes to compare results side by side.</p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 text-slate-500 text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-3 font-medium">Mode</th>
                  <th className="text-left px-4 py-3 font-medium">Behavioral Compat</th>
                  <th className="text-left px-4 py-3 font-medium">Skill Coverage</th>
                  <th className="text-left px-4 py-3 font-medium">Confidence</th>
                  <th className="text-left px-4 py-3 font-medium">Conflict Risk</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {(['behavioral', 'skill', 'random'] as const).map(mode => {
                  const r = comparisonResults[mode]
                  if (!r) return null
                  return (
                    <tr key={mode} className={mode === 'behavioral' ? 'bg-blue-50' : ''}>
                      <td className="px-4 py-3 font-medium capitalize text-slate-800">{mode}</td>
                      <td className="px-4 py-3 text-slate-600">{Math.round(r.avg_behavioral_compat * 100)}%</td>
                      <td className="px-4 py-3 text-slate-600">{Math.round(r.avg_skill_coverage * 100)}%</td>
                      <td className="px-4 py-3 text-slate-600">{Math.round(r.avg_confidence * 100)}%</td>
                      <td className="px-4 py-3 text-slate-600">{Math.round(r.avg_conflict_risk * 100)}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {/* Stats bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Students" value={loadingStudents ? '…' : String(students.length)} />
        <StatCard label="Team Size" value={String(teamSize)} />
        <StatCard label="Teams to Form" value={loadingStudents ? '…' : String(teamCount)} />
        <StatCard label="Leftover Students" value={loadingStudents ? '…' : String(students.length % teamSize)} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Weight sliders */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
          <h2 className="font-semibold text-slate-800 mb-1 text-base">Matching Weights</h2>
          <p className="text-xs text-slate-400 mb-5">
            Adjust the importance of each factor. Weights are auto-normalized to sum to 100%.
          </p>
          <div className="flex flex-col gap-5">
            {WEIGHT_META.map(({ key, label, description }) => (
              <div key={key}>
                <div className="flex justify-between items-center mb-1">
                  <div>
                    <span className="text-sm font-medium text-slate-700">{label}</span>
                    <p className="text-xs text-slate-400">{description}</p>
                  </div>
                  <span className="text-sm font-bold text-[#1e3a8a] w-10 text-right">
                    {Math.round(weights[key] * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={weights[key]}
                  onChange={(e) => handleWeightChange(key, parseFloat(e.target.value))}
                  className="w-full accent-[#1e3a8a]"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Team size + generate */}
        <div className="flex flex-col gap-4">
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
            <h2 className="font-semibold text-slate-800 mb-4 text-base">Team Configuration</h2>
            <label className="block text-sm text-slate-600 mb-1">Team Size</label>
            <div className="flex items-center gap-3 mb-2">
              <input
                type="number"
                min={2}
                max={8}
                value={teamSize}
                onChange={(e) => setTeamSize(Math.max(2, Math.min(8, parseInt(e.target.value) || 4)))}
                className="w-20 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-800 text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <span className="text-xs text-slate-400">members per team (2–8)</span>
            </div>
            {teamCount === 0 && !loadingStudents && (
              <p className="text-xs text-amber-600 mt-1">
                Not enough students for a team of {teamSize}. Add more students via the assessment.
              </p>
            )}
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col gap-3">
            <h2 className="font-semibold text-slate-800 text-base">Algorithm</h2>
            <div className="text-xs text-slate-500 leading-relaxed">
              <p className="mb-1">• Greedy optimization seeded by connector students</p>
              <p className="mb-1">• 2-opt improvement swaps between teams</p>
              <p>• Match confidence score per team</p>
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={generating || teamCount === 0 || loadingStudents}
            className="w-full py-4 bg-[#1e3a8a] text-white rounded-xl font-semibold text-sm hover:bg-blue-900 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shadow-md"
          >
            {generating
              ? 'Generating Teams…'
              : `Generate ${teamCount} Team${teamCount !== 1 ? 's' : ''}`}
          </button>
        </div>
      </div>

      {/* Student roster */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <h2 className="font-semibold text-slate-800 text-base">Student Pool</h2>
          <span className="text-xs text-slate-400">{students.length} students</span>
        </div>
        {loadingStudents ? (
          <div className="text-center text-slate-400 py-8 text-sm">Loading…</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 text-slate-500 text-xs uppercase tracking-wide">
                  <th className="text-left px-6 py-3 font-medium">Name</th>
                  <th className="text-left px-6 py-3 font-medium">Skills</th>
                  <th className="text-left px-6 py-3 font-medium">Level</th>
                  <th className="text-left px-6 py-3 font-medium">Conflict Style</th>
                  <th className="text-left px-6 py-3 font-medium">Leadership</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {students.map((s) => (
                  <tr key={s.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-3 font-medium text-slate-800">{s.name}</td>
                    <td className="px-6 py-3 text-slate-500">
                      {s.competence_signature.skills.slice(0, 3).join(', ')}
                    </td>
                    <td className="px-6 py-3 capitalize text-slate-500">
                      {s.competence_signature.experience_level}
                    </td>
                    <td className="px-6 py-3 capitalize text-slate-500">
                      {s.collaboration_signature.conflict_style}
                    </td>
                    <td className="px-6 py-3 capitalize text-slate-500">
                      {s.collaboration_signature.leadership_style}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 px-5 py-4 shadow-sm">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className="text-2xl font-bold text-[#1e3a8a]">{value}</p>
    </div>
  )
}
