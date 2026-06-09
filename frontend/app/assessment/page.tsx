'use client'

import { useEffect, useState } from 'react'
import RadarChart from '@/components/RadarChart'
import { analyzeAssessment, getScenarios } from '@/lib/api'
import type { AssessmentResult, BehavioralSignature, RadarData, Scenario } from '@/lib/types'

const RADAR_LABELS = ['Communication', 'Conflict Handling', 'Leadership', 'Accountability', 'Planning']

const COMM_MAP: Record<string, number>    = { async: 0.35, mixed: 0.65, sync: 0.9 }
const CONFLICT_MAP: Record<string, number> = { avoidant: 0.2, confrontational: 0.6, collaborative: 0.85 }
const LEADER_MAP: Record<string, number>  = { emergent: 0.3, facilitative: 0.7, directive: 0.92 }
const ACCOUNT_MAP: Record<string, number> = { low: 0.2, medium: 0.6, high: 0.9 }
const PLAN_MAP: Record<string, number>    = { spontaneous: 0.2, adaptive: 0.6, planner: 0.9 }

function signatureToRadar(sig: BehavioralSignature): RadarData {
  return {
    labels: RADAR_LABELS,
    values: [
      COMM_MAP[sig.communication_style] ?? 0.5,
      CONFLICT_MAP[sig.conflict_style] ?? 0.5,
      LEADER_MAP[sig.leadership_style] ?? 0.5,
      ACCOUNT_MAP[sig.accountability] ?? 0.5,
      PLAN_MAP[sig.planning_style] ?? 0.5,
    ],
    members_radar: [],
  }
}

function Badge({ text, variant = 'blue' }: { text: string; variant?: 'blue' | 'green' | 'amber' }) {
  const colors: Record<string, string> = {
    blue:  'bg-blue-100 text-blue-800',
    green: 'bg-emerald-100 text-emerald-800',
    amber: 'bg-amber-100 text-amber-800',
  }
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${colors[variant]}`}>
      {text}
    </span>
  )
}

export default function AssessmentPage() {
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [studentName, setStudentName] = useState('')
  const [responses, setResponses] = useState<Record<string, { option_id: string; response_text: string }>>({})
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<AssessmentResult | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getScenarios()
      .then(setScenarios)
      .catch(() => setError('Could not load scenarios. Is the backend running?'))
      .finally(() => setLoading(false))
  }, [])

  const setOptionId = (scenarioId: string, optionId: string) =>
    setResponses((prev) => ({
      ...prev,
      [scenarioId]: { ...(prev[scenarioId] ?? { response_text: '' }), option_id: optionId },
    }))

  const setResponseText = (scenarioId: string, text: string) =>
    setResponses((prev) => ({
      ...prev,
      [scenarioId]: { ...(prev[scenarioId] ?? { option_id: '' }), response_text: text },
    }))

  const allAnswered =
    scenarios.length > 0 &&
    scenarios.every((s) => responses[s.id]?.option_id)

  const handleSubmit = async () => {
    if (!studentName.trim()) { setError('Please enter your name.'); return }
    if (!allAnswered) { setError('Please answer all scenarios before submitting.'); return }
    setError('')
    setSubmitting(true)
    try {
      const res = await analyzeAssessment({
        student_name: studentName.trim(),
        responses: scenarios.map((s) => ({
          scenario_id: s.id,
          option_id: responses[s.id]?.option_id ?? '',
          response_text: responses[s.id]?.response_text ?? '',
        })),
      })
      setResult(res)
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Submission failed.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800 mb-2">Behavioral Assessment</h1>
        <p className="text-slate-500 text-sm">
          Respond to each scenario honestly. Your answers shape your behavioral signature, which the team formation
          engine uses to find your best-fit teammates.
        </p>
      </div>

      {/* Result panel */}
      {result && (
        <div className="mb-8 bg-white border border-emerald-200 rounded-2xl shadow-sm overflow-hidden">
          <div className="bg-emerald-50 border-b border-emerald-200 px-6 py-4">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-emerald-800">Assessment Complete</h2>
              <Badge text={`ID: ${result.student_id}`} variant="green" />
            </div>
            <p className="text-sm text-emerald-700 mt-1">
              Your behavioral signature has been saved. You are now part of the student pool.
            </p>
          </div>
          <div className="px-6 py-5 flex flex-col md:flex-row gap-6 items-start">
            <div className="flex-shrink-0">
              <RadarChart data={signatureToRadar(result.signature)} size={260} />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-slate-700 mb-3">Your Behavioral Profile</h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(result.signature)
                  .filter(([k]) => k !== 'confidence_score')
                  .map(([key, val]) => (
                    <div key={key} className="flex flex-col">
                      <span className="text-slate-400 text-xs capitalize">{key.replace(/_/g, ' ')}</span>
                      <span className="font-medium text-slate-700 capitalize">{String(val)}</span>
                    </div>
                  ))}
              </div>
              <div className="mt-3 flex items-center gap-2">
                <span className="text-slate-400 text-xs">Confidence score</span>
                <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#1e3a8a] rounded-full"
                    style={{ width: `${Math.round(result.signature.confidence_score * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-medium text-slate-600">
                  {Math.round(result.signature.confidence_score * 100)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Name input */}
      <div className="mb-6 bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
        <label className="block text-sm font-semibold text-slate-700 mb-2">Your Name</label>
        <input
          type="text"
          value={studentName}
          onChange={(e) => setStudentName(e.target.value)}
          placeholder="Enter your full name"
          className="w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-300"
        />
      </div>

      {/* Scenarios */}
      {loading ? (
        <div className="text-center text-slate-400 py-12">Loading scenarios…</div>
      ) : (
        <div className="flex flex-col gap-6">
          {scenarios.map((scenario, idx) => (
            <div key={scenario.id} className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-100">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-semibold text-slate-400">Scenario {idx + 1} of {scenarios.length}</span>
                  {responses[scenario.id]?.option_id && (
                    <Badge text="Answered" variant="green" />
                  )}
                </div>
                <h2 className="text-base font-semibold text-slate-800">{scenario.title}</h2>
              </div>
              <div className="px-6 py-4">
                <p className="text-sm text-slate-600 leading-relaxed mb-4">{scenario.description}</p>
                <div className="flex flex-col gap-2 mb-4">
                  {scenario.options.map((opt) => (
                    <label
                      key={opt.id}
                      className={`flex items-center gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors text-sm ${
                        responses[scenario.id]?.option_id === opt.id
                          ? 'border-blue-400 bg-blue-50 text-blue-800'
                          : 'border-slate-200 text-slate-700 hover:bg-slate-50'
                      }`}
                    >
                      <input
                        type="radio"
                        name={scenario.id}
                        value={opt.id}
                        checked={responses[scenario.id]?.option_id === opt.id}
                        onChange={() => setOptionId(scenario.id, opt.id)}
                        className="accent-blue-700"
                      />
                      {opt.label}
                    </label>
                  ))}
                </div>
                <textarea
                  rows={3}
                  value={responses[scenario.id]?.response_text ?? ''}
                  onChange={(e) => setResponseText(scenario.id, e.target.value)}
                  placeholder="Optional: elaborate on your choice or describe what you'd actually do…"
                  className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-700 resize-none focus:outline-none focus:ring-2 focus:ring-blue-200 placeholder-slate-300"
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Submit */}
      {!loading && scenarios.length > 0 && (
        <div className="mt-8 flex items-center justify-between">
          <span className="text-sm text-slate-400">
            {scenarios.filter((s) => responses[s.id]?.option_id).length} / {scenarios.length} answered
          </span>
          <button
            onClick={handleSubmit}
            disabled={submitting || !allAnswered || !studentName.trim()}
            className="px-8 py-3 bg-[#1e3a8a] text-white rounded-lg font-semibold text-sm hover:bg-blue-900 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {submitting ? 'Analyzing…' : 'Submit Assessment'}
          </button>
        </div>
      )}
    </div>
  )
}
