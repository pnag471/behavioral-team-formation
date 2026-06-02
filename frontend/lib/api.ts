import type {
  AssessmentRequest,
  AssessmentResult,
  RadarData,
  Scenario,
  Student,
  Team,
  TeamExplanation,
  TeamGenerationRequest,
} from './types'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http:localhost:8000'

const JSON_HEADERS = {
  'Content-Type': 'application/json',
  Accept: 'application/json',
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: JSON_HEADERS,
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${path} failed (${res.status}): ${text}`)
  }
  return res.json() as Promise<T>
}

// Students
export const getStudents = (): Promise<Student[]> => request('/students')

export const addStudent = (student: Student): Promise<Student> =>
  request('/students', { method: 'POST', body: JSON.stringify(student) })

// Assessment
export const getScenarios = (): Promise<Scenario[]> => request('/assessment/scenarios')

export const analyzeAssessment = (req: AssessmentRequest): Promise<AssessmentResult> =>
  request('/assessment/analyze', { method: 'POST', body: JSON.stringify(req) })

// Teams
export const generateTeams = (
  req: TeamGenerationRequest,
): Promise<{ teams: Team[] }> =>
  request('/teams/generate', { method: 'POST', body: JSON.stringify(req) })

export const getTeams = (): Promise<{ teams: Team[] }> => request('/teams/teams')

export const getTeam = (teamId: string): Promise<Team> => request(`/teams/${teamId}`)

export const getTeamExplanation = (teamId: string): Promise<TeamExplanation> =>
  request(`/teams/${teamId}/explanation`)

export const getTeamRadar = (teamId: string): Promise<RadarData> =>
  request(`/teams/${teamId}/radar`)
