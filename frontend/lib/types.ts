// ---------------------------------------------------------------------------
// Backend model mirrors — keep in sync with backend/app/models.py
// ---------------------------------------------------------------------------

export interface CompetenceSignature {
  skills: string[]
  roles: string[]
  experience_level: string
}

export interface WorkRhythmSignature {
  planning_style: string
  communication_style: string
  execution_style: string
  availability: string[]
}

export interface CollaborationSignature {
  conflict_style: string
  leadership_style: string
  accountability: string
  help_seeking: string
}

export interface MotivationLayer {
  interests: string[]
  learning_goals: string[]
}

export interface ConfidenceLayer {
  confidence_score: number
}

export interface Student {
  id: string
  name: string
  competence_signature: CompetenceSignature
  work_rhythm_signature: WorkRhythmSignature
  collaboration_signature: CollaborationSignature
  motivation_layer: MotivationLayer
  confidence_layer: ConfidenceLayer
}

// Assessment

export interface ScenarioOption {
  id: string
  label: string
}

export interface Scenario {
  id: string
  title: string
  description: string
  options: ScenarioOption[]
}

export interface AssessmentResponse {
  scenario_id: string
  option_id: string
  response_text: string
}

export interface AssessmentRequest {
  student_name: string
  student_id?: string
  responses: AssessmentResponse[]
}

export interface BehavioralSignature {
  planning_style: string
  communication_style: string
  execution_style: string
  conflict_style: string
  leadership_style: string
  accountability: string
  help_seeking: string
  confidence_score: number
}

export interface AssessmentResult {
  student_id: string
  signature: BehavioralSignature
}

// Teams

export interface TeamMember {
  student_id: string
  name: string
  roles: string[]
  experience_level: string
}

export interface TeamScoreBreakdown {
  skill_coverage: number
  behavioral_compat: number
  availability_overlap: number
  conflict_risk: number
  match_confidence: number
  total: number
}

export interface TeamNorm {
  category: string
  norm: string
}

export interface Team {
  id: string
  members: TeamMember[]
  score_breakdown: TeamScoreBreakdown
  team_norms: TeamNorm[]
}

export interface TeamGenerationRequest {
  team_size: number
  weights: {
    skill_coverage: number
    behavioral_compat: number
    availability_overlap: number
    shared_interests: number
  }
}

export interface TeamExplanation {
  team_id: string
  compatibility_score: number
  match_confidence: number
  strengths: string[]
  risks: string[]
  explanation: string
  team_norms: TeamNorm[]
}

// Radar

export interface MemberRadarData {
  student_id: string
  name: string
  values: number[]
}

export interface RadarData {
  labels: string[]
  values: number[]
  members_radar: MemberRadarData[]
}
