import Link from 'next/link'

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="bg-gradient-to-br from-[#1e3a8a] to-[#1e40af] text-white py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-800 text-blue-200 text-xs font-medium mb-6 tracking-wide">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 inline-block"></span>
            CS Research Prototype · 2026
          </div>
          <h1 className="text-5xl font-bold tracking-tight mb-4 leading-tight">
            Behavioral Team Formation
          </h1>
          <p className="text-xl text-blue-100 max-w-2xl mx-auto mb-3 leading-relaxed">
            AI-assisted team formation using behavioral signatures and explainable matching.
          </p>
          <p className="text-sm text-blue-300 mb-10">
            Grounded in team dynamics research · Designed for CS project courses
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link
              href="/assessment"
              className="px-6 py-3 bg-white text-[#1e3a8a] rounded-lg font-semibold text-sm hover:bg-blue-50 transition-colors shadow-md"
            >
              Start Assessment
            </Link>
            <Link
              href="/dashboard"
              className="px-6 py-3 border border-blue-300 text-white rounded-lg font-semibold text-sm hover:bg-blue-800 transition-colors"
            >
              Instructor Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* Feature cards */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <h2 className="text-2xl font-semibold text-center text-slate-700 mb-10">
          How it works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <FeatureCard
            icon="🧠"
            title="Behavioral Assessment"
            description="Students respond to six realistic team scenarios. A behavioral signature is inferred automatically — covering communication, conflict resolution, leadership, accountability, and planning style."
          />
          <FeatureCard
            icon="⬡"
            title="Intelligent Matching"
            description="A greedy optimization algorithm forms teams of four by maximizing a weighted score across skill coverage, behavioral compatibility, availability overlap, and shared interests."
          />
          <FeatureCard
            icon="🔍"
            title="Explainable Teams"
            description="Every team comes with a compatibility score, match confidence rating, behavioral radar chart, strengths, risk factors, and auto-generated working norms."
          />
        </div>
      </section>

      {/* Research context */}
      <section className="bg-white border-t border-slate-200 py-12 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-xl font-semibold text-slate-700 mb-3">Research Context</h2>
          <p className="text-slate-500 text-sm leading-relaxed">
            This prototype explores how behavioral data — beyond GPA, skill lists, or self-reported preferences — can
            improve team composition in academic settings. The matching engine is designed for transparency: every
            grouping decision is scored and explained, making it auditable by instructors and legible to students.
            Future versions will incorporate LLM-powered behavioral inference and longitudinal performance feedback.
          </p>
        </div>
      </section>
    </div>
  )
}

function FeatureCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow">
      <div className="text-3xl mb-3">{icon}</div>
      <h3 className="text-base font-semibold text-slate-800 mb-2">{title}</h3>
      <p className="text-sm text-slate-500 leading-relaxed">{description}</p>
    </div>
  )
}
