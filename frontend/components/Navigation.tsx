import Link from 'next/link'

export default function Navigation() {
  return (
    <header className="bg-[#1e3a8a] text-white shadow-md">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-semibold text-lg tracking-tight hover:opacity-90">
          <span className="text-[#93c5fd]">⬡</span>
          <span>Behavioral Team Formation</span>
        </Link>
        <nav className="flex items-center gap-6 text-sm font-medium">
          <Link href="/dashboard" className="text-blue-100 hover:text-white transition-colors">
            Dashboard
          </Link>
          <Link href="/interview" className="text-blue-100 hover:text-white transition-colors">
            Assessment
          </Link>
          <Link href="/teams" className="text-blue-100 hover:text-white transition-colors">
            Teams
          </Link>
          <span className="ml-2 px-2 py-0.5 rounded-full bg-blue-800 text-blue-200 text-xs font-medium tracking-wide">
            Research Prototype
          </span>
        </nav>
      </div>
    </header>
  )
}
