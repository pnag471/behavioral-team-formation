'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'

interface Message {
  role: 'interviewer' | 'student'
  content: string
}

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function InterviewPage() {
  const router = useRouter()
  const [name, setName] = useState('')
  const [nameSubmitted, setNameSubmitted] = useState(false)
  const [sessionId, setSessionId] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [extracting, setExtracting] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const startInterview = async () => {
    if (!name.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${BASE_URL}/conversation/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_name: name }),
      })
      const data = await res.json()
      setSessionId(data.session_id)
      setMessages([{ role: 'interviewer', content: data.message }])
      setNameSubmitted(true)
    } catch {
      setError('Could not connect to the server. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const studentMessage = input.trim()
    setInput('')

    const newMessages: Message[] = [
      ...messages,
      { role: 'student', content: studentMessage },
    ]
    setMessages(newMessages)
    setLoading(true)

    try {
      const res = await fetch(`${BASE_URL}/conversation/turn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          student_name: name,
          message: studentMessage,
          history: newMessages,
        }),
      })
      const data = await res.json()
      const updated: Message[] = [
        ...newMessages,
        { role: 'interviewer', content: data.message },
      ]
      setMessages(updated)
      setIsComplete(data.is_complete)
    } catch {
      setError('Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const extractProfile = async () => {
    setExtracting(true)
    setError('')
    try {
      const res = await fetch(`${BASE_URL}/conversation/extract?session_id=${sessionId}&student_name=${encodeURIComponent(name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(messages),
      })
      const data = await res.json()
      router.push(`/profile/${data.student_id}`)
    } catch {
      setError('Failed to extract profile. Please try again.')
    } finally {
      setExtracting(false)
    }
  }

  if (!nameSubmitted) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4">
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-8 w-full max-w-md">
          <h1 className="text-2xl font-bold text-slate-800 mb-2">Behavioral Interview</h1>
          <p className="text-slate-500 text-sm mb-6">
            You'll have a short conversation about how you work in teams. There are no right or wrong answers.
          </p>
          <label className="block text-sm font-medium text-slate-700 mb-2">Your name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && startInterview()}
            placeholder="Enter your full name"
            className="w-full border border-slate-200 rounded-lg px-4 py-3 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-300 mb-4"
          />
          {error && <p className="text-red-500 text-xs mb-3">{error}</p>}
          <button
            onClick={startInterview}
            disabled={loading || !name.trim()}
            className="w-full py-3 bg-[#1e3a8a] text-white rounded-xl font-semibold text-sm hover:bg-blue-900 transition-colors disabled:opacity-40"
          >
            {loading ? 'Starting...' : 'Start Interview'}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="font-bold text-slate-800">Behavioral Interview</h1>
          <p className="text-xs text-slate-400">Talking with {name}</p>
        </div>
        {isComplete && (
          <button
            onClick={extractProfile}
            disabled={extracting}
            className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm font-semibold hover:bg-emerald-700 transition-colors disabled:opacity-40"
          >
            {extracting ? 'Building profile...' : 'Generate My Profile →'}
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 max-w-3xl mx-auto w-full">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`mb-4 flex ${msg.role === 'student' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'student'
                  ? 'bg-[#1e3a8a] text-white rounded-br-sm'
                  : 'bg-white border border-slate-200 text-slate-700 rounded-bl-sm shadow-sm'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start mb-4">
            <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
              <div className="flex gap-1 items-center h-4">
                <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        {isComplete && !extracting && (
          <div className="text-center py-4">
            <p className="text-sm text-slate-500 mb-3">Interview complete! Ready to build your behavioral profile.</p>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      {!isComplete && (
        <div className="bg-white border-t border-slate-200 px-4 py-4">
          <div className="max-w-3xl mx-auto flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !loading && sendMessage()}
              placeholder="Type your response..."
              disabled={loading}
              className="flex-1 border border-slate-200 rounded-xl px-4 py-3 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-300 disabled:opacity-40"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="px-5 py-3 bg-[#1e3a8a] text-white rounded-xl font-semibold text-sm hover:bg-blue-900 transition-colors disabled:opacity-40"
            >
              Send
            </button>
          </div>
          {error && <p className="text-red-500 text-xs text-center mt-2">{error}</p>}
        </div>
      )}
    </div>
  )
}