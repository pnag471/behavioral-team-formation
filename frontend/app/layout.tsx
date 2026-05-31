import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import Navigation from '@/components/Navigation'
import './globals.css'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'Behavioral Team Formation',
  description:
    'AI-assisted team formation using behavioral signatures and explainable matching — research prototype',
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-[--color-surface] text-[--color-foreground]">
        <Navigation />
        <main className="flex-1">{children}</main>
        <footer className="border-t border-[--color-border] mt-auto py-4 text-center text-xs text-[--color-muted]">
          Behavioral Team Formation · Research Prototype · CS Department
        </footer>
      </body>
    </html>
  )
}
