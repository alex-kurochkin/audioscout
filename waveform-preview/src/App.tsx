import { useState } from 'react'
import Architecture from './components/Architecture'
import WaveformDemo from './components/WaveformDemo'
import CodeExplorer from './components/CodeExplorer'
import PeakTree from './components/PeakTree'
import Pipeline from './components/Pipeline'
import Performance from './components/Performance'
import CLIProgram from './components/CLIProgram'

type Tab = 'cli' | 'demo' | 'overview' | 'tree' | 'code' | 'pipeline' | 'perf'

const tabs: { id: Tab; label: string; icon: string; desc: string }[] = [
  { id: 'cli', label: 'CLI программа', icon: '🦀', desc: 'Rust исходники' },
  { id: 'demo', label: 'Генератор PNG', icon: '🖼️', desc: 'Работает в браузере' },
  { id: 'overview', label: 'Архитектура', icon: '🏗️', desc: 'Обзор' },
  { id: 'pipeline', label: 'Пайплайн', icon: '🔄', desc: 'От файла до пикселя' },
  { id: 'tree', label: 'Дерево Peaks', icon: '🌳', desc: 'Визуализация' },
  { id: 'code', label: 'Код SoundCraft', icon: '💻', desc: 'Исходники DAW' },
  { id: 'perf', label: 'Производительность', icon: '⚡', desc: 'Метрики' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('cli')

  return (
    <div className="min-h-screen bg-[#0a0e17] text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0d1220]">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center text-2xl shadow-lg shadow-emerald-500/20">
              🦀
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Waveform Preview Generator
              </h1>
              <p className="text-sm text-gray-400 mt-0.5">
                CLI-программа на Rust + интерактивная веб-демонстрация · алгоритм SoundCraft DAW
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b border-gray-800 bg-[#0d1220]/80 sticky top-0 z-50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 overflow-x-auto py-2">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                  tab === t.id
                    ? 'bg-emerald-500/15 text-emerald-400 shadow-inner'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                }`}
              >
                <span className="mr-1.5">{t.icon}</span>
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {tab === 'cli' && <CLIProgram />}
        {tab === 'demo' && <WaveformDemo />}
        {tab === 'overview' && <Architecture />}
        {tab === 'pipeline' && <Pipeline />}
        {tab === 'tree' && <PeakTree />}
        {tab === 'code' && <CodeExplorer />}
        {tab === 'perf' && <Performance />}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-6 text-center text-sm text-gray-500">
        <p>
          Алгоритм из open-source проекта{' '}
          <a href="https://github.com/storytold/soundcraft" className="text-emerald-400 hover:underline">
            storytold/soundcraft
          </a>{' '}
          · Rust DAW
        </p>
      </footer>
    </div>
  )
}
