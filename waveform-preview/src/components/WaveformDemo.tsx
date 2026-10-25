import { useState, useRef, useCallback, useEffect } from 'react'

// ─── Алгоритм SoundCraft Peaks ─────────────────────────────────────────────
const BLOCK = 32
const FANOUT = 4

interface Block {
  min: number
  max: number
  squares: number
}

const EMPTY_BLOCK: Block = { min: Infinity, max: -Infinity, squares: 0 }

function blockPoint(s: number): Block {
  return { min: s, max: s, squares: Number.isFinite(s) ? s * s : 0 }
}

function blockOf(samples: Float32Array | number[], start: number, end: number): Block {
  let b = { ...EMPTY_BLOCK }
  for (let i = start; i < end; i++) {
    const s = samples[i]
    const p = blockPoint(s)
    b = blockMerge(b, p)
  }
  return b
}

function blockMerge(a: Block, b: Block): Block {
  return {
    min: Math.min(a.min, b.min),
    max: Math.max(a.max, b.max),
    squares: a.squares + b.squares,
  }
}

function buildLevels(samples: Float32Array): Block[][] {
  if (samples.length === 0) return []
  const levels: Block[][] = []
  // Level 0
  const level0: Block[] = []
  for (let i = 0; i < samples.length; i += BLOCK) {
    level0.push(blockOf(samples, i, Math.min(i + BLOCK, samples.length)))
  }
  levels.push(level0)
  // Higher levels
  let current = level0
  while (current.length > 1) {
    const next: Block[] = []
    for (let i = 0; i < current.length; i += FANOUT) {
      const group = current.slice(i, Math.min(i + FANOUT, current.length))
      let merged = { ...EMPTY_BLOCK }
      for (const b of group) merged = blockMerge(merged, b)
      next.push(merged)
    }
    levels.push(next)
    current = next
  }
  return levels
}

function querySum(levels: Block[][], samples: Float32Array, start: number, end: number): Block {
  end = Math.min(end, samples.length)
  const first = Math.ceil(start / BLOCK)
  const last = Math.floor(end / BLOCK)
  const raw = (a: number, b: number) => blockOf(samples, a, b)
  if (first >= last) return raw(start, end)

  // Combine raw edges + tree middle
  let acc = blockMerge(raw(start, first * BLOCK), raw(last * BLOCK, end))

  // Tree walk for middle
  let lo = first, hi = last
  for (let lvl = 0; lvl < levels.length; lvl++) {
    const level = levels[lvl]
    const add = (i: number) => {
      if (i >= 0 && i < level.length) acc = blockMerge(acc, level[i])
    }
    if (lvl + 1 === levels.length || hi - lo < 2 * FANOUT) {
      for (let i = lo; i < hi; i++) add(i)
      break
    }
    while (lo % FANOUT !== 0) { add(lo); lo++ }
    while (hi % FANOUT !== 0) { hi--; add(hi) }
    lo = Math.floor(lo / FANOUT)
    hi = Math.floor(hi / FANOUT)
  }
  return acc
}

function querySpans(levels: Block[][], samples: Float32Array, start: number, step: number, cols: number): [number, number][] {
  const result: [number, number][] = []
  for (let c = 0; c < cols; c++) {
    const a = Math.max(0, Math.floor(start + step * c))
    const b = Math.min(samples.length, Math.ceil(start + step * (c + 1)))
    const block = querySum(levels, samples, a, b)
    result.push(block.min > block.max ? [0, 0] : [block.min, block.max])
  }
  return result
}

function queryRmsSpans(levels: Block[][], samples: Float32Array, start: number, step: number, cols: number): number[] {
  const result: number[] = []
  for (let c = 0; c < cols; c++) {
    const a = Math.max(0, Math.floor(start + step * c))
    const b = Math.min(samples.length, Math.ceil(start + step * (c + 1)))
    const block = querySum(levels, samples, a, b)
    const count = Math.max(1, b - a)
    result.push(Math.sqrt(block.squares / count))
  }
  return result
}

// ─── Рендеринг ─────────────────────────────────────────────────────────────
type Style = 'dark' | 'light' | 'gradient'
type Mode = 'peak' | 'rms'

interface RenderOpts {
  width: number
  height: number
  style: Style
  mode: Mode
  padding: number
}

function renderWaveform(
  ctx: CanvasRenderingContext2D,
  samples: Float32Array,
  levels: Block[][],
  opts: RenderOpts
) {
  const { width, height, style, mode, padding } = opts
  const w = width
  const h = height

  // Background
  if (style === 'dark') {
    ctx.fillStyle = '#0d1220'
  } else if (style === 'light') {
    ctx.fillStyle = '#ffffff'
  } else {
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, '#0a0e17')
    grad.addColorStop(0.5, '#111827')
    grad.addColorStop(1, '#0a0e17')
    ctx.fillStyle = grad
  }
  ctx.fillRect(0, 0, w, h)

  // Grid lines
  ctx.strokeStyle = style === 'light' ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.05)'
  ctx.lineWidth = 1
  for (let i = 1; i < 4; i++) {
    const y = (h / 4) * i
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(w, y)
    ctx.stroke()
  }

  // Center line
  ctx.strokeStyle = style === 'light' ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.1)'
  ctx.beginPath()
  ctx.moveTo(0, h / 2)
  ctx.lineTo(w, h / 2)
  ctx.stroke()

  if (samples.length === 0) return

  // Get spans
  const step = samples.length / w
  const columns: [number, number][] = mode === 'rms'
    ? queryRmsSpans(levels, samples, 0, step, w).map(r => [-r, r])
    : querySpans(levels, samples, 0, step, w)

  // Draw columns
  const margin = padding
  const drawH = h - margin * 2
  const cy = h / 2

  // Gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, h)
  if (style === 'light') {
    grad.addColorStop(0, '#059669')
    grad.addColorStop(0.5, '#0d9488')
    grad.addColorStop(1, '#059669')
  } else {
    grad.addColorStop(0, '#10b981')
    grad.addColorStop(0.5, '#06b6d4')
    grad.addColorStop(1, '#10b981')
  }
  ctx.fillStyle = grad

  for (let x = 0; x < w; x++) {
    const [lo, hi] = columns[x]
    const clamp = (v: number) => Math.max(-1, Math.min(1, v))
    const y1 = cy - clamp(hi) * (drawH / 2)
    const y2 = cy - clamp(lo) * (drawH / 2)
    const barH = Math.max(1, y2 - y1)
    ctx.fillRect(x, y1, 1, barH)
  }
}

// ─── Компонент ─────────────────────────────────────────────────────────────
export default function WaveformDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [samples, setSamples] = useState<Float32Array | null>(null)
  const [levels, setLevels] = useState<Block[][] | null>(null)
  const [fileName, setFileName] = useState('')
  const [sampleRate, setSampleRate] = useState(0)
  const [duration, setDuration] = useState(0)
  const [buildTime, setBuildTime] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  // Render options
  const [width, setWidth] = useState(1200)
  const [height, setHeight] = useState(200)
  const [style, setStyle] = useState<Style>('dark')
  const [mode, setMode] = useState<Mode>('peak')
  const [padding, setPadding] = useState(8)

  const processAudio = useCallback(async (buffer: ArrayBuffer, name: string) => {
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
      const audioBuffer = await audioCtx.decodeAudioData(buffer)
      // Mixdown to mono
      const numCh = audioBuffer.numberOfChannels
      const len = audioBuffer.length
      const mono = new Float32Array(len)
      for (let ch = 0; ch < numCh; ch++) {
        const data = audioBuffer.getChannelData(ch)
        for (let i = 0; i < len; i++) mono[i] += data[i] / numCh
      }

      setFileName(name)
      setSampleRate(audioBuffer.sampleRate)
      setDuration(audioBuffer.duration)

      // Build peaks
      const t0 = performance.now()
      const lvls = buildLevels(mono)
      setBuildTime(performance.now() - t0)

      setSamples(mono)
      setLevels(lvls)
    } catch (e) {
      console.error('Failed to decode:', e)
      alert('Не удалось декодировать аудиофайл. Попробуйте WAV или MP3.')
    }
  }, [])

  const handleFile = useCallback((file: File) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        processAudio(reader.result, file.name)
      }
    }
    reader.readAsArrayBuffer(file)
  }, [processAudio])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  // Generate demo signal
  const generateDemo = useCallback((type: string) => {
    const sr = 44100
    const durationSec = 5
    const len = sr * durationSec
    const s = new Float32Array(len)
    for (let i = 0; i < len; i++) {
      const t = i / sr
      const p = i / len
      switch (type) {
        case 'sine':
          s[i] = Math.sin(t * Math.PI * 2 * 440) * 0.6 * Math.exp(-p * 2)
          break
        case 'complex':
          s[i] = (Math.sin(t * Math.PI * 2 * 3) * 0.5 +
            Math.sin(t * Math.PI * 2 * 12) * 0.3 +
            Math.sin(t * Math.PI * 2 * 27) * 0.15) * Math.exp(-p * 2)
          break
        case 'drums': {
          const beat = (i % 11025) / 11025
          const kick = Math.sin(beat * Math.PI * 2 * 2) * Math.exp(-beat * 8) * 0.9
          const hihat = (Math.random() * 2 - 1) * Math.exp(-((i % 5512) / 5512) * 20) * 0.3
          const snare = ((i % 22050) > 11025) ? (Math.random() * 2 - 1) * Math.exp(-((i % 11025) / 11025) * 10) * 0.5 : 0
          s[i] = kick + hihat + snare
          break
        }
        default:
          s[i] = Math.sin(t * Math.PI * 2 * 5) * 0.7
      }
    }
    const t0 = performance.now()
    const lvls = buildLevels(s)
    setBuildTime(performance.now() - t0)
    setSamples(s)
    setLevels(lvls)
    setSampleRate(sr)
    setDuration(durationSec)
    setFileName(`demo-${type}.wav`)
  }, [])

  // Render
  useEffect(() => {
    if (!canvasRef.current || !samples || !levels) return
    const canvas = canvasRef.current
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${Math.min(width, 1200)}px`
    canvas.style.height = `${height}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    renderWaveform(ctx, samples, levels, { width, height, style, mode, padding })
  }, [samples, levels, width, height, style, mode, padding])

  // Download PNG
  const downloadPNG = useCallback(() => {
    if (!canvasRef.current) return
    canvasRef.current.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = fileName.replace(/\.[^.]+$/, '') + '-waveform.png'
      a.click()
      URL.revokeObjectURL(url)
    }, 'image/png')
  }, [fileName])

  return (
    <div className="space-y-6">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
          isDragging
            ? 'border-emerald-400 bg-emerald-500/10'
            : 'border-gray-700 bg-[#111827] hover:border-gray-600'
        }`}
      >
        <div className="text-4xl mb-3">🎵</div>
        <h3 className="text-xl font-semibold mb-2">Перетащите аудиофайл сюда</h3>
        <p className="text-gray-400 text-sm mb-4">
          Поддерживаются WAV, MP3, OGG, FLAC, AAC, M4A (декодируется через Web Audio API)
        </p>
        <label className="inline-block">
          <input
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <span className="px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-lg cursor-pointer hover:bg-emerald-500/30 transition-colors text-sm font-medium">
            Выбрать файл
          </span>
        </label>
        <div className="mt-4 text-xs text-gray-500">или выберите демо-сигнал:</div>
        <div className="flex gap-2 justify-center mt-2 flex-wrap">
          {['complex', 'sine', 'drums'].map(type => (
            <button
              key={type}
              onClick={() => generateDemo(type)}
              className="px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 transition-colors"
            >
              {type === 'complex' ? '🎼 Комплексный' : type === 'sine' ? '🎵 Синус' : '🥁 Барабаны'}
            </button>
          ))}
        </div>
      </div>

      {/* File info */}
      {samples && (
        <div className="rounded-xl bg-[#111827] border border-gray-800 p-4">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-2xl">📄</span>
            <div>
              <div className="font-semibold">{fileName}</div>
              <div className="text-xs text-gray-500">
                {samples.length.toLocaleString()} семплов · {sampleRate} Hz · {duration.toFixed(2)}s · Peaks built in {buildTime.toFixed(2)}ms
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="rounded-xl bg-[#111827] border border-gray-800 p-6">
        <h3 className="text-lg font-semibold mb-4">⚙️ Настройки рендеринга</h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Ширина: {width}px</label>
            <input
              type="range"
              min={200}
              max={3000}
              step={50}
              value={width}
              onChange={(e) => setWidth(Number(e.target.value))}
              className="w-full accent-emerald-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Высота: {height}px</label>
            <input
              type="range"
              min={50}
              max={600}
              step={10}
              value={height}
              onChange={(e) => setHeight(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Отступ: {padding}px</label>
            <input
              type="range"
              min={0}
              max={40}
              value={padding}
              onChange={(e) => setPadding(Number(e.target.value))}
              className="w-full accent-purple-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-2">Режим</label>
            <div className="flex gap-1">
              {(['peak', 'rms'] as const).map(m => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`flex-1 py-1.5 rounded text-xs font-medium ${
                    mode === m ? 'bg-emerald-500/20 text-emerald-400' : 'bg-gray-800 text-gray-400'
                  }`}
                >
                  {m === 'peak' ? 'Peak' : 'RMS'}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex gap-1 mb-4">
          <label className="text-xs text-gray-500 mr-2 self-center">Стиль:</label>
          {(['dark', 'light', 'gradient'] as const).map(s => (
            <button
              key={s}
              onClick={() => setStyle(s)}
              className={`px-3 py-1.5 rounded text-xs font-medium ${
                style === s ? 'bg-emerald-500/20 text-emerald-400' : 'bg-gray-800 text-gray-400'
              }`}
            >
              {s === 'dark' ? '🌙 Dark' : s === 'light' ? '☀️ Light' : '🌈 Gradient'}
            </button>
          ))}
        </div>

        {/* Canvas */}
        {samples && (
          <div className="rounded-lg overflow-hidden border border-gray-700 mb-4 bg-[#0a0e17]">
            <canvas ref={canvasRef} />
          </div>
        )}

        {/* Download */}
        {samples && (
          <button
            onClick={downloadPNG}
            className="w-full py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-white font-semibold rounded-lg transition-all shadow-lg shadow-emerald-500/20"
          >
            💾 Скачать PNG ({width}×{height})
          </button>
        )}
      </div>

      {/* Info */}
      <div className="rounded-xl bg-[#111827] border border-gray-800 p-6">
        <h3 className="text-lg font-semibold mb-3">🔬 Что здесь происходит</h3>
        <div className="text-sm text-gray-400 space-y-2">
          <p>
            <strong className="text-white">1.</strong> Аудио декодируется через Web Audio API (тот же принцип, что symphonia в Rust — packet-by-packet → planar f32)
          </p>
          <p>
            <strong className="text-white">2.</strong> Многоканальное → моно (mixdown, как в CLI программе)
          </p>
          <p>
            <strong className="text-white">3.</strong> Строится дерево Peaks: Level 0 = 32 семпла/блок, FANOUT = 4
          </p>
          <p>
            <strong className="text-white">4.</strong> Для каждого пикселя ширины — запрос <code className="text-emerald-400">spans()</code> или <code className="text-emerald-400">rms_spans()</code>
          </p>
          <p>
            <strong className="text-white">5.</strong> Canvas рендерит столбцы → экспорт в PNG через <code className="text-emerald-400">canvas.toBlob()</code>
          </p>
          <p className="text-emerald-400 pt-2">
            💡 Это та же логика, что в Rust CLI программе на вкладке «🦀 CLI программа». Попробуйте загрузить свой файл!
          </p>
        </div>
      </div>
    </div>
  )
}
