import { useMemo, useState } from 'react'
import type { JobParams } from '../api/client'

type Props = {
  onSubmit: (target: File, sources: File[], params: JobParams) => void
  busy: boolean
}

const EMPTY = [null, null, null, null, null] as (File | null)[]
const PIGMENTS = ['#C45C26', '#C9A227', '#2F6F5E', '#3D5A80', '#A24B6F']

export function UploadPanel({ onSubmit, busy }: Props) {
  const [target, setTarget] = useState<File | null>(null)
  const [sources, setSources] = useState<(File | null)[]>(EMPTY)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [params, setParams] = useState<JobParams>({
    window_s: 0.5,
    hop_s: 0.25,
    lambda_switch: 0.35,
  })

  const ready = useMemo(
    () => !!target && sources.every(Boolean),
    [target, sources],
  )

  const setSource = (i: number, f: File | null) => {
    setSources((prev) => prev.map((x, j) => (j === i ? f : x)))
  }

  return (
    <section className="upload">
      <div className="hero-copy">
        <p className="eyebrow">Acoustic collage</p>
        <h1 className="brand">Music Mosaic</h1>
        <p className="lede">
          Rebuild one song as a quilt of fragments — using only five other tracks.
        </p>
        <div className="hero-mosaic" aria-hidden>
          {Array.from({ length: 36 }, (_, i) => (
            <span
              key={i}
              style={{
                background: PIGMENTS[i % PIGMENTS.length],
                opacity: 0.35 + ((i * 17) % 50) / 100,
                animationDelay: `${(i % 12) * 40}ms`,
              }}
            />
          ))}
        </div>
      </div>

      <div className="drop-grid">
        <label className={`drop target-drop${target ? ' has-file' : ''}`}>
          <span className="drop-label">Target song · MP3</span>
          <span className="drop-file">{target?.name ?? 'Drop or choose the song to reconstruct'}</span>
          <input
            type="file"
            accept=".mp3,audio/mpeg"
            onChange={(e) => setTarget(e.target.files?.[0] ?? null)}
          />
        </label>
        {sources.map((s, i) => (
          <label key={i} className={`drop${s ? ' has-file' : ''}`}>
            <span
              className="drop-swatch"
              style={{ background: PIGMENTS[i] }}
              aria-hidden
            />
            <span className="drop-label">Source {String.fromCharCode(65 + i)}</span>
            <span className="drop-file">{s?.name ?? 'MP3'}</span>
            <input
              type="file"
              accept=".mp3,audio/mpeg"
              onChange={(e) => setSource(i, e.target.files?.[0] ?? null)}
            />
          </label>
        ))}
      </div>

      <div className="actions">
        <button
          className="cta"
          disabled={!ready || busy}
          onClick={() => ready && onSubmit(target!, sources as File[], params)}
        >
          Compose mosaic
        </button>
        <button
          type="button"
          className="linkish"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          {showAdvanced ? 'Hide' : 'Advanced'} parameters
        </button>
      </div>

      {showAdvanced && (
        <div className="advanced">
          <label>
            Window (s)
            <input
              type="number"
              step={0.05}
              min={0.1}
              max={2}
              value={params.window_s}
              onChange={(e) =>
                setParams((p) => ({ ...p, window_s: Number(e.target.value) }))
              }
            />
          </label>
          <label>
            Hop (s)
            <input
              type="number"
              step={0.05}
              min={0.05}
              max={1}
              value={params.hop_s}
              onChange={(e) =>
                setParams((p) => ({ ...p, hop_s: Number(e.target.value) }))
              }
            />
          </label>
          <label>
            Switch penalty
            <input
              type="number"
              step={0.05}
              min={0}
              max={2}
              value={params.lambda_switch}
              onChange={(e) =>
                setParams((p) => ({
                  ...p,
                  lambda_switch: Number(e.target.value),
                }))
              }
            />
          </label>
        </div>
      )}
    </section>
  )
}
