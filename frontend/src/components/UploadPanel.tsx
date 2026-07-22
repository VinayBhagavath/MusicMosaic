import { useMemo, useRef, useState } from 'react'
import type { JobParams } from '../api/client'

type Props = {
  onSubmit: (target: File, sources: File[], params: JobParams) => void
  busy: boolean
}

const EMPTY = [null, null, null, null, null] as (File | null)[]

export function UploadPanel({ onSubmit, busy }: Props) {
  const [target, setTarget] = useState<File | null>(null)
  const [sources, setSources] = useState<(File | null)[]>(EMPTY)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [params, setParams] = useState<JobParams>({
    window_s: 0.5,
    hop_s: 0.25,
    lambda_switch: 0.35,
  })
  const targetRef = useRef<HTMLInputElement>(null)

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
          Rebuild one song using only fragments from five others.
        </p>
      </div>

      <div className="drop-grid">
        <label className="drop target-drop">
          <span className="drop-label">Target</span>
          <span className="drop-file">{target?.name ?? 'Drop MP3'}</span>
          <input
            ref={targetRef}
            type="file"
            accept="audio/*,.mp3,.wav,.flac,.m4a"
            onChange={(e) => setTarget(e.target.files?.[0] ?? null)}
          />
        </label>
        {sources.map((s, i) => (
          <label key={i} className="drop">
            <span className="drop-label">Source {String.fromCharCode(65 + i)}</span>
            <span className="drop-file">{s?.name ?? 'Drop MP3'}</span>
            <input
              type="file"
              accept="audio/*,.mp3,.wav,.flac,.m4a"
              onChange={(e) => setSource(i, e.target.files?.[0] ?? null)}
            />
          </label>
        ))}
      </div>

      <div className="actions">
        <button
          className="cta"
          disabled={!ready || busy}
          onClick={() =>
            ready && onSubmit(target!, sources as File[], params)
          }
        >
          {busy ? 'Building…' : 'Build mosaic'}
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
