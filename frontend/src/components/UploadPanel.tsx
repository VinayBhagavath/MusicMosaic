import { useMemo, useState } from 'react'
import type { AudioInput, JobParams } from '../api/client'

type Props = {
  onSubmit: (target: AudioInput, sources: AudioInput[], params: JobParams) => void
  busy: boolean
}

const EMPTY: AudioInput[] = [
  { file: null, url: '' },
  { file: null, url: '' },
  { file: null, url: '' },
  { file: null, url: '' },
  { file: null, url: '' },
]
const PIGMENTS = ['#C45C26', '#C9A227', '#2F6F5E', '#3D5A80', '#A24B6F']

function filled(a: AudioInput) {
  return !!(a.file || a.url.trim())
}

export function UploadPanel({ onSubmit, busy }: Props) {
  const [target, setTarget] = useState<AudioInput>({ file: null, url: '' })
  const [sources, setSources] = useState<AudioInput[]>(EMPTY)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [params, setParams] = useState<JobParams>({
    window_s: 0.5,
    hop_s: 0.25,
    lambda_switch: 0.35,
  })

  const ready = useMemo(
    () => filled(target) && sources.every(filled) && params.hop_s <= params.window_s,
    [target, sources, params],
  )

  const setSource = (i: number, patch: Partial<AudioInput>) => {
    setSources((prev) =>
      prev.map((x, j) => {
        if (j !== i) return x
        const next = { ...x, ...patch }
        // File and URL are mutually exclusive
        if (patch.file) next.url = ''
        if (patch.url !== undefined && patch.url !== '') next.file = null
        return next
      }),
    )
  }

  return (
    <section className="upload">
      <div className="hero-copy">
        <p className="eyebrow">Acoustic collage</p>
        <h1 className="brand">Music Mosaic</h1>
        <p className="lede">
          Rebuild one song as a quilt of fragments — upload MP3s or paste YouTube links.
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
        <div className={`drop target-drop${filled(target) ? ' has-file' : ''}`}>
          <span className="drop-label">Target · MP3 or YouTube</span>
          <label className="file-hit">
            <span className="drop-file">
              {target.file?.name ?? 'Choose MP3'}
            </span>
            <input
              type="file"
              accept=".mp3,audio/mpeg"
              disabled={busy}
              onChange={(e) =>
                setTarget({ file: e.target.files?.[0] ?? null, url: '' })
              }
            />
          </label>
          <input
            className="url-input"
            type="url"
            placeholder="or paste a YouTube URL"
            value={target.url}
            disabled={busy}
            onChange={(e) => setTarget({ file: null, url: e.target.value })}
          />
        </div>

        {sources.map((s, i) => (
          <div key={i} className={`drop${filled(s) ? ' has-file' : ''}`}>
            <span
              className="drop-swatch"
              style={{ background: PIGMENTS[i] }}
              aria-hidden
            />
            <span className="drop-label">Source {String.fromCharCode(65 + i)}</span>
            <label className="file-hit">
              <span className="drop-file">{s.file?.name ?? 'MP3'}</span>
              <input
                type="file"
                accept=".mp3,audio/mpeg"
                disabled={busy}
                onChange={(e) =>
                  setSource(i, { file: e.target.files?.[0] ?? null, url: '' })
                }
              />
            </label>
            <input
              className="url-input"
              type="url"
              placeholder="YouTube URL"
              value={s.url}
              disabled={busy}
              onChange={(e) => setSource(i, { url: e.target.value, file: null })}
            />
          </div>
        ))}
      </div>

      <p className="hint">
        Tracks must be 5 seconds–8 minutes. YouTube downloads need ffmpeg + network.
      </p>

      <div className="actions">
        <button
          className="cta"
          disabled={!ready || busy || submitting}
          onClick={() => {
            if (!ready || busy || submitting) return
            setSubmitting(true)
            onSubmit(target, sources, params)
          }}
        >
          {busy || submitting ? 'Starting…' : 'Compose mosaic'}
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
          {params.hop_s > params.window_s && (
            <p className="error" style={{ gridColumn: '1 / -1' }}>
              Hop must be ≤ window or the mosaic will have silent gaps.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
