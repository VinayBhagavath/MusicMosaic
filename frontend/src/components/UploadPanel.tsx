import { useEffect, useMemo, useState } from 'react'
import {
  searchYouTube,
  type AudioInput,
  type JobParams,
  type YouTubeHit,
} from '../api/client'

type Props = {
  onSubmit: (target: AudioInput, sources: AudioInput[], params: JobParams) => void
  busy: boolean
}

type SlotKey = 'target' | 0 | 1 | 2 | 3 | 4

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

function fmtDur(s: number | null | undefined) {
  if (s == null || !Number.isFinite(s)) return ''
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
    .toString()
    .padStart(2, '0')
  return `${m}:${sec}`
}

export function UploadPanel({ onSubmit, busy }: Props) {
  const [target, setTarget] = useState<AudioInput>({ file: null, url: '' })
  const [sources, setSources] = useState<AudioInput[]>(EMPTY)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [params, setParams] = useState<JobParams>({
    window_s: 1.0,
    hop_s: 0.5,
    lambda_switch: 0.85,
  })

  const [searchQ, setSearchQ] = useState('piano')
  const [searching, setSearching] = useState(false)
  const [hits, setHits] = useState<YouTubeHit[]>([])
  const [searchErr, setSearchErr] = useState<string | null>(null)
  const [assignTo, setAssignTo] = useState<SlotKey>('target')

  const ready = useMemo(
    () => filled(target) && sources.every(filled) && params.hop_s <= params.window_s,
    [target, sources, params],
  )

  const setSource = (i: number, patch: Partial<AudioInput>) => {
    setSources((prev) =>
      prev.map((x, j) => {
        if (j !== i) return x
        const next = { ...x, ...patch }
        if (patch.file) next.url = ''
        if (patch.url !== undefined && patch.url !== '') next.file = null
        return next
      }),
    )
  }

  const assignUrl = (url: string, slot: SlotKey) => {
    if (slot === 'target') setTarget({ file: null, url })
    else setSource(slot, { file: null, url })
  }

  const runSearch = async () => {
    setSearching(true)
    setSearchErr(null)
    try {
      const res = await searchYouTube(searchQ)
      setHits(res)
      if (!res.length) setSearchErr('No results found.')
    } catch (e) {
      setSearchErr(e instanceof Error ? e.message : String(e))
      setHits([])
    } finally {
      setSearching(false)
    }
  }

  useEffect(() => {
    // gentle default search on first paint so the picker isn't empty
    void runSearch()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <section className="upload">
      <div className="hero-copy">
        <p className="eyebrow">Acoustic collage</p>
        <h1 className="brand">Music Mosaic</h1>
        <p className="lede">
          Rebuild one song from five others — search YouTube for tracks or drop MP3s.
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

      <div className="yt-search">
        <div className="yt-search-row">
          <input
            className="yt-query"
            value={searchQ}
            disabled={busy}
            placeholder='Search songs — e.g. "piano", "lofi", "jazz guitar"'
            onChange={(e) => setSearchQ(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void runSearch()}
          />
          <button
            type="button"
            className="cta small"
            disabled={busy || searching || searchQ.trim().length < 2}
            onClick={() => void runSearch()}
          >
            {searching ? 'Searching…' : 'Search'}
          </button>
          <select
            className="yt-assign"
            value={String(assignTo)}
            disabled={busy}
            onChange={(e) => {
              const v = e.target.value
              setAssignTo(v === 'target' ? 'target' : (Number(v) as 0 | 1 | 2 | 3 | 4))
            }}
          >
            <option value="target">Insert → Target</option>
            {PIGMENTS.map((_, i) => (
              <option key={i} value={i}>
                Insert → Source {String.fromCharCode(65 + i)}
              </option>
            ))}
          </select>
        </div>
        {searchErr && <p className="error">{searchErr}</p>}
        <div className="yt-results">
          {hits.map((h) => (
            <button
              key={h.id}
              type="button"
              className="yt-hit"
              disabled={busy}
              onClick={() => assignUrl(h.url, assignTo)}
              title={h.url}
            >
              <span className="yt-hit-title">{h.title}</span>
              <span className="yt-hit-meta mono">
                {fmtDur(h.duration_s)}
                {h.channel ? ` · ${h.channel}` : ''}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="drop-grid">
        <div className={`drop target-drop${filled(target) ? ' has-file' : ''}`}>
          <span className="drop-label">Target · MP3 or YouTube</span>
          <label className="file-hit">
            <span className="drop-file">{target.file?.name ?? 'Choose MP3'}</span>
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
            <span className="drop-swatch" style={{ background: PIGMENTS[i] }} aria-hidden />
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
        Search returns songs between 5 seconds and 5 minutes (e.g. “piano” → “piano songs”).
        Matching uses key-invariant chroma + pretrained CLAP when available.
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
              min={0.25}
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
              min={0.1}
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
