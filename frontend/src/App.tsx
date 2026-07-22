import { useCallback, useEffect, useState } from 'react'
import {
  audioUrl,
  createJob,
  getJob,
  getMosaic,
  targetUrl,
  type JobParams,
  type Mosaic,
  type MosaicTile,
} from './api/client'
import { MosaicGrid } from './components/MosaicGrid'
import { PlaybackBar } from './components/PlaybackBar'
import { StatsPanel } from './components/StatsPanel'
import { TileDetail } from './components/TileDetail'
import { TimelineStrip } from './components/TimelineStrip'
import { UploadPanel } from './components/UploadPanel'
import { useSyncedPlayback } from './hooks/useSyncedPlayback'
import './styles/theme.css'

type Phase = 'upload' | 'running' | 'done' | 'error'

export default function App() {
  const [phase, setPhase] = useState<Phase>('upload')
  const [jobId, setJobId] = useState<string | null>(null)
  const [pct, setPct] = useState(0)
  const [message, setMessage] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [mosaic, setMosaic] = useState<Mosaic | null>(null)
  const [selected, setSelected] = useState<MosaicTile | null>(null)
  const [useTarget, setUseTarget] = useState(false)

  const hop = mosaic?.hop_s ?? 0.25
  const dur = mosaic?.duration_s ?? 0
  const playback = useSyncedPlayback(hop, dur)

  const onSubmit = useCallback(async (target: File, sources: File[], params: JobParams) => {
    setPhase('running')
    setError(null)
    setPct(0)
    setMessage('Uploading…')
    try {
      const id = await createJob(target, sources, params)
      setJobId(id)
    } catch (e) {
      setPhase('error')
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    if (!jobId || phase !== 'running') return
    const id = setInterval(async () => {
      try {
        const st = await getJob(jobId)
        setPct(st.pct)
        setMessage(st.message)
        if (st.stage === 'done') {
          const m = await getMosaic(jobId)
          setMosaic(m)
          setPhase('done')
        } else if (st.stage === 'error') {
          setPhase('error')
          setError(st.error || st.message)
        }
      } catch (e) {
        setPhase('error')
        setError(e instanceof Error ? e.message : String(e))
      }
    }, 500)
    return () => clearInterval(id)
  }, [jobId, phase])

  return (
    <div className="app">
      <div className="grain" aria-hidden />
      {phase === 'upload' && <UploadPanel onSubmit={onSubmit} busy={false} />}

      {phase === 'running' && (
        <section className="progress-panel">
          <h1 className="brand sm">Music Mosaic</h1>
          <p className="lede">{message}</p>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <p className="mono muted">{pct.toFixed(0)}%</p>
        </section>
      )}

      {phase === 'error' && (
        <section className="progress-panel">
          <h1 className="brand sm">Music Mosaic</h1>
          <p className="error">{error}</p>
          <button type="button" className="cta" onClick={() => setPhase('upload')}>
            Try again
          </button>
        </section>
      )}

      {phase === 'done' && mosaic && jobId && (
        <section className="result">
          <header className="result-head">
            <h1 className="brand sm">Music Mosaic</h1>
            <button type="button" className="linkish" onClick={() => window.location.reload()}>
              New mosaic
            </button>
          </header>

          <MosaicGrid
            mosaic={mosaic}
            activeTile={playback.activeTile}
            onSelect={setSelected}
          />
          <TimelineStrip mosaic={mosaic} time={playback.time} onSeek={playback.seek} />
          <PlaybackBar
            audioRef={playback.audioRef}
            src={audioUrl(jobId)}
            targetSrc={targetUrl(jobId)}
            playing={playback.playing}
            time={playback.time}
            duration={dur}
            useTarget={useTarget}
            onToggleSource={() => {
              const t = playback.time
              setUseTarget((v) => !v)
              requestAnimationFrame(() => playback.seek(t))
            }}
            onPlay={playback.play}
            onPause={playback.pause}
            onSeek={playback.seek}
            onEnded={() => playback.setPlaying(false)}
          />
          <div className="result-bottom">
            <StatsPanel mosaic={mosaic} />
            <TileDetail
              tile={selected}
              songs={mosaic.songs}
              hopS={mosaic.hop_s}
              onSeekTo={playback.seek}
              onClose={() => setSelected(null)}
            />
          </div>
        </section>
      )}
    </div>
  )
}
