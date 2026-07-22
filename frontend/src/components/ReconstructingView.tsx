import { useEffect, useMemo, useState } from 'react'

const PIGMENTS = ['#C45C26', '#C9A227', '#2F6F5E', '#3D5A80', '#A24B6F']

type Props = {
  pct: number
  message: string
  stage: string
}

/** Gallery-stage animation: splice waveform → quilt tiles fill in. */
export function ReconstructingView({ pct, message, stage }: Props) {
  const cols = 18
  const rows = 10
  const total = cols * rows

  // Deterministic pseudo-random fill order so the quilt feels organic
  const order = useMemo(() => {
    const idx = Array.from({ length: total }, (_, i) => i)
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(((Math.sin(i * 12.9898) * 43758.5453) % 1) * (i + 1))
      ;[idx[i], idx[j]] = [idx[j], idx[i]]
    }
    return idx
  }, [total])

  const orderRank = useMemo(() => {
    const rank = new Array<number>(total)
    order.forEach((cell, rankIdx) => {
      rank[cell] = rankIdx
    })
    return rank
  }, [order, total])

  const fillCount = Math.floor((Math.min(100, Math.max(0, pct)) / 100) * total)
  const filled = useMemo(() => new Set(order.slice(0, fillCount)), [order, fillCount])

  const cutCount = Math.min(24, Math.floor((pct / 40) * 24))
  const splicing = pct < 55 || stage === 'load' || stage === 'segment' || stage === 'features'

  const bars = useMemo(
    () =>
      Array.from({ length: 64 }, (_, i) => {
        const h = 18 + Math.abs(Math.sin(i * 0.45)) * 70 + (i % 5) * 4
        return h
      }),
    [],
  )

  const [tick, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 120)
    return () => clearInterval(id)
  }, [])

  return (
    <section className="rebuild">
      <div className="rebuild-head">
        <div>
          <h1 className="brand sm">Music Mosaic</h1>
          <p className="stage-label">{message || 'Composing…'}</p>
        </div>
        <span className="mono">{pct.toFixed(0)}%</span>
      </div>

      {splicing && (
        <div className="splice-stage">
          <p className="splice-caption">Splicing target into overlapping windows</p>
          <div className="waveform" aria-hidden>
            {bars.map((h, i) => (
              <div
                key={i}
                className={`wave-bar${i < (pct / 100) * 64 + (tick % 3) ? ' on' : ''}`}
                style={{
                  height: `${h}%`,
                  animationDelay: `${(i % 8) * 0.08}s`,
                }}
              />
            ))}
            <div className="splice-cuts">
              {Array.from({ length: cutCount }, (_, i) => (
                <div
                  key={i}
                  className="cut"
                  style={{
                    left: `${((i + 1) / 25) * 100}%`,
                    animationDelay: `${i * 0.04}s`,
                    background: PIGMENTS[i % PIGMENTS.length],
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      <div>
        <p className="splice-caption" style={{ marginBottom: '0.75rem' }}>
          {pct < 45 ? 'Preparing quilt' : 'Painting mosaic from source fragments'}
        </p>
        <div
          className="quilt-preview"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        >
          {Array.from({ length: total }, (_, i) => {
            const isFilled = filled.has(i)
            const color = PIGMENTS[((orderRank[i] ?? 0) + i) % PIGMENTS.length]
            return (
              <div
                key={i}
                className={`quilt-cell${isFilled ? ' filled' : ''}`}
                style={isFilled ? { background: color } : undefined}
              />
            )
          })}
        </div>
      </div>

      <div>
        <div className="progress-meta">
          <span>{stage}</span>
          <span>reconstructing</span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
      </div>
    </section>
  )
}
