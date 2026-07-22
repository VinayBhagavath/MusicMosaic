import { useEffect, useMemo, useRef, useState } from 'react'
import type { Mosaic, MosaicTile } from '../api/client'

type Props = {
  mosaic: Mosaic
  activeTile: number
  onSelect: (tile: MosaicTile) => void
  /** When true, tiles cascade in on mount (post-reconstruction reveal). */
  animateReveal?: boolean
}

export function MosaicGrid({ mosaic, activeTile, onSelect, animateReveal = true }: Props) {
  const colorById = useMemo(
    () => Object.fromEntries(mosaic.songs.map((s) => [s.id, s.color])),
    [mosaic.songs],
  )
  const cols = Math.min(28, Math.max(14, Math.ceil(Math.sqrt(mosaic.tiles.length * 1.5))))
  const cellRefs = useRef<(HTMLButtonElement | null)[]>([])
  const [visible, setVisible] = useState(animateReveal ? 0 : mosaic.tiles.length)

  useEffect(() => {
    if (!animateReveal) return
    let i = 0
    const step = Math.max(1, Math.ceil(mosaic.tiles.length / 80))
    const id = setInterval(() => {
      i = Math.min(mosaic.tiles.length, i + step)
      setVisible(i)
      if (i >= mosaic.tiles.length) clearInterval(id)
    }, 28)
    return () => clearInterval(id)
  }, [animateReveal, mosaic.tiles.length])

  useEffect(() => {
    cellRefs.current[activeTile]?.scrollIntoView({
      block: 'nearest',
      inline: 'nearest',
      behavior: 'smooth',
    })
  }, [activeTile])

  return (
    <div
      className="mosaic-grid"
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
    >
      {mosaic.tiles.map((t) => {
        if (t.i >= visible) {
          return <div key={t.i} className="quilt-cell" />
        }
        const color = colorById[t.song_id] ?? '#888'
        const bright = 0.42 + 0.58 * Math.max(0, Math.min(1, t.similarity))
        return (
          <button
            key={t.i}
            type="button"
            ref={(el) => {
              cellRefs.current[t.i] = el
            }}
            className={`tile reveal${t.i === activeTile ? ' active' : ''}`}
            title={`${t.song_id} @ ${t.source_start_s.toFixed(2)}s · ${t.similarity.toFixed(2)}`}
            style={{
              background: color,
              opacity: bright,
              animationDelay: `${Math.min(t.i * 4, 400)}ms`,
            }}
            onClick={() => onSelect(t)}
          />
        )
      })}
    </div>
  )
}
