import { useEffect, useMemo, useRef } from 'react'
import type { Mosaic, MosaicTile } from '../api/client'

type Props = {
  mosaic: Mosaic
  activeTile: number
  onSelect: (tile: MosaicTile) => void
}

export function MosaicGrid({ mosaic, activeTile, onSelect }: Props) {
  const colorById = useMemo(
    () => Object.fromEntries(mosaic.songs.map((s) => [s.id, s.color])),
    [mosaic.songs],
  )
  const cols = Math.min(32, Math.max(12, Math.ceil(Math.sqrt(mosaic.tiles.length * 1.6))))
  const cellRefs = useRef<(HTMLButtonElement | null)[]>([])

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
        const color = colorById[t.song_id] ?? '#888'
        const bright = 0.35 + 0.65 * Math.max(0, Math.min(1, t.similarity))
        return (
          <button
            key={t.i}
            type="button"
            ref={(el) => {
              cellRefs.current[t.i] = el
            }}
            className={`tile${t.i === activeTile ? ' active' : ''}`}
            title={`${t.song_id} @ ${t.source_start_s.toFixed(2)}s · sim ${t.similarity.toFixed(2)}`}
            style={{
              background: color,
              opacity: bright,
            }}
            onClick={() => onSelect(t)}
          />
        )
      })}
    </div>
  )
}
