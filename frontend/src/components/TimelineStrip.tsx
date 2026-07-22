import { useMemo } from 'react'
import type { Mosaic } from '../api/client'

type Props = {
  mosaic: Mosaic
  time: number
  onSeek: (t: number) => void
}

export function TimelineStrip({ mosaic, time, onSeek }: Props) {
  const colorById = useMemo(
    () => Object.fromEntries(mosaic.songs.map((s) => [s.id, s.color])),
    [mosaic.songs],
  )
  const dur = mosaic.duration_s || 1

  return (
    <div
      className="timeline"
      onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect()
        const x = (e.clientX - rect.left) / rect.width
        onSeek(x * dur)
      }}
    >
      <div className="timeline-segs">
        {mosaic.tiles.map((t) => (
          <div
            key={t.i}
            className="timeline-seg"
            style={{
              flex: 1,
              background: colorById[t.song_id],
            }}
          />
        ))}
      </div>
      <div className="playhead" style={{ left: `${(time / dur) * 100}%` }} />
    </div>
  )
}
