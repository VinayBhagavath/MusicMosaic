import type { MosaicTile, MosaicSong } from '../api/client'

type Props = {
  tile: MosaicTile | null
  songs: MosaicSong[]
  windowS: number
  hopS: number
  onSeekTo: (t: number) => void
  onClose: () => void
}

function fmt(s: number) {
  const m = Math.floor(s / 60)
  const sec = (s % 60).toFixed(2).padStart(5, '0')
  return `${m}:${sec}`
}

export function TileDetail({ tile, songs, windowS, hopS, onSeekTo, onClose }: Props) {
  if (!tile) {
    return (
      <aside className="tile-detail">
        <p className="hint">Select a tile to inspect its source fragment.</p>
      </aside>
    )
  }
  const song = songs.find((s) => s.id === tile.song_id)
  return (
    <aside className="tile-detail">
      <button type="button" className="linkish" onClick={onClose}>
        Close
      </button>
      <h3>Tile {tile.i}</h3>
      <p>
        <span className="swatch" style={{ background: song?.color }} />
        Source <strong>{tile.song_id}</strong> ({song?.name})
      </p>
      <dl>
        <div>
          <dt>Target</dt>
          <dd>{fmt(tile.target_start_s)}</dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd>{fmt(tile.source_start_s)}</dd>
        </div>
        <div>
          <dt>Similarity</dt>
          <dd>{tile.similarity.toFixed(3)}</dd>
        </div>
      </dl>
      <button type="button" className="cta small" onClick={() => onSeekTo(tile.target_start_s)}>
        Jump to tile
      </button>
      <p className="hint">
        Window {windowS}s · hop {hopS}s
      </p>
    </aside>
  )
}
