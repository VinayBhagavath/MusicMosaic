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
  const layers = tile.layers?.length ? tile.layers : null
  return (
    <aside className="tile-detail">
      <button type="button" className="linkish" onClick={onClose}>
        Close
      </button>
      <h3>Tile {tile.i}</h3>
      <p>
        <span className="swatch" style={{ background: song?.color }} />
        Primary <strong>{tile.song_id}</strong> ({song?.name})
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
        <div>
          <dt>Pitch shift</dt>
          <dd>{(tile.key_shift ?? 0).toFixed(2)} st</dd>
        </div>
        {tile.target_duration_s != null && (
          <div>
            <dt>Event length</dt>
            <dd>{tile.target_duration_s.toFixed(2)}s</dd>
          </div>
        )}
      </dl>
      {layers && layers.length > 1 && (
        <div className="layers">
          <h4>Layers</h4>
          <ul>
            {layers.map((L, idx) => {
              const ls = songs.find((s) => s.id === L.song_id)
              return (
                <li key={`${L.song_id}-${idx}`}>
                  <span className="swatch" style={{ background: ls?.color }} />
                  <strong>{L.song_id}</strong>
                  <span>
                    {(L.weight * 100).toFixed(0)}% · {L.similarity.toFixed(2)} ·{' '}
                    {L.role ?? 'full'} · {(L.key_shift ?? 0).toFixed(2)} st
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      )}
      <button type="button" className="cta small" onClick={() => onSeekTo(tile.target_start_s)}>
        Jump to tile
      </button>
      <p className="hint">
        Window {windowS}s · hop {hopS}s
      </p>
    </aside>
  )
}
