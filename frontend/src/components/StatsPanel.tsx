import type { Mosaic } from '../api/client'

type Props = { mosaic: Mosaic }

export function StatsPanel({ mosaic }: Props) {
  const s = mosaic.stats
  return (
    <div className="stats">
      <div className="stat">
        <span className="stat-k">Avg similarity</span>
        <span className="stat-v">{s.avg_similarity.toFixed(3)}</span>
      </div>
      <div className="stat">
        <span className="stat-k">Transitions</span>
        <span className="stat-v">
          {s.transitions_viterbi}
          <span className="muted"> / greedy {s.transitions_greedy}</span>
        </span>
      </div>
      <div className="contrib">
        {mosaic.songs.map((song) => (
          <div key={song.id} className="contrib-row">
            <span className="swatch" style={{ background: song.color }} />
            <span>{song.id}</span>
            <div className="bar">
              <div
                style={{
                  width: `${s.contribution_pct[song.id] ?? 0}%`,
                  background: song.color,
                }}
              />
            </div>
            <span className="pct">{(s.contribution_pct[song.id] ?? 0).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}
