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
      {s.quality && (
        <>
          <div className="stat">
            <span className="stat-k">Chroma fidelity</span>
            <span className="stat-v">{s.quality.chroma_similarity.toFixed(3)}</span>
          </div>
          <div className="stat">
            <span className="stat-k">Log-mel distance</span>
            <span className="stat-v">{s.quality.log_mel_distance.toFixed(3)}</span>
          </div>
          <div className="stat">
            <span className="stat-k">Onset correlation</span>
            <span className="stat-v">{s.quality.onset_correlation.toFixed(3)}</span>
          </div>
        </>
      )}
      <div className="stat">
        <span className="stat-k">Matcher</span>
        <span className="stat-v">
          {s.fidelity_first ? 'fidelity first' : 'balanced'} · {s.embedding_backend ?? 'mosaic'}
        </span>
      </div>
      <div className="stat">
        <span className="stat-k">Renderer</span>
        <span className="stat-v">
          {s.reconstruction_backend === 'nmf' ? 'sparse diagonal NMF' : 'unit selection'}
          {s.reconstruction_backend_requested === 'auto' && (
            <span className="muted"> · auto {s.nmf_accepted ? 'accepted NMF' : 'kept baseline'}</span>
          )}
        </span>
      </div>
      {s.nmf && (
        <div className="stat">
          <span className="stat-k">NMF activations</span>
          <span className="stat-v">
            {s.nmf.active_polyphony.toFixed(1)} voices · {s.nmf.source_frames.toLocaleString()} source
            frames
          </span>
        </div>
      )}
      {s.stage_timings_s && (
        <div className="stat">
          <span className="stat-k">Stage timings</span>
          <span className="stat-v">
            {Object.entries(s.stage_timings_s)
              .map(([stage, seconds]) => `${stage} ${seconds.toFixed(1)}s`)
              .join(' · ')}
          </span>
        </div>
      )}
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
