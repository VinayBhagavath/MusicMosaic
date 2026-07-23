export type JobStatus = {
  job_id: string
  stage: string
  pct: number
  message: string
  error?: string | null
  stats?: Record<string, unknown> | null
  elapsed_s?: number | null
}

export type MosaicSong = { id: string; name: string; color: string }

export type MosaicLayer = {
  song_id: string
  source_start_s: number
  similarity: number
  weight: number
  key_shift?: number
  role?: string
}

export type MosaicTile = {
  i: number
  target_start_s: number
  target_duration_s?: number
  song_id: string
  source_start_s: number
  similarity: number
  key_shift?: number
  layers?: MosaicLayer[]
}

export type QualityMetrics = {
  log_mel_distance: number
  chroma_similarity: number
  onset_correlation: number
  boundary_discontinuity: number
}

export type Mosaic = {
  window_s: number
  hop_s: number
  duration_s: number
  songs: MosaicSong[]
  tiles: MosaicTile[]
  stats: {
    contribution_pct: Record<string, number>
    avg_similarity: number
    num_transitions: number
    transitions_viterbi: number
    transitions_greedy: number
    num_tiles: number
    n_layers?: number
    max_share?: number
    fidelity_first?: boolean
    embedding_backend?: string
    reconstruction_backend?: 'unit' | 'nmf'
    reconstruction_backend_requested?: 'auto' | 'unit' | 'nmf'
    nmf_accepted?: boolean
    unit_quality?: QualityMetrics | null
    nmf_quality?: QualityMetrics | null
    nmf?: {
      spectral_error: number
      active_polyphony: number
      source_frames: number
      target_frames: number
    } | null
    use_stems?: boolean
    stage_timings_s?: Record<string, number>
    quality?: QualityMetrics
  }
}

export type JobParams = {
  window_s: number
  hop_s: number
  lambda_switch: number
  lambda_balance: number
  max_share: number
  n_layers: number
  fidelity_first: boolean
  reconstruction_backend: 'auto' | 'unit' | 'nmf'
  use_stems?: boolean
}

export type AudioInput = {
  file: File | null
  url: string
}

export type YouTubeHit = {
  id: string
  title: string
  url: string
  duration_s?: number | null
  channel?: string | null
}

const BASE = ''

export async function searchYouTube(q: string): Promise<YouTubeHit[]> {
  const params = new URLSearchParams({
    q,
    limit: '10',
  })
  const res = await fetch(`${BASE}/api/youtube/search?${params}`)
  if (!res.ok) throw new Error(await res.text())
  const data = await res.json()
  return (data.results || []) as YouTubeHit[]
}

export async function createJob(
  target: AudioInput,
  sources: AudioInput[],
  params: JobParams,
): Promise<string> {
  const fd = new FormData()
  if (target.url.trim()) fd.append('target_url', target.url.trim())
  else if (target.file) fd.append('target', target.file)

  sources.forEach((s, i) => {
    if (s.url.trim()) fd.append(`source_${i}_url`, s.url.trim())
    else if (s.file) fd.append(`source_${i}`, s.file)
  })

  fd.append('window_s', String(params.window_s))
  fd.append('hop_s', String(params.hop_s))
  fd.append('lambda_switch', String(params.lambda_switch))
  fd.append('lambda_balance', String(params.lambda_balance))
  fd.append('max_share', String(params.max_share))
  fd.append('n_layers', String(params.n_layers))
  fd.append('layer_primary_weight', params.n_layers > 1 ? '0.62' : '1.0')
  fd.append('fidelity_first', String(params.fidelity_first))
  fd.append('reconstruction_backend', params.reconstruction_backend)
  fd.append('use_stems', String(Boolean(params.use_stems)))
  // Fidelity defaults: exact global beam, short onset units, polyphonic fill.
  fd.append('min_run_tiles', '1')
  fd.append('lambda_concat', '0.55')
  fd.append('lambda_join', '0.7')
  fd.append('harmonic_match', 'true')
  fd.append('harmonic_strength', '0.42')
  fd.append('rerank_spectral', 'true')
  fd.append('onset_sync_xf', 'true')
  const res = await fetch(`${BASE}/api/jobs`, { method: 'POST', body: fd })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  const data = await res.json()
  return data.job_id as string
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/api/jobs/${jobId}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getMosaic(jobId: string): Promise<Mosaic> {
  const res = await fetch(`${BASE}/api/jobs/${jobId}/mosaic`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function audioUrl(jobId: string): string {
  return `${BASE}/api/jobs/${jobId}/audio`
}

export function targetUrl(jobId: string): string {
  return `${BASE}/api/jobs/${jobId}/target`
}
