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

export type MosaicTile = {
  i: number
  target_start_s: number
  song_id: string
  source_start_s: number
  similarity: number
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
  }
}

export type JobParams = {
  window_s: number
  hop_s: number
  lambda_switch: number
}

export type AudioInput = {
  file: File | null
  url: string
}

const BASE = ''

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
