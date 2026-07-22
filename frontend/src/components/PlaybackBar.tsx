type Props = {
  mosaicRef: React.RefObject<HTMLAudioElement | null>
  targetRef: React.RefObject<HTMLAudioElement | null>
  src: string
  targetSrc: string
  playing: boolean
  time: number
  duration: number
  useTarget: boolean
  onToggleSource: () => void
  onPlay: () => void
  onPause: () => void
  onSeek: (t: number) => void
  onEnded: () => void
}

function fmt(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
    .toString()
    .padStart(2, '0')
  return `${m}:${sec}`
}

export function PlaybackBar({
  mosaicRef,
  targetRef,
  src,
  targetSrc,
  playing,
  time,
  duration,
  useTarget,
  onToggleSource,
  onPlay,
  onPause,
  onSeek,
  onEnded,
}: Props) {
  return (
    <div className="playback">
      <audio ref={mosaicRef} src={src} preload="auto" onEnded={onEnded} />
      <audio ref={targetRef} src={targetSrc} preload="auto" onEnded={onEnded} />
      <button type="button" className="cta small" onClick={playing ? onPause : onPlay}>
        {playing ? 'Pause' : 'Play'}
      </button>
      <button type="button" className="linkish" onClick={onToggleSource}>
        {useTarget ? 'Hearing: target' : 'Hearing: mosaic'}
      </button>
      <input
        className="scrub"
        type="range"
        min={0}
        max={duration || 1}
        step={0.01}
        value={time}
        onChange={(e) => onSeek(Number(e.target.value))}
      />
      <span className="time mono">
        {fmt(time)} / {fmt(duration)}
      </span>
    </div>
  )
}
