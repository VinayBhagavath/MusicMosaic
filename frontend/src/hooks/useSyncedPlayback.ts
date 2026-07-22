import { useEffect, useRef, useState } from 'react'

/** Sync tile index to audio currentTime via rAF. */
export function useSyncedPlayback(hopS: number, durationS: number) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [activeTile, setActiveTile] = useState(0)

  useEffect(() => {
    let raf = 0
    const tick = () => {
      const a = audioRef.current
      if (a && !a.paused) {
        const t = a.currentTime
        setTime(t)
        setActiveTile(Math.min(Math.floor(t / hopS), Math.max(0, Math.floor(durationS / hopS) - 1)))
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [hopS, durationS])

  const play = async () => {
    await audioRef.current?.play()
    setPlaying(true)
  }
  const pause = () => {
    audioRef.current?.pause()
    setPlaying(false)
  }
  const seek = (t: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = t
      setTime(t)
      setActiveTile(Math.floor(t / hopS))
    }
  }

  return { audioRef, playing, time, activeTile, play, pause, seek, setPlaying }
}
