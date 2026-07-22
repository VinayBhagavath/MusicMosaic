import { useEffect, useRef, useState } from 'react'

/** Sync tile index to audio currentTime via rAF. Dual-source A/B safe. */
export function useSyncedPlayback(hopS: number, durationS: number) {
  const mosaicRef = useRef<HTMLAudioElement | null>(null)
  const targetRef = useRef<HTMLAudioElement | null>(null)
  const [useTarget, setUseTarget] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [activeTile, setActiveTile] = useState(0)
  const pending = useRef<{ t: number; play: boolean } | null>(null)

  const active = () => (useTarget ? targetRef.current : mosaicRef.current)

  useEffect(() => {
    let raf = 0
    const tick = () => {
      const a = active()
      if (a && !a.paused) {
        const t = a.currentTime
        setTime(t)
        setActiveTile(
          Math.min(Math.floor(t / hopS), Math.max(0, Math.floor(durationS / hopS) - 1)),
        )
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hopS, durationS, useTarget])

  useEffect(() => {
    const p = pending.current
    const a = active()
    if (!p || !a) return
    const apply = () => {
      try {
        a.currentTime = p.t
      } catch {
        /* ignore seek before ready */
      }
      setTime(p.t)
      setActiveTile(Math.floor(p.t / hopS))
      if (p.play) {
        a.play()
          .then(() => setPlaying(true))
          .catch(() => setPlaying(false))
      }
      pending.current = null
    }
    if (a.readyState >= 2) apply()
    else a.addEventListener('loadeddata', apply, { once: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useTarget, hopS])

  const play = async () => {
    await active()?.play()
    setPlaying(true)
  }
  const pause = () => {
    mosaicRef.current?.pause()
    targetRef.current?.pause()
    setPlaying(false)
  }
  const seek = (t: number) => {
    const a = active()
    if (a) {
      a.currentTime = t
      setTime(t)
      setActiveTile(Math.floor(t / hopS))
    }
  }
  const toggleSource = () => {
    const a = active()
    pending.current = { t: a?.currentTime ?? time, play: !!(a && !a.paused) }
    mosaicRef.current?.pause()
    targetRef.current?.pause()
    setUseTarget((v) => !v)
  }

  return {
    mosaicRef,
    targetRef,
    useTarget,
    playing,
    time,
    activeTile,
    play,
    pause,
    seek,
    toggleSource,
    setPlaying,
  }
}
