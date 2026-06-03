// hifi/walker.jsx — rollout video player
// Plays the real eval rollout MP4. When a rollout has no video.mp4 (or it fails
// to load) we show a "missing video" placeholder rather than a synthetic
// stand-in animation — the app never fabricates rollout content.

const { useRef, useEffect, useState } = React;

function WalkerPlayer({
  run, ckpt, rollout, frame, setFrame,
  playing, setPlaying, speed = 1, setSpeed,
}) {
  const videoRef = useRef(null);
  const rafRef = useRef(null);
  const lastTimeRef = useRef(0);

  // Video support: show real mp4 when available, fall back to a missing-state.
  const [videoFailed, setVideoFailed] = useState(false);
  useEffect(() => { setVideoFailed(false); }, [rollout?.dir]); // reset when rollout changes

  const videoUrl  = rollout.dir ? `${rollout.dir}/video.mp4` : null;
  const showVideo = videoUrl && !videoFailed;

  // Sync video position to frame (fraction-based so we don't need to know fps)
  useEffect(() => {
    if (!showVideo || !videoRef.current) return;
    const video = videoRef.current;
    const onLoaded = () => {
      const targetTime = (frame / Math.max(1, rollout.length - 1)) * video.duration;
      if (Math.abs(video.currentTime - targetTime) > 0.04) {
        video.currentTime = targetTime;
      }
    };
    if (video.readyState >= 1) onLoaded();
    else video.addEventListener('loadedmetadata', onLoaded, { once: true });
  }, [frame, showVideo, rollout.length]);

  // Animation loop — advances frame while playing (drives the video + charts)
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }
    const fps = 30 * speed;
    const tick = (t) => {
      if (!lastTimeRef.current) lastTimeRef.current = t;
      const dt = t - lastTimeRef.current;
      if (dt >= 1000 / fps) {
        lastTimeRef.current = t;
        setFrame(f => {
          const next = f + 1;
          if (next >= rollout.length) {
            setPlaying(false);
            return rollout.length - 1;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTimeRef.current = 0;
    };
  }, [playing, speed, rollout, setFrame, setPlaying]);

  // Compute timecode
  const fps = 30;
  const totalSecs = rollout.length / fps;
  const curSecs = frame / fps;
  const tc = `${D.fmtTime(curSecs)} / ${D.fmtTime(totalSecs)}`;

  return (
    <div className="col" style={{ width: '100%', gap: 8 }}>
      {/* Player: real video when available, missing-state otherwise */}
      <div className="player" style={{ aspectRatio: '16/9', borderRadius: 4, maxHeight: 'clamp(260px, calc(21vw + 140px), 520px)' }}>
        {showVideo ? (
          <video
            ref={videoRef}
            src={videoUrl}
            onError={() => setVideoFailed(true)}
            style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
          />
        ) : (
          <div className="player-missing">
            <span className="player-missing__mark">⊘</span>
            <span className="player-missing__title">no rollout video</span>
            <span className="player-missing__detail">
              {videoFailed ? 'video.mp4 failed to load' : 'video.mp4 not recorded for this rollout'}
            </span>
          </div>
        )}
        {/* Top-left corner caption */}
        <div style={{ position: 'absolute', top: 8, left: 10, color: 'rgba(244,241,234,.85)', font: '500 11px var(--mono)', textShadow: '0 1px 2px rgba(0,0,0,.6)' }}>
          {run.env}  ·  ckpt {D.fmtStep(ckpt.step)}  ·  {rollout.kind} ep  ·  r {rollout.return}
        </div>
        {/* Top-right corner */}
        <div style={{ position: 'absolute', top: 8, right: 10, color: 'rgba(244,241,234,.7)', font: '500 11px var(--mono)', textShadow: '0 1px 2px rgba(0,0,0,.6)' }}>
          {tc}
        </div>
        {/* Bottom overlay strip */}
        <div className="ph">
          <span>frame {frame.toLocaleString()} / {rollout.length.toLocaleString()}</span>
          <span style={{ display: 'flex', gap: 12 }}>
            <span>{playing ? '▶ playing' : '∥ paused'} · {speed.toFixed(2)}×</span>
          </span>
        </div>
      </div>

      {/* Transport controls */}
      <div className="row gap-2" style={{ paddingTop: 2 }}>
        <button className="btn icon" title="prev frame (←)" onClick={() => setFrame(f => Math.max(0, f - 1))}>◀</button>
        <button className={"btn icon" + (playing ? ' solid' : '')} title="play/pause (space)"
                onClick={() => setPlaying(p => !p)} style={{ width: 32 }}>
          {playing ? '∥∥' : '▶'}
        </button>
        <button className="btn icon" title="next frame (→)" onClick={() => setFrame(f => Math.min(rollout.length - 1, f + 1))}>▶</button>
        <button className="btn icon" title="restart" onClick={() => setFrame(0)}>⟲</button>

        <div className="scrub" style={{ flex: 1, marginLeft: 6 }}>
          <div className="track"
            onMouseDown={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const t = (e.clientX - rect.left) / rect.width;
              setFrame(Math.floor(t * (rollout.length - 1)));
              const onMove = (ev) => {
                const t2 = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
                setFrame(Math.floor(t2 * (rollout.length - 1)));
              };
              const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);
              };
              window.addEventListener('mousemove', onMove);
              window.addEventListener('mouseup', onUp);
            }}
          >
            <div className="fill" style={{ width: `${(frame / (rollout.length - 1)) * 100}%` }} />
            <div className="head" style={{ left: `${(frame / (rollout.length - 1)) * 100}%` }} />
          </div>
          <span className="num">frame {frame}</span>
        </div>

        <select className="dropdown" value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))}
                style={{ font: '500 11.5px var(--ui)' }}>
          <option value="0.25">0.25×</option>
          <option value="0.5">0.5×</option>
          <option value="1">1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
        </select>

      </div>
    </div>
  );
}

window.WalkerPlayer = WalkerPlayer;
