// hifi/walker.jsx — procedural side-view walker animation
// Acts as the "video" for our prototype. Frame-driven so it's
// deterministic and scrubbable. In production this is replaced by
// a real <video> tag pointing at the eval rollout MP4.

const { useRef, useEffect, useCallback, useState } = React;

// ── Drawing primitives ───────────────────────────────────────────────
function roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// Walker: a stylised 2D biped (matches walker2d-v4 vibe).
// scenarioBeats define per-rollout drama: stumble at X, fall at Y, etc.
function drawWalker(ctx, w, h, frame, rolloutKind, opts = {}) {
  const { showSaliency = false, showTraj = false, env = 'walker2d-v4' } = opts;

  // Background — warm dark wash
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, '#1f1a15');
  bg.addColorStop(0.6, '#15110d');
  bg.addColorStop(1, '#0c0a08');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Soft vignette
  const vg = ctx.createRadialGradient(w / 2, h / 2, h * 0.3, w / 2, h / 2, h * 0.9);
  vg.addColorStop(0, 'rgba(0,0,0,0)');
  vg.addColorStop(1, 'rgba(0,0,0,0.55)');
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, w, h);

  // Ground geometry
  const groundY = Math.floor(h * 0.82);

  // Ground hatch (subtle, scrolling)
  ctx.strokeStyle = 'rgba(192,168,135,0.18)';
  ctx.lineWidth = 1;
  const scroll = -((frame * 0.85) % 32);
  for (let x = scroll - 32; x < w + 32; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, groundY + 1);
    ctx.lineTo(x - 12, h);
    ctx.stroke();
  }

  // Major ground tick lines every 4 "steps" of progress
  ctx.strokeStyle = 'rgba(232,210,175,0.45)';
  ctx.lineWidth = 1;
  const majorSpace = 128;
  const majorScroll = -((frame * 0.85) % majorSpace);
  for (let x = majorScroll - majorSpace; x < w + majorSpace; x += majorSpace) {
    ctx.beginPath();
    ctx.moveTo(x, groundY);
    ctx.lineTo(x, groundY + 6);
    ctx.stroke();
  }

  // Ground line itself
  ctx.strokeStyle = 'rgba(232,210,175,0.75)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, groundY + 0.5);
  ctx.lineTo(w, groundY + 0.5);
  ctx.stroke();

  // ─── Stumble / fall logic per scenario ─────────────────────────
  let tilt = 0;
  let crashed = false;
  if (rolloutKind === 'best' || rolloutKind === 'q3') {
    // Recovered stumble around frame 610–650
    if (frame >= 605 && frame <= 645) {
      const t = (frame - 605) / 40;
      tilt = Math.sin(t * Math.PI) * 0.42;
    }
  } else if (rolloutKind === 'median' || rolloutKind === 'q1') {
    // Smaller wobbles, no big stumble
    tilt = Math.sin(frame / 35) * 0.08;
  } else if (rolloutKind === 'worst') {
    // Falls progressively after frame 720
    if (frame > 720) {
      tilt = Math.min(1.45, (frame - 720) * 0.018);
      if (tilt > 1.2) crashed = true;
    }
  }

  // ─── Walker geometry ───────────────────────────────────────────
  const cx = Math.floor(w / 2);
  const gaitPhase = (frame / 14) * Math.PI;
  const bob = (rolloutKind === 'best' ? 5 : 4) * Math.abs(Math.sin(gaitPhase * 2));

  // Hip position (subject to tilt)
  const hipY = groundY - 56 + bob;
  ctx.save();
  ctx.translate(cx, hipY);
  ctx.rotate(tilt);

  const colorLine = '#ece2cc';
  const colorAccent = '#d97a55';
  const lineW = 5.5;

  // Body torso (rounded rectangle)
  ctx.fillStyle = '#1b140f';
  ctx.strokeStyle = colorLine;
  ctx.lineWidth = lineW;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  roundedRect(ctx, -9, -34, 18, 36, 5);
  ctx.fill();
  ctx.stroke();

  // Head
  ctx.beginPath();
  ctx.arc(0, -42, 7.5, 0, Math.PI * 2);
  ctx.fillStyle = '#1b140f';
  ctx.fill();
  ctx.stroke();

  // Visor accent on head
  ctx.beginPath();
  ctx.arc(2, -42, 2, 0, Math.PI * 2);
  ctx.fillStyle = colorAccent;
  ctx.fill();

  // Arms (swing opposite to legs)
  const armPhase = gaitPhase + Math.PI;
  drawArm(ctx, 0, -28, armPhase * 0.6, colorLine, lineW);
  drawArm(ctx, 0, -28, armPhase * 0.6 + Math.PI, colorLine, lineW);

  // Legs
  drawLeg(ctx, 0, 0, gaitPhase, colorLine, lineW, crashed);
  drawLeg(ctx, 0, 0, gaitPhase + Math.PI, colorLine, lineW, crashed);

  // Spine accent dot
  ctx.beginPath();
  ctx.arc(0, -18, 2.2, 0, Math.PI * 2);
  ctx.fillStyle = colorAccent;
  ctx.fill();

  ctx.restore();

  // ─── Saliency overlay ─────────────────────────────────────────
  if (showSaliency) {
    drawSaliency(ctx, cx, hipY, frame, tilt);
  }

  // ─── Trajectory / heel-strike markers ─────────────────────────
  if (showTraj) {
    drawTrajectory(ctx, w, h, frame, groundY);
  }
}

function drawLeg(ctx, x, y, phase, color, lineW, crashed) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineW;
  ctx.lineCap = 'round';

  // Hip → knee → foot
  const thighA = Math.sin(phase) * (crashed ? 1.2 : 0.55);
  const kneeBend = Math.max(0.05, Math.cos(phase) * 0.7 + 0.3);
  const thighLen = 22;
  const shinLen = 24;

  const knee = {
    x: x + Math.sin(thighA) * thighLen,
    y: y + Math.cos(thighA) * thighLen,
  };
  const foot = {
    x: knee.x + Math.sin(thighA - kneeBend) * shinLen,
    y: knee.y + Math.cos(thighA - kneeBend) * shinLen,
  };

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(knee.x, knee.y);
  ctx.lineTo(foot.x, foot.y);
  ctx.stroke();

  // Foot shoe
  ctx.beginPath();
  ctx.ellipse(foot.x + 1, foot.y - 0.5, 4.5, 2.2, 0, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawArm(ctx, x, y, phase, color, lineW) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineW - 1;
  ctx.lineCap = 'round';

  const upperA = Math.sin(phase) * 0.45;
  const lowerBend = Math.max(0.1, Math.cos(phase) * 0.4 + 0.2);
  const upperLen = 14;
  const lowerLen = 14;

  const elbow = {
    x: x + Math.sin(upperA) * upperLen,
    y: y + Math.cos(upperA) * upperLen,
  };
  const hand = {
    x: elbow.x + Math.sin(upperA + lowerBend) * lowerLen,
    y: elbow.y + Math.cos(upperA + lowerBend) * lowerLen,
  };

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(elbow.x, elbow.y);
  ctx.lineTo(hand.x, hand.y);
  ctx.stroke();
}

function drawSaliency(ctx, cx, cy, frame, tilt) {
  // Soft radial blobs at attention points — they slowly migrate.
  const seed = Math.floor(frame / 5);
  const points = [
    { x: cx, y: cy - 40, r: 32, a: 0.20 + 0.05 * Math.sin(frame / 30) },     // head/visor
    { x: cx, y: cy - 5,  r: 40, a: 0.16 + 0.05 * Math.cos(frame / 25) },     // hip
    { x: cx + 18 * Math.sin(frame / 14 + 0.7), y: cy + 40, r: 28, a: 0.22 }, // active foot
  ];
  // Spike during stumble
  const stumble = (frame >= 605 && frame <= 645);
  if (stumble) {
    points.forEach(p => p.a *= 1.8);
    points.push({ x: cx + 25, y: cy + 30, r: 50, a: 0.35 });
  }
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (const p of points) {
    const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r);
    grad.addColorStop(0, `rgba(217, 122, 85, ${p.a})`);
    grad.addColorStop(1, 'rgba(217, 122, 85, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(p.x - p.r, p.y - p.r, p.r * 2, p.r * 2);
  }
  ctx.restore();
}

function drawTrajectory(ctx, w, h, frame, groundY) {
  // Past heel-strikes as small ticks fading out behind walker
  ctx.fillStyle = 'rgba(217,122,85,0.5)';
  const center = w / 2;
  const stride = 32;
  for (let i = 1; i < 8; i++) {
    const x = center - i * stride - ((frame * 0.85) % stride);
    const alpha = Math.max(0, 1 - i / 8) * 0.7;
    ctx.fillStyle = `rgba(217,122,85,${alpha})`;
    ctx.fillRect(x, groundY - 1, 4, 2);
  }
}

// ── React component ──────────────────────────────────────────────────
function WalkerPlayer({
  run, ckpt, rollout, frame, setFrame,
  playing, setPlaying, speed = 1, setSpeed,
  overlay = 'none', setOverlay,
}) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const lastTimeRef = useRef(0);

  // DPR-aware canvas sizing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawWalker(ctx, rect.width, rect.height, frame, rollout.kind, {
        showSaliency: overlay === 'saliency',
        showTraj: overlay === 'trajectory',
        env: run.env,
      });
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [frame, rollout, overlay, run.env]);

  // Animation loop
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

  // Click + drag scrub on canvas surface
  const dragRef = useRef(false);
  const onScrub = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    setFrame(Math.floor(t * (rollout.length - 1)));
  }, [rollout, setFrame]);

  // Compute timecode
  const fps = 30;
  const totalSecs = rollout.length / fps;
  const curSecs = frame / fps;
  const tc = `${D.fmtTime(curSecs)} / ${D.fmtTime(totalSecs)}`;

  return (
    <div className="col" style={{ width: '100%', gap: 8 }}>
      {/* Canvas player */}
      <div className="player" style={{ aspectRatio: '16/9', borderRadius: 4, maxHeight: 296 }}>
        <canvas
          ref={canvasRef}
          style={{ width: '100%', height: '100%' }}
          onMouseDown={(e) => { dragRef.current = true; onScrub(e); }}
          onMouseMove={(e) => { if (dragRef.current) onScrub(e); }}
          onMouseUp={() => { dragRef.current = false; }}
          onMouseLeave={() => { dragRef.current = false; }}
        />
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
            <span>overlay: {overlay}</span>
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

        <select className="dropdown" value={overlay} onChange={(e) => setOverlay(e.target.value)}
                style={{ font: '500 11.5px var(--ui)' }}>
          <option value="none">overlay: none</option>
          <option value="saliency">overlay: saliency</option>
          <option value="trajectory">overlay: trajectory</option>
        </select>
      </div>
    </div>
  );
}

window.WalkerPlayer = WalkerPlayer;
window.drawWalker = drawWalker;
