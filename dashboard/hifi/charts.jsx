// hifi/charts.jsx — interactive chart suite

const { useMemo, useRef, useCallback: useCB, useState: useSt } = React;

// ── Path helpers ─────────────────────────────────────────────────────
// Build a polyline SVG path with optional decimation for perf.
function buildPath(values, xScale, yScale, decim = 1) {
  let out = '';
  for (let i = 0; i < values.length; i += decim) {
    out += (i === 0 ? 'M' : ' L') + xScale(i).toFixed(1) + ',' + yScale(values[i]).toFixed(2);
  }
  // Always include final point
  const lastI = values.length - 1;
  if (lastI % decim !== 0) {
    out += ' L' + xScale(lastI).toFixed(1) + ',' + yScale(values[lastI]).toFixed(2);
  }
  return out;
}

// Run pin → ghost class index (matches CSS --run-N)
const GHOST_CLASS = { 0: 'ghost-1', 1: 'ghost-2', 2: 'ghost-3' };

// ── Hover tooltip (shared by FrameLevelChart + FrameLevelChartBare) ──
function ChartTooltip({ lines, hoverFrame, hoverX }) {
  const fmt = (v) => {
    if (v == null || isNaN(v)) return '–';
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 10)  return v.toFixed(1);
    return v.toFixed(2);
  };
  const anchor = hoverX > 0.55
    ? { right: `calc(${((1 - hoverX) * 100).toFixed(1)}% + 8px)` }
    : { left:  `calc(${(hoverX       * 100).toFixed(1)}% + 8px)` };
  return (
    <div style={{
      position: 'absolute', top: 8, pointerEvents: 'none', zIndex: 20,
      background: 'var(--paper-warm)', border: '1px solid var(--hairline)',
      borderRadius: 6, padding: '6px 10px', minWidth: 150,
      boxShadow: '0 3px 14px rgba(0,0,0,0.30)',
      ...anchor,
    }}>
      <div style={{ fontSize: 10, color: 'var(--ink-3)', marginBottom: 5,
                    fontFamily: 'var(--mono)', letterSpacing: '0.03em' }}>
        frame {hoverFrame}
      </div>
      {lines.map((ln) => {
        const val = ln.values[Math.min(hoverFrame, ln.values.length - 1)];
        return (
          <div key={ln.runId} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
            <svg width={18} height={8} viewBox="0 0 18 8"
                 style={{ flexShrink: 0, display: 'block', overflow: 'visible' }}>
              <line x1={1} y1={4} x2={17} y2={4}
                stroke={ln.strokeColor || (ln.isFocal ? 'var(--ink)' : 'var(--ink-3)')}
                strokeWidth={ln.isFocal ? 2 : 1.4}
                strokeDasharray={ln.dash || undefined}
                strokeLinecap="round"
              />
            </svg>
            <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                           color: 'var(--ink-2)', maxWidth: 104, fontSize: 11,
                           fontFamily: 'var(--ui)' }}>
              {ln.name}
            </span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 11,
                           fontWeight: ln.isFocal ? 600 : 400,
                           color: ln.isFocal ? 'var(--accent)' : 'var(--ink)',
                           flexShrink: 0 }}>
              {fmt(val)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Frame-level chart (cumulative_return or metric) ──────────────────
// Each comparison line is: { run, ckpt, rollout, color, isFocal }.
function FrameLevelChart({
  title, label, lines, frame, focalLength, setFrame, height = 152,
  yLabel, yMin, yMax, metric, valueAtCursor,
}) {
  const svgRef = useRef(null);
  const [hover, setHover] = useSt(null); // { frame: int, pct: float 0-1 }
  const w = 480;
  const h = height;
  const padL = 32, padR = 8, padT = 8, padB = 18;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  // Compute x_max = max length of any line in view
  const xMax = useMemo(() => Math.max(...lines.map(l => l.values.length), focalLength || 0), [lines, focalLength]);

  // Compute y range if not provided
  const [yLo, yHi] = useMemo(() => {
    if (yMin != null && yMax != null) return [yMin, yMax];
    let lo = Infinity, hi = -Infinity;
    for (const ln of lines) {
      for (let i = 0; i < ln.values.length; i += 4) {
        if (ln.values[i] < lo) lo = ln.values[i];
        if (ln.values[i] > hi) hi = ln.values[i];
      }
    }
    if (lo === Infinity) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.08;
    return [lo - pad, hi + pad];
  }, [lines, yMin, yMax]);

  const xScale = (f) => padL + (f / xMax) * innerW;
  const yScale = (v) => padT + (1 - (v - yLo) / (yHi - yLo)) * innerH;

  // Y ticks (3 lines)
  const yTicks = useMemo(() => {
    const ticks = [];
    for (let i = 0; i < 3; i++) {
      const v = yLo + ((2 - i) / 2) * (yHi - yLo);
      ticks.push(v);
    }
    return ticks;
  }, [yLo, yHi]);

  // X ticks at sensible frame counts
  const xTicks = useMemo(() => {
    const step = xMax > 1500 ? 400 : (xMax > 500 ? 200 : 100);
    const out = [0];
    for (let i = step; i < xMax; i += step) out.push(i);
    out.push(xMax);
    return out;
  }, [xMax]);

  // Scrub on chart click
  const onPointer = useCB((e) => {
    if (!svgRef.current || !setFrame) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * w;
    const f = Math.max(0, Math.min(focalLength - 1, Math.round(((x - padL) / innerW) * xMax)));
    setFrame(f);
  }, [setFrame, focalLength, xMax]);

  const fmtNum = (v) => {
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 10)  return v.toFixed(1);
    return v.toFixed(2);
  };

  const onHover = (e) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const xVB = pct * w;
    const f = Math.max(0, Math.min(xMax - 1, Math.round(((xVB - padL) / innerW) * xMax)));
    setHover({ frame: f, pct });
  };

  return (
    <div className="col" style={{ minWidth: 0, flex: 1 }}>
      {/* Header */}
      <div className="row" style={{ alignItems: 'baseline', gap: 8, marginBottom: 4, height: 22 }}>
        <span className="display" style={{ fontSize: 13, fontWeight: 600 }}>{title}</span>
        {label && <span className="muted" style={{ fontSize: 11 }}>{label}</span>}
        <span className="grow" />
        {valueAtCursor != null && (
          <span className="num" style={{ fontSize: 11.5, color: 'var(--accent)' }}>
            @{frame}: {fmtNum(valueAtCursor)}
          </span>
        )}
      </div>
      <div style={{ position: 'relative' }}>
      <div className="card" style={{ borderRadius: 3, padding: 0 }}>
        <svg
          ref={svgRef}
          className="chart-svg"
          viewBox={`0 0 ${w} ${h}`}
          width="100%"
          height={h}
          preserveAspectRatio="none"
          onMouseDown={(e) => {
            onPointer(e);
            const move = (ev) => onPointer(ev);
            const up = () => {
              window.removeEventListener('mousemove', move);
              window.removeEventListener('mouseup', up);
            };
            window.addEventListener('mousemove', move);
            window.addEventListener('mouseup', up);
          }}
          onMouseMove={onHover}
          onMouseLeave={() => setHover(null)}
          style={{ cursor: 'col-resize', display: 'block', height: 'clamp(160px, calc(13vw + 55px), 320px)' }}
        >
          {/* Y gridlines + labels */}
          {yTicks.map((v, i) => (
            <g key={i}>
              <line x1={padL} y1={yScale(v)} x2={w - padR} y2={yScale(v)} className="grid" />
              <text x={4} y={yScale(v) + 3.5} className="axis-text">{fmtNum(v)}</text>
            </g>
          ))}
          {/* zero line emphasis if range straddles 0 */}
          {yLo < 0 && yHi > 0 && (
            <line x1={padL} y1={yScale(0)} x2={w - padR} y2={yScale(0)} stroke="rgba(255,255,255,.18)" strokeWidth="0.8" />
          )}
          {/* X axis */}
          <line x1={padL} y1={padT + innerH} x2={w - padR} y2={padT + innerH} className="axis" />
          {xTicks.map((f, i) => (
            <text key={i} x={xScale(f) - 8} y={h - 4} className="axis-text">{f >= 1000 ? `${(f/1000).toFixed(f % 1000 === 0 ? 0 : 1)}k` : f}</text>
          ))}

          {/* Area fill + ghost lines (drawn first so focal stays on top) */}
          {lines.filter(l => !l.isFocal).map((ln, i) => {
            const cls = GHOST_CLASS[i % 3];
            return (
              <g key={ln.runId}>
                <path d={buildPath(ln.values, xScale, yScale, 3)} className={cls}
                  style={{ stroke: ln.color }} />
                {/* End marker */}
                <line
                  x1={xScale(ln.values.length - 1)}
                  y1={yScale(ln.values[ln.values.length - 1]) - 4}
                  x2={xScale(ln.values.length - 1)}
                  y2={yScale(ln.values[ln.values.length - 1]) + 4}
                  stroke={ln.color} opacity="0.7"
                />
              </g>
            );
          })}
          {/* Focal — area fill then line */}
          {lines.filter(l => l.isFocal).map((ln) => {
            const baseline = padT + innerH;
            const pts = [];
            for (let i = 0; i < ln.values.length; i += 2) {
              pts.push([xScale(i), yScale(ln.values[i])]);
            }
            const lastI = ln.values.length - 1;
            if (lastI % 2 !== 0) pts.push([xScale(lastI), yScale(ln.values[lastI])]);
            const areaD = pts.length === 0 ? '' :
              `M ${pts[0][0].toFixed(1)},${baseline} L ${pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(2)}`).join(' L ')} L ${pts[pts.length-1][0].toFixed(1)},${baseline} Z`;
            return (
              <g key={ln.runId}>
                <defs>
                  <linearGradient id="focalFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor="var(--accent)" stopOpacity="0.22" />
                    <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <path d={areaD} fill="url(#focalFill)" />
                <path d={buildPath(ln.values, xScale, yScale, 2)} className="focal" />
              </g>
            );
          })}

          {/* Hover crosshair */}
          {hover != null && (
            <line
              x1={xScale(hover.frame)} y1={padT}
              x2={xScale(hover.frame)} y2={padT + innerH}
              stroke="var(--ink-3)" strokeWidth="0.7" strokeDasharray="2 3"
              style={{ pointerEvents: 'none' }}
            />
          )}
          {/* Playhead */}
          {focalLength > 0 && (
            <g>
              <line x1={xScale(frame)} y1={padT} x2={xScale(frame)} y2={padT + innerH} className="playhead" />
              <circle cx={xScale(frame)} cy={padT + innerH - 6} r="3" fill="var(--accent)"
                filter="drop-shadow(0 0 3px var(--accent))" />
            </g>
          )}
        </svg>
      </div>
      {hover != null && (
        <ChartTooltip lines={lines} hoverFrame={hover.frame} hoverX={hover.pct} />
      )}
      </div>
    </div>
  );
}

// ── Loss strip (training-step x-axis, ckpt playhead) ─────────────────
function LossChart({ title, values, atCkptValue, width, height = 80, ckptStepFrac }) {
  const w = width;
  const h = height;
  const padL = 4, padR = 4, padT = 6, padB = 16;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const [yLo, yHi] = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i += 2) {
      if (values[i] < lo) lo = values[i];
      if (values[i] > hi) hi = values[i];
    }
    if (lo === Infinity) { lo = 0; hi = 1; }
    return [lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.05];
  }, [values]);

  const xScale = (i) => padL + (i / (values.length - 1)) * innerW;
  const yScale = (v) => padT + (1 - (v - yLo) / (yHi - yLo)) * innerH;

  return (
    <div className="col" style={{ flex: 1, minWidth: 0, padding: '6px 10px' }}>
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 2 }}>
        <span className="label-eyebrow">{title}</span>
        <span className="num" style={{ fontSize: 10.5, color: 'var(--ink-2)' }}>at ckpt: {atCkptValue}</span>
      </div>
      <svg className="chart-svg" viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none">
        <line x1={padL} y1={padT + innerH} x2={w - padR} y2={padT + innerH} className="axis" />
        <path d={buildPath(values, xScale, yScale, 2)} className="focal" />
        <line
          x1={padL + ckptStepFrac * innerW}
          y1={padT}
          x2={padL + ckptStepFrac * innerW}
          y2={padT + innerH}
          className="playhead-soft"
        />
      </svg>
    </div>
  );
}

// ── Checkpoint sparkbar ──────────────────────────────────────────────
function CheckpointSparkbar({ checkpoints, activeStep, onSelect, height = 32 }) {
  // Use a fixed internal coordinate width for maths; the SVG renders at 100% of its container.
  const w = 360;
  const h = height;
  const n = checkpoints.length;
  const bw = Math.max(2, (w - 4) / n - 2);
  const minMean = Math.min(...checkpoints.map(c => c.mean));
  const maxMean = Math.max(...checkpoints.map(c => c.mean), 1);
  // Normalise against the actual range so early low-reward bars are still visible
  const range = Math.max(maxMean - Math.min(minMean, 0), 1);

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      width="100%" height={h}
      preserveAspectRatio="none"
      style={{ display: 'block', cursor: 'pointer', overflow: 'visible' }}
    >
      {/* Baseline */}
      <line x1={0} y1={h - 1} x2={w} y2={h - 1} stroke="var(--hairline)" strokeWidth="1" />

      {checkpoints.map((c, i) => {
        const x = 2 + i * (bw + 2);
        const bh = Math.max(2, ((c.mean - Math.min(minMean, 0)) / range) * (h - 8));
        const active = c.step === activeStep;
        return (
          <g key={c.step} onClick={() => onSelect(c.step)}>
            {/* Full-height transparent hit area */}
            <rect x={x - 1} y={0} width={bw + 2} height={h} fill="transparent" />
            {/* Bar */}
            <rect
              x={x} y={h - 2 - bh}
              width={bw} height={bh}
              rx="1"
              fill={active ? 'var(--accent)' : 'var(--ink)'}
              opacity={active ? 1 : (c.isDip ? 0.35 : 0.5)}
            />
            {/* Active indicator — small dot above bar instead of triangle */}
            {active && (
              <circle
                cx={x + bw / 2} cy={h - 4 - bh}
                r="2.5"
                fill="var(--accent)"
              />
            )}
            <title>{`ckpt ${D.fmtStep(c.step)} · μ ${c.mean} ± ${c.std}`}</title>
          </g>
        );
      })}
    </svg>
  );
}

// ── Shared run line-style table ──────────────────────────────────────
// Slot 0 = focused (solid ink). Slots 1-3 match ghost-1/2/3 CSS classes.
// Used by both sparklines (left rail) and chart ghost curves.
const RUN_LINE_STYLES = [
  { color: 'var(--ink)',   dash: null,    width: 1.6 }, // 0 focused
  { color: 'var(--run-2)', dash: '4 2',  width: 1.4 }, // 1 ghost-1
  { color: 'var(--run-3)', dash: '6 3',  width: 1.4 }, // 2 ghost-2
  { color: 'var(--run-4)', dash: '2 2',  width: 1.4 }, // 3 ghost-3
];
const RUN_LINE_DEFAULT = { color: 'var(--ink-4)', dash: null, width: 0.0 }; // unpinned

// ── Sparkline (tiny chart for runs list) ─────────────────────────────
function Sparkline({ values, width = 64, height = 16, color = 'var(--ink-2)', strokeDasharray = null, strokeWidth = 1 }) {
  const w = width;
  const h = height;
  if (!values || values.length === 0) return null;
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (lo === hi) hi = lo + 1;
  const xScale = (i) => (i / (values.length - 1)) * (w - 2) + 1;
  const yScale = (v) => 1 + (1 - (v - lo) / (hi - lo)) * (h - 2);
  let d = '';
  for (let i = 0; i < values.length; i++) {
    d += (i === 0 ? 'M' : ' L') + xScale(i).toFixed(1) + ',' + yScale(values[i]).toFixed(1);
  }
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      <path d={d} stroke={color} strokeWidth={strokeWidth} fill="none"
            strokeDasharray={strokeDasharray || undefined} />
    </svg>
  );
}

window.ChartTooltip = ChartTooltip;
window.FrameLevelChart = FrameLevelChart;
window.LossChart = LossChart;
window.CheckpointSparkbar = CheckpointSparkbar;
window.Sparkline = Sparkline;
window.buildPath = buildPath;
window.RUN_LINE_STYLES = RUN_LINE_STYLES;
window.RUN_LINE_DEFAULT = RUN_LINE_DEFAULT;
