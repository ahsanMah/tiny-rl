// hifi/charts.jsx — interactive chart suite

const { useMemo, useRef, useCallback: useCB } = React;

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

// ── Frame-level chart (cumulative_return or metric) ──────────────────
// Each comparison line is: { run, ckpt, rollout, color, isFocal }.
function FrameLevelChart({
  title, label, lines, frame, focalLength, setFrame, height = 152,
  yLabel, yMin, yMax, metric, valueAtCursor,
}) {
  const svgRef = useRef(null);
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
          style={{ cursor: 'col-resize', display: 'block' }}
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
            <line x1={padL} y1={yScale(0)} x2={w - padR} y2={yScale(0)} stroke="rgba(28,24,19,.25)" strokeWidth="0.8" />
          )}
          {/* X axis */}
          <line x1={padL} y1={padT + innerH} x2={w - padR} y2={padT + innerH} className="axis" />
          {xTicks.map((f, i) => (
            <text key={i} x={xScale(f) - 8} y={h - 4} className="axis-text">{f >= 1000 ? `${(f/1000).toFixed(f % 1000 === 0 ? 0 : 1)}k` : f}</text>
          ))}

          {/* Ghost lines (drawn first so focal stays on top) */}
          {lines.filter(l => !l.isFocal).map((ln, i) => {
            const cls = GHOST_CLASS[i % 3];
            return (
              <g key={ln.runId}>
                <path d={buildPath(ln.values, xScale, yScale, 3)} className={`chart-svg ${cls}`}
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
          {/* Focal */}
          {lines.filter(l => l.isFocal).map((ln) => (
            <path key={ln.runId} d={buildPath(ln.values, xScale, yScale, 2)} className="focal" />
          ))}

          {/* Playhead */}
          {focalLength > 0 && (
            <g>
              <line x1={xScale(frame)} y1={padT} x2={xScale(frame)} y2={padT + innerH} className="playhead" />
              <circle cx={xScale(frame)} cy={padT + innerH - 6} r="3" fill="var(--accent)" />
            </g>
          )}
        </svg>
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
function CheckpointSparkbar({ checkpoints, activeStep, onSelect, width = 360, height = 32 }) {
  const w = width;
  const h = height;
  const bw = (w - 4) / checkpoints.length - 2;
  const maxMean = Math.max(...checkpoints.map(c => c.mean), 1);

  return (
    <div style={{ display: 'flex', alignItems: 'center' }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
        <line x1={0} y1={h - 1} x2={w} y2={h - 1} stroke="var(--hairline)" />
        {checkpoints.map((c, i) => {
          const x = 2 + i * (bw + 2);
          const bh = Math.max(2, (c.mean / maxMean) * (h - 6));
          const active = c.step === activeStep;
          return (
            <g key={c.step} style={{ cursor: 'pointer' }} onClick={() => onSelect(c.step)}>
              {/* Hover hit area */}
              <rect x={x - 1} y={0} width={bw + 2} height={h} fill="transparent" />
              <rect
                x={x} y={h - 2 - bh}
                width={bw} height={bh}
                fill={active ? 'var(--accent)' : 'var(--ink)'}
                opacity={active ? 1 : (c.isDip ? 0.42 : 0.55)}
              />
              {active && (
                <polygon
                  points={`${x + bw / 2 - 3},0 ${x + bw / 2 + 3},0 ${x + bw / 2},4`}
                  fill="var(--accent)"
                />
              )}
              <title>{`ckpt ${D.fmtStep(c.step)} · μ ${c.mean}`}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Sparkline (tiny chart for runs list) ─────────────────────────────
function Sparkline({ values, width = 64, height = 16, color = 'var(--ink-2)' }) {
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
      <path d={d} stroke={color} strokeWidth="1" fill="none" />
    </svg>
  );
}

// ── Action distribution bars ─────────────────────────────────────────
function ActionBars({ probs, labels, width = 246, height = 64 }) {
  const w = width;
  const h = height;
  const n = probs.length;
  const gap = 3;
  const bw = (w - gap * (n - 1)) / n;
  const max = Math.max(...probs);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      {probs.map((p, i) => {
        const bh = (p / max) * (h - 14);
        return (
          <g key={i}>
            <rect x={i * (bw + gap)} y={h - 12 - bh} width={bw} height={bh}
                  fill={i === probs.indexOf(max) ? 'var(--accent)' : 'var(--ink)'} />
            <text x={i * (bw + gap) + bw / 2 - 4} y={h - 2}
                  style={{ font: '9px var(--mono)', fill: 'var(--ink-3)' }}>{labels[i]}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ── TD-error strip (1-row heatmap synced to frame) ───────────────────
function TdErrorStrip({ values, frame, totalFrames, width = 246, height = 38 }) {
  const w = width;
  const h = height;
  const cols = Math.min(64, values.length);
  const cw = (w - 2) / cols;
  // Bin the values to columns
  const binned = useMemo(() => {
    const out = new Array(cols).fill(0);
    const binSize = Math.ceil(values.length / cols);
    for (let i = 0; i < cols; i++) {
      let max = 0;
      for (let j = i * binSize; j < Math.min(values.length, (i + 1) * binSize); j++) {
        const v = Math.abs(values[j]);
        if (v > max) max = v;
      }
      out[i] = max;
    }
    return out;
  }, [values, cols]);
  const maxV = Math.max(...binned, 0.01);
  const activeCol = Math.floor((frame / (totalFrames || 1)) * cols);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      {binned.map((v, i) => {
        const alpha = Math.min(1, v / maxV);
        return (
          <rect key={i}
            x={1 + i * cw}
            y={2}
            width={cw - 0.5}
            height={h - 4}
            fill={`rgba(28,24,19,${(alpha * 0.85).toFixed(2)})`}
          />
        );
      })}
      {/* Frame indicator */}
      <line
        x1={1 + activeCol * cw + cw / 2}
        y1={0}
        x2={1 + activeCol * cw + cw / 2}
        y2={h}
        stroke="var(--accent)" strokeWidth="1"
      />
    </svg>
  );
}

window.FrameLevelChart = FrameLevelChart;
window.LossChart = LossChart;
window.CheckpointSparkbar = CheckpointSparkbar;
window.Sparkline = Sparkline;
window.ActionBars = ActionBars;
window.TdErrorStrip = TdErrorStrip;
window.buildPath = buildPath;
