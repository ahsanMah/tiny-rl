// hifi/charts.jsx — interactive chart suite

const { useMemo, useRef, useCallback: useCB, useState: useSt, useEffect: useEff } = React;

// ── Responsive sizing ────────────────────────────────────────────────
// Measure a container's real pixel size so SVGs can be drawn at 1:1 (1 user
// unit = 1px). This avoids preserveAspectRatio="none", whose anisotropic
// stretch distorts text glyphs and stroke widths. Returns [ref, {width,height}].
let _clipSeq = 0;
function nextClipId() { return 'clip-' + (++_clipSeq); }
function useMeasuredSize(fallback = { width: 480, height: 160 }) {
  const ref = useRef(null);
  const [size, setSize] = useSt(fallback);
  useEff(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      if (cr.width > 0 && cr.height > 0) {
        setSize({ width: cr.width, height: cr.height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

// ── Drag lifecycle ───────────────────────────────────────────────────
// Shared pointer-drag helper used by chart scrubbing and the resizable rail.
// Suppresses text selection (and optionally forces a cursor) for the drag's
// duration, then restores both on mouseup. `onMove` is called with each event,
// starting with the initiating mousedown.
function startDrag(e, onMove, cursor) {
  e.preventDefault();
  onMove(e);
  const prevSelect = document.body.style.userSelect;
  const prevCursor = document.body.style.cursor;
  document.body.style.userSelect = 'none';
  if (cursor) document.body.style.cursor = cursor;
  const move = (ev) => onMove(ev);
  const up = () => {
    document.body.style.userSelect = prevSelect;
    document.body.style.cursor = prevCursor;
    window.removeEventListener('mousemove', move);
    window.removeEventListener('mouseup', up);
  };
  window.addEventListener('mousemove', move);
  window.addEventListener('mouseup', up);
}

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

// ── Missing-data placeholder ─────────────────────────────────────────
// Shown wherever an artifact (signal / metric / video) is absent. We never
// synthesize stand-in data; we state plainly that it isn't there.
function EmptyState({ title, detail, mark = '∅', minHeight }) {
  return (
    <div className="empty-state" style={minHeight ? { minHeight } : undefined}>
      <span className="empty-state__mark">{mark}</span>
      <span className="label-eyebrow empty-state__title">{title}</span>
      {detail && <span className="empty-state__detail">{detail}</span>}
    </div>
  );
}

// ── Hover tooltip (shared by the frame-level charts) ──
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
      background: 'var(--surface)', border: '1px solid var(--hairline)',
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

// ── Loss strip (training-step x-axis, ckpt playhead) ─────────────────
function LossChart({ title, values, atCkptValue, width, height = 80, ckptStepFrac }) {
  // Measure the real rendered size so we can draw at 1:1 (no aspect-ratio stretch).
  const [boxRef, size] = useMeasuredSize({ width, height });
  const w = size.width;
  const h = size.height;
  const padL = 4, padR = 4, padT = 6, padB = 16;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;
  const clipId = useMemo(() => nextClipId(), []);

  // Full scan so the drawn line's extrema are always within range (no overshoot).
  const [yLo, yHi] = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i++) {
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
      <div ref={boxRef} style={{ width: '100%', height }}>
        <svg className="chart-svg" viewBox={`0 0 ${w} ${h}`} width="100%" height="100%">
          <defs>
            <clipPath id={clipId}>
              <rect x={padL} y={padT} width={innerW} height={innerH} />
            </clipPath>
          </defs>
          <line x1={padL} y1={padT + innerH} x2={w - padR} y2={padT + innerH} className="axis" />
          <path d={buildPath(values, xScale, yScale, 2)} className="focal" clipPath={`url(#${clipId})`} />
          <line
            x1={padL + ckptStepFrac * innerW}
            y1={padT}
            x2={padL + ckptStepFrac * innerW}
            y2={padT + innerH}
            className="playhead-soft"
          />
        </svg>
      </div>
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
  { color: 'var(--ink)',   dash: null, width: 1.6 }, // 0 focused
  { color: 'var(--run-2)', dash: null, width: 1.4 }, // 1 ghost-1
  { color: 'var(--run-3)', dash: null, width: 1.4 }, // 2 ghost-2
  { color: 'var(--run-4)', dash: null, width: 1.4 }, // 3 ghost-3
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
window.EmptyState = EmptyState;
window.LossChart = LossChart;
window.CheckpointSparkbar = CheckpointSparkbar;
window.Sparkline = Sparkline;
window.buildPath = buildPath;
window.useMeasuredSize = useMeasuredSize;
window.nextClipId = nextClipId;
window.startDrag = startDrag;
window.RUN_LINE_STYLES = RUN_LINE_STYLES;
window.RUN_LINE_DEFAULT = RUN_LINE_DEFAULT;
