// wf-primitives.jsx — shared lo-fi wireframe building blocks
// All exported on window so other Babel files can use them.

// Generate a reward-like curve path
function rewardPath(w, h, seed = 1, opts = {}) {
  const { noise = 0.06, target = 0.85, samples = 80, dip = false } = opts;
  let s = seed;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const pts = [];
  for (let i = 0; i < samples; i++) {
    const t = i / (samples - 1);
    // Sigmoid-ish learning curve
    const base = target * (1 - Math.exp(-3.2 * t));
    const wiggle = (rand() - 0.5) * noise * (1 - t * 0.5);
    let y = base + wiggle;
    if (dip && t > 0.45 && t < 0.6) y -= 0.18;
    pts.push([t * w, h - y * h * 0.95 - 2]);
  }
  return 'M' + pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' L');
}

function lossPath(w, h, seed = 2, opts = {}) {
  const { noise = 0.05, floor = 0.12, samples = 80 } = opts;
  let s = seed;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const pts = [];
  for (let i = 0; i < samples; i++) {
    const t = i / (samples - 1);
    const base = floor + (1 - floor) * Math.exp(-2.8 * t);
    const wiggle = (rand() - 0.5) * noise * (1 - t * 0.3);
    const y = base + wiggle;
    pts.push([t * w, y * h * 0.9 + 4]);
  }
  return 'M' + pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' L');
}

// Chart frame with grid + 1-3 lines
function Chart({ w = 240, h = 100, lines = [{ kind: 'reward', seed: 1 }], xLabel, yLabel, title, yTicks = ['1.0','.5','0'], xTicks = ['0','5M','10M'] }) {
  const innerW = w - 22;
  const innerH = h - 16;
  return (
    <div style={{ position: 'relative' }}>
      {title && <div className="label" style={{ marginBottom: 2 }}>{title}</div>}
      <svg className="chart" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        {/* y ticks */}
        {yTicks.map((t, i) => {
          const y = 4 + (i / (yTicks.length - 1)) * (innerH - 4);
          return (
            <g key={`y${i}`}>
              <line className="grid" x1={20} y1={y} x2={w - 2} y2={y} />
              <text className="label-sm" x={2} y={y + 3}>{t}</text>
            </g>
          );
        })}
        {/* x axis */}
        <line className="axis" x1={20} y1={innerH} x2={w - 2} y2={innerH} />
        <line className="axis" x1={20} y1={4} x2={20} y2={innerH} />
        {xTicks.map((t, i) => {
          const x = 20 + (i / (xTicks.length - 1)) * (innerW - 2);
          return <text key={`x${i}`} className="label-sm" x={x - 6} y={h - 1}>{t}</text>;
        })}
        {/* lines */}
        {lines.map((ln, i) => {
          const d = ln.kind === 'loss'
            ? lossPath(innerW - 2, innerH - 6, ln.seed || 1, ln.opts)
            : rewardPath(innerW - 2, innerH - 6, ln.seed || 1, ln.opts);
          return (
            <g key={i} transform={`translate(20, 4)`}>
              <path d={d} className={ln.cls || 'line-a'} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Sparkline({ kind = 'reward', seed = 3, w = 60, h = 14 }) {
  const d = kind === 'loss' ? lossPath(w, h, seed) : rewardPath(w, h, seed);
  return <svg className="spark" viewBox={`0 0 ${w} ${h}`}><path d={d} /></svg>;
}

// Heatmap (animated value/attention placeholder)
function Heatmap({ cols = 16, rows = 10, seed = 1, scale = 1 }) {
  let s = seed;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const cells = [];
  for (let i = 0; i < rows * cols; i++) {
    const v = Math.pow(rand(), 1.4) * scale;
    cells.push(v);
  }
  return (
    <div className="heatmap" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gridTemplateRows: `repeat(${rows}, 1fr)`, width: '100%', height: '100%' }}>
      {cells.map((v, i) => (
        <div key={i} style={{ backgroundColor: `rgba(26,23,20,${v.toFixed(2)})` }} />
      ))}
    </div>
  );
}

// Bar chart (action distribution)
function ActionBars({ n = 8, seed = 5, w = 220, h = 70, labels }) {
  let s = seed;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const vals = Array.from({ length: n }, () => 0.2 + rand() * 0.8);
  const max = Math.max(...vals);
  const gap = 3;
  const bw = (w - gap * (n - 1)) / n;
  return (
    <svg className="chart" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {vals.map((v, i) => {
        const bh = (v / max) * (h - 14);
        return (
          <g key={i}>
            <rect x={i * (bw + gap)} y={h - 12 - bh} width={bw} height={bh} fill="#1a1714" />
            <text className="label-sm" x={i * (bw + gap) + bw / 2 - 3} y={h - 2}>{labels ? labels[i] : i}</text>
          </g>
        );
      })}
    </svg>
  );
}

// Parallel coords sketch
function ParallelCoords({ w = 720, h = 160, axes = ['lr', 'γ', 'λ', 'clip', 'batch', 'entropy', 'reward'], lines = 14, seed = 9 }) {
  let s = seed;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const axW = (w - 40) / (axes.length - 1);
  const top = 14, bot = h - 22;
  const xs = axes.map((_, i) => 20 + i * axW);
  const runs = [];
  for (let r = 0; r < lines; r++) {
    const ys = xs.map(() => top + rand() * (bot - top));
    runs.push(ys);
  }
  // Highlight 2-3
  const hi = [1, 4, 8];
  return (
    <svg className="chart" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {xs.map((x, i) => (
        <g key={i}>
          <line className="axis" x1={x} y1={top} x2={x} y2={bot} />
          <text className="label-sm" x={x - 8} y={top - 4}>{axes[i]}</text>
          <text className="label-sm" x={x - 4} y={bot + 10}>—</text>
        </g>
      ))}
      {runs.map((ys, r) => {
        const isHi = hi.includes(r);
        const d = 'M' + ys.map((y, i) => `${xs[i]},${y.toFixed(1)}`).join(' L');
        return (
          <path
            key={r}
            d={d}
            stroke={isHi ? (r === 1 ? '#c96442' : '#1a1714') : 'rgba(26,23,20,.25)'}
            strokeWidth={isHi ? 1.3 : 0.8}
            fill="none"
            strokeDasharray={r === 4 ? '3 2' : undefined}
          />
        );
      })}
    </svg>
  );
}

// Run row component
function RunRow({ name, id, status = 'done', steps, reward, active, cols = '8px 1fr 60px 56px 36px' }) {
  return (
    <div className={`run-row ${active ? 'active' : ''}`} style={{ gridTemplateColumns: cols }}>
      <span className={`status-dot ${status}`} />
      <span className="num"><span className="strong">{name}</span> <span className="muted">{id}</span></span>
      <Sparkline seed={(name || '').length * 3 + 1} />
      <span className="num muted" style={{ textAlign:'right' }}>{reward}</span>
      <span className="num muted" style={{ textAlign:'right' }}>{steps}</span>
    </div>
  );
}

// Sticky-note ish caption block
function Caption({ optimizes, tradeoffs, watchouts }) {
  return (
    <div className="caption-block">
      <span className="h">{optimizes}</span>
      <div className="row">
        {tradeoffs && (<div className="col"><div className="ttl">Trade-offs</div>{tradeoffs}</div>)}
        {watchouts && (<div className="col"><div className="ttl">Watch-outs</div>{watchouts}</div>)}
      </div>
    </div>
  );
}

Object.assign(window, {
  Chart, Sparkline, Heatmap, ActionBars, ParallelCoords, RunRow, Caption,
  rewardPath, lossPath,
});
