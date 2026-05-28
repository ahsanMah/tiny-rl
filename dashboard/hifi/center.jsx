// hifi/center.jsx — center column composition

const { useMemo: useM, useState: useSt } = React;

// ── Icon primitives (inline SVG — no library needed) ─────────────────
function IconSun({ size = 15, strokeWidth = 1.8 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4"/>
      <line x1="12" y1="2"  x2="12" y2="6"/>
      <line x1="12" y1="18" x2="12" y2="22"/>
      <line x1="2"  y1="12" x2="6"  y2="12"/>
      <line x1="18" y1="12" x2="22" y2="12"/>
      <line x1="4.93"  y1="4.93"  x2="7.76"  y2="7.76"/>
      <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>
      <line x1="4.93"  y1="19.07" x2="7.76"  y2="16.24"/>
      <line x1="16.24" y1="7.76"  x2="19.07" y2="4.93"/>
    </svg>
  );
}
function IconMoon({ size = 15, strokeWidth = 1.8 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  );
}
// ── Top breadcrumb / actions bar ─────────────────────────────────────
function TopBar({ run, ckpt, pinnedCount, diffBaselineName, onChangeBaseline, allRuns, pinnedRuns, darkMode, onToggleDark }) {
  return (
    <div className="row border-b" style={{ padding: '12px 22px', gap: 12, height: 80, flex: '0 0 auto', minWidth: 0 }}>
      <div className="doc-title" style={{ flex: '1 1 auto', minWidth: 0 }}>
        <span className="crumb">
          <span style={{ color: 'var(--ink)' }} className="strong">tracker</span>
          <span className="sep">/</span>
          <span>{run.env.split('-')[0]}</span>
          <span className="sep">/</span>
          <span className="current">{run.name}</span>
          <span className="sep">@</span>
          <span className="num" style={{ color: 'var(--ink-2)' }}>{D.fmtStep(ckpt.step)}</span>
        </span>
      </div>
            <select
        className="dropdown"
        value={diffBaselineName || ''}
        onChange={(e) => onChangeBaseline(e.target.value || null)}
        title="hyperparam diff baseline"
        style={{ font: '500 var(--t-sm) var(--ui)', flex: '0 0 auto', maxWidth: 180, padding: '6px 10px', height: 34 }}
      >
        <option value="">no baseline</option>
        {pinnedRuns.filter(r => r.id !== run.id).map(r => (
          <option key={r.id} value={r.name}>vs {r.name}</option>
        ))}
      </select>
      <button
        className="btn icon"
        onClick={onToggleDark}
        title={darkMode ? 'switch to light mode' : 'switch to dark mode'}
        style={{ flex: '0 0 auto', width: 34, height: 34 }}
      >
        {darkMode ? <IconSun /> : <IconMoon />}
      </button>
    </div>
  );
}

// ── Checkpoint navigator row (sparkbar + arrows + stats) ─────────────
function CkptNav({ run, ckpt, onSelectCkpt }) {
  const idx = run.checkpoints.findIndex(c => c.step === ckpt.step);
  const total = run.checkpoints.length;
  const prev = () => idx > 0 && onSelectCkpt(run.checkpoints[idx - 1].step);
  const next = () => idx < total - 1 && onSelectCkpt(run.checkpoints[idx + 1].step);
  return (
    <div className="row border-b" style={{ padding: '18px 22px', gap: 14, flex: '0 0 auto', background: 'var(--surface)' }}>
      <span className="label-eyebrow">Checkpoint</span>
      <span className="num strong" style={{ fontSize: 15, fontFamily: 'var(--mono)' }}>{D.fmtStep(ckpt.step)}</span>
      <span className="muted" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>step {idx + 1} of {total}</span>
      <div className="row gap-1">
        <button className="btn icon" onClick={prev} disabled={idx === 0} title="previous (J)">◀</button>
        <button className="btn icon" onClick={next} disabled={idx === total - 1} title="next (L)">▶</button>
      </div>
      <div style={{ flex: 1, minWidth: 0, marginLeft: 6 }}>
        <CheckpointSparkbar
          checkpoints={run.checkpoints}
          activeStep={ckpt.step}
          onSelect={onSelectCkpt}
          height={32}
        />
      </div>
      <span className="grow" />
      <div className="row gap-3" style={{ alignItems: 'baseline' }}>
        <span className="col" style={{ alignItems: 'flex-end' }}>
          <span className="label-eyebrow">μ ± σ</span>
          <span className="num strong" style={{ fontSize: 13 }}>{ckpt.mean} <span className="muted" style={{ fontSize: 11 }}>± {ckpt.std}</span></span>
        </span>
        <span className="col" style={{ alignItems: 'flex-end' }}>
          <span className="label-eyebrow">best</span>
          <span className="num strong" style={{ fontSize: 13 }}>{ckpt.best}</span>
        </span>
        <span className="col" style={{ alignItems: 'flex-end' }}>
          <span className="label-eyebrow">worst</span>
          <span className="num strong" style={{ fontSize: 13 }}>{ckpt.worst}</span>
        </span>
      </div>
    </div>
  );
}

// ── Episode picker buttons (no thumbnails) ──────────────────────────
function EpisodePicker({ ckpt, selected, onSelect }) {
  const rollouts = ckpt.rollouts;
  // Buttons: best, median, worst (the 3 the user explicitly asked for)
  const visible = ['best','median','worst'].map(k => rollouts.find(r => r.kind === k)).filter(Boolean);
  const iconMap = { best: '★', median: '◇', worst: '✕' };
  const labelMap = { best: 'best episode', median: 'median', worst: 'worst' };
  return (
    <div className="row gap-2">
      {visible.map(r => (
        <button
          key={r.kind}
          className={'ep-btn' + (selected === r.kind ? ' active' : '')}
          onClick={() => onSelect(r.kind)}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="ep-icon">{iconMap[r.kind]}</span>
            <span>{labelMap[r.kind]}</span>
          </span>
          <span className="ep-meta">{r.length.toLocaleString()}f · r {r.return}</span>
        </button>
      ))}
      
    </div>
  );
}

// ── Frame-level chart pair (cumulative + metric) ─────────────────────
const METRIC_OPTIONS = [
  { key: 'value',       label: 'V(s_t)' },
  { key: 'step_reward', label: 'step_reward' },
  { key: 'action_logp', label: 'action_logp' },
  { key: 'advantage',   label: 'advantage' },
  { key: 'td_error',    label: 'TD-error' },
  { key: 'entropy',     label: 'entropy' },
];

function FrameChartPair({ focalRun, focalCkpt, focalRollout, frame, setFrame, pinnedRuns, metric, setMetric, signalVersion }) {
  // ── Line builder — shared helper adds name/strokeColor/dash for tooltip ──
  const buildLines = (sig) => {
    const out = [];
    out.push({
      runId: focalRun.id, name: focalRun.name,
      values: D.frameSignal(focalRun, focalCkpt, focalRollout, sig),
      isFocal: true, strokeColor: RUN_LINE_STYLES[0].color, dash: null, color: 'var(--ink)',
    });
    let gi = 0;
    for (const pr of pinnedRuns) {
      if (pr.id === focalRun.id) continue;
      const c = pr.checkpoints[pr.checkpoints.length - 1];
      const ro = c.rollouts.find(r => r.kind === 'best') || c.rollouts[0];
      const ls = RUN_LINE_STYLES[gi + 1] || RUN_LINE_STYLES[RUN_LINE_STYLES.length - 1];
      out.push({
        runId: pr.id, name: pr.name,
        values: D.frameSignal(pr, c, ro, sig),
        isFocal: false, strokeColor: ls.color, dash: ls.dash, color: '',
      });
      gi++;
    }
    return out;
  };

  const cumLines    = useM(() => buildLines('cumulative_return'), [focalRun, focalCkpt, focalRollout, pinnedRuns, signalVersion]);
  const metricLines = useM(() => buildLines(metric),              [focalRun, focalCkpt, focalRollout, pinnedRuns, metric, signalVersion]);

  const cumAtCursor   = cumLines[0]?.values?.[Math.min(frame, cumLines[0].values.length - 1)];
  const metricAtCursor = metricLines[0]?.values?.[Math.min(frame, metricLines[0].values.length - 1)];

  const metricLabel = METRIC_OPTIONS.find(m => m.key === metric)?.label || metric;
  const ghostsCount = cumLines.length - 1;

  return (
    <div className="row gap-3" style={{ padding: '20px 22px 16px' }}>
      <FrameLevelChart
        title="cumulative_return"
        label={`- ${cumLines.length} pinned runs`}
        lines={cumLines}
        frame={frame}
        focalLength={focalRollout.length}
        setFrame={setFrame}
        height={160}
        valueAtCursor={cumAtCursor}
      />
      <div className="col" style={{ flex: 1, minWidth: 0 }}>
        <div className="row" style={{ alignItems: 'baseline', gap: 8, marginBottom: 4, height: 22 }}>
          <span className="display" style={{ fontSize: 13, fontWeight: 600 }}>{metricLabel}</span>
          <span className="muted" style={{ fontSize: 11 }}>per-frame</span>
          <span className="grow" />
          {metricAtCursor != null && (
            <span className="num" style={{ fontSize: 11.5, color: 'var(--accent)' }}>
              @{frame}: {Math.abs(metricAtCursor) >= 10 ? metricAtCursor.toFixed(0) : metricAtCursor.toFixed(2)}
            </span>
          )}
          <select className="dropdown" value={metric} onChange={(e) => setMetric(e.target.value)}>
            {METRIC_OPTIONS
              .filter(o => {
                const realSignals = focalRun.capabilities?.signals || [];
                // If the run has real signals, only show those + cumulative_return (always derived).
                // If no real signals (pure synthetic run), show everything.
                return realSignals.length === 0 || realSignals.includes(o.key) || o.key === 'cumulative_return';
              })
              .map(o => <option key={o.key} value={o.key}>{o.label}</option>)}
          </select>
        </div>
        {/* Reuse FrameLevelChart minus its own header (we built the header above) */}
        <FrameLevelChartBare
          lines={metricLines}
          frame={frame}
          focalLength={focalRollout.length}
          setFrame={setFrame}
          height={160}
        />
      </div>
    </div>
  );
}

// Bare version (no header) — used when the parent renders its own title row
function FrameLevelChartBare({ lines, frame, focalLength, setFrame, height = 160 }) {
  const svgRef = React.useRef(null);
  const [hover, setHover] = useSt(null); // { frame: int, pct: float 0-1 }
  const w = 480;
  const h = height;
  const padL = 32, padR = 8, padT = 8, padB = 18;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const xMax = useM(() => Math.max(...lines.map(l => l.values.length), focalLength || 0), [lines, focalLength]);
  const [yLo, yHi] = useM(() => {
    let lo = Infinity, hi = -Infinity;
    for (const ln of lines) for (let i = 0; i < ln.values.length; i += 4) {
      if (ln.values[i] < lo) lo = ln.values[i];
      if (ln.values[i] > hi) hi = ln.values[i];
    }
    if (lo === Infinity) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.08;
    return [lo - pad, hi + pad];
  }, [lines]);

  const xScale = (f) => padL + (f / xMax) * innerW;
  const yScale = (v) => padT + (1 - (v - yLo) / (yHi - yLo)) * innerH;

  const yTicks = [yHi, (yLo + yHi) / 2, yLo];
  const step = xMax > 1500 ? 400 : (xMax > 500 ? 200 : 100);
  const xTicks = [0]; for (let i = step; i < xMax; i += step) xTicks.push(i); xTicks.push(xMax);

  const onPointer = (e) => {
    if (!svgRef.current || !setFrame) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * w;
    const f = Math.max(0, Math.min(focalLength - 1, Math.round(((x - padL) / innerW) * xMax)));
    setFrame(f);
  };

  const onHover = (e) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const xVB = pct * w;
    const f = Math.max(0, Math.min(xMax - 1, Math.round(((xVB - padL) / innerW) * xMax)));
    setHover({ frame: f, pct });
  };

  const fmt = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

  return (
    <div style={{ position: 'relative' }}>
    <div className="card" style={{ borderRadius: 3 }}>
      <svg
        ref={svgRef}
        className="chart-svg"
        viewBox={`0 0 ${w} ${h}`}
        width="100%" height={h}
        preserveAspectRatio="none"
        style={{ cursor: 'col-resize', display: 'block', height: 'clamp(160px, calc(13vw + 55px), 320px)' }}
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
      >
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={padL} y1={yScale(v)} x2={w - padR} y2={yScale(v)} className="grid" />
            <text x={4} y={yScale(v) + 3.5} className="axis-text">{fmt(v)}</text>
          </g>
        ))}
        {yLo < 0 && yHi > 0 && (
          <line x1={padL} y1={yScale(0)} x2={w - padR} y2={yScale(0)} stroke="var(--chart-zero-stroke)" strokeWidth="0.8" />
        )}
        <line x1={padL} y1={padT + innerH} x2={w - padR} y2={padT + innerH} className="axis" />
        {xTicks.map((f, i) => (
          <text key={i} x={xScale(f) - 8} y={h - 4} className="axis-text">
            {f >= 1000 ? `${(f / 1000).toFixed(f % 1000 === 0 ? 0 : 1)}k` : f}
          </text>
        ))}
        {/* Ghost lines */}
        {lines.filter(l => !l.isFocal).map((ln, i) => {
          const cls = ['ghost-1','ghost-2','ghost-3'][i % 3];
          return (
            <g key={ln.runId}>
              <path d={buildPath(ln.values, xScale, yScale, 3)} className={cls} />
              <line
                x1={xScale(ln.values.length - 1)}
                y1={yScale(ln.values[ln.values.length - 1]) - 4}
                x2={xScale(ln.values.length - 1)}
                y2={yScale(ln.values[ln.values.length - 1]) + 4}
                stroke="var(--ink-3)" opacity="0.6"
              />
            </g>
          );
        })}
        {lines.filter(l => l.isFocal).map((ln) => (
          <path key={ln.runId} d={buildPath(ln.values, xScale, yScale, 2)} className="focal" />
        ))}
        {/* Hover crosshair */}
        {hover != null && (
          <line
            x1={xScale(hover.frame)} y1={padT}
            x2={xScale(hover.frame)} y2={padT + innerH}
            stroke="var(--ink-3)" strokeWidth="0.7" strokeDasharray="2 3"
            style={{ pointerEvents: 'none' }}
          />
        )}
        {focalLength > 0 && (
          <g>
            <line x1={xScale(frame)} y1={padT} x2={xScale(frame)} y2={padT + innerH} className="playhead" />
            <circle cx={xScale(frame)} cy={padT + innerH - 6} r="3" fill="var(--accent)" />
          </g>
        )}
      </svg>
    </div>
    {hover != null && (
      <ChartTooltip lines={lines} hoverFrame={hover.frame} hoverX={hover.pct} />
    )}
    </div>
  );
}

// ── Loss strip (3 small charts) ─────────────────────────────────────
function LossStrip({ run, ckpt }) {
  const totalSteps = run.steps;
  const ckptStepFrac = ckpt.step / totalSteps;

// Synthesize a per-training-step loss curve for the focal run
  const losses = useM(() => {
    function gen(seed, opts) {
      const { floor = 0.1, samples = 200 } = opts;
      const r = D.rng(seed);
      const out = new Float32Array(samples);
      for (let i = 0; i < samples; i++) {
        const t = i / (samples - 1);
        const base = floor + (1 - floor) * Math.exp(-2.8 * t);
        out[i] = base + (r() - 0.5) * 0.05;
      }
      return out;
    }
        return {
      policy_loss: gen(parseInt(run.id, 16) * 7, { floor: 0.08, samples: 200 }),
      value_loss:  gen(parseInt(run.id, 16) * 11, { floor: 0.12, samples: 200 }),
      entropy:     gen(parseInt(run.id, 16) * 13, { floor: 0.40, samples: 200 }),
    };
  }, [run]);

  const valAt = (arr) => {
    const idx = Math.floor(ckptStepFrac * (arr.length - 1));
    return arr[idx];
};

  return (
    <div className="col border-t" style={{ flex: '0 0 auto', padding: '0 22px' }}>
      <div className="row" style={{ alignItems: 'stretch' }}>
        <LossChart title="policy_loss" values={losses.policy_loss} atCkptValue={valAt(losses.policy_loss).toFixed(3)} width={300} height={120} ckptStepFrac={ckptStepFrac} />
        <div className="hr-v" />
            <LossChart title="value_loss" values={losses.value_loss} atCkptValue={valAt(losses.value_loss).toFixed(3)} width={300} height={120} ckptStepFrac={ckptStepFrac} />
          <div className="hr-v" />
        <LossChart title="entropy" values={losses.entropy} atCkptValue={valAt(losses.entropy).toFixed(3)} width={300} height={120} ckptStepFrac={ckptStepFrac} />
      </div>
    </div>
  );
}

window.TopBar = TopBar;
window.CkptNav = CkptNav;
window.EpisodePicker = EpisodePicker;
window.FrameChartPair = FrameChartPair;
window.LossStrip = LossStrip;
