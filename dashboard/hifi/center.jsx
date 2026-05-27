// hifi/center.jsx — center column composition

const { useMemo: useM, useState: useSt } = React;

// ── Top breadcrumb / actions bar ─────────────────────────────────────
function TopBar({ run, ckpt, pinnedCount, diffBaselineName, onChangeBaseline, allRuns }) {
  return (
    <div className="row border-b" style={{ padding: '8px 16px', gap: 10, height: 44, flex: '0 0 auto', minWidth: 0 }}>
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
      <span className="tag" style={{ flex: '0 0 auto' }}>⊕ {pinnedCount} pinned</span>
      <select
        className="dropdown"
        value={diffBaselineName || ''}
        onChange={(e) => onChangeBaseline(e.target.value || null)}
        title="hyperparam diff baseline"
        style={{ font: '500 11.5px var(--ui)', flex: '0 0 auto', maxWidth: 200 }}
      >
        <option value="">no diff baseline</option>
        {allRuns.map(r => (
          <option key={r.id} value={r.name}>diff vs {r.name}</option>
        ))}
      </select>
      <button className="btn icon" title="share" style={{ flex: '0 0 auto' }}>↗</button>
      <button className="btn solid" style={{ flex: '0 0 auto' }}>＋ new run</button>
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
    <div className="row border-b" style={{ padding: '10px 16px', gap: 12, flex: '0 0 auto', background: 'var(--paper-warm)' }}>
      <span className="label-eyebrow">Checkpoint</span>
      <span className="num strong" style={{ fontSize: 15, fontFamily: 'var(--mono)' }}>{D.fmtStep(ckpt.step)}</span>
      <span className="muted" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>step {idx + 1} of {total}</span>
      <div className="row gap-1">
        <button className="btn icon" onClick={prev} disabled={idx === 0} title="previous (J)">◀</button>
        <button className="btn icon" onClick={next} disabled={idx === total - 1} title="next (L)">▶</button>
      </div>
      <div style={{ marginLeft: 6 }}>
        <CheckpointSparkbar
          checkpoints={run.checkpoints}
          activeStep={ckpt.step}
          onSelect={onSelectCkpt}
          width={360} height={32}
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
      <span className="muted" style={{ fontSize: 11, alignSelf: 'center', paddingLeft: 4 }}>
        {ckpt.rollouts.length} rollouts at this ckpt
      </span>
    </div>
  );
}

// ── Frame-level chart pair (cumulative + metric) ─────────────────────
const METRIC_OPTIONS = [
  { key: 'value',       label: 'V̂(s_t)' },
  { key: 'step_reward', label: 'step_reward' },
  { key: 'action_logp', label: 'action_logp' },
  { key: 'advantage',   label: 'advantage' },
  { key: 'td_error',    label: 'TD-error' },
  { key: 'entropy',     label: 'entropy' },
];

function FrameChartPair({ focalRun, focalCkpt, focalRollout, frame, setFrame, pinnedRuns, metric, setMetric }) {
  // Build "lines" for cumulative_return chart
  const cumLines = useM(() => {
    const out = [];
    // Focal
    out.push({
      runId: focalRun.id,
      values: D.frameSignal(focalRun, focalCkpt, focalRollout, 'cumulative_return'),
      isFocal: true,
      color: 'var(--ink)',
    });
    // Each pinned run contributes its best rollout @ its latest ckpt
    for (const pr of pinnedRuns) {
      if (pr.id === focalRun.id) continue;
      const c = pr.checkpoints[pr.checkpoints.length - 1];
      const ro = c.rollouts.find(r => r.kind === 'best') || c.rollouts[0];
      out.push({
        runId: pr.id,
        values: D.frameSignal(pr, c, ro, 'cumulative_return'),
        isFocal: false,
        color: '',
      });
    }
    return out;
  }, [focalRun, focalCkpt, focalRollout, pinnedRuns]);

  // Build lines for metric chart
  const metricLines = useM(() => {
    const out = [];
    out.push({
      runId: focalRun.id,
      values: D.frameSignal(focalRun, focalCkpt, focalRollout, metric),
      isFocal: true,
      color: 'var(--ink)',
    });
    for (const pr of pinnedRuns) {
      if (pr.id === focalRun.id) continue;
      const c = pr.checkpoints[pr.checkpoints.length - 1];
      const ro = c.rollouts.find(r => r.kind === 'best') || c.rollouts[0];
      out.push({
        runId: pr.id,
        values: D.frameSignal(pr, c, ro, metric),
        isFocal: false,
        color: '',
      });
    }
    return out;
  }, [focalRun, focalCkpt, focalRollout, pinnedRuns, metric]);

  const cumAtCursor   = cumLines[0]?.values?.[Math.min(frame, cumLines[0].values.length - 1)];
  const metricAtCursor = metricLines[0]?.values?.[Math.min(frame, metricLines[0].values.length - 1)];

  const metricLabel = METRIC_OPTIONS.find(m => m.key === metric)?.label || metric;
  const ghostsCount = cumLines.length - 1;

  return (
    <div className="row gap-3" style={{ padding: '10px 16px 6px' }}>
      <FrameLevelChart
        title="cumulative_return"
        label={ghostsCount > 0 ? `─ focal · ┄ ${ghostsCount} ghost${ghostsCount > 1 ? 's' : ''}` : '─ focal'}
        lines={cumLines}
        frame={frame}
        focalLength={focalRollout.length}
        setFrame={setFrame}
        height={148}
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
            {METRIC_OPTIONS.map(o => <option key={o.key} value={o.key}>{o.label}</option>)}
          </select>
        </div>
        {/* Reuse FrameLevelChart minus its own header (we built the header above) */}
        <FrameLevelChartBare
          lines={metricLines}
          frame={frame}
          focalLength={focalRollout.length}
          setFrame={setFrame}
          height={148}
        />
      </div>
    </div>
  );
}

// Bare version (no header) — used when the parent renders its own title row
function FrameLevelChartBare({ lines, frame, focalLength, setFrame, height = 148 }) {
  const svgRef = React.useRef(null);
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

  const fmt = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

  return (
    <div className="card" style={{ borderRadius: 3 }}>
      <svg
        ref={svgRef}
        className="chart-svg"
        viewBox={`0 0 ${w} ${h}`}
        width="100%" height={h}
        preserveAspectRatio="none"
        style={{ cursor: 'col-resize', display: 'block' }}
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
      >
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={padL} y1={yScale(v)} x2={w - padR} y2={yScale(v)} className="grid" />
            <text x={4} y={yScale(v) + 3.5} className="axis-text">{fmt(v)}</text>
          </g>
        ))}
        {yLo < 0 && yHi > 0 && (
          <line x1={padL} y1={yScale(0)} x2={w - padR} y2={yScale(0)} stroke="rgba(28,24,19,.25)" />
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
        {focalLength > 0 && (
          <g>
            <line x1={xScale(frame)} y1={padT} x2={xScale(frame)} y2={padT + innerH} className="playhead" />
            <circle cx={xScale(frame)} cy={padT + innerH - 6} r="3" fill="var(--accent)" />
          </g>
        )}
      </svg>
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
    <div className="col border-t" style={{ flex: '0 0 auto' }}>
      <div className="row" style={{ alignItems: 'stretch' }}>
        <LossChart title="policy_loss" values={losses.policy_loss} atCkptValue={valAt(losses.policy_loss).toFixed(3)} width={300} height={78} ckptStepFrac={ckptStepFrac} />
        <div className="hr-v" />
        <LossChart title="value_loss" values={losses.value_loss} atCkptValue={valAt(losses.value_loss).toFixed(3)} width={300} height={78} ckptStepFrac={ckptStepFrac} />
        <div className="hr-v" />
        <LossChart title="entropy" values={losses.entropy} atCkptValue={valAt(losses.entropy).toFixed(3)} width={300} height={78} ckptStepFrac={ckptStepFrac} />
      </div>
      <div className="row border-t" style={{ padding: '4px 14px', fontSize: 10.5, color: 'var(--ink-3)' }}>
        <span className="italic" style={{ fontFamily: 'var(--display)' }}>↳ continuous train metrics — vertical mark at active ckpt step</span>
        <span className="grow" />
        <a href="#" style={{ color: 'var(--ink-3)', textDecoration: 'underline' }}>open in Tensorboard ↗</a>
      </div>
    </div>
  );
}

window.TopBar = TopBar;
window.CkptNav = CkptNav;
window.EpisodePicker = EpisodePicker;
window.FrameChartPair = FrameChartPair;
window.LossStrip = LossStrip;
