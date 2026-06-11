// hifi/center.jsx — center column composition

const { useMemo: useM, useState: useSt } = React;

// ── Icon primitives (inline SVG — no library needed) ─────────────────
function IconSun({ size = 15, strokeWidth = 1.8 }) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <circle cx="12" cy="12" r="4" />
            <line x1="12" y1="2" x2="12" y2="6" />
            <line x1="12" y1="18" x2="12" y2="22" />
            <line x1="2" y1="12" x2="6" y2="12" />
            <line x1="18" y1="12" x2="22" y2="12" />
            <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
            <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
            <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
            <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
        </svg>
    );
}
function IconMoon({ size = 15, strokeWidth = 1.8 }) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
    );
}
// ── Top breadcrumb / actions bar ─────────────────────────────────────
function TopBar({
    run,
    ckpt,
    pinnedCount,
    diffBaselineName,
    onChangeBaseline,
    allRuns,
    pinnedRuns,
    darkMode,
    onToggleDark,
    mode,
    onOpenLeft,
    rightDocked,
    onOpenRight,
}) {
    const isPhone = mode === "phone";
    return (
        <div
            className="row border-b"
            style={{
                padding: isPhone ? "10px 12px" : "12px 22px",
                gap: isPhone ? 8 : 12,
                height: 80,
                flex: "0 0 auto",
                minWidth: 0,
            }}
        >
            {mode === "phone" && (
                <button
                    className="btn icon"
                    onClick={onOpenLeft}
                    title="show runs"
                    aria-label="show runs"
                    style={{ flex: "0 0 auto", width: 34, height: 34 }}
                >
                    <IconMenu />
                </button>
            )}
            <div
                className="doc-title"
                style={{ flex: "1 1 auto", minWidth: 0 }}
            >
                <span className="crumb">
                    {/* On phones the prefix segments are dropped to leave room
                        for the run name — the essential identifier. */}
                    {!isPhone && (
                        <React.Fragment>
                            <span style={{ color: "var(--ink)" }} className="strong">
                                tracker
                            </span>
                            <span className="sep">/</span>
                            <span>{run.env.split("-")[0]}</span>
                            <span className="sep">/</span>
                        </React.Fragment>
                    )}
                    <span className="current">{run.name}</span>
                    <span className="sep">@</span>
                    <span className="num" style={{ color: "var(--ink-2)" }}>
                        {D.fmtStep(ckpt.step)}
                    </span>
                </span>
            </div>
            <select
                className="dropdown"
                value={diffBaselineName || ""}
                onChange={(e) => onChangeBaseline(e.target.value || null)}
                title="hyperparam diff baseline"
                style={{
                    font: "500 var(--t-sm) var(--ui)",
                    flex: "0 1 auto",
                    minWidth: 0,
                    maxWidth: isPhone ? 116 : 180,
                    padding: "6px 10px",
                    height: 34,
                }}
            >
                <option value="">no baseline</option>
                {pinnedRuns
                    .filter((r) => r.id !== run.id)
                    .map((r) => (
                        <option key={r.id} value={r.name}>
                            vs {r.name}
                        </option>
                    ))}
            </select>
            <button
                className="btn icon"
                onClick={onToggleDark}
                title={
                    darkMode ? "switch to light mode" : "switch to dark mode"
                }
                style={{ flex: "0 0 auto", width: 34, height: 34 }}
            >
                {darkMode ? <IconSun /> : <IconMoon />}
            </button>
            {!rightDocked && (
                <button
                    className="btn icon"
                    onClick={onOpenRight}
                    title="show details"
                    aria-label="show run details"
                    style={{ flex: "0 0 auto", width: 34, height: 34 }}
                >
                    <IconInfo />
                </button>
            )}
        </div>
    );
}

// ── Checkpoint navigator row (sparkbar + arrows + stats) ─────────────
function CkptNav({ run, ckpt, onSelectCkpt, mode }) {
    const idx = run.checkpoints.findIndex((c) => c.step === ckpt.step);
    const total = run.checkpoints.length;
    const prev = () => idx > 0 && onSelectCkpt(run.checkpoints[idx - 1].step);
    const next = () =>
        idx < total - 1 && onSelectCkpt(run.checkpoints[idx + 1].step);
    const f2 = (v) => (v == null || isNaN(v) ? "–" : Number(v).toFixed(2));
    const isPhone = mode === "phone";
    return (
        <div
            className="row border-b"
            style={{
                padding: isPhone ? "10px 12px" : "11px 22px",
                gap: isPhone ? 10 : 14,
                rowGap: isPhone ? 10 : undefined,
                flexWrap: isPhone ? "wrap" : "nowrap",
                flex: "0 0 auto",
                background: "var(--surface)",
            }}
        >
            <span className="label-eyebrow">Checkpoint</span>
            <span
                className="num strong"
                style={{
                    fontSize: 15,
                    fontFamily: "var(--mono)",
                    minWidth: "4.5ch",
                    display: "inline-block",
                }}
            >
                {D.fmtStep(ckpt.step)}
            </span>
            {!isPhone && (
                <span
                    className="muted"
                    style={{ fontSize: 11, whiteSpace: "nowrap" }}
                >
                    step {idx + 1} of {total}
                </span>
            )}
            <div className="row gap-1">
                <button
                    className="btn icon"
                    onClick={prev}
                    disabled={idx === 0}
                    title="previous (J)"
                >
                    ◀
                </button>
                <button
                    className="btn icon"
                    onClick={next}
                    disabled={idx === total - 1}
                    title="next (L)"
                >
                    ▶
                </button>
            </div>
            <div
                style={{
                    // On phones the sparkbar takes its own full-width row so it
                    // stays scrubbable; the stats wrap onto a line below it.
                    flex: isPhone ? "1 1 100%" : 1,
                    minWidth: 0,
                    marginLeft: isPhone ? 0 : 6,
                    order: isPhone ? 1 : 0,
                }}
            >
                <CheckpointSparkbar
                    checkpoints={run.checkpoints}
                    activeStep={ckpt.step}
                    onSelect={onSelectCkpt}
                    height={32}
                />
            </div>
            {!isPhone && <span className="grow" />}
            <div
                className="row gap-3"
                style={{ alignItems: "baseline", order: isPhone ? 2 : 0 }}
            >
                <span className="col" style={{ alignItems: "flex-end" }}>
                    <span className="label-eyebrow">μ ± σ</span>
                    <span className="num strong" style={{ fontSize: 13 }}>
                        {f2(ckpt.mean)}{" "}
                        <span className="muted" style={{ fontSize: 11 }}>
                            ± {f2(ckpt.std)}
                        </span>
                    </span>
                </span>
                <span className="col" style={{ alignItems: "flex-end" }}>
                    <span className="label-eyebrow">best</span>
                    <span className="num strong" style={{ fontSize: 13 }}>
                        {f2(ckpt.best)}
                    </span>
                </span>
                <span className="col" style={{ alignItems: "flex-end" }}>
                    <span className="label-eyebrow">worst</span>
                    <span className="num strong" style={{ fontSize: 13 }}>
                        {f2(ckpt.worst)}
                    </span>
                </span>
            </div>
        </div>
    );
}

// ── Episode picker buttons (no thumbnails) ──────────────────────────
function EpisodePicker({ ckpt, selected, onSelect, mode }) {
    const rollouts = ckpt.rollouts;
    // Buttons: best, median, worst (the 3 the user explicitly asked for)
    const visible = ["best", "median", "worst"]
        .map((k) => rollouts.find((r) => r.kind === k))
        .filter(Boolean);
    const iconMap = { best: "★", median: "◇", worst: "✕" };
    const labelMap = { best: "best episode", median: "median", worst: "worst" };
    const isPhone = mode === "phone";
    return (
        <div
            className="gap-2"
            style={{ display: "flex", flexDirection: isPhone ? "column" : "row" }}
        >
            {visible.map((r) => (
                <button
                    key={r.kind}
                    className={
                        "ep-btn" + (selected === r.kind ? " active" : "")
                    }
                    onClick={() => onSelect(r.kind)}
                >
                    <span
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                        }}
                    >
                        <span className="ep-icon">{iconMap[r.kind]}</span>
                        <span>{labelMap[r.kind]}</span>
                    </span>
                    <span className="ep-meta">
                        {r.length.toLocaleString()}f · r {Number(r.return).toFixed(2)}
                    </span>
                </button>
            ))}
        </div>
    );
}

// ── Frame-level chart pair (cumulative + metric) ─────────────────────
// Pretty labels for known signal keys. Any signal present in a run's
// signals.npz but not listed here falls back to its raw name.
const SIGNAL_LABELS = {
    value: "V(s_t)",
    value_estimate: "V(s_t)",
    step_reward: "step_reward",
    action_logp: "action_logp",
    advantage: "advantage",
    td_error: "TD-error",
    entropy: "entropy",
};
const signalLabel = (key) => SIGNAL_LABELS[key] || key;
// Responsive height shared by the frame charts and their empty states.
const CHART_FILL = "clamp(160px, calc(13vw + 55px), 320px)";
const fmtCursor = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2));

function FrameChartPair({
    focalRun,
    focalCkpt,
    focalRollout,
    frame,
    setFrame,
    pinnedRuns,
    metric,
    setMetric,
    signalVersion,
    mode,
}) {
    // ── Line builder — one focal line + a ghost per pinned run. Lines whose real
    // signal is missing are omitted (we never fabricate); the ghost slot/color
    // still advances per pinned run so colors stay aligned with the left rail. ──
    const buildLines = (sig) => {
        const out = [];
        const focalVals = D.frameSignal(focalRun, focalCkpt, focalRollout, sig);
        if (focalVals && focalVals.length) {
            out.push({
                runId: focalRun.id,
                name: focalRun.name,
                values: focalVals,
                isFocal: true,
                strokeColor: RUN_LINE_STYLES[0].color,
                dash: null,
                color: "var(--ink)",
            });
        }
        let gi = 0;
        for (const pr of pinnedRuns) {
            if (pr.id === focalRun.id) continue;
            const ls =
                RUN_LINE_STYLES[gi + 1] ||
                RUN_LINE_STYLES[RUN_LINE_STYLES.length - 1];
            gi++;
            const c = pr.checkpoints[pr.checkpoints.length - 1];
            const ro =
                c &&
                (c.rollouts.find((r) => r.kind === "best") || c.rollouts[0]);
            const vals = ro ? D.frameSignal(pr, c, ro, sig) : null;
            if (!vals || !vals.length) continue;
            out.push({
                runId: pr.id,
                name: pr.name,
                values: vals,
                isFocal: false,
                strokeColor: ls.color,
                dash: ls.dash,
                color: "",
            });
        }
        return out;
    };

    // Dropdown reflects what's actually in this run's signals.npz (capabilities.signals),
    // minus cumulative_return which has its own dedicated chart on the left.
    const metricOptions = useM(
        () =>
            (focalRun.capabilities?.signals || []).filter(
                (k) => k !== "cumulative_return",
            ),
        [focalRun],
    );
    // The globally-selected metric may not exist for this run — fall back to the first.
    const activeMetric = metricOptions.includes(metric)
        ? metric
        : metricOptions[0];

    const cumLines = useM(
        () => buildLines("cumulative_return"),
        [focalRun, focalCkpt, focalRollout, pinnedRuns, signalVersion],
    );
    const metricLines = useM(
        () => buildLines(activeMetric),
        [
            focalRun,
            focalCkpt,
            focalRollout,
            pinnedRuns,
            activeMetric,
            signalVersion,
        ],
    );

    const cumFocal = cumLines.find((l) => l.isFocal);
    const metricFocal = metricLines.find((l) => l.isFocal);
    const cumAtCursor =
        cumFocal &&
        cumFocal.values[Math.min(frame, cumFocal.values.length - 1)];
    const metricAtCursor =
        metricFocal &&
        metricFocal.values[Math.min(frame, metricFocal.values.length - 1)];

    const metricLabel = activeMetric
        ? signalLabel(activeMetric)
        : "per-frame signal";

    const isPhone = mode === "phone";
    return (
        <div
            className="gap-3"
            style={{
                display: "flex",
                flexDirection: isPhone ? "column" : "row",
                padding: isPhone ? "20px 14px 16px" : "20px 22px 16px",
            }}
        >
            {/* cumulative_return (left) — its own dedicated chart */}
            <div className="col" style={{ minWidth: 0, flex: 1 }}>
                <div
                    className="row"
                    style={{
                        alignItems: "baseline",
                        gap: 8,
                        marginBottom: 4,
                        height: 22,
                    }}
                >
                    <span
                        className="display"
                        style={{ fontSize: 13, fontWeight: 600 }}
                    >
                        cumulative_return
                    </span>
                    <span className="muted" style={{ fontSize: 11 }}>
                        {cumLines.length} run{cumLines.length === 1 ? "" : "s"}
                    </span>
                    <span className="grow" />
                    {cumAtCursor != null && (
                        <span
                            className="num"
                            style={{ fontSize: 11.5, color: "var(--accent)" }}
                        >
                            @{frame}: {fmtCursor(cumAtCursor)}
                        </span>
                    )}
                </div>
                <FrameLevelChartBare
                    lines={cumLines}
                    frame={frame}
                    focalLength={focalRollout.length}
                    setFrame={setFrame}
                    height={160}
                    emptyTitle="cumulative_return unavailable"
                    emptyDetail="This rollout has no step-level signals (signals.npz absent)."
                />
            </div>

            {/* selectable per-frame metric (right) */}
            <div className="col" style={{ flex: 1, minWidth: 0 }}>
                <div
                    className="row"
                    style={{
                        alignItems: "baseline",
                        gap: 8,
                        marginBottom: 4,
                        height: 22,
                    }}
                >
                    <span
                        className="display"
                        style={{ fontSize: 13, fontWeight: 600 }}
                    >
                        {metricLabel}
                    </span>
                    <span className="muted" style={{ fontSize: 11 }}>
                        per-frame
                    </span>
                    <span className="grow" />
                    {metricAtCursor != null && (
                        <span
                            className="num"
                            style={{ fontSize: 11.5, color: "var(--accent)" }}
                        >
                            @{frame}: {fmtCursor(metricAtCursor)}
                        </span>
                    )}
                    {metricOptions.length > 0 && (
                        <select
                            className="dropdown"
                            value={activeMetric}
                            onChange={(e) => setMetric(e.target.value)}
                        >
                            {metricOptions.map((k) => (
                                <option key={k} value={k}>
                                    {signalLabel(k)}
                                </option>
                            ))}
                        </select>
                    )}
                </div>
                <FrameLevelChartBare
                    lines={metricLines}
                    frame={frame}
                    focalLength={focalRollout.length}
                    setFrame={setFrame}
                    height={160}
                    emptyTitle={
                        activeMetric
                            ? `${signalLabel(activeMetric)} unavailable`
                            : "no per-frame signals"
                    }
                    emptyDetail={
                        activeMetric
                            ? "This rollout has no signals.npz, or it omits this signal."
                            : "This run logged no per-frame signals (signals.npz absent)."
                    }
                />
            </div>
        </div>
    );
}

// Bare version (no header) — used when the parent renders its own title row.
// Renders a missing-data placeholder when there's no focal signal to draw,
// rather than fabricating a curve.
function FrameLevelChartBare({
    lines,
    frame,
    focalLength,
    setFrame,
    height = 160,
    emptyTitle = "no signal data",
    emptyDetail,
}) {
    const svgRef = React.useRef(null);
    const [hover, setHover] = useSt(null); // { frame: int, pct: float 0-1 }

    // Measure the real rendered size so the SVG draws at 1:1 (1 unit = 1px),
    // avoiding the preserveAspectRatio="none" stretch that distorted text/strokes.
    const [boxRef, size] = useMeasuredSize({ width: 480, height });
    const clipId = useM(() => nextClipId(), []);

    const hasFocal = lines.some(
        (l) => l.isFocal && l.values && l.values.length,
    );
    const w = size.width;
    const h = size.height;
    const padL = 32,
        padR = 8,
        padT = 8,
        padB = 18;
    const innerW = w - padL - padR;
    const innerH = h - padT - padB;

    const xMax = useM(
        () => Math.max(...lines.map((l) => l.values.length), focalLength || 0),
        [lines, focalLength],
    );
    // Full scan — sampling missed extrema that the drawn path still rendered,
    // spilling the line out of the plot box.
    const [yLo, yHi] = useM(() => {
        let lo = Infinity,
            hi = -Infinity;
        for (const ln of lines)
            for (let i = 0; i < ln.values.length; i++) {
                if (ln.values[i] < lo) lo = ln.values[i];
                if (ln.values[i] > hi) hi = ln.values[i];
            }
        if (lo === Infinity) {
            lo = 0;
            hi = 1;
        }
        const pad = (hi - lo) * 0.08;
        return [lo - pad, hi + pad];
    }, [lines]);

    // No focal signal for this rollout/metric — show a missing-data placeholder
    // (after all hooks, to satisfy the Rules of Hooks).
    if (!hasFocal) {
        return (
            <EmptyState
                minHeight={CHART_FILL}
                title={emptyTitle}
                detail={emptyDetail}
            />
        );
    }

    const xScale = (f) => padL + (f / xMax) * innerW;
    const yScale = (v) => padT + (1 - (v - yLo) / (yHi - yLo)) * innerH;

    const yTicks = [yHi, (yLo + yHi) / 2, yLo];
    const step = xMax > 1500 ? 400 : xMax > 500 ? 200 : 100;
    const xTicks = [0];
    for (let i = step; i < xMax; i += step) xTicks.push(i);
    xTicks.push(xMax);

    const onPointer = (e) => {
        if (!svgRef.current || !setFrame) return;
        const rect = svgRef.current.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * w;
        const f = Math.max(
            0,
            Math.min(focalLength - 1, Math.round(((x - padL) / innerW) * xMax)),
        );
        setFrame(f);
    };

    const onHover = (e) => {
        if (!svgRef.current) return;
        const rect = svgRef.current.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        const xVB = pct * w;
        const f = Math.max(
            0,
            Math.min(xMax - 1, Math.round(((xVB - padL) / innerW) * xMax)),
        );
        setHover({ frame: f, pct });
    };

    const fmt = (v) =>
        Math.abs(v) >= 100
            ? v.toFixed(0)
            : Math.abs(v) >= 10
              ? v.toFixed(1)
              : v.toFixed(2);

    return (
        <div style={{ position: "relative" }}>
            <div
                ref={boxRef}
                className="card"
                style={{ borderRadius: 3, height: CHART_FILL }}
            >
                <svg
                    ref={svgRef}
                    className="chart-svg"
                    viewBox={`0 0 ${w} ${h}`}
                    width="100%"
                    height="100%"
                    style={{ cursor: "col-resize", display: "block" }}
                    onMouseDown={(e) => startDrag(e, onPointer, "col-resize")}
                    onMouseMove={onHover}
                    onMouseLeave={() => setHover(null)}
                >
                    <defs>
                        <clipPath id={clipId}>
                            <rect
                                x={padL}
                                y={padT}
                                width={innerW}
                                height={innerH}
                            />
                        </clipPath>
                    </defs>
                    {yTicks.map((v, i) => (
                        <g key={i}>
                            <line
                                x1={padL}
                                y1={yScale(v)}
                                x2={w - padR}
                                y2={yScale(v)}
                                className="grid"
                            />
                            <text
                                x={4}
                                y={yScale(v) + 3.5}
                                className="axis-text"
                            >
                                {fmt(v)}
                            </text>
                        </g>
                    ))}
                    {yLo < 0 && yHi > 0 && (
                        <line
                            x1={padL}
                            y1={yScale(0)}
                            x2={w - padR}
                            y2={yScale(0)}
                            stroke="var(--chart-zero-stroke)"
                            strokeWidth="0.8"
                        />
                    )}
                    <line
                        x1={padL}
                        y1={padT + innerH}
                        x2={w - padR}
                        y2={padT + innerH}
                        className="axis"
                    />
                    {xTicks.map((f, i) => (
                        <text
                            key={i}
                            x={xScale(f) - 8}
                            y={h - 4}
                            className="axis-text"
                        >
                            {f >= 1000
                                ? `${(f / 1000).toFixed(f % 1000 === 0 ? 0 : 1)}k`
                                : f}
                        </text>
                    ))}
                    <g clipPath={`url(#${clipId})`}>
                        {/* Ghost lines */}
                        {lines
                            .filter((l) => !l.isFocal)
                            .map((ln, i) => {
                                const cls = ["ghost-1", "ghost-2", "ghost-3"][
                                    i % 3
                                ];
                                return (
                                    <g key={ln.runId}>
                                        <path
                                            d={buildPath(
                                                ln.values,
                                                xScale,
                                                yScale,
                                                3,
                                            )}
                                            className={cls}
                                        />
                                        <line
                                            x1={xScale(ln.values.length - 1)}
                                            y1={
                                                yScale(
                                                    ln.values[
                                                        ln.values.length - 1
                                                    ],
                                                ) - 4
                                            }
                                            x2={xScale(ln.values.length - 1)}
                                            y2={
                                                yScale(
                                                    ln.values[
                                                        ln.values.length - 1
                                                    ],
                                                ) + 4
                                            }
                                            stroke="var(--ink-3)"
                                            opacity="0.6"
                                        />
                                    </g>
                                );
                            })}
                        {lines
                            .filter((l) => l.isFocal)
                            .map((ln) => (
                                <path
                                    key={ln.runId}
                                    d={buildPath(ln.values, xScale, yScale, 2)}
                                    className="focal"
                                />
                            ))}
                    </g>
                    {/* Hover crosshair */}
                    {hover != null && (
                        <line
                            x1={xScale(hover.frame)}
                            y1={padT}
                            x2={xScale(hover.frame)}
                            y2={padT + innerH}
                            stroke="var(--ink-3)"
                            strokeWidth="0.7"
                            strokeDasharray="2 3"
                            style={{ pointerEvents: "none" }}
                        />
                    )}
                    {focalLength > 0 && (
                        <g>
                            <line
                                x1={xScale(frame)}
                                y1={padT}
                                x2={xScale(frame)}
                                y2={padT + innerH}
                                className="playhead"
                            />
                            <circle
                                cx={xScale(frame)}
                                cy={padT + innerH - 6}
                                r="3"
                                fill="var(--accent)"
                            />
                        </g>
                    )}
                </svg>
            </div>
            {hover != null && (
                <ChartTooltip
                    lines={lines}
                    hoverFrame={hover.frame}
                    hoverX={hover.pct}
                />
            )}
        </div>
    );
}

// ── Loss strip (3 small charts) ─────────────────────────────────────
function LossStrip({ run, ckpt }) {
    const totalSteps = run.steps;
    const ckptStepFrac = totalSteps > 0 ? ckpt.step / totalSteps : 1;

    const losses = useM(() => {
        const tm = run.trainMetrics;
        if (!tm || Object.keys(tm).length === 0) return {};

        // Skip non-loss metrics; prefer the most informative ones
        const skip = new Set(["sps", "epoch", "wall_time_s"]);
        const preferred = [
            "policy_loss",
            "value_loss",
            "entropy",
            "q1_loss",
            "q2_loss",
        ];
        const available = Object.keys(tm).filter((k) => !skip.has(k));
        const ordered = [
            ...preferred.filter((k) => available.includes(k)),
            ...available.filter((k) => !preferred.includes(k)),
        ].slice(0, 3);

        const result = {};
        for (const key of ordered) {
            const entries = tm[key];
            const arr = new Float32Array(entries.length);
            entries.forEach((e, i) => {
                arr[i] = e.value;
            });
            result[key] = arr;
        }
        return result;
    }, [run]);

    const metricKeys = Object.keys(losses);
    const valAt = (arr) =>
        arr[
            Math.min(
                Math.floor(ckptStepFrac * (arr.length - 1)),
                arr.length - 1,
            )
        ];

    return (
        <div
            className="col border-t"
            style={{
                flex: "0 0 auto",
                padding: metricKeys.length ? "0 22px" : "16px 22px",
            }}
        >
            {metricKeys.length === 0 ? (
                <EmptyState
                    title="no training metrics"
                    detail="This run has no train_metrics.jsonl to chart."
                />
            ) : (
                <div className="row" style={{ alignItems: "stretch" }}>
                    {metricKeys.map((key, i) => (
                        <React.Fragment key={key}>
                            {i > 0 && <div className="hr-v" />}
                            <LossChart
                                title={key}
                                values={losses[key]}
                                atCkptValue={valAt(losses[key]).toFixed(3)}
                                width={300}
                                height={120}
                                ckptStepFrac={ckptStepFrac}
                            />
                        </React.Fragment>
                    ))}
                </div>
            )}
        </div>
    );
}

window.TopBar = TopBar;
window.CkptNav = CkptNav;
window.EpisodePicker = EpisodePicker;
window.FrameChartPair = FrameChartPair;
window.LossStrip = LossStrip;
