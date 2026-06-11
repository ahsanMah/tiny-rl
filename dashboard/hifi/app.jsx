// hifi/app.jsx — top-level app: state + composition + keyboard nav
// localStorage persists session state across reloads (timed content guidance).

const { useState: uS, useEffect: uE, useCallback: uCB, useMemo: uM } = React;

// ── localStorage helpers ─────────────────────────────────────────────
const LS_KEY = 'rl-tracker-session';
function loadSession() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return {};
    return JSON.parse(raw);
  } catch { return {}; }
}
function saveSession(state) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(state));
  } catch {}
}

// ── Responsive breakpoints ───────────────────────────────────────────
// Drives whether the rails render docked (in-flow) or as slide-in drawers.
//   wide   (>= 1080px): both rails docked
//   tablet (760–1079px): left rail docked, right rail is a drawer
//   phone  (< 760px):    both rails are drawers
const BP_TABLET = 760;
const BP_WIDE = 1080;
function modeForWidth(w) {
  if (w < BP_TABLET) return 'phone';
  if (w < BP_WIDE) return 'tablet';
  return 'wide';
}
// Returns the current responsive mode, updating on resize (rAF-throttled).
function useViewport() {
  const [mode, setMode] = uS(() => modeForWidth(window.innerWidth));
  uE(() => {
    let raf = 0;
    const onResize = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        setMode(modeForWidth(window.innerWidth));
      });
    };
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); if (raf) cancelAnimationFrame(raf); };
  }, []);
  return mode;
}

// ── Slide-in drawer ──────────────────────────────────────────────────
// Wraps a rail when it can't be docked. Always mounted so the panel can
// animate both directions;  The dim backdrop only renders while open and dismisses on click.
function Drawer({ side, open, onClose, children }) {
  return (
    <React.Fragment>
      {open && <div className="drawer-backdrop" onClick={onClose} />}
      <div className={`drawer drawer-${side}${open ? ' open' : ''}`} aria-hidden={!open}>
        {children}
      </div>
    </React.Fragment>
  );
}

// Hamburger / panel-toggle glyph for the mobile rail triggers.
function IconMenu() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}


function App() {
  const init = uM(loadSession, []);
  const mode = useViewport();

  // ── Loading state — wait for D.ready before rendering ──────────
  const [loaded, setLoaded] = uS(false);
  uE(() => { D.ready.then(() => setLoaded(true)); }, []);

  // ── State (with localStorage restore) ──────────────────────────
  // Default IDs now point to the real runs. If localStorage has old
  // synthetic IDs from a previous session, they won't match any real
  // run — the focusedRun guard below will fall back to the first run.
  const [focusedId, setFocusedId] = uS(init.focusedId || 'BipedalWalker-v3-sac_1779998141');
  const [pinnedIds, setPinnedIds] = uS(init.pinnedIds || ['BipedalWalker-v3-sac_1779998141', 'BipedalWalker-v3-ppo_2026-05-28_16-14']);
  const [diffBaselineName, setDiffBaselineName] = uS(init.diffBaselineName ?? 'BipedalWalker-v3-ppo_2026-05-28_16-14');
  const [ckptStep, setCkptStep] = uS(init.ckptStep || 0);
  const [episodeKind, setEpisodeKind] = uS(init.episodeKind || 'best');
  const [frame, setFrame] = uS(init.frame || 0);
  const [playing, setPlaying] = uS(false);
  const [speed, setSpeed] = uS(1);
  const [metric, setMetric] = uS(init.metric || 'value');
  const [railWidth, setRailWidth] = uS(init.railWidth || 260);
  const [query, setQuery] = uS('');

  // ── Responsive drawers ─────────────────────────────────────────
  // When a rail can't be docked at the current width it renders as a
  // slide-in drawer. `leftDocked` is true on tablet+wide; phones get a
  // hamburger that opens the runs list as a drawer.
  const leftDocked = mode !== 'phone';
  const [leftDrawerOpen, setLeftDrawerOpen] = uS(false);

  const [darkMode, setDarkMode] = uS(() => {
    try { return localStorage.getItem('rl-dark-mode') !== 'false'; } catch { return true; }
  });

  // Apply theme to document
  uE(() => {
    document.documentElement.classList.toggle('dark', darkMode);

    try { localStorage.setItem('rl-dark-mode', String(darkMode)); } catch {}
  }, [darkMode]);

  const toggleDark = uCB(() => setDarkMode(d => !d), []);

  // ── Derived ────────────────────────────────────────────────────
  const focusedRun = uM(() => D.RUNS.find(r => r.id === focusedId) || D.RUNS[0], [focusedId, loaded]);

  // Snap ckptStep to nearest available checkpoint when run changes
  uE(() => {
    if (!focusedRun) return;
    const valid = focusedRun.checkpoints.find(c => c.step === ckptStep);
    if (!valid) {
      let best = focusedRun.checkpoints[0];
      let bestDist = Math.abs(best.step - ckptStep);
      for (const c of focusedRun.checkpoints) {
        const d = Math.abs(c.step - ckptStep);
        if (d < bestDist) { best = c; bestDist = d; }
      }
      setCkptStep(best.step);
    }
  }, [focusedRun]);

  const ckpt = uM(() => D.getCheckpoint(focusedRun, ckptStep), [focusedRun, ckptStep]);
  const rollout = uM(() => D.getRollout(ckpt, episodeKind), [ckpt, episodeKind]);

  // Clamp frame when rollout changes
  uE(() => {
    if (!rollout) return;
    setFrame(f => Math.min(f, rollout.length - 1));
  }, [rollout]);

  const [signalVersion, setSignalVersion] = uS(0);
  const baselineRun = uM(() => D.RUNS.find(r => r.name === diffBaselineName) || null, [diffBaselineName]);
  const pinnedRuns = uM(() => D.RUNS.filter(r => pinnedIds.includes(r.id)), [pinnedIds]);

  // Load real signals for the focal rollout AND each pinned run's charted rollout
  // (last checkpoint, best episode — the ones FrameChartPair draws as ghosts), then
  // bump signalVersion so the charts redraw. D.loadSignals caches, so re-runs are cheap.
  // signalVersion is passed to FrameChartPair and used as a useMemo dependency.
  uE(() => {
    const rollouts = [];
    if (rollout) rollouts.push(rollout);
    for (const pr of pinnedRuns) {
      const c = pr.checkpoints[pr.checkpoints.length - 1];
      const ro = c && (c.rollouts.find(r => r.kind === 'best') || c.rollouts[0]);
      if (ro) rollouts.push(ro);
    }
    if (rollouts.length === 0) return;
    Promise.all(rollouts.map(r => D.loadSignals(r))).then(() => setSignalVersion(v => v + 1));
  }, [rollout, pinnedRuns]);

  // Actions
  const togglePin = uCB((id) => {
    setPinnedIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  }, []);

  const handleFocus = uCB((id) => {
    setFocusedId(id);
    setPlaying(false);
    setEpisodeKind('best');
    setFrame(0);
    setLeftDrawerOpen(false); // picking a run dismisses the mobile drawer
  }, []);

  // ── Persist on relevant state changes ──────────────────────────
  uE(() => {
    saveSession({ focusedId, pinnedIds, diffBaselineName, ckptStep, episodeKind, frame, metric, railWidth });
  }, [focusedId, pinnedIds, diffBaselineName, ckptStep, episodeKind, frame, metric, railWidth]);

  // ── Keyboard ───────────────────────────────────────────────────
  uE(() => {
    const isEditable = (el) => el && (
      el.tagName === 'INPUT' ||
      el.tagName === 'TEXTAREA' ||
      el.tagName === 'SELECT' ||
      el.isContentEditable
    );

    const onKey = (e) => {
      if (isEditable(document.activeElement)) return;
      switch (e.key) {
        case 'Escape':
          setLeftDrawerOpen(false);
          break;
        case ' ':
          e.preventDefault();
          setPlaying(p => !p);
          break;
        case 'ArrowLeft':
          if (e.shiftKey) setFrame(f => Math.max(0, f - 30));
          else            setFrame(f => Math.max(0, f - 1));
          break;
        case 'ArrowRight':
          if (e.shiftKey) setFrame(f => Math.min((rollout?.length || 1) - 1, f + 30));
          else            setFrame(f => Math.min((rollout?.length || 1) - 1, f + 1));
          break;
        case 'j':
        case 'J': {
          const idx = focusedRun.checkpoints.findIndex(c => c.step === ckptStep);
          if (idx > 0) setCkptStep(focusedRun.checkpoints[idx - 1].step);
          break;
        }
        case 'l':
        case 'L': {
          const idx = focusedRun.checkpoints.findIndex(c => c.step === ckptStep);
          if (idx < focusedRun.checkpoints.length - 1) setCkptStep(focusedRun.checkpoints[idx + 1].step);
          break;
        }
        case '1': setEpisodeKind('best');   setFrame(0); break;
        case '2': setEpisodeKind('median'); setFrame(0); break;
        case '3': setEpisodeKind('worst');  setFrame(0); break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [rollout, focusedRun, ckptStep]);

  if (!loaded || !focusedRun || !ckpt || !rollout) {
    return <div style={{ padding: 40 }}>Loading runs…</div>;
  }

  return (
    <React.Fragment>
      <div style={{ display: 'flex', width: '100%', height: '100%' }}>
        {/* LEFT RAIL — docked on tablet/wide, slide-in drawer on phone */}
        {leftDocked ? (
          <React.Fragment>
            <RailLeft
              runs={D.RUNS}
              focusedId={focusedId}
              pinnedIds={pinnedIds}
              onFocus={handleFocus}
              onTogglePin={togglePin}
              query={query} setQuery={setQuery}
              width={railWidth}
            />

            {/* RAIL RESIZE HANDLE (docked only) */}
            <div
              title="drag to resize"
              onMouseDown={(e) => startDrag(e, (ev) => {
                setRailWidth(Math.max(180, Math.min(480, ev.clientX)));
              }, 'col-resize')}
              style={{ flex: '0 0 auto', width: 5, marginLeft: -2, marginRight: -2,
                       cursor: 'col-resize', zIndex: 5 }}
            />
          </React.Fragment>
        ) : (
          <Drawer side="left" open={leftDrawerOpen} onClose={() => setLeftDrawerOpen(false)}>
            <RailLeft
              runs={D.RUNS}
              focusedId={focusedId}
              pinnedIds={pinnedIds}
              onFocus={handleFocus}
              onTogglePin={togglePin}
              query={query} setQuery={setQuery}
              width={Math.min(300, window.innerWidth - 56)}
            />
          </Drawer>
        )}

        {/* CENTER */}
        <div className="col grow" style={{ minWidth: 0, height: '100%' }}>
          <TopBar
            run={focusedRun}
            ckpt={ckpt}
            pinnedCount={pinnedIds.length}
            diffBaselineName={diffBaselineName}
            onChangeBaseline={setDiffBaselineName}
            allRuns={D.RUNS}
            pinnedRuns={pinnedRuns}
            darkMode={darkMode}
            onToggleDark={toggleDark}
            mode={mode}
            onOpenLeft={() => setLeftDrawerOpen(true)}
          />
          <CkptNav
            run={focusedRun}
            ckpt={ckpt}
            onSelectCkpt={setCkptStep}
          />

          {/* Scrollable body */}
          <div className="scroll col grow" style={{ minHeight: 0 }}>
            <div className="col" style={{ padding: '28px 20px 20px', gap: 20, flex: '0 0 auto' }}>
              <WalkerPlayer
                run={focusedRun} ckpt={ckpt} rollout={rollout}
                frame={frame} setFrame={setFrame}
                playing={playing} setPlaying={setPlaying}
                speed={speed} setSpeed={setSpeed}
              />
              <EpisodePicker
                ckpt={ckpt}
                selected={episodeKind}
                onSelect={(k) => { setEpisodeKind(k); setFrame(0); setPlaying(false); }}
              />
            </div>

            <FrameChartPair
              focalRun={focusedRun}
              focalCkpt={ckpt}
              focalRollout={rollout}
              frame={frame}
              setFrame={setFrame}
              pinnedRuns={pinnedRuns}
              metric={metric} setMetric={setMetric}
              signalVersion={signalVersion}
            />

            <LossStrip run={focusedRun} ckpt={ckpt} />
          </div>{/* end scrollable body */}

        </div>

        {/* RIGHT RAIL */}
        <RailRight
          run={focusedRun}
          ckpt={ckpt}
          rollout={rollout}
          frame={frame}
          baselineRun={baselineRun}
          allRuns={D.RUNS}
        />
      </div>

    </React.Fragment>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
