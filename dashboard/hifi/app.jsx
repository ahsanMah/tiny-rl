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


function App() {
  const init = uM(loadSession, []);

  // ── State (with localStorage restore) ──────────────────────────
  const [focusedId, setFocusedId] = uS(init.focusedId || 'a4ee');
  const [pinnedIds, setPinnedIds] = uS(init.pinnedIds || ['a4f2', 'a4ee', 'a4ec']);
  const [diffBaselineName, setDiffBaselineName] = uS(init.diffBaselineName ?? 'ppo-walker-v35');
  const [ckptStep, setCkptStep] = uS(init.ckptStep || 3500000);
  const [episodeKind, setEpisodeKind] = uS(init.episodeKind || 'best');
  const [frame, setFrame] = uS(init.frame || 0);
  const [playing, setPlaying] = uS(false);
  const [speed, setSpeed] = uS(1);
  const [overlay, setOverlay] = uS(TWEAK_DEFAULTS.playerOverlay);
  const [metric, setMetric] = uS(init.metric || 'value');
  const [query, setQuery] = uS('');
  const [darkMode, setDarkMode] = uS(() => {
    try { return localStorage.getItem('rl-dark-mode') !== 'false'; } catch { return true; }
  });

  // Apply theme to document
  uE(() => {
    document.documentElement.classList.toggle('light-medium-contrast', !darkMode);

    try { localStorage.setItem('rl-dark-mode', String(darkMode)); } catch {}
  }, [darkMode]);

  const toggleDark = uCB(() => setDarkMode(d => !d), []);

  // ── Derived ────────────────────────────────────────────────────
  const focusedRun = uM(() => D.RUNS.find(r => r.id === focusedId), [focusedId]);

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

  const baselineRun = uM(() => D.RUNS.find(r => r.name === diffBaselineName) || null, [diffBaselineName]);
  const pinnedRuns = uM(() => D.RUNS.filter(r => pinnedIds.includes(r.id)), [pinnedIds]);

  // Actions
  const togglePin = uCB((id) => {
    setPinnedIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  }, []);

  const handleFocus = uCB((id) => {
    setFocusedId(id);
    setPlaying(false);
    setEpisodeKind('best');
    setFrame(0);
  }, []);

  // ── Persist on relevant state changes ──────────────────────────
  uE(() => {
    saveSession({ focusedId, pinnedIds, diffBaselineName, ckptStep, episodeKind, frame, metric });
  }, [focusedId, pinnedIds, diffBaselineName, ckptStep, episodeKind, frame, metric]);

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

  if (!focusedRun || !ckpt || !rollout) {
    return <div style={{ padding: 40 }}>Loading…</div>;
  }

  return (
    <React.Fragment>
      <div style={{ display: 'flex', width: '100%', height: '100%' }}>
        {/* LEFT RAIL */}
        <RailLeft
          runs={D.RUNS}
          focusedId={focusedId}
          pinnedIds={pinnedIds}
          onFocus={handleFocus}
          onTogglePin={togglePin}
          query={query} setQuery={setQuery}
        />

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
                overlay={overlay} setOverlay={setOverlay}
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
