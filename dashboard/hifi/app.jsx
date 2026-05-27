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

// ── Tweak defaults — EDITMODE block lets the host persist tweaks ─────
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#b85a3e",
  "density": "regular",
  "paperTone": "warm",
  "playerOverlay": "none"
}/*EDITMODE-END*/;

// Theme inject — translate tweaks to CSS variable overrides
function ThemeInject({ tweaks }) {
  const css = `
    :root {
      --accent: ${tweaks.accent};
      ${tweaks.paperTone === 'cool' ? `
        --paper:      #ecf0eb;
        --paper-warm: #f2f6f0;
        --paper-cool: #e2e7e0;
      ` : tweaks.paperTone === 'neutral' ? `
        --paper:      #efece7;
        --paper-warm: #f5f3ee;
        --paper-cool: #e6e2db;
      ` : ''}
      ${tweaks.density === 'compact' ? `
        --t-xs:  9px;
        --t-sm:  10.5px;
        --t-md:  12px;
        --t-lg:  14px;
      ` : tweaks.density === 'spacious' ? `
        --t-xs:  11px;
        --t-sm:  12.5px;
        --t-md:  14px;
        --t-lg:  16px;
      ` : ''}
    }
  `;
  return <style dangerouslySetInnerHTML={{ __html: css }} />;
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

  // Tweaks
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);

  // When tweak default overlay changes, propagate to player state
  uE(() => {
    if (tweaks.playerOverlay !== overlay) setOverlay(tweaks.playerOverlay);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tweaks.playerOverlay]);

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
      <ThemeInject tweaks={tweaks} />

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
          />
          <CkptNav
            run={focusedRun}
            ckpt={ckpt}
            onSelectCkpt={setCkptStep}
          />

          <div className="col" style={{ padding: '14px 16px 10px', gap: 10, flex: '0 0 auto' }}>
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

          <div className="grow" style={{ minHeight: 0 }} />
          <LossStrip run={focusedRun} ckpt={ckpt} />

          <div className="row border-t" style={{ padding: '5px 16px', fontSize: 10.5, color: 'var(--ink-3)', flex: '0 0 auto' }}>
            <span><span className="kbd">space</span> play  ·  <span className="kbd">← →</span> frame  ·  <span className="kbd">⇧← →</span> ±30  ·  <span className="kbd">J L</span> ckpt  ·  <span className="kbd">1 2 3</span> best/median/worst</span>
            <span className="grow" />
            <span className="italic" style={{ fontFamily: 'var(--display)' }}>"player is procedural — real video would render here"</span>
          </div>
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

      {/* Tweaks */}
      <TweaksPanel title="Tweaks">
        <TweakSection label="Accent">
          <TweakColor
            label="hue"
            value={tweaks.accent}
            options={['#b85a3e', '#3e6db8', '#3e8a5a', '#7a4a8a', '#1c1813']}
            onChange={(v) => setTweak('accent', v)}
          />
        </TweakSection>
        <TweakSection label="Theme">
          <TweakRadio
            label="paper tone"
            value={tweaks.paperTone}
            options={[
              { value: 'warm',    label: 'warm' },
              { value: 'cool',    label: 'cool' },
              { value: 'neutral', label: 'neutral' },
            ]}
            onChange={(v) => setTweak('paperTone', v)}
          />
        </TweakSection>
        <TweakSection label="Density">
          <TweakRadio
            label="text"
            value={tweaks.density}
            options={[
              { value: 'compact',  label: 'compact' },
              { value: 'regular',  label: 'regular' },
              { value: 'spacious', label: 'spacious' },
            ]}
            onChange={(v) => setTweak('density', v)}
          />
        </TweakSection>
        <TweakSection label="Player">
          <TweakRadio
            label="default overlay"
            value={tweaks.playerOverlay}
            options={[
              { value: 'none',       label: 'none' },
              { value: 'saliency',   label: 'saliency' },
              { value: 'trajectory', label: 'traj' },
            ]}
            onChange={(v) => setTweak('playerOverlay', v)}
          />
        </TweakSection>
      </TweaksPanel>
    </React.Fragment>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
