// hifi/rail-right.jsx — focused run metadata

function HpRow({ k, v, base, changed }) {
  return (
    <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center', minHeight: 32 }}>
      <span className="muted" style={{ fontSize: 11 }}>{k}</span>
      <span style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
        {changed && base !== undefined && (
          <span className="muted-soft" style={{ fontFamily: 'var(--mono)', fontSize: 10, textDecoration: 'line-through' }}>{base}</span>
        )}
        <span className="num" style={{ fontWeight: changed ? 600 : 400, color: changed ? 'var(--accent)' : 'var(--ink)' }}>
          {v}
        </span>
      </span>
    </div>
  );
}

function Section({ title, right, children }) {
  return (
    <div className="col" style={{ padding: '22px 20px', borderBottom: '1px solid var(--hairline)' }}>
      <div className="row" style={{ alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 16, gap: 8 }}>
        <span className="label-eyebrow" style={{ flex: '1 1 auto', minWidth: 0 }}>{title}</span>
        {right && <span style={{ fontSize: 10.5, color: 'var(--ink-3)', whiteSpace: 'nowrap', flex: '0 0 auto' }}>{right}</span>}
      </div>
      {children}
    </div>
  );
}

function RailRight({ run, ckpt, rollout, frame, baselineRun, allRuns }) {
  if (!run) return null;

  // HP diff
  const baseHp = baselineRun ? baselineRun.hp : {};
  const hpKeys = Object.keys(run.hp);
  const changes = hpKeys.filter(k => baselineRun && baseHp[k] !== run.hp[k]);

  // Action probs at current frame
  // const probs = D.actionProbs(run, ckpt, rollout, frame);
  // const labels = D.ACTION_LABELS[run.env] || ['0','1','2','3','4','5','6','7'];

  // TD-error for this rollout
  const tdValues = D.frameSignal(run, ckpt, rollout, 'td_error');

  return (
    <div className="col scroll" style={{ width: 'clamp(190px, 15vw, 280px)', flexShrink: 0, background: 'var(--paper)', borderLeft: '1px solid var(--hairline)', height: '100%' }}>

      {/* Focused run */}
      <Section title="Focused"
        right={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span className={`status-dot ${run.status}`} />
            <span style={{ textTransform: 'capitalize' }}>{run.status}</span>
          </span>
        }
      >
        <div className="display" style={{ fontSize: 20, fontWeight: 600, lineHeight: 1.2 }}>{run.name}</div>
        <div className="num muted" style={{ fontSize: 11, marginTop: 4 }}>
          #{run.id} · {run.env} · {run.alg.toUpperCase()}
        </div>
        <div className="row" style={{ marginTop: 20, gap: 18, flexWrap: 'wrap' }}>
          <div className="col">
            <span className="muted" style={{ fontSize: 10 }}>steps</span>
            <span className="num strong" style={{ fontSize: 14 }}>{D.fmtStep(run.steps)}</span>
          </div>
          <div className="col">
            <span className="muted" style={{ fontSize: 10 }}>final reward</span>
            <span className="num strong" style={{ fontSize: 14 }}>{D.fmtReward(run.reward)}</span>
          </div>
          <div className="col">
            <span className="muted" style={{ fontSize: 10 }}>updated</span>
            <span className="num strong" style={{ fontSize: 14 }}>{run.ago}</span>
          </div>
          <div className="col">
            <span className="muted" style={{ fontSize: 10 }}>Rollouts</span>
            <span className="num strong" style={{ fontSize: 14 }}>{ckpt.rollouts.length}</span>
          </div>
        </div>
      </Section>

      {/* Hyperparams */}
      <Section title="Hyperparams">
        {/* Diff subtitle */}
        <div style={{ fontFamily: 'var(--ui)', fontStyle: 'italic', fontSize: 11.5,
                      color: 'var(--ink-3)', marginBottom: 8, lineHeight: 1.3 }}>
          {baselineRun ? (
            <>
              diff vs{' '}
              <span style={{ fontStyle: 'normal', fontWeight: 600, color: 'var(--ink-2)',
                             fontFamily: 'var(--mono)' }}>
                {baselineRun.name}
              </span>
              {' '}
              <span style={{ fontStyle: 'normal',
                             color: changes.length ? 'var(--accent)' : 'var(--ink-4)' }}>
                {changes.length}Δ
              </span>
            </>
          ) : (
            <span style={{ color: 'var(--ink-4)' }}>no baseline selected</span>
          )}
        </div>
        <div className="col gap-1">
          {hpKeys.map(k => (
            <HpRow
              key={k}
              k={k}
              v={run.hp[k]}
              base={baseHp[k]}
              changed={baselineRun && baseHp[k] !== run.hp[k]}
            />
          ))}
        </div>
      </Section>

      
      {/* Tags + notes */}
      <Section title="Tags + notes">
        <div className="row gap-1" style={{ flexWrap: 'wrap', marginBottom: 6 }}>
          {run.tags.map(t => <span key={t} className="tag">{t}</span>)}
          <span className="tag ghost dotted">＋ add</span>
        </div>
        {run.note && (
          <div style={{ fontFamily: 'var(--ui)', fontStyle: 'italic', fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.45,
                        borderLeft: '2px solid var(--accent-soft)', paddingLeft: 10, marginTop: 6 }}>
            "{run.note}"
          </div>
        )}
      </Section>
    </div>
  );
}

window.RailRight = RailRight;
