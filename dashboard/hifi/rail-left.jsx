// hifi/rail-left.jsx — runs list with search, pin/unpin, focus

const { useState: useS } = React;

function IconPin({ filled = false, size = 11 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24"
         fill={filled ? 'currentColor' : 'none'}
         stroke="currentColor" strokeWidth="2"
         strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="17" x2="12" y2="22" />
      <path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24Z" />
    </svg>
  );
}

function RunRow({ run, isFocused, isPinned, onFocus, onTogglePin, lineSlot }) {
  const sparkValues = run.checkpoints.map(c => c.mean);
  // Pick the matching line style: slot 0 = focused, 1-3 = ghosts, null = unpinned default
  const ls = lineSlot != null
    ? (RUN_LINE_STYLES[lineSlot] || RUN_LINE_STYLES[RUN_LINE_STYLES.length - 1])
    : RUN_LINE_DEFAULT;

  return (
    <div className={`run-row ${isFocused ? 'active' : ''}`}
         style={{ gridTemplateColumns: '16px minmax(0,1fr) 50px 40px', columnGap: 6, padding: '9px 12px' }}
         onClick={onFocus}>
      <button
        onClick={(e) => { e.stopPropagation(); onTogglePin(); }}
        title={isPinned ? 'unpin' : 'pin'}
        style={{
          border: 'none', background: 'transparent', padding: 0, cursor: 'pointer',
          color: isPinned ? 'var(--green)' : 'var(--ink-4)',
          width: 16, height: 16, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        <IconPin filled={isPinned} size={15} />
      </button>
      <span className="strong" style={{ overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis', minWidth: 0 }}>
        {run.name}
      </span>
      <Sparkline
        values={sparkValues} width={50} height={14}
        color={ls.color}
        strokeWidth={ls.width}
        strokeDasharray={ls.dash}
      />
      <span className="num muted" style={{ textAlign: 'right', fontSize: 10.5 }}>{D.fmtReward(run.reward)}</span>
    </div>
  );
}

function RailLeft({ runs, focusedId, pinnedIds, onFocus, onTogglePin, query, setQuery }) {
  const [groupBy, setGroupBy] = useS('alg'); // 'alg' | 'env' | 'flat'

  // ── Slot map: focused run = 0, other pinned in D.RUNS order = 1, 2, 3 ──
  // Must match the order FrameChartPair iterates pinnedRuns (D.RUNS order).
  const slotMap = {};
  slotMap[focusedId] = 0;
  let nextSlot = 1;
  for (const r of runs) {
    if (r.id !== focusedId && pinnedIds.includes(r.id)) {
      slotMap[r.id] = nextSlot++;
    }
  }

  const filtered = runs.filter(r => {
    if (!query) return true;
    const q = query.toLowerCase();
    return r.name.toLowerCase().includes(q)
        || r.id.includes(q)
        || r.alg.includes(q)
        || r.env.toLowerCase().includes(q)
        || r.tags.some(t => t.includes(q));
  });

  // Pinned section
  const pinned = filtered.filter(r => pinnedIds.includes(r.id));

  // Group remaining
  const remaining = filtered.filter(r => !pinnedIds.includes(r.id));
  const grouped = (() => {
    if (groupBy === 'flat') return [['All', remaining]];
    const m = new Map();
    for (const r of remaining) {
      const key = groupBy === 'alg' ? r.alg : r.env;
      if (!m.has(key)) m.set(key, []);
      m.get(key).push(r);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  })();

  return (
    <div className="col" style={{ width: 'clamp(150px, 30vw, 340px)', flexShrink: 0, background: 'color-mix(in srgb, var(--accent) 7%, var(--surface))', borderRight: '1px solid var(--hairline)', height: '100%' }}>
      {/* Header */}
      <div className="col" style={{ padding: '22px 16px 16px', gap: 12, borderBottom: '1px solid var(--hairline)' }}>
        <div className="row" style={{ gap: 8 }}>
          <span className="display" style={{ fontSize: 15, fontWeight: 700 }}>tracker</span>
          <span className="muted" style={{ fontSize: 11 }}>/ 247 runs</span>
          <span className="grow" />
          <button className="btn tert" style={{ padding: '4px 10px', fontSize: 12 }}>＋ new run</button>
        </div>
        <input
          className="input"
          placeholder="filter: name, id, tag…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ fontFamily: 'var(--mono)', fontSize: 11 }}
        />
        <div className="row" style={{ gap: 4, fontSize: 14 }}>
          <span className="label-eyebrow">group</span>
          {[
            { key: 'alg',  label: 'alg'  },
            { key: 'env',  label: 'env' },
            { key: 'flat', label: 'all'  },
          ].map(({ key, label }) => (
            <button key={key}
              className={'btn' + (groupBy === key ? ' active' : '')}
              style={{ padding: '1px 7px', fontSize: 12 }}
              onClick={() => setGroupBy(key)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* List */}
      <div className="scroll grow">
        {/* Pinned section */}
        <div className="col">
          <div className="row" style={{ padding: '20px 16px 8px', alignItems: 'baseline', justifyContent: 'space-between', gap: 8 }}>
            <span className="label-eyebrow">Pinned · {pinned.length}</span>
          </div>
          {pinned.length === 0 && (
            <div className="muted" style={{ padding: '8px 12px', fontSize: 11, fontStyle: 'italic' }}>none pinned</div>
          )}
          {pinned.map(r => (
            <RunRow
              key={r.id}
              run={r}
              isFocused={r.id === focusedId}
              isPinned={true}
              onFocus={() => onFocus(r.id)}
              onTogglePin={() => onTogglePin(r.id)}
              lineSlot={slotMap[r.id] ?? null}
            />
          ))}
        </div>

        {/* Groups */}
        {grouped.map(([groupName, groupRuns]) => (
          <div key={groupName} className="col">
            <div className="row" style={{ padding: '20px 16px 8px', alignItems: 'baseline', justifyContent: 'space-between', gap: 8 }}>
              <span className="label-eyebrow">{groupName} · {groupRuns.length}</span>
              <span className="muted" style={{ fontSize: 10, whiteSpace: 'nowrap' }}>↕ recent</span>
            </div>
            {groupRuns.map(r => (
              <RunRow
                key={r.id}
                run={r}
                isFocused={r.id === focusedId}
                isPinned={false}
                onFocus={() => onFocus(r.id)}
                onTogglePin={() => onTogglePin(r.id)}
              />
            ))}
          </div>
        ))}

        <div style={{ height: 16 }} />
      </div>

      {/* Footer */}
      <div className="row" style={{ padding: '14px 16px', borderTop: '1px solid var(--hairline)', fontSize: 10.5, color: 'var(--ink-3)' }}>
        <span>{filtered.length} of {runs.length}</span>
      </div>
    </div>
  );
}

window.RailLeft = RailLeft;
