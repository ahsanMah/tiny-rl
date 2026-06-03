// hifi/data.jsx — real data loader
// Replaces the synthetic data generator with actual files from dashboard_artifacts.

const D = {};

// ── Format helpers (unchanged — still used by the UI) ───────────────────
function fmtStep(s) {
  if (s >= 1e6) return `${(s / 1e6).toFixed(s % 1e6 === 0 ? 0 : 2)}M`;
  if (s >= 1e3) return `${(s / 1e3).toFixed(0)}k`;
  return `${s}`;
}
function fmtReward(r) { return r == null ? '—' : (r >= 1000 ? `+${(r / 1000).toFixed(1)}k` : `${r >= 0 ? '+' : ''}${r.toFixed(1)}`); }
function fmtTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// ── Compute how long ago a timestamp was (like "2h" or "3d") ────────────
// This replaces the hardcoded "ago" strings in the old synthetic data.
function computeAgo(isoString) {
  const diffMs = Date.now() - new Date(isoString).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 60) return `${mins}m`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ${mins % 60}m`;
  return `${Math.floor(hours / 24)}d`;
}

// ── Parse a .jsonl file (one JSON object per line) ───────────────────────
// Equivalent to: [json.loads(line) for line in open(file)]
async function fetchJsonl(url) {
  const text = await fetch(url).then(r => r.text());
  return text.trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
}

// ── Load one run from its files ──────────────────────────────────────────
async function loadRun(id) {
  // 1. Load the main metadata file (run.json)
  const runDoc = await fetch(`/runs/${id}/run.json`).then(r => r.json());

  // 2. Load the list of checkpoints (checkpoints.jsonl)
  //    Each line is one checkpoint event with return statistics.
  let ckptEvents = [];
  try {
    ckptEvents = await fetchJsonl(`/runs/${id}/checkpoints.jsonl`);
  } catch {
    // Some runs may not have any checkpoints yet — that's fine
  }

  // 3. For each checkpoint event, fetch checkpoint.json to get rollout details
  const checkpoints = await Promise.all(
    ckptEvents.map(async (evt) => {
      let rollouts = [];
      try {
        const ckptDoc = await fetch(`/runs/${id}/${evt.checkpoint_dir}/checkpoint.json`).then(r => r.json());

        // Map rollout entries to the shape the UI expects.
        // "idx" is used internally for seeding the synthetic signal fallback.
        rollouts = ckptDoc.rollouts.map((r) => ({
          kind:   r.kind,
          return: r.return,
          length: r.length,
          idx:    ['best', 'median', 'worst'].indexOf(r.kind),
          // Full URL path so the browser can fetch signals/video later (Steps 4 & 5)
          dir: `/runs/${id}/${evt.checkpoint_dir}/${r.dir}`,
        }));
      } catch {
        // No checkpoint.json or no rollouts — leave rollouts empty
      }

      // Rename fields from real schema → what the UI expects
      return {
        step:     evt.step,
        mean:     evt.mean_return,
        std:      evt.std_return,
        best:     evt.best_return,
        median:   evt.median_return,
        worst:    evt.worst_return,
        rollouts,
        isDip:    false,
      };
    })
  );

  // 4. Derive a few summary fields the UI shows in the left rail
  const lastStep    = checkpoints.length > 0 ? checkpoints[checkpoints.length - 1].step : 0;
  const allBests    = checkpoints.map(c => c.best).filter(v => v != null);
  const bestReward  = allBests.length > 0 ? Math.max(...allBests) : null;

  // Convert hparam values to strings — the UI renders them as text
  const hp = {};
  for (const [k, v] of Object.entries(runDoc.hparams || {})) {
    hp[k] = String(v);
  }

  // Load training metrics (loss curves, sps, etc.) for the LossStrip.
  // Group all events by metric name: { policy_loss: [{step, value}, ...], ... }
  let trainMetrics = {};
  try {
    const tmEvents = await fetchJsonl(`/runs/${id}/train_metrics.jsonl`);
    for (const evt of tmEvents) {
      for (const [key, val] of Object.entries(evt.metrics || {})) {
        if (!trainMetrics[key]) trainMetrics[key] = [];
        trainMetrics[key].push({ step: evt.step, value: val });
      }
    }
  } catch {
    // Not all runs have train_metrics.jsonl — fine to skip
  }

  return {
    id:           runDoc.run_id,
    name:         runDoc.name,
    alg:          runDoc.algorithm,
    env:          runDoc.env_id,
    status:       runDoc.status,
    steps:        lastStep,
    reward:       bestReward != null ? Math.round(bestReward * 10) / 10 : null,
    ago:          computeAgo(runDoc.created_at),
    tags:         [runDoc.algorithm, runDoc.env_id].filter(Boolean),
    hp,
    capabilities:  runDoc.capabilities || { signals: [] },
    trainMetrics,
    checkpoints,
  };
}

// ── Load all runs ────────────────────────────────────────────────────────
async function loadAllRuns() {
  const runIds = await fetch('/runs/index.json').then(r => r.json());
  console.log('Loading runs:', runIds);
  const runs = await Promise.all(runIds.map(loadRun));
  console.log('All runs loaded:', runs);
  return runs;
}
// ── Real signal loading (.npz files) ────────────────────────────────────
// We use JSZip to unzip  and then parse each .npy 

// Cache: rollout.dir → { step_reward: Float32Array, cumulative_return: Float32Array, ... }
const SIGNAL_CACHE = new Map();

// Parse a single .npy binary buffer into a Float32Array.
function parseNpy(buffer) {
  const view = new DataView(buffer);
  const major = view.getUint8(6);
  const headerLen = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const headerStart = major >= 2 ? 12 : 10;
  const headerStr = new TextDecoder().decode(new Uint8Array(buffer, headerStart, headerLen));
  const dataBuffer = buffer.slice(headerStart + headerLen);

  const dtypeMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const dtype = dtypeMatch ? dtypeMatch[1] : '<f4';

  switch (dtype) {
    case '<f4': return new Float32Array(dataBuffer);
    case '<f8': return new Float32Array(new Float64Array(dataBuffer)); // downcast to float32
    case '<i4': return new Int32Array(dataBuffer);
    default:    return new Float32Array(dataBuffer);
  }
}

// Fetch and parse a rollout's signals.npz.
// Returns an object like { step_reward: Float32Array, cumulative_return: Float32Array }.
// Returns {} if the file doesn't exist (e.g. PPO runs have no signals.npz).
async function loadSignals(rollout) {
  if (!rollout || !rollout.dir) return {};
  if (SIGNAL_CACHE.has(rollout.dir)) return SIGNAL_CACHE.get(rollout.dir);

  try {
    const response = await fetch(`${rollout.dir}/signals.npz`);
    if (!response.ok) throw new Error('no signals.npz');

    const buffer   = await response.arrayBuffer();
    const zip      = await JSZip.loadAsync(buffer);
    const signals  = {};

    // Each file in the zip is one signal array, e.g. "step_reward.npy"
    for (const filename of Object.keys(zip.files)) {
      const signalName = filename.replace('.npy', '');
      const npyBuffer  = await zip.files[filename].async('arraybuffer');
      signals[signalName] = parseNpy(npyBuffer);
    }

    // Prefer the stored cumulative_return; only derive it from step_reward
    // when the run didn't write it directly (never fabricate when neither exists).
    if (!signals.cumulative_return && signals.step_reward) {
      const sr  = signals.step_reward;
      const cum = new Float32Array(sr.length);
      let total = 0;
      for (let i = 0; i < sr.length; i++) { total += sr[i]; cum[i] = total; }
      signals.cumulative_return = cum;
    }

    console.log(`Loaded signals for ${rollout.dir}:`, Object.keys(signals));
    SIGNAL_CACHE.set(rollout.dir, signals);
    return signals;
  } catch {
    // Not an error — just means this rollout has no signal file (e.g. PPO runs)
    SIGNAL_CACHE.set(rollout.dir, {});
    return {};
  }
}

// ── Per-frame signal lookup ──────────────────────────────────────────────
// Returns the real signal array for this rollout/metric, or null when the
// rollout has no signals.npz or the metric isn't present in it. We never
// synthesize stand-in data — callers render a "missing" state instead.
function frameSignal(run, ckpt, rollout, metric) {
  if (!rollout || !rollout.dir || !metric) return null;
  const real = SIGNAL_CACHE.get(rollout.dir);
  return real && real[metric] !== undefined ? real[metric] : null;
}

// ── Lookup helpers (unchanged) ───────────────────────────────────────────
function getRun(id) { return D.RUNS.find(r => r.id === id); }
function getCheckpoint(run, step) {
  if (!run || !run.checkpoints.length) return null;
  return run.checkpoints.find(c => c.step === step) || run.checkpoints[run.checkpoints.length - 1];
}
function getRollout(ckpt, kind) {
  if (!ckpt || !ckpt.rollouts.length) return null;
  return ckpt.rollouts.find(r => r.kind === kind) || ckpt.rollouts[0];
}

// ── Bootstrap ────────────────────────────────────────────────────────────
D.RUNS = [];  // starts empty; populated when D.ready resolves
D.ready = loadAllRuns().then(runs => { D.RUNS = runs; });

D.frameSignal    = frameSignal;
D.loadSignals    = loadSignals;
D.getRun         = getRun;
D.getCheckpoint  = getCheckpoint;
D.getRollout     = getRollout;
D.fmtStep        = fmtStep;
D.fmtReward      = fmtReward;
D.fmtTime        = fmtTime;

window.D = D;
