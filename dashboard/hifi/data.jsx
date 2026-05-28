// hifi/data.jsx — real data loader
// Replaces the synthetic data generator with actual files from dashboard_artifacts.

const D = {};

// ── String hash (replaces the old parseInt(id, 16) which only worked for hex IDs) ──
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (Math.imul(31, hash) + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

// ── Deterministic RNG — still used by LossStrip for now ─────────────────
function rng(seed) {
  let s = (seed | 0) || 1;
  return () => {
    s = (s + 0x9E3779B9) | 0;
    let t = Math.imul(s ^ (s >>> 16), 0x21f0aaad);
    t = Math.imul(t ^ (t >>> 15), 0x735a2d97);
    return ((t ^ (t >>> 15)) >>> 0) / 4294967296;
  };
}

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

    // Derive cumulative_return by summing step_reward — it's not stored directly
    if (signals.step_reward) {
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

// ── Per-frame signal synthesis ───────────────────────────────────────────
const FRAME_CACHE = new Map();

function frameKey(runId, step, kind, metric) {
  return `${runId}-${step}-${kind}-${metric}`;
}

function frameSignal(run, ckpt, rollout, metric) {
  // If real signals are already loaded for this rollout, use them
  if (rollout.dir) {
    const real = SIGNAL_CACHE.get(rollout.dir);
    if (real && real[metric] !== undefined) return real[metric];
  }

  // Otherwise fall back to the synthetic generator (cached to avoid recomputing)
  const key = frameKey(run.id, ckpt.step, rollout.kind, metric);
  if (FRAME_CACHE.has(key)) return FRAME_CACHE.get(key);

  const len = rollout.length;
  const arr = new Float32Array(len);
  // Use hashString instead of parseInt(id, 16) so non-hex IDs work
  const seed = (hashString(run.id) * 31 + ckpt.step / 1e5 + rollout.idx * 7 + metric.length) | 0;
  const r = rng(seed);

  const meanR = rollout.return / len;
  const stumbleAt = Math.floor(len * (0.35 + r() * 0.45));
  const stumbleWidth = 8 + Math.floor(r() * 8);

  if (metric === 'cumulative_return') {
    let cum = 0;
    for (let i = 0; i < len; i++) {
      let step_r = meanR + (r() - 0.5) * Math.abs(meanR) * 1.4;
      if (Math.abs(i - stumbleAt) < stumbleWidth) step_r -= 1.0;
      cum += step_r;
      arr[i] = cum;
    }
  } else if (metric === 'step_reward') {
    for (let i = 0; i < len; i++) {
      let s = meanR + (r() - 0.5) * Math.abs(meanR) * 1.4;
      if (Math.abs(i - stumbleAt) < stumbleWidth) s -= 1.0;
      arr[i] = s;
    }
  } else if (metric === 'value') {
    let v = 0.5;
    const target = 0.4 + rollout.return / 2000;
    for (let i = 0; i < len; i++) {
      v = v * 0.92 + (r() * 0.4 + target - 0.2) * 0.08;
      if (Math.abs(i - stumbleAt) < stumbleWidth * 3) v -= 0.018;
      arr[i] = v;
    }
  } else if (metric === 'td_error') {
    for (let i = 0; i < len; i++) {
      let e = (r() - 0.5) * 0.4;
      if (Math.abs(i - stumbleAt) < stumbleWidth) e += (r() - 0.5) * 2.5;
      arr[i] = e;
    }
  } else if (metric === 'advantage') {
    for (let i = 0; i < len; i++) {
      let a = (r() - 0.5) * 0.6 + (i % 50 === 0 ? r() : 0);
      if (Math.abs(i - stumbleAt) < stumbleWidth) a -= 1.2;
      arr[i] = a;
    }
  } else if (metric === 'action_logp') {
    for (let i = 0; i < len; i++) arr[i] = -0.5 - r() * 0.8;
  } else if (metric === 'entropy') {
    for (let i = 0; i < len; i++) {
      const decay = 0.7 - (i / len) * 0.2;
      arr[i] = decay + (r() - 0.5) * 0.1;
    }
  } else {
    for (let i = 0; i < len; i++) arr[i] = r();
  }

  FRAME_CACHE.set(key, arr);
  return arr;
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
D.hashString     = hashString;
D.getRun         = getRun;
D.getCheckpoint  = getCheckpoint;
D.getRollout     = getRollout;
D.fmtStep        = fmtStep;
D.fmtReward      = fmtReward;
D.fmtTime        = fmtTime;
D.rng            = rng;

window.D = D;
