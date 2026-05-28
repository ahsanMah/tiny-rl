// hifi/data.jsx — synthetic data layer for the prototype
// All deterministic. In production these come from a backend.
//
// Hierarchy: Project → Run → Checkpoint → Rollout (best/median/worst + 2 more)
// Each rollout has per-frame signals: step_reward, value, action_logp,
// advantage, td_error, entropy, action_probs[8].

const D = {};

// ── Deterministic RNG (mulberry32-ish) ───────────────────────────────
function rng(seed) {
  let s = (seed | 0) || 1;
  return () => {
    s = (s + 0x9E3779B9) | 0;
    let t = Math.imul(s ^ (s >>> 16), 0x21f0aaad);
    t = Math.imul(t ^ (t >>> 15), 0x735a2d97);
    return ((t ^ (t >>> 15)) >>> 0) / 4294967296;
  };
}

// ── Algorithms / env presets ─────────────────────────────────────────
const ALGS = ['ppo', 'sac', 'dqn', 'rainbow', 'impala'];

// ── Runs ─────────────────────────────────────────────────────────────
const RAW_RUNS = [
  // Walker family (focal)
  { name: 'ppo-walker-v37', id: 'a4f2', alg: 'ppo', env: 'walker2d-v4', status: 'running', steps: 3.2e6,  reward: 842,  ago: '12m',  tags: ['locomotion','ppo','lr-sweep'] },
  { name: 'ppo-walker-v36', id: 'a4ee', alg: 'ppo', env: 'walker2d-v4', status: 'done',    steps: 10e6,  reward: 811,  ago: '4h 22m', tags: ['locomotion','ppo','baseline'], note: 'entropy collapsed at ~3M, recovered after lr warmup' },
  { name: 'ppo-walker-v35', id: 'a4ec', alg: 'ppo', env: 'walker2d-v4', status: 'done',    steps: 10e6,  reward: 774,  ago: '1d',     tags: ['locomotion','ppo'] },
  { name: 'ppo-walker-v34', id: 'a4e9', alg: 'ppo', env: 'walker2d-v4', status: 'failed',  steps: 1.1e6, reward: null, ago: '1d 4h', tags: ['locomotion','ppo','crashed'], note: 'NaN gradients at 1.1M' },
  { name: 'ppo-walker-v33', id: 'a4e6', alg: 'ppo', env: 'walker2d-v4', status: 'done',    steps: 10e6,  reward: 712,  ago: '2d',     tags: ['locomotion','ppo'] },
  { name: 'ppo-walker-v32', id: 'a4e3', alg: 'ppo', env: 'walker2d-v4', status: 'done',    steps: 10e6,  reward: 688,  ago: '2d',     tags: ['locomotion','ppo'] },
  // Ant family
  { name: 'sac-ant-tuned-3', id: 'b201', alg: 'sac',     env: 'ant-v4',    status: 'running', steps: 6.7e6,  reward: 1218,  ago: '40m', tags: ['locomotion','sac'] },
  { name: 'sac-ant-tuned-2', id: 'b200', alg: 'sac',     env: 'ant-v4',    status: 'done',    steps: 10e6,   reward: 1102,  ago: '1d',  tags: ['locomotion','sac'] },
  { name: 'sac-ant-tuned-1', id: 'b1ff', alg: 'sac',     env: 'ant-v4',    status: 'done',    steps: 10e6,   reward: 1004,  ago: '3d',  tags: ['locomotion','sac'] },
  { name: 'sac-ant-base',    id: 'b1fc', alg: 'sac',     env: 'ant-v4',    status: 'done',    steps: 10e6,   reward: 894,   ago: '5d',  tags: ['locomotion','sac','baseline'] },
  // Atari family
  { name: 'dqn-breakout-v9', id: 'c012', alg: 'dqn',     env: 'Breakout',  status: 'done',    steps: 50e6,   reward: 312,   ago: '1d',  tags: ['atari','dqn'] },
  { name: 'dqn-breakout-v8', id: 'c011', alg: 'dqn',     env: 'Breakout',  status: 'done',    steps: 50e6,   reward: 289,   ago: '4d',  tags: ['atari','dqn'] },
  { name: 'dqn-breakout-v7', id: 'c010', alg: 'dqn',     env: 'Breakout',  status: 'done',    steps: 50e6,   reward: 241,   ago: '6d',  tags: ['atari','dqn'] },
  { name: 'rainbow-pong-v2', id: 'c104', alg: 'rainbow', env: 'Pong',      status: 'done',    steps: 20e6,   reward: 21,    ago: '7d',  tags: ['atari','rainbow'] },
  { name: 'rainbow-pong-v1', id: 'c103', alg: 'rainbow', env: 'Pong',      status: 'done',    steps: 20e6,   reward: 19,    ago: '7d',  tags: ['atari','rainbow'] },
  { name: 'impala-mujoco-1', id: 'd044', alg: 'impala',  env: 'humanoid-v4', status: 'done',  steps: 30e6,   reward: 602,   ago: '9d',  tags: ['locomotion','impala'] },
];

// ── Hyperparams (focal + relatives) ──────────────────────────────────
const HP_DEFAULTS = {
  ppo: { lr: '3e-4', lr_warmup: '—', gamma: '0.99', gae_lambda: '0.95', clip: '0.2', entropy_c: '0.005', batch: '2048', minibatch: '64', epochs: '10', vf_coef: '0.5', n_envs: '16', max_grad: '0.5' },
  sac: { lr: '3e-4', tau: '0.005', gamma: '0.99', alpha: '0.2', batch: '256', buffer: '1M', target_freq: '1', n_envs: '8' },
  dqn: { lr: '6.25e-5', gamma: '0.99', batch: '32', buffer: '1M', eps_start: '1.0', eps_end: '0.01', target_freq: '8000', n_envs: '1' },
  rainbow: { lr: '6.25e-5', gamma: '0.99', batch: '32', buffer: '1M', noisy: 'true', n_atoms: '51', n_step: '3', n_envs: '1' },
  impala: { lr: '6e-4', gamma: '0.99', entropy_c: '0.01', batch: '32', n_envs: '256', unroll: '20' },
};

// Per-run hp overrides (where they differ from algo default)
const HP_OVERRIDES = {
  'a4f2': { lr: '1e-4', n_envs: '32' },                                          // v37
  'a4ee': { lr: '3e-4', lr_warmup: '50k', entropy_c: '0.01', n_envs: '32' },     // v36 — focal
  'a4ec': { lr: '1e-4', entropy_c: '0.005', n_envs: '16' },                      // v35 — baseline
  'a4e9': { lr: '3e-3', n_envs: '32' },                                          // v34 — failed
  'a4e6': { lr: '1e-4', n_envs: '16' },                                          // v33
  'b201': { lr: '5e-4' },
};

// ── Checkpoints per run ──────────────────────────────────────────────
// 13 ckpts spanning 0.5M → 10M for finished walker runs, fewer for in-progress.
// Each ckpt holds 5 rollouts: best, q3, median, q1, worst.
function genCheckpoints(run) {
  const r = rng(parseInt(run.id, 16));
  const totalSteps = run.steps;
  const baselineMax = run.reward != null ? run.reward * 1.08 : 100;
  const ckptCount = run.status === 'failed' ? 2 : (run.status === 'running' ? Math.max(3, Math.floor(totalSteps / 1e6) + 1) : 13);
  const ckpts = [];
  for (let i = 0; i < ckptCount; i++) {
    const tFrac = (i + 1) / ckptCount;
    const step = Math.round(totalSteps * tFrac / 1e5) * 1e5;
    // Sigmoid-ish growth toward baselineMax
    const base = baselineMax * (1 - Math.exp(-3.2 * tFrac));
    // Inject one entropy-collapse dip for the focal run (v36, i=5)
    const isDip = run.id === 'a4ee' && i === 5;
    const dipMul = isDip ? 0.55 : 1;
    const mean = Math.round(base * dipMul + (r() - 0.5) * baselineMax * 0.05);
    const std = Math.round(Math.max(20, baselineMax * 0.05 + (r() * 30)));
    const rollouts = ['best','q3','median','q1','worst'].map((kind, j) => {
      const k = [1.18, 1.04, 0.96, 0.86, 0.78][j];
      const ret = Math.round(mean * k);
      const len = Math.round(1200 + (k - 0.78) * 600 + r() * 80); // best is longer
      return { kind, return: ret, length: len, idx: j };
    });
    ckpts.push({ step, mean, std, best: rollouts[0].return, median: rollouts[2].return, worst: rollouts[4].return, rollouts, isDip });
  }
  return ckpts;
}

// Compose runs with checkpoints
const RUNS = RAW_RUNS.map(r => ({
  ...r,
  hp: { ...HP_DEFAULTS[r.alg], ...(HP_OVERRIDES[r.id] || {}) },
  checkpoints: genCheckpoints(r),
}));

// ── Per-frame signal synthesis ───────────────────────────────────────
// Cache keyed by `${runId}-${step}-${rolloutKind}-${metric}`.
const FRAME_CACHE = new Map();

function frameKey(runId, step, kind, metric) {
  return `${runId}-${step}-${kind}-${metric}`;
}

const N_ACTIONS = 8;

// Returns Float32Array of length = rolloutLen for one metric
function frameSignal(run, ckpt, rollout, metric) {
  const key = frameKey(run.id, ckpt.step, rollout.kind, metric);
  if (FRAME_CACHE.has(key)) return FRAME_CACHE.get(key);

  const len = rollout.length;
  const arr = new Float32Array(len);
  const seed = (parseInt(run.id, 16) * 31 + ckpt.step / 1e5 + rollout.idx * 7 + metric.length) | 0;
  const r = rng(seed);

  // mean per-step reward such that sum ≈ rollout.return
  const meanR = rollout.return / len;
  // Locate a "stumble" event (negative-reward spike) somewhere in second half
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
    for (let i = 0; i < len; i++) {
      arr[i] = -0.5 - r() * 0.8;
    }
  } else if (metric === 'entropy') {
    // Action entropy — decays over rollout, with noise
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

// Look up run by id
function getRun(id) { return RUNS.find(r => r.id === id); }
function getCheckpoint(run, step) { return run.checkpoints.find(c => c.step === step) || run.checkpoints[run.checkpoints.length - 1]; }
function getRollout(ckpt, kind) { return ckpt.rollouts.find(r => r.kind === kind) || ckpt.rollouts[0]; }

// Format helpers
function fmtStep(s) {
  if (s >= 1e6) return `${(s / 1e6).toFixed(s % 1e6 === 0 ? 0 : 2)}M`;
  if (s >= 1e3) return `${(s / 1e3).toFixed(0)}k`;
  return `${s}`;
}
function fmtReward(r) { return r == null ? '—' : (r >= 1000 ? `+${(r / 1000).toFixed(1)}k` : `+${r}`); }
function fmtTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

D.RUNS = RUNS;
D.ALGS = ALGS;
D.N_ACTIONS = N_ACTIONS;
D.frameSignal = frameSignal;
D.getRun = getRun;
D.getCheckpoint = getCheckpoint;
D.getRollout = getRollout;
D.fmtStep = fmtStep;
D.fmtReward = fmtReward;
D.fmtTime = fmtTime;
D.rng = rng;

window.D = D;
