/**
 * TypeScript definitions for the rl-dashboard-v1 schema.
 * Represents the data structures output by the Python DashboardRunWriter.
 */

export const DASHBOARD_SCHEMA_VERSION = "rl-dashboard-v1";

/**
 * run.json
 */
export interface RunDoc {
  schema_version: string;
  run_id: string;
  name: string;
  algorithm: string | null;
  env_id: string | null;
  status: "running" | "done" | "failed" | string;
  seed: number | null;
  hparams: Record<string, any>;
  capabilities: Capabilities;
  created_at: string; // ISO 8601 date string
  updated_at: string;
  ended_at?: string;
  [extra: string]: any; // Allows for extra_metadata
}

export interface Capabilities {
  signals: string[];
  signal_semantics?: Record<string, SignalSemantic>;
}

export interface SignalSemantic {
  unit?: string;
  min?: number;
  max?: number;
  labels?: Record<number, string>;
  // Additional rendering hints can be added here
}

/**
 * train_metrics.jsonl (lines)
 */
export interface TrainMetricsEvent {
  type: "train_metrics";
  time: string;
  step: number;
  metrics: Record<string, number>;
  epoch?: number;
  wall_time_s?: number;
}

/**
 * episodes.jsonl (lines)
 */
export interface EpisodeEndEvent {
  type: "episode_end";
  time: string;
  step: number;
  episode_return: number;
  episode_length: number;
  env_index?: number;
}

/**
 * checkpoints.jsonl (lines)
 */
export interface CheckpointEvent {
  type: "checkpoint";
  time: string;
  step: number;
  num_eval_episodes: number;
  mean_return: number | null;
  std_return: number | null;
  best_return: number | null;
  median_return: number | null;
  worst_return: number | null;
  checkpoint_dir: string; // Relative path to checkpoint folder
}

/**
 * checkpoints/<step>/checkpoint.json
 */
export interface CheckpointDoc {
  schema_version: string;
  step: number;
  created_at: string;
  num_eval_episodes: number;
  mean_return: number | null;
  std_return: number | null;
  best_return: number | null;
  median_return: number | null;
  worst_return: number | null;
  rollouts: RolloutDirRef[];
  episodes: EpisodeSummary[];
}

export interface EpisodeSummary {
  episode_index: number;
  return: number | null;
  length: number | null;
}

export interface RolloutDirRef {
  kind: "best" | "median" | "worst" | string;
  episode_index: number;
  return: number | null;
  length: number | null;
  dir: string; // Relative path to rollout folder (e.g., "rollouts/best")
}

/**
 * checkpoints/<step>/rollouts/<kind>/meta.json
 */
export interface RolloutMetaDoc {
  schema_version: string;
  step: number;
  kind: string;
  episode_index: number;
  return: number | null;
  length: number | null;
  video_file: string | null; // usually "video.mp4"
  signals_file: string | null; // usually "signals.npz"
  available_signals: string[];
  signal_shapes: Record<string, number[]>;
  signal_semantics: Record<string, SignalSemantic>;
}

/**
 * Utility type describing the parsed payload of a .npz file in the browser.
 * Note: This is an interpretation of the binary data, not part of the JSON schema.
 */
export type ParsedSignals = Record<string, Float32Array | Int32Array | Uint8Array>;
