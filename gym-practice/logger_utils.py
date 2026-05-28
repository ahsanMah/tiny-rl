import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from loguru import logger
from tensorboardX import SummaryWriter

DASHBOARD_SCHEMA_VERSION = "rl-dashboard-v1"


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return value

    return value


def _to_jsonable(value: Any) -> Any:
    value = _coerce_scalar(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    return str(value)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(payload), separators=(",", ":")))
        f.write("\n")


def _infer_env_and_algorithm(exp_name: str) -> tuple[Optional[str], Optional[str]]:
    # Best-effort parser for names shaped like "<env>-<algo>".
    known_algo_suffixes = (
        "ppo-jax",
        "vpg-gae",
        "vectorized-gae",
        "ppo",
        "sac",
    )
    for suffix in known_algo_suffixes:
        marker = f"-{suffix}"
        if exp_name.endswith(marker):
            env_id = exp_name[: -len(marker)]
            return env_id, suffix
    return None, None


class DashboardRunWriter:
    """Writes dashboard-oriented run artifacts using the rl-dashboard-v1 schema."""

    def __init__(
        self,
        base_dir: str,
        run_id: str,
        *,
        name: Optional[str] = None,
        algorithm: Optional[str] = None,
        env_id: Optional[str] = None,
        seed: Optional[int] = None,
        hparams: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        status: str = "running",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = base_dir
        self.run_id = run_id
        self.run_dir = os.path.join(base_dir, "runs", run_id)
        self.checkpoints_root = os.path.join(self.run_dir, "checkpoints")

        self.run_json_path = os.path.join(self.run_dir, "run.json")
        self.train_metrics_path = os.path.join(self.run_dir, "train_metrics.jsonl")
        self.episodes_path = os.path.join(self.run_dir, "episodes.jsonl")
        self.checkpoints_path = os.path.join(self.run_dir, "checkpoints.jsonl")

        os.makedirs(self.checkpoints_root, exist_ok=True)

        cap = dict(capabilities or {})
        cap.setdefault("signals", [])

        self._known_signals = set(str(s) for s in cap.get("signals", []))
        cap["signals"] = sorted(self._known_signals)

        now = _utc_now_iso()
        self.run_doc: Dict[str, Any] = {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "run_id": run_id,
            "name": name or run_id,
            "algorithm": algorithm,
            "env_id": env_id,
            "status": status,
            "seed": seed,
            "hparams": hparams or {},
            "capabilities": cap,
            "created_at": now,
            "updated_at": now,
        }

        if extra_metadata:
            self.run_doc.update(_to_jsonable(extra_metadata))

        self._flush_run_doc()

    def _flush_run_doc(self) -> None:
        self.run_doc["updated_at"] = _utc_now_iso()
        _write_json(self.run_json_path, self.run_doc)

    def _checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.checkpoints_root, f"{int(step):09d}")

    def update_status(self, status: str, *, ended: bool = False) -> None:
        self.run_doc["status"] = status
        if ended:
            self.run_doc["ended_at"] = _utc_now_iso()
        self._flush_run_doc()

    def register_signals(
        self,
        signals: Iterable[str],
        signal_semantics: Optional[Dict[str, Any]] = None,
    ) -> None:
        changed = False

        new_keys = {str(k) for k in signals if k}
        if new_keys - self._known_signals:
            self._known_signals |= new_keys
            self.run_doc.setdefault("capabilities", {})["signals"] = sorted(
                self._known_signals
            )
            changed = True

        if signal_semantics:
            cap = self.run_doc.setdefault("capabilities", {})
            semantics = cap.setdefault("signal_semantics", {})
            semantics.update(_to_jsonable(signal_semantics))
            changed = True

        if changed:
            self._flush_run_doc()

    def log_train_metrics(
        self,
        step: int,
        metrics: Dict[str, Any],
        *,
        epoch: Optional[int] = None,
        wall_time_s: Optional[float] = None,
    ) -> None:
        metric_values = {
            str(k): _coerce_scalar(v)
            for k, v in metrics.items()
            if _coerce_scalar(v) is not None
        }
        if not metric_values:
            return

        event = {
            "type": "train_metrics",
            "time": _utc_now_iso(),
            "step": int(step),
            "metrics": metric_values,
        }
        if epoch is not None:
            event["epoch"] = int(epoch)
        if wall_time_s is not None:
            event["wall_time_s"] = float(wall_time_s)

        _append_jsonl(self.train_metrics_path, event)

    def log_episode_end(
        self,
        step: int,
        episode_return: float,
        episode_length: int,
        *,
        env_index: Optional[int] = None,
    ) -> None:
        event = {
            "type": "episode_end",
            "time": _utc_now_iso(),
            "step": int(step),
            "episode_return": float(_coerce_scalar(episode_return)),
            "episode_length": int(_coerce_scalar(episode_length)),
        }
        if env_index is not None:
            event["env_index"] = int(env_index)

        _append_jsonl(self.episodes_path, event)

    def _select_rollouts(self, episodes: list[Dict[str, Any]]) -> list[tuple[str, Dict[str, Any]]]:
        if not episodes:
            return []

        episodes_with_return = [ep for ep in episodes if ep.get("return") is not None]
        ranked = (
            sorted(episodes_with_return, key=lambda ep: ep["return"], reverse=True)
            if episodes_with_return
            else sorted(episodes, key=lambda ep: ep["episode_index"])
        )

        selected: list[tuple[str, Dict[str, Any]]] = [("best", ranked[0])]
        if len(ranked) >= 3:
            selected.append(("median", ranked[len(ranked) // 2]))
        if len(ranked) >= 2:
            selected.append(("worst", ranked[-1]))

        return selected

    def write_rollout(
        self,
        *,
        step: int,
        kind: str,
        episode_index: int,
        episode_return: Optional[float],
        episode_length: Optional[int],
        video_src_path: Optional[str] = None,
        signals_src_path: Optional[str] = None,
        signals: Optional[Dict[str, Any]] = None,
        signal_semantics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        checkpoint_dir = self._checkpoint_dir(step)
        rollout_dir = os.path.join(checkpoint_dir, "rollouts", kind)
        os.makedirs(rollout_dir, exist_ok=True)

        # Video handling
        video_file = None
        if video_src_path:
            if os.path.exists(video_src_path):
                dst = os.path.join(rollout_dir, "video.mp4")
                if os.path.abspath(video_src_path) != os.path.abspath(dst):
                    shutil.copy2(video_src_path, dst)
                video_file = "video.mp4"
            else:
                logger.warning(
                    f"Dashboard writer: video source not found for {kind} at step={step}: {video_src_path}"
                )

        # Signal handling (either copy .npz or build one from provided arrays)
        signals_file = None
        available_signals: list[str] = []
        signal_shapes: Dict[str, list[int]] = {}

        if signals_src_path:
            if os.path.exists(signals_src_path):
                dst = os.path.join(rollout_dir, "signals.npz")
                if os.path.abspath(signals_src_path) != os.path.abspath(dst):
                    shutil.copy2(signals_src_path, dst)
                signals_file = "signals.npz"
                with np.load(dst) as loaded:
                    for key in loaded.files:
                        arr = loaded[key]
                        available_signals.append(key)
                        signal_shapes[key] = list(arr.shape)
            else:
                logger.warning(
                    f"Dashboard writer: signals source not found for {kind} at step={step}: {signals_src_path}"
                )
        elif signals:
            normalized_signals: Dict[str, np.ndarray] = {}
            for key, value in signals.items():
                if value is None:
                    continue
                try:
                    arr = np.asarray(value)
                except Exception:
                    continue

                if arr.size == 0:
                    continue
                if arr.dtype.kind in {"O", "U", "S"}:
                    continue
                if arr.dtype.kind == "f":
                    arr = arr.astype(np.float32)

                normalized_signals[str(key)] = arr

            if normalized_signals:
                dst = os.path.join(rollout_dir, "signals.npz")
                np.savez_compressed(dst, **normalized_signals)
                signals_file = "signals.npz"
                for key, arr in normalized_signals.items():
                    available_signals.append(key)
                    signal_shapes[key] = list(arr.shape)

        if available_signals:
            self.register_signals(available_signals, signal_semantics)

        rollout_meta = {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "step": int(step),
            "kind": kind,
            "episode_index": int(episode_index),
            "return": episode_return,
            "length": episode_length,
            "video_file": video_file,
            "signals_file": signals_file,
            "available_signals": sorted(available_signals),
            "signal_shapes": signal_shapes,
            "signal_semantics": _to_jsonable(signal_semantics or {}),
        }
        _write_json(os.path.join(rollout_dir, "meta.json"), rollout_meta)

        return {
            "kind": kind,
            "episode_index": int(episode_index),
            "return": episode_return,
            "length": episode_length,
            "dir": os.path.relpath(rollout_dir, checkpoint_dir),
        }

    def log_checkpoint(self, step: int, episodes: list[Dict[str, Any]]) -> None:
        if not episodes:
            return

        step = int(step)
        normalized: list[Dict[str, Any]] = []
        for i, ep in enumerate(episodes):
            episode_index = int(ep.get("episode_index", i))

            episode_return = ep.get("return")
            if episode_return is not None:
                episode_return = float(_coerce_scalar(episode_return))

            episode_length = ep.get("length")
            if episode_length is not None:
                episode_length = int(_coerce_scalar(episode_length))

            normalized.append(
                {
                    "episode_index": episode_index,
                    "return": episode_return,
                    "length": episode_length,
                    "video_path": ep.get("video_path"),
                    "signals_path": ep.get("signals_path"),
                    "signals": ep.get("signals"),
                    "signal_semantics": ep.get("signal_semantics"),
                }
            )

        checkpoint_dir = self._checkpoint_dir(step)
        os.makedirs(checkpoint_dir, exist_ok=True)

        returns = [ep["return"] for ep in normalized if ep["return"] is not None]
        mean_return = float(np.mean(returns)) if returns else None
        std_return = float(np.std(returns)) if returns else None
        best_return = max(returns) if returns else None
        median_return = float(np.median(returns)) if returns else None
        worst_return = min(returns) if returns else None

        selected_rollouts = self._select_rollouts(normalized)
        checkpoint_rollouts: list[Dict[str, Any]] = []
        for kind, ep in selected_rollouts:
            checkpoint_rollouts.append(
                self.write_rollout(
                    step=step,
                    kind=kind,
                    episode_index=ep["episode_index"],
                    episode_return=ep["return"],
                    episode_length=ep["length"],
                    video_src_path=ep.get("video_path"),
                    signals_src_path=ep.get("signals_path"),
                    signals=ep.get("signals"),
                    signal_semantics=ep.get("signal_semantics"),
                )
            )

        checkpoint_payload = {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "step": step,
            "created_at": _utc_now_iso(),
            "num_eval_episodes": len(normalized),
            "mean_return": mean_return,
            "std_return": std_return,
            "best_return": best_return,
            "median_return": median_return,
            "worst_return": worst_return,
            "rollouts": checkpoint_rollouts,
            "episodes": [
                {
                    "episode_index": ep["episode_index"],
                    "return": ep["return"],
                    "length": ep["length"],
                }
                for ep in normalized
            ],
        }
        _write_json(os.path.join(checkpoint_dir, "checkpoint.json"), checkpoint_payload)

        checkpoint_event = {
            "type": "checkpoint",
            "time": _utc_now_iso(),
            "step": step,
            "num_eval_episodes": len(normalized),
            "mean_return": mean_return,
            "std_return": std_return,
            "best_return": best_return,
            "median_return": median_return,
            "worst_return": worst_return,
            "checkpoint_dir": os.path.relpath(checkpoint_dir, self.run_dir),
        }
        _append_jsonl(self.checkpoints_path, checkpoint_event)

    def close(self, status: str = "done") -> None:
        self.update_status(status, ended=True)


class RLLogger:
    def __init__(
        self,
        log_dir: str,
        exp_name: str,
        *,
        dashboard_log_dir: Optional[str] = None,
        dashboard_run_metadata: Optional[Dict[str, Any]] = None,
        dashboard_hparams: Optional[Dict[str, Any]] = None,
        dashboard_capabilities: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the TensorBoard SummaryWriter.

        Args:
            log_dir: Base directory for logs (e.g., 'runs')
            exp_name: Name of the experiment/algorithm
            dashboard_log_dir: Optional base dir for dashboard-schema artifacts.
                If omitted, the logger checks RL_DASHBOARD_LOG_DIR env var.
        """
        # Append a timestamp so multiple runs of the same experiment don't overwrite each other
        self.run_name = f"{exp_name}_{int(time.time())}"
        self.writer = SummaryWriter(os.path.join(log_dir, self.run_name))
        print(f"Logging to {self.writer.logdir}")

        self.dashboard_writer: Optional[DashboardRunWriter] = None
        dashboard_root = dashboard_log_dir or os.environ.get("RL_DASHBOARD_LOG_DIR")
        if dashboard_root:
            env_guess, algo_guess = _infer_env_and_algorithm(exp_name)
            metadata = dashboard_run_metadata or {}
            self.dashboard_writer = DashboardRunWriter(
                base_dir=dashboard_root,
                run_id=self.run_name,
                name=metadata.get("name", self.run_name),
                algorithm=metadata.get("algorithm", algo_guess),
                env_id=metadata.get("env_id", env_guess),
                seed=metadata.get("seed"),
                hparams=dashboard_hparams or metadata.get("hparams"),
                capabilities=dashboard_capabilities or metadata.get("capabilities"),
                extra_metadata=metadata.get("extra"),
            )
            logger.info(
                f"Dashboard schema logging enabled: {self.dashboard_writer.run_dir}"
            )

    def log_episode(self, global_step: int, reward: float, length: int):
        """Logs episode-level metrics (how well the agent is doing)."""
        self.writer.add_scalar("Rollout/Episode_Reward", reward, global_step)
        self.writer.add_scalar("Rollout/Episode_Length", length, global_step)

        if self.dashboard_writer is not None:
            self.dashboard_writer.log_episode_end(
                step=global_step,
                episode_return=reward,
                episode_length=length,
            )

    def log_train_metrics(self, global_step: int, metrics: Dict[str, Any]):
        """
        Logs a dictionary of training metrics (losses, entropy, lr, etc.).
        Metrics will automatically be grouped under the 'Train/' prefix in TensorBoard.
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"Train/{key}", value, global_step)

        if self.dashboard_writer is not None:
            self.dashboard_writer.log_train_metrics(step=global_step, metrics=metrics)

    def log_speed(self, global_step: int, steps_done: int, start_time: float):
        """
        Calculates and logs the Steps Per Second (SPS).

        Args:
            global_step: Current total environment steps
            steps_done: How many steps have been taken since start_time
            start_time: Wall-clock time when training started
        """
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            sps = int(steps_done / elapsed_time)
            self.writer.add_scalar("Time/SPS", sps, global_step)

            if self.dashboard_writer is not None:
                self.dashboard_writer.log_train_metrics(
                    step=global_step,
                    metrics={"sps": sps},
                    wall_time_s=elapsed_time,
                )

    def _load_video(self, video_path: str):
        """
        Loads a video file and converts it to a format suitable for TensorBoard.

        Args:
            video_path: Path to the video file (e.g., .mp4)
        Returns:
            A numpy array of shape (num_frames, height, width, channels) with pixel values in [0, 255].
        """

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV format) to RGB (TensorBoard format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        video = np.array(frames)

        if video.size == 0:
            raise ValueError(f"Video file has no frames: {video_path}")

        # Convert to (batch, num_frames, channels, height, width)
        video = np.transpose(video, (0, 3, 1, 2))[None, ...]
        return video

    def _build_dashboard_rollouts_from_summary(
        self,
        *,
        global_step: int,
        video_exp_folder: str,
        num_episodes: int,
    ) -> list[Dict[str, Any]]:
        summary = VideoLogger.load_eval_summary(video_exp_folder, global_step)

        by_episode: Dict[int, Dict[str, Any]] = {}
        if summary and isinstance(summary.get("episodes"), list):
            for ep in summary["episodes"]:
                try:
                    ep_idx = int(ep.get("episode_index"))
                except (TypeError, ValueError):
                    continue
                by_episode[ep_idx] = ep

        if by_episode:
            candidate_indices = sorted(by_episode.keys())
        else:
            candidate_indices = list(range(num_episodes))

        rollout_entries: list[Dict[str, Any]] = []
        for ep in candidate_indices:
            base_video_path = VideoLogger.get_video_filename(video_exp_folder, global_step, ep)
            summary_ep = by_episode.get(ep, {})

            episode_return = summary_ep.get("return")
            if episode_return is not None:
                episode_return = float(_coerce_scalar(episode_return))

            episode_length = summary_ep.get("length")
            if episode_length is not None:
                episode_length = int(_coerce_scalar(episode_length))

            video_path = summary_ep.get("video_path") or base_video_path
            if video_path and not os.path.isabs(video_path):
                video_path = os.path.join(video_exp_folder, video_path)
            if video_path and not os.path.exists(video_path):
                video_path = None

            signals_path = summary_ep.get("signals_path")
            if signals_path and not os.path.isabs(signals_path):
                signals_path = os.path.join(video_exp_folder, signals_path)
            if signals_path and not os.path.exists(signals_path):
                signals_path = None

            rollout_entries.append(
                {
                    "episode_index": ep,
                    "return": episode_return,
                    "length": episode_length,
                    "video_path": video_path,
                    "signals_path": signals_path,
                    "signal_semantics": summary_ep.get("signal_semantics"),
                }
            )

        return rollout_entries

    def log_video(self, global_step: int, video_exp_folder: str, num_episodes: int):
        """
        Logs a video to TensorBoard.

        Args:
            global_step: Current total environment steps
            video_exp_folder: Folder containing evaluation videos
            num_episodes: Number of episodes expected at this checkpoint
        """
        for ep in range(num_episodes):
            video_path = VideoLogger.get_video_filename(
                video_exp_folder, global_step, ep
            )
            if not os.path.exists(video_path):
                logger.warning(
                    f"Video file {video_path} does not exist and cannot be logged."
                )
                continue
            video = self._load_video(video_path)
            self.writer.add_video(f"Evaluation/episode-{ep}", video, global_step)

        if self.dashboard_writer is not None:
            rollout_entries = self._build_dashboard_rollouts_from_summary(
                global_step=global_step,
                video_exp_folder=video_exp_folder,
                num_episodes=num_episodes,
            )
            if rollout_entries:
                self.dashboard_writer.log_checkpoint(global_step, rollout_entries)

    def close(self):
        """Closes the TensorBoard writer to ensure all data is flushed to disk."""
        if self.dashboard_writer is not None:
            self.dashboard_writer.close(status="done")
        self.writer.close()


class VideoLogger:
    @staticmethod
    def get_video_filename(exp_folder: str, global_step: int, episode_num: int):
        """Constructs the expected video file path for a given episode number."""
        return os.path.join(
            exp_folder, f"step-{global_step:05d}-episode-{episode_num}.mp4"
        )

    @staticmethod
    def get_signals_filename(exp_folder: str, global_step: int, episode_num: int):
        return os.path.join(
            exp_folder, f"step-{global_step:05d}-episode-{episode_num}-signals.npz"
        )

    @staticmethod
    def get_eval_summary_filename(exp_folder: str, global_step: int):
        return os.path.join(exp_folder, f"step-{global_step:05d}-summary.json")

    @staticmethod
    def load_eval_summary(exp_folder: str, global_step: int) -> Optional[Dict[str, Any]]:
        summary_path = VideoLogger.get_eval_summary_filename(exp_folder, global_step)
        if not os.path.exists(summary_path):
            return None

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read eval summary from {summary_path}: {exc}")
            return None

    def __init__(
        self,
        env_name: str,
        exp_folder: str,
        num_eval_episodes: int = 4,
    ):

        # Configuration
        self.env_name = env_name
        self.num_eval_episodes = num_eval_episodes
        self.exp_folder = exp_folder
        os.makedirs(self.exp_folder, exist_ok=True)

    def record_evaluation(self, policy, global_step: int = 0):

        # Create environment with recording capabilities
        env = gym.make(
            self.env_name, render_mode="rgb_array"
        )  # rgb_array needed for video recording

        # Add video recording for every episode
        env = RecordVideo(
            env,
            video_folder=self.exp_folder,  # Folder to save videos
            name_prefix=f"step-{global_step:05d}",  # Prefix for video filenames
            episode_trigger=lambda x: True,  # Record every episode
        )

        # Add episode statistics tracking
        env = RecordEpisodeStatistics(env, buffer_length=self.num_eval_episodes)

        logger.info(f"Starting evaluation for {self.num_eval_episodes} episodes...")
        logger.info(f"Videos will be saved to: {self.exp_folder}/")

        episode_summaries: list[Dict[str, Any]] = []

        for episode_num in range(self.num_eval_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            step_rewards: list[float] = []

            episode_over = False
            while not episode_over:
                # Replace this with your trained agent's policy
                action = policy.get_action(obs, sample=False)

                obs, reward, terminated, truncated, info = env.step(action)
                reward_f = float(_coerce_scalar(reward))
                episode_reward += reward_f
                step_rewards.append(reward_f)
                step_count += 1

                episode_over = terminated or truncated

            signals_path = VideoLogger.get_signals_filename(
                self.exp_folder, global_step, episode_num
            )
            np.savez_compressed(
                signals_path,
                step_reward=np.asarray(step_rewards, dtype=np.float32),
            )

            video_path = VideoLogger.get_video_filename(
                self.exp_folder, global_step, episode_num
            )
            summary = {
                "episode_index": episode_num,
                "return": episode_reward,
                "length": step_count,
                "video_path": video_path,
                "signals_path": signals_path,
                "available_signals": ["step_reward"],
                "signal_semantics": {},
            }
            episode_summaries.append(summary)

            print(
                f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}"
            )

        # Forces videos to be saved
        env.close()

        returns = np.asarray([ep["return"] for ep in episode_summaries], dtype=np.float32)
        lengths = np.asarray([ep["length"] for ep in episode_summaries], dtype=np.float32)

        # Print summary statistics
        logger.info("\nEvaluation Summary:")
        logger.info(f"Episode durations: {lengths.tolist()}")

        # Calculate some useful metrics
        avg_reward = float(np.mean(returns)) if len(returns) else 0.0
        avg_length = float(np.mean(lengths)) if len(lengths) else 0.0
        std_reward = float(np.std(returns)) if len(returns) else 0.0

        logger.info(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Average episode length: {avg_length:.1f} steps")
        logger.info(
            f"Success rate: {sum(1 for r in returns if r > 0) / max(len(returns), 1):.1%}"
        )

        summary_payload = {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "type": "eval_summary",
            "global_step": int(global_step),
            "created_at": _utc_now_iso(),
            "num_eval_episodes": int(self.num_eval_episodes),
            "episodes": episode_summaries,
        }
        summary_path = VideoLogger.get_eval_summary_filename(
            self.exp_folder, global_step
        )
        _write_json(summary_path, summary_payload)

        return episode_summaries
