import datetime
import os
import time
from typing import Any, Dict

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from loguru import logger
from tensorboardX import SummaryWriter


class RLLogger:
    def __init__(self, log_dir: str, exp_name: str):
        """
        Initializes the TensorBoard SummaryWriter.

        Args:
            log_dir: Base directory for logs (e.g., 'runs')
            exp_name: Name of the experiment/algorithm
        """
        # Append a timestamp so multiple runs of the same experiment don't overwrite each other
        datestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{exp_name}/{datestr}"
        self.writer = SummaryWriter(os.path.join(log_dir, self.run_name))
        print(f"Logging to {self.writer.logdir}")

    def log_episode(self, global_step: int, reward: float, length: int):
        """Logs episode-level metrics (how well the agent is doing)."""
        self.writer.add_scalar("rollout/episode_reward", reward, global_step)
        self.writer.add_scalar("rollout/episode_length", length, global_step)

    def log_train_metrics(self, global_step: int, metrics: Dict[str, Any]):
        """
        Logs a dictionary of training metrics (losses, entropy, lr, etc.).
        Metrics will automatically be grouped under the 'train/' prefix in TensorBoard.
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, global_step)

    def log_validation_steps(self, global_step: int, metrics: Dict[float, Any]):
        """Logs validation losses keyed by diffusion timestep."""
        for timestep, value in metrics.items():
            self.writer.add_scalar(
                f"val_loss/t={float(timestep):.2f}", value, global_step
            )

    def log_validation_psnrs(self, global_step: int, metrics: Dict[float, Any]):
        """Logs validation x1-prediction PSNRs (dB) keyed by diffusion timestep."""
        for timestep, value in metrics.items():
            self.writer.add_scalar(
                f"val_psnr/t={float(timestep):.2f}", value, global_step
            )

    def log_reconstructions(
        self,
        global_step: int,
        x_true,
        x_preds: Dict[float, Any],
    ) -> None:
        """Log ground-truth and x1-prediction images to TensorBoard.

        Accepts any array-like supported by ``np.asarray`` (e.g. ``mlx.array``).

        Args:
            global_step: Current training step.
            x1_true: ``(B, H, W, C)`` ground-truth frames in ``[-1, 1]``.
            x1_preds: ``{timestep: (B, H, W, C)}`` one-step x1 predictions in
                ``[-1, 1]``, keyed by the diffusion timestep they were evaluated at.
        """

        def _to_tb(arr) -> np.ndarray:
            """Array-like ``(B, H, W, C)`` in ``[-1, 1]`` → ``(B, C, H, W)`` in ``[0, 1]``."""
            arr = np.asarray(arr)
            arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)  # broadcast grayscale → RGB
            return arr.transpose(0, 3, 1, 2)  # (B, C, H, W)

        self.writer.add_images(
            "reconstruction/ground_truth", _to_tb(x_true), global_step
        )
        for t, pred in sorted(x_preds.items()):
            self.writer.add_images(
                f"reconstruction/pred_t={t:.2f}", _to_tb(pred), global_step
            )

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

    def _load_video(self, video_path: str):
        """
        Loads a video file and converts it to a format suitable for TensorBoard.

        Args:
            video_path: Path to the video file (e.g., .mp4)
        Returns:
            A numpy array of shape (num_frames, height, width, channels) with pixel values in [0, 255].
        """

        frames = np.asarray(list(iio.imiter(video_path)), dtype=np.uint8)
        # Convert to (batch, num_frames, channels, height, width)
        video = np.transpose(frames, (0, 3, 1, 2))[None, ...]
        return video

    def log_video(self, global_step: int, video_exp_folder: str, num_episodes: int):
        """
        Logs a video to TensorBoard.

        Args:
            video_path: Path to the video file (e.g., .mp4) to log
            global_step: Current total environment steps
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

    def close(self):
        """Closes the TensorBoard writer to ensure all data is flushed to disk."""
        self.writer.close()


class VideoLogger:
    @staticmethod
    def get_video_filename(exp_folder: str, global_step: int, episode_num: int):
        """Constructs the expected video file path for a given episode number."""
        return os.path.join(
            exp_folder, f"step-{global_step:05d}-episode-{episode_num}.mp4"
        )

    def __init__(
        self,
        env_name: str,
        exp_folder: str,
        num_eval_episodes: int = 4,
    ):

        # Configuration
        self.env_name = env_name
        self.num_eval_episodes = 4
        self.exp_folder = exp_folder

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

        for episode_num in range(self.num_eval_episodes):
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0

            episode_over = False
            while not episode_over:
                # Replace this with your trained agent's policy
                action = policy.get_action(obs, sample=False)

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                episode_over = terminated or truncated

            print(
                f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}"
            )

        # Print summary statistics
        logger.info("\nEvaluation Summary:")
        logger.info(f"Episode durations: {list(env.time_queue)}")
        logger.info(f"Episode rewards: {list(env.return_queue)}")
        logger.info(f"Episode lengths: {list(env.length_queue)}")

        # Calculate some useful metrics
        avg_reward = np.mean(env.return_queue)
        avg_length = np.mean(env.length_queue)
        std_reward = np.std(env.return_queue)

        logger.info(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Average episode length: {avg_length:.1f} steps")
        logger.info(
            f"Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}"
        )

        # Forces videoes to be saved
        env.close()
