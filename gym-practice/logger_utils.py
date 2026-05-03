import os
import time
from typing import Any, Dict

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
        run_name = f"{exp_name}_{int(time.time())}"
        self.writer = SummaryWriter(os.path.join(log_dir, run_name))
        print(f"Logging to {self.writer.logdir}")

    def log_episode(self, global_step: int, reward: float, length: int):
        """Logs episode-level metrics (how well the agent is doing)."""
        self.writer.add_scalar("Rollout/Episode_Reward", reward, global_step)
        self.writer.add_scalar("Rollout/Episode_Length", length, global_step)

    def log_train_metrics(self, global_step: int, metrics: Dict[str, Any]):
        """
        Logs a dictionary of training metrics (losses, entropy, lr, etc.).
        Metrics will automatically be grouped under the 'Train/' prefix in TensorBoard.
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"Train/{key}", value, global_step)

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

    def close(self):
        """Closes the TensorBoard writer to ensure all data is flushed to disk."""
        self.writer.close()
