from collections import deque
from pprint import pprint

import gymnasium as gym
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from tqdm import tqdm

from data import Dataset, make_env, pad_frames_to_multiple, record_rollouts
from diffusion import FlowMatchingTrainer, load_model, sample_euler, save_model
from logger_utils import VideoLogger
from ppo import Batch, Categorical, PPOConfig, RunningNorm, make_net, update
from unet import UNet3D
from vae import WaveletVAE, decode_latents, encode_clips, load_vae
from video_utils import save_video_grid


class WorldModel:
    def __init__(self, model: UNet3D, vae: WaveletVAE | None = None):
        self.model = model
        self.vae = vae

    @property
    def null_action(self) -> int:
        return self.model.null_action

    @property
    def max_context_size(self) -> int:
        return self.model.max_context_size

    def decoder(self, latents: mx.array) -> mx.array:
        return decode_latents(self.vae, latents)

    def image_encoder(self, x: mx.array) -> mx.array:
        if self.vae is not None:
            return encode_clips(self.vae, x)
        return x

    def encode(self, x: mx.array, t: mx.array, context: mx.array) -> mx.array:
        x = self.image_encoder(x)
        xmid, _, _ = self.model.encode(x, t, context=context)
        return xmid


def collect_and_encode_rollout(
    env: gym.Env,
    model: WorldModel,
    *,
    num_steps: int = 256,
    seed: int = 0,
    encode_batch_size: int = 64,
) -> tuple[Dataset, tuple[mx.array, np.ndarray, np.ndarray, np.ndarray]]:
    """Roll out ``env`` and encode it into per-step world-model state embeddings.

    Collects a raw ``(frames, actions, rewards)`` stream via ``rollout_env`` and
    slides a ``model.max_context_size``-frame window over it (respecting episode
    boundaries) to produce a bottleneck embedding per step. Each window is fed to
    ``model.encode`` with ``t=1`` and its context's last action set to the NULL
    action, yielding the deterministic "state embedding for acting" described in
    ``UNet3D.encode``.

    Window ``[s, s+L)`` represents the state at frame ``s+L-1``; its aligned
    ``action``/``reward`` are the ones recorded at that same frame.

    If ``model.vae`` is set, frames are latent-encoded with ``encode_clips``
    before being passed to the world model (latent-space models). The returned
    ``Dataset`` carries the same encoder so its raw-pixel clips are encoded to
    latents on the fly when sampled for world-model training.

    The embedding is the raw ``xmid`` bottleneck, intentionally left spatial
    (not pooled/flattened) so a downstream policy can consume it convolutionally.

    Args:
        encode_batch_size: number of windows encoded per ``model.encode`` call.

    Returns:
        rollout_dataset: ``Dataset`` over the raw ``(frames, actions, rewards)``
            clips, for training the world model on real-env transitions.
        policy_rollouts: ``(embeddings, actions, rewards, dones)`` where
            - embeddings: ``(num_windows, S/4, H/4, W/4, 4 * base_channels)``
              float32 state embeddings,
            - actions: ``(num_windows,)`` int32 — action taken from each state,
            - rewards: ``(num_windows,)`` float32 — reward received at each state,
            - dones: ``(num_windows,)`` float32 — 1.0 if the window ends an
              episode (terminal), else 0.0. The windows form a single ordered
              stream of concatenated episodes (stride-1, no boundary crossing),
              so ``dones`` is the GAE reset mask for that stream.
    """
    context_size = model.max_context_size + 1

    videos, action_sequence, reward_sequence, save_dir, done_sequence = record_rollouts(
        env=env,
        num_steps=num_steps,
        seed=seed,
        clip_length=context_size,
        clip_stride=1,
        save_to_disk=True,
        save_dir="/tmp/dreamer/test",
        recompute=True,
        pad_multiple=32,
        return_dones=True,
    )

    rollout_dataset = Dataset(save_dir, encoder=model.image_encoder)

    _embeddings: list[mx.array] = []
    null_action = model.null_action
    for batch_start in tqdm(
        range(0, len(videos), encode_batch_size), desc="encoding rollout"
    ):
        video_batch = videos[batch_start : batch_start + encode_batch_size]
        video_batch = mx.array(video_batch)

        action_batch = action_sequence[batch_start : batch_start + encode_batch_size]
        # Last action is the one we are about to choose -> NULL it out so the
        # embedding is the pre-action state.
        action_batch = mx.array(action_batch)
        action_batch[:, -1] = null_action

        t = mx.ones((video_batch.shape[0],))
        embs = model.encode(video_batch, t, context=action_batch)
        _embeddings.append(embs)

    embeddings = mx.concatenate(_embeddings, axis=0)

    # Align action/reward/done to the frame each window ends on (s + L - 1).
    aligned_actions = action_sequence[:, -1].astype(np.int32)
    aligned_rewards = reward_sequence[:, -1].astype(np.float32)
    aligned_dones = done_sequence[:, -1].astype(np.float32)

    policy_rollouts = (embeddings, aligned_actions, aligned_rewards, aligned_dones)

    return rollout_dataset, policy_rollouts


def update_world_model(
    trainer: FlowMatchingTrainer,
    dataset: Dataset,
    *,
    num_steps: int = 100,
    batch_size: int = 8,
) -> float:
    """Run ``num_steps`` flow-matching updates on real-env clips.

    Samples latent-encoded ``(clips, actions, rewards)`` batches from ``dataset``
    and steps the (compiled) flow-matching trainer, mutating the world model and
    its EMA shadow in place. Returns the mean flow loss over the updates.
    """
    total_loss = mx.array(0.0)
    pbar = tqdm(range(num_steps), desc="world model update")
    for step in pbar:
        batch, actions, rewards = dataset.sample_train_batch(batch_size)
        loss, _reward_loss = trainer.compiled_train_step(batch, actions, rewards)
        total_loss = total_loss + loss
        mx.async_eval(total_loss)
        if step % 10 == 0:
            pbar.set_postfix(loss=f"{float(loss):.4f}")

    return float(total_loss) / num_steps


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over a single ordered stream of concatenated episodes.

    Mirrors the reverse scan in ``ppo.collect_rollout`` but for a flat ``(N,)``
    sequence whose episode resets are marked by ``dones`` (1.0 on terminal
    steps). The successor of step ``t`` is step ``t+1`` (windows are contiguous
    within an episode); ``dones[t]`` zeroes the bootstrap at episode ends. A
    non-terminal final step is treated as truncated with a zero bootstrap.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def build_policy_agent(embedding_dim: int, num_actions: int, cfg: PPOConfig):
    """Flat-MLP actor-critic over flattened world-model embeddings.

    Returns ``(policy, value_net, normalizer, policy_opt, value_opt)`` — built
    once and reused across iterations so optimizer/normalizer state persists.
    """
    normalizer = RunningNorm((embedding_dim,)) if cfg.normalize_obs else None
    policy_net = make_net(embedding_dim, num_actions, cfg, normalizer)
    value_net = make_net(embedding_dim, 1, cfg, normalizer)
    mx.eval(policy_net.parameters(), value_net.parameters())

    policy = Categorical(policy_net)
    policy_opt = optim.AdamW(learning_rate=cfg.policy_lr)
    value_opt = optim.AdamW(learning_rate=cfg.value_lr)
    return policy, value_net, normalizer, policy_opt, value_opt


def update_policy(
    policy,
    value_net,
    normalizer,
    policy_opt,
    value_opt,
    policy_rollouts: tuple[mx.array, np.ndarray, np.ndarray, np.ndarray],
    cfg: PPOConfig,
) -> dict:
    """Run a PPO update on a real-env batch of world-model embeddings.

    Flattens the spatial embeddings to vector observations, computes GAE
    advantages/returns with the current value net (obs stats frozen for the
    whole update, as in ``ppo``), runs the clipped PPO update, then folds the
    batch into the obs normalizer for the next iteration.
    """
    embeddings, actions, rewards, dones = policy_rollouts
    obs = embeddings.reshape(embeddings.shape[0], -1)

    values = np.array(value_net(obs).reshape(-1))
    advantages, returns = compute_gae(
        np.asarray(rewards),
        values,
        np.asarray(dones),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    batch = Batch(
        obs=obs,
        actions=mx.array(actions),
        returns=mx.array(returns),
        advantages=mx.array(advantages),
    )
    metrics = update(policy, value_net, policy_opt, value_opt, batch, cfg)

    if normalizer is not None:
        normalizer.update(np.array(obs))
    return metrics


class EvalAgent:
    """Stateful adapter that lets ``VideoLogger.record_evaluation`` drive an
    embedding-space policy on raw env frames.

    ``record_evaluation`` owns the eval loop: it hands us one raw observation at
    a time via ``get_action(obs, sample=False)`` and expects an ``int`` action.
    Our policy, however, acts on the *flattened world-model state embedding* of a
    ``max_context_size + 1``-frame window (see ``collect_and_encode_rollout``), so
    each call we:

    1. Preprocess the frame exactly as the training pipeline does
       (``screen / 127.5 - 1`` → window → ``pad_frames_to_multiple(.., 32)``).
    2. Append it to a rolling window of the last ``context_size`` frames, left-
       padded with the first frame until the window fills (cold start).
    3. Encode the window through the world model with ``t=1`` and an action
       context whose final entry is the NULL action, yielding the same ``xmid``
       state embedding the policy was trained on.
    4. Flatten it and defer to the wrapped policy.

    ``reset`` (called by ``record_evaluation`` at each episode start) clears the
    frame/action buffers so windows never bleed across episodes.
    """

    def __init__(self, world_model: WorldModel, policy: Categorical):
        self.world_model = world_model
        self.policy = policy
        self.context_size = world_model.max_context_size + 1
        self.null_action = world_model.null_action
        self.reset()

    def reset(self) -> None:
        self._frames: deque = deque(maxlen=self.context_size)
        # Actions taken at the (context_size - 1) frames preceding the current one.
        self._actions: deque = deque(maxlen=self.context_size - 1)

    @staticmethod
    def _preprocess(obs) -> np.ndarray:
        screen = obs["screen"] if isinstance(obs, dict) else obs
        return np.asarray(screen, dtype=np.float32) / 127.5 - 1.0

    def get_action(self, obs, sample: bool = False) -> int:
        self._frames.append(self._preprocess(obs))

        # Left-pad the window with the first available frame until it fills, then
        # pad spatially to match the world model's expected (multiple-of-32) size.
        frames = list(self._frames)
        frames = [frames[0]] * (self.context_size - len(frames)) + frames
        window = pad_frames_to_multiple(np.stack(frames, axis=0), multiple=32)

        # Action context aligned to the window: action taken at each frame, with
        # the current frame's action NULLed (it's the one we're about to choose).
        actions = list(self._actions)
        actions = [self.null_action] * (self.context_size - 1 - len(actions)) + actions
        action_ctx = np.asarray(actions + [self.null_action], dtype=np.int32)
        # print(f"{action_ctx = }")

        emb = self.world_model.encode(
            mx.array(window)[None],
            mx.ones((1,)),
            context=mx.array(action_ctx)[None],
        )
        action = self.policy.get_action(np.array(emb.reshape(-1)), sample=sample)

        self._actions.append(int(action))

        return int(action)


def dream(
    world_model: WorldModel,
    policy: Categorical,
    seed_rollout: Dataset,
    *,
    num_trajectory_steps: int = 256,
    num_parallel_dreams: int = 4,
    sample_steps: int = 16,
    grid_path: str | None = None,
    save_seeds: bool = False,
) -> tuple[mx.array, np.ndarray, np.ndarray, np.ndarray]:
    """Roll the policy forward *inside* the world model and return imagined
    experience in the same ``(embeddings, actions, rewards, dones)`` layout as
    ``collect_and_encode_rollout`` — ready to hand straight to ``update_policy``.

    Starting from ``num_parallel_dreams`` seed windows sampled from
    ``seed_rollout`` (already VAE-encoded latents, since the dataset carries the
    world model's image encoder), each dream step:

    1. Encodes the current ``context_size``-frame window at ``t=1`` with the last
       action NULLed to get the pre-action state embedding — identical to the
       real-env path in ``collect_and_encode_rollout``.
    2. Samples an action from ``policy`` for every parallel dream.
    3. Generates the next latent frame with ``sample_euler`` conditioned on that
       action and slides it into the rolling window.
    4. Re-encodes the window now ending on the new frame — last action kept (not
       NULLed, since it produced that frame) — and reads the world model's reward
       head as the imagined reward for the transition.

    Because the world model has no termination head, dreams are fixed-horizon:
    every trajectory runs the full ``num_trajectory_steps`` and is marked ``done`` only on
    its final step (a GAE reset between concatenated dreams, see ``compute_gae``).

    All work happens in latent space, so we call ``world_model.model`` directly
    rather than ``world_model.encode`` (which would re-encode the latents through
    the VAE).

    Returns ``(embeddings, actions, rewards, dones)`` with the parallel dreams
    concatenated into one ordered stream (dream 0's steps, then dream 1's, ...),
    matching the contract ``update_policy`` already consumes.
    """
    model = world_model.model
    if not getattr(model, "has_reward_head", False):
        raise ValueError(
            "dream needs a world model built with predict_reward=True to score "
            "imagined transitions"
        )

    context_size = world_model.max_context_size + 1
    null_action = world_model.null_action

    # Seed windows are already latent (the dataset carries the image encoder).
    seed_frames, seed_actions, _ = seed_rollout.sample_val_batch(num_parallel_dreams)

    if grid_path is not None and save_seeds:
        videos = world_model.decoder(seed_frames)  # (B, T, H, W, C)
        mx.eval(videos)
        _path = grid_path.replace(".mp4", "-seed.mp4")
        save_video_grid(np.array(videos), _path, grid_size=4)

    frames = seed_frames[:, -context_size:]
    actions = np.array(seed_actions[:, -context_size:], dtype=np.int32)
    batch = int(frames.shape[0])
    ones = mx.ones((batch,))

    step_embeddings: list[mx.array] = []
    step_actions: list[np.ndarray] = []
    step_rewards: list[np.ndarray] = []
    step_frames: list[mx.array] = []  # generated latent frame per step, for viz

    for _ in tqdm(range(num_trajectory_steps), desc="dreaming"):
        # 1. Pre-action state embedding: full window, last action NULLed.
        ctx = actions.copy()
        ctx[:, -1] = null_action
        xmid, _, _ = model.encode(frames, ones, context=mx.array(ctx))

        # 2. Sample an action per parallel dream from the policy.
        obs = np.array(xmid.reshape(batch, -1))
        act = np.asarray(policy.get_action(obs, sample=True), dtype=np.int32)

        # 3. Generate the next latent frame conditioned on the chosen action.
        #    Conditioning is the last max_context_size frames; the action window
        #    aligns those frames' actions and appends the new action.
        gen_actions = mx.array(np.concatenate([actions[:, 1:], act[:, None]], axis=1))
        sample = sample_euler(
            model,
            conditioning_clips=frames[:, 1:],
            actions=gen_actions,
            num_steps=sample_steps,
        )

        # Slide the rolling window forward one frame / action.
        frames = mx.concatenate([frames[:, 1:], sample[:, -1:]], axis=1)
        actions = np.concatenate([actions[:, 1:], act[:, None]], axis=1)

        # 4. Reward for the transition: window now ending on the new frame, with
        #    the action that produced it kept (matches the reward-head target).
        xmid_r, _, _ = model.encode(frames, ones, context=mx.array(actions))
        reward = model.predict_reward(xmid_r)
        mx.eval(frames, xmid, reward)

        step_embeddings.append(xmid)
        step_actions.append(act)
        step_rewards.append(np.array(reward))
        if grid_path is not None:
            step_frames.append(sample[:, -1:])

    # (T, B, ...) -> (B, T, ...) -> (B*T, ...): one ordered stream per dream,
    # dreams concatenated back-to-back so compute_gae scans each in order.
    embeddings = mx.stack(step_embeddings, axis=0).swapaxes(0, 1)
    embeddings = embeddings.reshape(batch * num_trajectory_steps, *embeddings.shape[2:])
    actions_out = np.stack(step_actions, axis=0).swapaxes(0, 1).reshape(-1)
    rewards_out = (
        np.stack(step_rewards, axis=0).swapaxes(0, 1).reshape(-1).astype(np.float32)
    )

    # Mark only each dream's final step terminal, resetting GAE between dreams.
    dones = np.zeros((batch, num_trajectory_steps), dtype=np.float32)
    dones[:, -1] = 1.0
    dones_out = dones.reshape(-1)

    # Decode the imagined latent trajectories into pixel videos and tile them
    # into a 4x4 grid for inspection.
    if grid_path is not None and step_frames:
        latents = mx.concatenate(step_frames, axis=1)  # (B, T, ...latent...)
        videos = world_model.decoder(latents)  # (B, T, H, W, C)
        mx.eval(videos)
        save_video_grid(np.array(videos), grid_path, grid_size=4)

    return embeddings, actions_out.astype(np.int32), rewards_out, dones_out


def train():

    load_dir = "logs/vizdoom-latent-rewards"
    gym_env = make_env("VizdoomBasic-v1")
    # breakpoint()
    # Online + EMA copies start from the same pretrained weights.
    model = load_model(load_dir)
    ema_model = load_model(load_dir)
    model.set_dtype(mx.bfloat16)
    ema_model.set_dtype(mx.bfloat16)

    vae = load_vae("logs/vizdoom-vae")
    world_model = WorldModel(ema_model, vae)
    ppo_config = PPOConfig(num_envs=1, update_epochs=10)
    world_model_trainer = FlowMatchingTrainer(
        model,
        ema_model,
        learning_rate=1e-5,
        reward_loss_weight=1e-1,
        weight_decay=1e-2,
        ema_decay=0.99,
    )
    experiment_dir = "logs/doom-dreamer-v1"

    print("building policy agent")
    rollout_dataset, policy_rollouts = collect_and_encode_rollout(
        gym_env, world_model, num_steps=10
    )
    embedding_dim = int(np.prod(policy_rollouts[0].shape[1:]))
    num_actions = int(gym_env.action_space.n)
    policy, value_net, normalizer, policy_opt, value_opt = build_policy_agent(
        embedding_dim, num_actions, ppo_config
    )
    print("policy agent built")

    gym_sessions = 5
    global_step = 0
    for sess in range(gym_sessions):
        rollout_dataset, policy_rollouts = collect_and_encode_rollout(
            gym_env, world_model, num_steps=1000
        )
        # train world (diffusion) model using real frames
        wm_loss = update_world_model(
            world_model_trainer, rollout_dataset, num_steps=100, batch_size=32
        )
        print(f"world model loss: {wm_loss:.4f}")

        # train policy using world model embeddings
        metrics = update_policy(
            policy,
            value_net,
            normalizer,
            policy_opt,
            value_opt,
            policy_rollouts,
            ppo_config,
        )
        global_step += ppo_config.num_iterations * ppo_config.update_epochs
        print("policy update (real):")
        pprint(metrics)

        # Dream loop: roll the policy forward inside the world model and train it on
        # the imagined experience. Each dream produces the same embedding/action/
        # reward/done tuples as the real rollout, so update_policy is reused as-is.
        num_dream_iters = 5
        dream_steps = 8 * (sess + 1)
        num_parallel_dreams = 16
        for dream_iter in range(num_dream_iters):
            dream_rollouts = dream(
                world_model,
                policy,
                rollout_dataset,
                num_trajectory_steps=dream_steps,
                num_parallel_dreams=num_parallel_dreams,
                grid_path=f"{experiment_dir}/dream_grid_{sess}-{dream_iter:03d}.mp4",
                save_seeds=dream_iter == 0,
            )
            metrics = update_policy(
                policy,
                value_net,
                normalizer,
                policy_opt,
                value_opt,
                dream_rollouts,
                ppo_config,
            )
            print(f"policy update (dream {dream_iter + 1}/{num_dream_iters}):")
            pprint(metrics)
            global_step += ppo_config.num_iterations * ppo_config.update_epochs

        # Evaluate the policy in the *real* env, recording videos via the logger.
        # frame_skip/continuous mirror make_env's doom path so eval dynamics match
        # the transitions the world model was trained on.
        eval_agent = EvalAgent(world_model, policy)
        video_logger = VideoLogger("VizdoomBasic-v1", exp_folder=experiment_dir)
        video_logger.record_evaluation(
            eval_agent,
            global_step=global_step,
            continuous=False,
        )


if __name__ == "__main__":
    train()
