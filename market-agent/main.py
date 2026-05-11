import numpy as np

from market_agent.data import (
    DatasetSplitConfig,
    ensure_datetime_sorted,
    filter_ticker,
    load_nasdaq_dataset,
    split_train_test,
)
from market_agent.env import MarketEnv, MarketEnvConfig


def run_random_episodes(
    env: MarketEnv,
    *,
    episodes: int = 2,
    max_steps: int = 10,
    seed: int = 7,
) -> None:
    action_labels = {0: "sell", 1: "hold", 2: "buy"}
    rng = np.random.default_rng(seed)

    for episode in range(1, episodes + 1):
        _, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        print(f"Episode {episode}")
        print("step  idx  action  reward     position   cash")
        for step in range(1, max_steps + 1):
            action = int(rng.integers(0, env.action_space.n))
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(
                f"{step:>4}  {info['index']:>3}  {action_labels[action]:<6}"
                f"  {reward:>8.2f}  {info['position']:>9}"
                f"  {info['cash']:>7.2f}"
            )
            if terminated or truncated:
                break
        print(f"Total reward: {total_reward:.2f}\n")


def main() -> None:
    dataset = load_nasdaq_dataset(split="train", data_files="trade_data_2019_2023.csv")
    dataset = filter_ticker(dataset, ticker="AAPL", ticker_column_name="tic")
    dataset = ensure_datetime_sorted(dataset, date_column="date")
    splits = split_train_test(
        dataset,
        config=DatasetSplitConfig(test_size=0.2, shuffle=False),
    )
    env = MarketEnv(splits["train"], config=MarketEnvConfig(window_size=30))
    observation, info = env.reset()
    print("Observation shape:", observation.shape)
    print("Info:", info)
    print()
    run_random_episodes(env, episodes=2, max_steps=12, seed=7)


if __name__ == "__main__":
    main()
