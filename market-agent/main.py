from market_agent.data import (
    DatasetSplitConfig,
    ensure_datetime_sorted,
    filter_ticker,
    load_nasdaq_dataset,
    split_train_test,
)
from market_agent.env import MarketEnv, MarketEnvConfig


def main() -> None:
    dataset = load_nasdaq_dataset(split="train")
    dataset = filter_ticker(dataset, "AAPL")
    dataset = ensure_datetime_sorted(dataset, date_column="date")
    splits = split_train_test(
        dataset,
        config=DatasetSplitConfig(test_size=0.2, shuffle=False),
    )
    env = MarketEnv(splits["train"], config=MarketEnvConfig(window_size=30))
    observation, info = env.reset()
    print("Observation shape:", observation.shape)
    print("Info:", info)


if __name__ == "__main__":
    main()
