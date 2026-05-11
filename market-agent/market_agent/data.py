"""Dataset loading and preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Configuration for train/test split."""

    test_size: float = 0.2
    shuffle: bool = False
    seed: int = 42


def load_nasdaq_dataset(
    *,
    data_dir: str = "data",
    data_files: Optional[str] = None,
    split: Optional[str] = None,
    streaming: bool = False,
) -> Dataset | DatasetDict:
    """Load the NASDAQ dataset from Hugging Face.

    Args:
        data_dir: Local cache directory.
        data_files: Optional data file name to load from the dataset.
        split: Optional dataset split name to load directly.
        streaming: Whether to stream the dataset.
    """

    dataset = load_dataset(
        "benstaf/nasdaq_2013_2023",
        cache_dir=data_dir,
        data_files=data_files,
        split=split,
        streaming=streaming,
    )
    return dataset


def filter_ticker(
    dataset: Dataset, ticker: str, ticker_column_name="ticker"
) -> Dataset:
    """Filter a dataset to a single ticker symbol."""

    ticker_norm = ticker.strip().upper()
    filtered = dataset.filter(
        lambda row: row.get(ticker_column_name) == ticker_norm
    )
    print("Filtered samples:", len(filtered))
    return filtered


def split_train_test(
    dataset: Dataset,
    *,
    config: DatasetSplitConfig | None = None,
) -> DatasetDict:
    """Split a dataset into train/test splits.

    For time-series data, keep shuffle=False to avoid leakage.
    """

    config = config or DatasetSplitConfig()
    return dataset.train_test_split(
        test_size=config.test_size,
        shuffle=config.shuffle,
        seed=config.seed,
    )


def ensure_datetime_sorted(
    dataset: Dataset,
    *,
    date_column: str = "date",
) -> Dataset:
    """Sort dataset rows by date column if present."""

    if date_column not in dataset.column_names:
        return dataset
    df = dataset.to_pandas()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    return Dataset.from_pandas(df, preserve_index=False)


def main() -> None:
    dataset = load_nasdaq_dataset(
        data_files="trade_data_2019_2023.csv",
        split="train",
    )
    print("Columns:", dataset.column_names)


if __name__ == "__main__":
    main()
