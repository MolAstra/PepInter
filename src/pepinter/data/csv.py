from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from typing import Callable


class CSVDataset(Dataset):
    def __init__(
        self,
        data_path: Path | str,
        select_columns: list[str] | None = None,
        make_toy: bool = False,
        transform: Callable = None,
        split_column: str | None = None,
        split_value: str | None = None,
    ):
        data_path = Path(data_path)
        if data_path.suffix == ".parquet":
            self.df = pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            self.df = pd.read_csv(data_path)
        elif data_path.suffix == ".pkl":
            self.df = pd.read_pickle(data_path)
        else:
            raise ValueError(f"Invalid data path: {data_path}")
        if make_toy:
            self.df = self.df.sample(1000)
        if select_columns is not None:
            self.df = self.df[select_columns]
        if split_column is not None and split_value is not None:
            self.df = self.df[self.df[split_column] == split_value]

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.transform is not None:
            return self.transform(row.to_dict())
        else:
            return row.to_dict()
