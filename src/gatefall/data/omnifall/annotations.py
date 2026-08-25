"""Operações reutilizáveis sobre configurações de anotação do OmniFall."""

import os
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import DatasetDict, load_dataset


def load_annotation_config(
    dataset_repo_id: str,
    config_name: str,
    revision: str,
) -> DatasetDict:
    return cast(
        DatasetDict,
        load_dataset(dataset_repo_id, config_name, revision=revision),
    )


def annotation_split_dataframe(config: DatasetDict, split: str) -> pd.DataFrame:
    return cast(pd.DataFrame, config[split].to_pandas())


def write_annotation_csv(dataframe: pd.DataFrame, path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"skip {path} (já existe, use --force para sobrescrever)")
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    print(f"{path}: {len(dataframe)} linhas, colunas={list(dataframe.columns)}")
