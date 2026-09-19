"""Hash de arquivos, compartilhado entre o pacote e os scripts de dados."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.util.hashing import hash_pandas_object


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe(dataframe: pd.DataFrame) -> str:
    # hash do conteúdo, não dos bytes do arquivo: parquet não garante bytes
    # estáveis entre gravações — quem chama deve passar um frame já ordenado
    # de forma determinística.
    digest = hashlib.sha256()
    digest.update(hash_pandas_object(dataframe, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
