"""Perda simulada de detecção da pessoa no treino das dobras por sujeito (Etapa 4C).

Intervenção exploratória e não canônica de robustez de pose, aplicada só às
janelas do treino de desenvolvimento do braço A. Cada janela amostrada é
sorteada de forma independente: com probabilidade `DROPOUT_PROBABILITY`,
exatamente `MASKED_FRAMES` quadros consecutivos da janela de `WINDOW_FRAMES`
viram detecções ausentes, numa posição uniforme entre os deslocamentos
válidos.

A ausência é imposta na pose bruta, numa cópia de `PoseArrays`, no mesmo
formato que a extração grava quando o YOLO-Pose não encontra ninguém
(`person_found` falso, keypoints e bbox zerados). As 134 features são então
recalculadas pelo pipeline inalterado (imputação, dt efetivo e cinemática)
sobre o prefixo causal do vídeo até o fim da janela, de modo que o
forward-fill e o dt efetivo enxergam a última observação real anterior à
janela e nada posterior a ela. Os HDF5, o cache de features limpas e o código
de pose não mudam.

O sorteio usa um gerador próprio, semeado por (`RANDOM_STREAM_KEY`, seed,
índice da janela, visita da janela), e não consome o RNG global do NumPy nem
o do PyTorch: inicialização do modelo e embaralhamento do treino ficam
idênticos aos do controle da Etapa 4B.
"""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd

from gatefall.config import WINDOW_FRAMES
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import window_frame_indices
from gatefall.pose.kinematics import build_pose_features_from_arrays
from gatefall.pose.loading import PoseArrays
from gatefall.train.internal_folds.folds import FeatureLoader

RawPoseLoader = Callable[[str], PoseArrays]

STAGE = "4C"
DROPOUT_PROBABILITY = 0.20
MASKED_FRAMES = 3
RANDOM_STREAM_KEY = 0x4C
MIN_POSITIVE_FOLDS = 2
MIN_FOLD_DELTA = -0.02

DECISION_RULE: dict[str, object] = {
    "metric": "retained_macro_f1_restricted",
    "comparison": "delta pareado por dobra contra o controle da Etapa 4B (mesma dobra, seed, receita e estatística)",
    "min_positive_folds": MIN_POSITIVE_FOLDS,
    "min_delta": MIN_FOLD_DELTA,
    "pooled": False,
    "statement": (
        "Critério exploratório congelado antes dos treinos: delta de macro-F1 restrita positivo em pelo menos duas "
        "das três dobras e nenhum delta abaixo de -0,02. Não é teste de significância nem validação do alvo de "
        "0,800; val e test oficiais não entram em nenhuma decisão."
    ),
}


def intervention_record(seed: int) -> dict[str, object]:
    return {
        "name": "person_detection_dropout",
        "stage": STAGE,
        "canonical": False,
        "scope": "development_train",
        "probability": DROPOUT_PROBABILITY,
        "masked_frames": MASKED_FRAMES,
        "window_frames": WINDOW_FRAMES,
        "position": (
            "uniforme entre os deslocamentos da janela cujos quadros mascarados são quadros reais distintos do vídeo; "
            "janelas com fim antes do quadro MASKED_FRAMES - 1 não têm posição válida e ficam limpas"
        ),
        "representation": (
            "cópia de PoseArrays com person_found falso e keypoints e bbox zerados nos quadros mascarados, antes da "
            "imputação, do dt efetivo e das 134 features"
        ),
        "causal_context": "features recalculadas sobre o prefixo do vídeo até o último quadro da janela",
        "random_stream": {
            "generator": "numpy.random.default_rng (PCG64)",
            "seed_sequence": ["RANDOM_STREAM_KEY", "seed", "window_index", "visit"],
            "random_stream_key": RANDOM_STREAM_KEY,
            "seed": seed,
        },
        "retained_evaluation": "limpa, idêntica à da Etapa 4B",
        "decision_rule": DECISION_RULE,
    }


def valid_offsets(k_end: int) -> range:
    """Deslocamentos na janela em que os `MASKED_FRAMES` quadros são reais, sem a replicação de borda do início."""
    return range(max(0, WINDOW_FRAMES - 1 - k_end), WINDOW_FRAMES - MASKED_FRAMES + 1)


def draw_dropout(seed: int, window_index: int, visit: int, k_end: int) -> int | None:
    """Deslocamento sorteado para a janela nesta visita, ou None quando ela fica limpa."""
    rng = np.random.default_rng([RANDOM_STREAM_KEY, seed, window_index, visit])
    if rng.random() >= DROPOUT_PROBABILITY:
        return None
    offsets = valid_offsets(k_end)
    if not offsets:
        return None
    return offsets[int(rng.integers(len(offsets)))]


def mask_person_detections(pose: PoseArrays, frames: np.ndarray) -> PoseArrays:
    keypoints = pose.keypoints.copy()
    bbox = pose.bbox.copy()
    person_found = pose.person_found.copy()
    keypoints[frames] = 0.0
    bbox[frames] = 0.0
    person_found[frames] = False
    return replace(pose, keypoints=keypoints, bbox=bbox, person_found=person_found)


def causal_prefix(pose: PoseArrays, k_end: int) -> PoseArrays:
    stop = k_end + 1
    return replace(
        pose,
        keypoints=pose.keypoints[:stop],
        bbox=pose.bbox[:stop],
        person_found=pose.person_found[:stop],
        k=stop,
    )


def masked_window_features(pose: PoseArrays, k_end: int, offset: int) -> np.ndarray:
    if offset not in valid_offsets(k_end):
        raise ValueError(f"deslocamento {offset} inválido para a janela que termina em {k_end}")
    indices = window_frame_indices(k_end, pose.k)
    masked = mask_person_detections(causal_prefix(pose, k_end), indices[offset : offset + MASKED_FRAMES])
    features, _ = build_pose_features_from_arrays(masked)
    return features[indices].astype(np.float32)


class DetectionDropoutWindowDataset(PoseWindowDataset):
    """Janelas do treino de desenvolvimento com perda simulada de detecção; o índice e os rótulos são os limpos."""

    def __init__(
        self,
        frames: pd.DataFrame,
        split: str,
        stride: int,
        feature_loader: FeatureLoader,
        raw_pose_loader: RawPoseLoader,
        seed: int,
    ) -> None:
        super().__init__(frames, split, stride, feature_loader)
        self._raw_pose_loader = raw_pose_loader
        self._raw_pose_cache: dict[str, PoseArrays] = {}
        self._seed = seed
        self._visits = np.zeros(len(self), dtype=np.int64)

    def _raw_pose(self, video_id: str, n_frames: int) -> PoseArrays:
        if video_id not in self._raw_pose_cache:
            pose = self._raw_pose_loader(video_id)
            if pose.k != n_frames:
                raise ValueError(f"{video_id}: pose bruta tem {pose.k} quadros, a grade temporal tem {n_frames}")
            self._raw_pose_cache[video_id] = pose
        return self._raw_pose_cache[video_id]

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, tuple[str, int]]:
        visit = int(self._visits[index])
        self._visits[index] += 1
        row = self._windows.iloc[index]
        k_end = int(row["k_end"])
        offset = draw_dropout(self._seed, index, visit, k_end)
        if offset is None:
            return super().__getitem__(index)
        video_id = str(row["video_id"])
        window = masked_window_features(self._raw_pose(video_id, int(row["n_frames"])), k_end, offset)
        return window, int(row["label"]), (video_id, k_end)
