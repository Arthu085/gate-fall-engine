"""Amostragem do treino balanceada por sujeito nas dobras por sujeito (Etapa 5A).

Intervenção exploratória e não canônica, aplicada só às épocas de treino do
braço A. Cada época sorteia, com reposição, exatamente o número original de
janelas do treino de desenvolvimento. A probabilidade de uma janela é
1 / (S · n_s), em que S é o número de sujeitos de desenvolvimento e n_s o
número de janelas de treino do seu sujeito; cada sujeito soma 1 / S.

O sorteio usa um gerador próprio, semeado por (`RANDOM_STREAM_KEY`, seed,
época), e não consome o RNG global do NumPy, do PyTorch nem do `random`: a
inicialização do modelo e o fluxo de dropout ficam idênticos aos do controle
da Etapa 4B. As janelas, os rótulos, a estatística, os pesos de classe e a
avaliação do retido são os do controle.
"""

import hashlib
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from gatefall.datasets.le2i import LE2I_LABEL_NAMES

STAGE = "5A"
RANDOM_STREAM_KEY = 0x5A
MIN_IMPROVEMENT = 0.03
MIN_IMPROVED_FOLDS = 2
MIN_FOLD_DELTA = -0.02

DECISION_RULE: dict[str, object] = {
    "metric": "retained_macro_f1_restricted",
    "comparison": "delta pareado por dobra contra o controle da Etapa 4B (mesma dobra, seed, receita e estatística)",
    "min_improvement": MIN_IMPROVEMENT,
    "min_improved_folds": MIN_IMPROVED_FOLDS,
    "min_delta": MIN_FOLD_DELTA,
    "pooled": False,
    "statement": (
        "Critério exploratório congelado antes dos treinos: delta de macro-F1 restrita de pelo menos +0,03 em pelo "
        "menos duas das três dobras e nenhum delta abaixo de -0,02 na dobra restante. Não é teste de significância "
        "nem validação do alvo de 0,800; val e test oficiais não entram em nenhuma decisão, e a variante segue não "
        "canônica mesmo que o critério seja atendido."
    ),
}


def intervention_record(seed: int) -> dict[str, object]:
    return {
        "name": "subject_balanced_sampling",
        "stage": STAGE,
        "canonical": False,
        "scope": "development_train_epochs",
        "replacement": True,
        "draws_per_epoch": "número original de janelas do treino de desenvolvimento (stride 4)",
        "window_probability": "1 / (S * n_s): S sujeitos de desenvolvimento, n_s janelas de treino do sujeito",
        "subject_probability": "1 / S para cada sujeito de desenvolvimento",
        "class_balancing": False,
        "augmentation": False,
        "random_stream": {
            "generator": "numpy.random.default_rng (PCG64)",
            "seed_sequence": ["RANDOM_STREAM_KEY", "seed", "epoch"],
            "random_stream_key": RANDOM_STREAM_KEY,
            "seed": seed,
        },
        "unchanged": (
            "janelas e rótulos limpos, estatística da dobra, pesos de classe do treino limpo completo, receita, "
            "inicialização, 30 épocas e checkpoint da última época"
        ),
        "clean_train_evaluation": "final.train avaliado sequencialmente sobre todas as janelas limpas, sem o sampler",
        "retained_evaluation": "limpa, idêntica à da Etapa 4B",
        "decision_rule": DECISION_RULE,
    }


def window_probabilities(window_subjects: Sequence[int] | np.ndarray) -> np.ndarray:
    subjects = np.asarray(window_subjects, dtype=np.int64)
    if subjects.size == 0:
        raise ValueError("a amostragem por sujeito exige ao menos uma janela de treino")
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse].astype(np.float64))


def draw_epoch(probabilities: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng([RANDOM_STREAM_KEY, seed, epoch])
    return rng.choice(len(probabilities), size=len(probabilities), replace=True, p=probabilities)


class SubjectBalancedSampler(Sampler[int]):
    """Sorteio com reposição por época; a i-ésima iteração usa a época i e registra os índices sorteados."""

    def __init__(self, window_subjects: Sequence[int] | np.ndarray, seed: int) -> None:
        self._probabilities = window_probabilities(window_subjects)
        self._seed = seed
        self.draws: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._probabilities)

    def __iter__(self) -> Iterator[int]:
        indices = draw_epoch(self._probabilities, self._seed, len(self.draws))
        self.draws.append(indices)
        return iter(indices.tolist())


def draws_sha256(draws: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for indices in draws:
        digest.update(np.asarray(indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def run_lengths(values: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in values.tolist():
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([int(value), 1])
    return runs


def expand_runs(runs: object) -> np.ndarray:
    if not isinstance(runs, list) or not all(
        isinstance(run, list)
        and len(run) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in run)
        and run[1] > 0
        for run in runs
    ):
        raise ValueError("codificação por trechos deve ser uma lista de pares [valor, contagem positiva]")
    return np.repeat(
        np.array([run[0] for run in runs], dtype=np.int64), np.array([run[1] for run in runs], dtype=np.int64)
    )


def sampling_record(
    window_subjects: Sequence[int] | np.ndarray, window_labels: Sequence[int] | np.ndarray, seed: int, epochs: int
) -> dict[str, object]:
    """Contagens que o sampler produz nas `epochs` épocas, calculadas antes do treino com o mesmo sorteio."""
    subjects = np.asarray(window_subjects, dtype=np.int64)
    labels = np.asarray(window_labels, dtype=np.int64)
    if subjects.shape != labels.shape:
        raise ValueError("sujeitos e rótulos das janelas de treino têm tamanhos diferentes")
    probabilities = window_probabilities(subjects)
    draws = [draw_epoch(probabilities, seed, epoch) for epoch in range(epochs)]
    ordered = sorted(int(subject) for subject in np.unique(subjects))

    def by_subject(indices: np.ndarray) -> dict[str, int]:
        drawn = subjects[indices]
        return {str(subject): int((drawn == subject).sum()) for subject in ordered}

    def by_label(values: np.ndarray) -> dict[str, int]:
        return {name: int((values == class_id).sum()) for class_id, name in enumerate(LE2I_LABEL_NAMES)}

    every = np.concatenate(draws)
    distinct = np.unique(every)
    return {
        "development_windows": int(subjects.size),
        "window_subject_runs": run_lengths(subjects),
        "window_label_runs": run_lengths(labels),
        "epochs": epochs,
        "draws_per_epoch": int(subjects.size),
        "subject_windows": {str(subject): int((subjects == subject).sum()) for subject in ordered},
        "window_probability": {
            str(subject): float(probabilities[np.flatnonzero(subjects == subject)[0]]) for subject in ordered
        },
        "clean_class_support": by_label(labels),
        "sampled_subject_windows_by_epoch": [by_subject(indices) for indices in draws],
        "sampled_subject_windows_total": by_subject(every),
        "sampled_class_windows_total": by_label(labels[every]),
        "distinct_windows_drawn_by_subject": {
            str(subject): int((subjects[distinct] == subject).sum()) for subject in ordered
        },
        "draws_sha256": draws_sha256(draws),
    }
