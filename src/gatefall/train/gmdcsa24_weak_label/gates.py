"""Portões de pré-treino do piloto T/S do GMDCSA24 com rótulos fracos (braço A).

Implementa o portão "Treino" de
`docs/analysis/gmdcsa24-weak-label-protocol.md#portoes-antes-de-cada-etapa`:
W1, W2 e V2–V5 para T e S, inclusive as partes de V2 que dependem do treino, o
controle pareado de V3 e a equivalência de configuração de V4. Tudo falha
fechado. A proveniência é a da extração validada no commit pinado em
`EXTRACTION_COMMIT`; um commit novo do código de treino não exige refazer o
adapter nem a pose. `test` só entra pelo hash do índice de janelas, sem
features, predições ou métricas.
"""

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import torch

from gatefall.config import IGNORE_LABEL
from gatefall.data.gmdcsa24.pose_extract import (
    PINNED_GATE,
    POSE_MANIFEST_FILE,
    ExtractionGate,
    read_json,
    section,
    validate_pose_output,
)
from gatefall.data.gmdcsa24.probe import FIRST_UNPADDED_INDEX
from gatefall.data.gmdcsa24.provenance import DEFAULT_REVIEW_DIR, ensure_ignored_output
from gatefall.data.gmdcsa24.source import sha256_of
from gatefall.data.gmdcsa24.weak_label import (
    DEFAULT_OUTPUT_DIR,
    DROPPED_LABELS,
    ENV,
    LE2I_CS_TRAIN_WINDOWS,
    MANIFEST_FILE,
    SPLIT,
    WINDOW_FILES,
    compare_expected,
    validate_outputs,
)
from gatefall.datasets.le2i import PROTOCOL_PROCESSED_DIRS
from gatefall.eval.analysis.multiseed_summary import _config_fingerprint
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, load_alarm_protocol
from gatefall.features.standardization import (
    StandardizationStats,
    load_stats,
    validate_stats_layout,
)
from gatefall.hashing import sha256_dataframe, sha256_file
from gatefall.runs import LOCAL_RUN_ROOTS, REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.baseline_a.cli import _resolve_config
from gatefall.train.baseline_a.artifacts import validate_training_run
from gatefall.train.baseline_a.config import TrainConfig
from gatefall.train.gmdcsa24_weak_label.assembly import (
    FeatureLoader,
    Le2iSide,
    PoseRoots,
    TreatmentSources,
    arm_a_features,
    build_sources,
    class_counts,
    class_weights,
    gmdcsa24_windows,
    treatment_stats,
)

# Commit em que o adapter e a pose foram gerados e validados (E0 e extração).
EXTRACTION_COMMIT = "5fc8204ae474d36d849e87000b5d4b52cd3d5177"

# Totais do treino aumentado: 5 219 janelas Le2i-CS mais T = 1 973 ou S = 1 562.
TRAINING_WINDOWS = {"T": 7192, "S": 6781}

# Hash dos índices de val/test do Le2i-CS publicado em "Resultado sobre os
# dados pinados" do protocolo.
LE2I_CS_EVAL_INDEX_SHA256 = {
    "val": "56f40fe5e32ac6558bcef7e157178ad81dd0fd497da9f53c91a6cffcce90422b",
    "test": "36168466d34f6bea11d601b60cbcc3d82d830baeb0b27d209a1cd1b4c617c984",
}

# Controle do braço A: tabela de checkpoint_sha256 e fingerprint da
# configuração fora da seed publicados em docs/analysis/multiseed-summary.md.
CONTROL_CHECKPOINT_SHA256 = {
    42: "78278b1a6a6eb929c27ae071ebf364a2032877eb0adbcb4e4efd2a6a26787ab8",
    43: "701ec9a941ccd3863931757c520cef4b496be2c64257bac43c0aaa16c014993e",
    44: "a068534f0b476ff9d181794623764fab319732ee1d12a9b612385066e361aa36",
    45: "940055adf3613f4e56ed7745da06ebea2200ed68f1922d2c03d5b62b2af4b4fa",
    46: "6ebe278ce22da41afed26d22c061c4b86919e43efb0511c02607276e135e1265",
}
CONTROL_CONFIG_FINGERPRINT = "b80b439f1878b1f14c2a879dd6bda6f0ce3dc5572229c0f248375609241b030a"

# Os três itens que podem diferir do controle são o manifesto de treino, os
# pesos da perda e o JSON de padronização; só o último aparece na configuração.
TREATMENT_CONFIG_FIELDS = frozenset({"standardization_stats_path", "standardization_stats_sha256"})

DEFAULT_TRAINING_DIR = Path("data/processed/gmdcsa24_training")
DEFAULT_RUN_ROOT = Path("runs/local/gmdcsa24_weak_label")
PREPARED_MANIFEST_FILE = "manifest.json"
PREPARED_STATS_FILE = "pose_stats.json"
TREATMENT_RECORD_FILE = "treatment.json"
DECISION_FILE = "decision.json"
TEST_DESCRIPTIVE_DIR = "test_descriptive"
VALIDATION_SPLITS = ("train", "val")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")

FORBIDDEN_TRAINING_DIRS = (
    *PROTOCOL_PROCESSED_DIRS.values(),
    DEFAULT_OUTPUT_DIR,
    Path("data/features"),
    Path("data/labels"),
    Path("data/raw"),
    Path("src/gatefall/features/stats"),
    REFERENCE_RUN_ROOT,
    *LOCAL_RUN_ROOTS.values(),
    DEFAULT_REVIEW_DIR,
)

DATA_CONDITIONS = ("checkout", "proveniência", "W1", "W2", "V2", "V5", "tratamento")
CONDITIONS = (
    "checkout",
    "proveniência",
    "W1",
    "W2",
    "V2",
    "V3",
    "V4",
    "V5",
    "tratamento",
    "padronização e pesos",
    "saídas",
)


@dataclass(frozen=True)
class TrainingGate:
    """Expectativas fixas do portão de treino; o selftest troca os valores publicados."""

    extraction: ExtractionGate
    extraction_commit: str
    le2i_train_windows: int
    training_windows: dict[str, int]
    eval_index_sha256: dict[str, str]
    control_checkpoint_sha256: dict[int, str]
    control_config_fingerprint: str

    def gmdcsa24_windows(self, treatment: str) -> int:
        funnel = cast(dict[str, int], self.extraction.expected_aggregates["funnel"])
        return int(funnel[treatment])


PINNED_TRAINING_GATE = TrainingGate(
    extraction=PINNED_GATE,
    extraction_commit=EXTRACTION_COMMIT,
    le2i_train_windows=LE2I_CS_TRAIN_WINDOWS,
    training_windows=TRAINING_WINDOWS,
    eval_index_sha256=LE2I_CS_EVAL_INDEX_SHA256,
    control_checkpoint_sha256=CONTROL_CHECKPOINT_SHA256,
    control_config_fingerprint=CONTROL_CONFIG_FINGERPRINT,
)


def git_lineage(ancestor: str, descendant: str) -> bool:
    """`ancestor` é o próprio `descendant` ou um commit anterior na história dele."""
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


@dataclass(frozen=True)
class TreatmentInputs:
    treatment: str
    adapter_dir: Path
    pose_dir: Path
    training_dir: Path
    le2i: Le2iSide
    checkout: tuple[str, list[str]]
    gate: TrainingGate = PINNED_TRAINING_GATE
    load_features: FeatureLoader = arm_a_features
    lineage: Callable[[str, str], bool] = git_lineage
    repository_root: Path = REPOSITORY_ROOT

    @property
    def prepared_dir(self) -> Path:
        return self.training_dir / self.treatment.lower()

    @property
    def roots(self) -> PoseRoots:
        return PoseRoots(le2i=self.le2i.pose_root, gmdcsa24=self.pose_dir)


@dataclass(frozen=True)
class ControlRun:
    run_dir: Path
    retrained: bool = False


@dataclass
class Assessment:
    results: dict[str, list[str]]
    sources: TreatmentSources | None = None
    record: dict[str, object] | None = None
    stats: StandardizationStats | None = None
    config: TrainConfig | None = None
    control_checkpoint_sha256: str | None = None

    @property
    def passed(self) -> bool:
        return all(not problems for problems in self.results.values())


def location_problems(path: Path) -> list[str]:
    """Saída fora dos artefatos do Le2i, do adapter, da pose, dos rótulos, das
    referências, dos runs do controle e da revisão, e ignorada pelo Git."""
    resolved = path.resolve()
    problems: list[str] = []
    for forbidden in FORBIDDEN_TRAINING_DIRS:
        target = forbidden.resolve()
        if resolved == target or target in resolved.parents or resolved in target.parents:
            problems.append(f"{path} se sobrepõe a {forbidden}")
    try:
        ensure_ignored_output(path)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def overlaps(first: Path, second: Path) -> bool:
    """Mesmo caminho resolvido, ou um é ancestral do outro."""
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def output_problems(
    inputs: TreatmentInputs,
    control: ControlRun | None = None,
    run_dir: Path | None = None,
    seed: int | None = None,
) -> list[str]:
    """Preparação, run do tratamento e run do controle separados entre si e das
    entradas; conferido antes de qualquer gravação."""
    problems = [f"--training-dir: {problem}" for problem in location_problems(inputs.training_dir)]
    inputs_read = {
        "adapter do GMDCSA24": inputs.adapter_dir,
        "pose do GMDCSA24": inputs.pose_dir,
        "pose do Le2i-CS": inputs.le2i.pose_root,
        "grade do Le2i-CS": inputs.le2i.frames_path,
        "JSON de padronização do controle": inputs.le2i.stats_path,
    }
    if control is not None:
        inputs_read["run do controle"] = control.run_dir
    for name, path in {**inputs_read, "raiz dos runs do tratamento": DEFAULT_RUN_ROOT}.items():
        if overlaps(inputs.training_dir, path):
            problems.append(f"--training-dir {inputs.training_dir} se sobrepõe a {name} ({path})")
    if run_dir is None:
        return problems
    problems.extend(f"--run-dir: {problem}" for problem in location_problems(run_dir))
    outputs = {"--training-dir": inputs.training_dir, "preparações padrão": DEFAULT_TRAINING_DIR}
    for name, path in {**inputs_read, **outputs}.items():
        if overlaps(run_dir, path):
            problems.append(f"--run-dir {run_dir} se sobrepõe a {name} ({path})")
    resolved = run_dir.resolve()
    for root in (DEFAULT_RUN_ROOT, *(DEFAULT_RUN_ROOT / name.lower() for name in WINDOW_FILES)):
        if resolved == root.resolve() or resolved in root.resolve().parents:
            problems.append(f"--run-dir {run_dir} contém os runs de outras seeds em {root}")
    if run_dir.parent.name != inputs.treatment.lower() or run_dir.name != f"seed_{seed}":
        problems.append(
            f"--run-dir {run_dir} fora do layout <raiz>/{inputs.treatment.lower()}/seed_{seed}"
        )
    if run_dir.exists() and not run_dir.is_dir():
        problems.append(f"--run-dir {run_dir} não é um diretório")
    elif run_dir.is_dir() and any(child.is_dir() for child in run_dir.iterdir()):
        problems.append(f"--run-dir {run_dir} contém subdiretórios; um run é um diretório plano")
    for parent in resolved.parents:
        if (parent / TREATMENT_RECORD_FILE).exists() or (parent / "checkpoint.pt").exists():
            problems.append(f"--run-dir {run_dir} está dentro de outro run ({parent})")
            break
    return problems


def canonical_run_dir(treatment: str, seed: int) -> Path:
    """O único local de execução do run de um tratamento e uma seed no CLI."""
    return DEFAULT_RUN_ROOT / treatment.lower() / f"seed_{seed}"


def execution_root_problems(inputs: TreatmentInputs, seed: int, control: ControlRun) -> list[str]:
    """No CLI, o diretório de execução é a raiz do repositório GateFall, que fixa o
    run canônico; nenhum ancestral nem outra raiz reaproveita as mesmas entradas."""
    root = inputs.repository_root.resolve()
    cwd = Path.cwd().resolve()
    problems = []
    if cwd != root:
        problems.append(
            f"diretório de execução {cwd} difere da raiz do repositório GateFall {root}, "
            "que guarda o run canônico de cada tratamento e seed"
        )
    paths = {
        "adapter do GMDCSA24": inputs.adapter_dir,
        "pose do GMDCSA24": inputs.pose_dir,
        "--training-dir": inputs.training_dir,
        "pose do Le2i-CS": inputs.le2i.pose_root,
        "grade do Le2i-CS": inputs.le2i.frames_path,
        "JSON de padronização do controle": inputs.le2i.stats_path,
        "run do controle": control.run_dir,
    }
    problems.extend(
        f"{name} ({path}) fora da raiz do repositório GateFall {root}"
        for name, path in paths.items()
        if root not in path.resolve().parents
    )
    run_dir = root / canonical_run_dir(inputs.treatment, seed)
    if root not in run_dir.resolve().parents:
        problems.append(
            f"run canônico {run_dir} resolve para {run_dir.resolve()}, fora da raiz do "
            f"repositório GateFall {root}"
        )
    return problems


def test_access_problems(root: Path = DEFAULT_RUN_ROOT) -> list[str]:
    """Depois de qualquer acesso a `test`, nada que sustenta a decisão é refeito."""
    path = root / TEST_DESCRIPTIVE_DIR
    if path.exists():
        return [
            f"{path}: test do Le2i-CS já acessado; a decisão, a preparação, os runs e as "
            "avaliações que a sustentam não são mais gravados, recuperados nem refeitos"
        ]
    return []


def decision_problems() -> list[str]:
    """Com a decisão GO/NO-GO registrada ou `test` acessado, nada que a sustenta é recriado."""
    path = DEFAULT_RUN_ROOT / DECISION_FILE
    problems = test_access_problems()
    if path.exists():
        problems.insert(
            0,
            f"{path}: decisão GO/NO-GO registrada; preparação, runs e avaliações que a "
            "sustentam não são mais gravados nem recuperados",
        )
    return problems


def checkout_problems(checkout: tuple[str, list[str]]) -> list[str]:
    """O treino registra o commit do código; só exige checkout limpo, não o commit da extração."""
    _, status = checkout
    if status:
        return [
            f"checkout do GateFall com {len(status)} arquivo(s) modificado(s) ou não rastreado(s)"
        ]
    return []


def provenance_results(inputs: TreatmentInputs) -> dict[str, list[str]]:
    """As seis condições do adapter, a validação da pose e o commit da extração validada."""
    gate = inputs.gate
    adapter = validate_outputs(
        inputs.adapter_dir,
        inputs.le2i.reference,
        gate.extraction.expected_exclusions,
        gate.extraction.expected_aggregates,
    )
    results = {
        "proveniência": [
            f"adapter {condition}: {problem}"
            for condition in ("integridade", "proveniência")
            for problem in adapter.get(condition, [])
        ],
        **{condition: list(adapter.get(condition, [])) for condition in ("W1", "W2", "V2", "V5")},
    }
    if any(adapter.values()):
        results["proveniência"].append("pose não verificada: saída do adapter reprovada")
        return results
    pose = validate_pose_output(
        inputs.pose_dir, inputs.adapter_dir, inputs.le2i.reference, gate.extraction
    )
    results["proveniência"].extend(
        f"pose {condition}: {problem}"
        for condition, problems in pose.items()
        for problem in problems
    )
    expected_code = {"commit": gate.extraction_commit, "dirty": False}
    for name, path in (
        ("adapter", inputs.adapter_dir / MANIFEST_FILE),
        ("pose", inputs.pose_dir / POSE_MANIFEST_FILE),
    ):
        if not path.is_file():
            continue
        code = section(read_json(path), "code")
        observed = {key: code.get(key) for key in expected_code}
        if observed != expected_code:
            results["proveniência"].append(
                f"{name}: gerado em {observed}, a extração validada é {expected_code}"
            )
    return results


def membership_problems(inputs: TreatmentInputs, sources: TreatmentSources) -> list[str]:
    """Janelas de T ou S, contagens do treino aumentado e alinhamento com a pose."""
    gate = inputs.gate
    treatment = inputs.treatment
    windows = sources.gmdcsa24
    problems: list[str] = []
    expected = {
        f"janelas do GMDCSA24 em {treatment}": gate.gmdcsa24_windows(treatment),
        "janelas de treino do Le2i-CS": gate.le2i_train_windows,
        f"janelas do treino aumentado de {treatment}": gate.training_windows[treatment],
    }
    observed = {
        f"janelas do GMDCSA24 em {treatment}": len(windows),
        "janelas de treino do Le2i-CS": len(sources.le2i_train),
        f"janelas do treino aumentado de {treatment}": len(sources.train),
    }
    problems.extend(compare_expected(expected, observed))
    expected_total = gate.le2i_train_windows + gate.gmdcsa24_windows(treatment)
    if expected_total != gate.training_windows[treatment]:
        problems.append(f"total esperado de {treatment} não é a soma do Le2i-CS e do GMDCSA24")
    keys = {
        name: set(zip(frame["video_id"].astype(str), frame["k_end"].astype(int)))
        for name, frame in (
            (name, gmdcsa24_windows(inputs.adapter_dir, name)) for name in WINDOW_FILES
        )
    }
    if not keys["S"] <= keys["T"]:
        problems.append(f"{len(keys['S'] - keys['T'])} janelas de S fora de T")
    if set(windows["env"].astype(str)) - {ENV} or set(windows["split"].astype(str)) - {SPLIT}:
        problems.append(f"janelas do GMDCSA24 fora de env = {ENV} e split = {SPLIT}")
    if ENV in set(sources.le2i_train["env"].astype(str)):
        problems.append(f"treino Le2i-CS com env = {ENV}")
    labels = windows["label"].astype(int)
    if bool(labels.isin([*DROPPED_LABELS, IGNORE_LABEL]).any()):
        problems.append("janela do GMDCSA24 com lie_down, lying ou IGNORE_LABEL")
    k_end = windows["k_end"].astype(int)
    outside = (k_end < FIRST_UNPADDED_INDEX) | (k_end >= windows["n_frames"].astype(int))
    if bool(outside.any()):
        problems.append("janela do GMDCSA24 com replicação de borda ou k_end fora da grade")
    pose_path = inputs.pose_dir / POSE_MANIFEST_FILE
    records = section(read_json(pose_path), "videos") if pose_path.is_file() else {}
    for video_id, n_frames in sorted(
        {str(video): int(k) for video, k in zip(windows["video_id"], windows["n_frames"])}.items()
    ):
        record = records.get(video_id)
        k = cast(dict[str, object], record).get("K") if isinstance(record, dict) else None
        if k != n_frames:
            problems.append(f"{video_id}: K da pose {k!r}, o índice de janelas registra {n_frames}")
    return problems


def split_problems(inputs: TreatmentInputs, sources: TreatmentSources) -> list[str]:
    """Partes de V2 que dependem do treino: val/test intactos e treino só com `train`."""
    reference = inputs.le2i.reference
    problems = compare_expected(
        inputs.gate.eval_index_sha256, reference.eval_index_sha256, "índice Le2i-CS atual "
    )
    val_hash = sha256_dataframe(sources.val.windows)
    if val_hash != inputs.gate.eval_index_sha256.get("val"):
        problems.append(f"índice de val usado no treino com hash {val_hash}, não o publicado")
    train = sources.train.windows
    if set(train["split"].astype(str)) != {"train"}:
        problems.append(f"treino aumentado com splits {sorted(set(train['split'].astype(str)))}")
    leaked = set(train["video_id"].astype(str)) & reference.eval_video_ids
    if leaked:
        problems.append(f"vídeos de val/test do Le2i-CS no treino: {sorted(leaked)}")
    shared = set(sources.gmdcsa24["video_id"].astype(str)) & set(
        inputs.le2i.frames["video_id"].astype(str)
    )
    if shared:
        problems.append(f"video_id do GMDCSA24 na grade do Le2i-CS: {sorted(shared)}")
    problems.extend(
        f"JSON do tratamento: {problem}" for problem in location_problems(inputs.prepared_dir)
    )
    return problems


def assess_data(inputs: TreatmentInputs) -> Assessment:
    """Condições que não dependem da seed: valem para `prepare` e para o treino."""
    results = {condition: [] for condition in DATA_CONDITIONS}
    results["checkout"] = checkout_problems(inputs.checkout)
    results.update(provenance_results(inputs))
    if any(results[condition] for condition in ("proveniência", "W1", "W2", "V2", "V5")):
        results["tratamento"] = ["não verificado: proveniência ou adapter reprovados"]
        return Assessment(results)
    sources = build_sources(
        inputs.le2i,
        gmdcsa24_windows(inputs.adapter_dir, inputs.treatment),
        inputs.roots,
        inputs.load_features,
    )
    results["tratamento"] = membership_problems(inputs, sources)
    results["V2"].extend(split_problems(inputs, sources))
    return Assessment(results, sources=sources, record=preparation_record(inputs, sources))


def preparation_record(inputs: TreatmentInputs, sources: TreatmentSources) -> dict[str, object]:
    """O que o manifesto preparado registra e o treino recalcula antes de começar."""
    adapter_manifest = read_json(inputs.adapter_dir / MANIFEST_FILE)
    pose_manifest = read_json(inputs.pose_dir / POSE_MANIFEST_FILE)
    return {
        "treatment": inputs.treatment,
        "protocol": {
            key: section(adapter_manifest, "protocol").get(key) for key in ("path", "sha256")
        },
        "adapter": {
            "dir": inputs.adapter_dir.as_posix(),
            "manifest_sha256": sha256_of(inputs.adapter_dir / MANIFEST_FILE),
            "windows_file": WINDOW_FILES[inputs.treatment],
            "windows_sha256": sha256_dataframe(sources.gmdcsa24),
            "code_commit": section(adapter_manifest, "code").get("commit"),
        },
        "pose": {
            "dir": inputs.pose_dir.as_posix(),
            "manifest_sha256": sha256_of(inputs.pose_dir / POSE_MANIFEST_FILE),
            "code_commit": section(pose_manifest, "code").get("commit"),
        },
        "windows": {
            "le2i_train": len(sources.le2i_train),
            "gmdcsa24": len(sources.gmdcsa24),
            "total": len(sources.train),
            "le2i_train_index_sha256": sha256_dataframe(sources.le2i_train),
            "train_index_sha256": sha256_dataframe(sources.train.windows),
            "class_counts": class_counts(sources.train.windows),
        },
        "eval_index_sha256": dict(inputs.le2i.reference.eval_index_sha256),
        "class_weights": class_weights(sources.train),
    }


def code_problems(
    inputs: TreatmentInputs, code: object, label: str, after: str | None = None
) -> list[str]:
    """Commit limpo e completo entre `after` (padrão: o da extração validada) e o HEAD atual."""
    if not isinstance(code, dict) or set(code) != {"commit", "dirty"}:
        return [f"{label}: code deve ter exatamente commit e dirty, obtido {code!r}"]
    commit, dirty = code["commit"], code["dirty"]
    problems = [] if dirty is False else [f"{label}: gerado com code.dirty = {dirty!r}"]
    if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
        return [*problems, f"{label}: code.commit {commit!r} não é um SHA-1 completo"]
    head = inputs.checkout[0]
    after = after or inputs.gate.extraction_commit
    if not inputs.lineage(after, commit):
        problems.append(f"{label}: commit {commit} não descende de {after}")
    if not inputs.lineage(commit, head):
        problems.append(f"{label}: commit {commit} não é ancestral do HEAD atual {head}")
    return problems


def preparation_problems(inputs: TreatmentInputs, assessment: Assessment) -> list[str]:
    """Manifesto e JSON preparados conferidos contra o treino aumentado recalculado,
    inclusive o commit da preparação e todos os campos de `standardization`."""
    sources = cast(TreatmentSources, assessment.sources)
    record = cast(dict[str, object], assessment.record)
    manifest_path = inputs.prepared_dir / PREPARED_MANIFEST_FILE
    stats_path = inputs.prepared_dir / PREPARED_STATS_FILE
    if not manifest_path.is_file() or not stats_path.is_file():
        return [
            f"{inputs.prepared_dir} não preparado; rode `prepare --treatment {inputs.treatment}`"
        ]
    try:
        manifest = read_json(manifest_path)
        stats = load_stats(stats_path)
        validate_stats_layout(stats)
    except (OSError, TypeError, KeyError, ValueError) as exc:
        return [f"{inputs.prepared_dir}: preparação ilegível ({exc})"]
    if not isinstance(manifest, dict):
        return [f"{manifest_path}: não é um objeto JSON"]
    recomputed = treatment_stats(sources.train, inputs.le2i.frames_path)
    expected = {
        **record,
        "standardization": {
            "path": stats_path.as_posix(),
            "sha256": sha256_file(stats_path),
            "window_count": recomputed.window_count,
            "frames_hash": recomputed.frames_hash,
        },
    }
    observed = {key: value for key, value in manifest.items() if key not in ("created_at", "code")}
    standardization = observed.get("standardization")
    if isinstance(standardization, dict) and isinstance(standardization.get("path"), str):
        recorded_path = Path(cast(str, standardization["path"]))
        if recorded_path.resolve() == stats_path.resolve():
            observed["standardization"] = {**standardization, "path": stats_path.as_posix()}
    problems = compare_expected(expected, observed, "manifesto preparado: ")
    created_at = manifest.get("created_at")
    try:
        datetime.fromisoformat(cast(str, created_at))
    except (TypeError, ValueError):
        problems.append(f"manifesto preparado: created_at inválido {created_at!r}")
    problems.extend(code_problems(inputs, manifest.get("code"), "manifesto preparado"))
    if stats != recomputed:
        problems.append(
            f"{stats_path} diverge da padronização recalculada sobre o treino aumentado de "
            f"{inputs.treatment}"
        )
    if not problems:
        assessment.stats = stats
    return problems


def control_results(
    inputs: TreatmentInputs, seed: int, control: ControlRun
) -> tuple[dict[str, list[str]], TrainConfig | None, str | None]:
    """V3 (controle pareado), a configuração congelada do controle (V4) e, em V2,
    as estatísticas do controle sem GMDCSA24 e o alarme não recalibrado."""
    gate = inputs.gate
    results: dict[str, list[str]] = {"V2": [], "V3": [], "V4": []}
    if seed not in gate.control_checkpoint_sha256:
        seeds = sorted(gate.control_checkpoint_sha256)
        results["V3"].append(f"seed {seed} fora das seeds do controle {seeds}")
        return results, None, None
    stats_path = inputs.le2i.stats_path
    expected = _resolve_config(seed, stats_path, sha256_file(stats_path))
    try:
        config = validate_training_run(control.run_dir, expected_config=expected)
    except RuntimeError as exc:
        results["V3"].append(f"controle inválido para a seed {seed}: {exc}")
        return results, None, None
    if _config_fingerprint(config) != gate.control_config_fingerprint:
        results["V4"].append("configuração do controle fora da receita congelada publicada")
    checkpoint = sha256_file(control.run_dir / "checkpoint.pt")
    if checkpoint != gate.control_checkpoint_sha256[seed]:
        if not control.retrained:
            results["V3"].append(
                f"checkpoint_sha256 do controle {checkpoint} diverge do publicado para a "
                f"seed {seed}; "
                "use um controle publicado ou um retreinado nesta máquina (--control-retrained)"
            )
        else:
            metrics = json.loads((control.run_dir / "metrics.json").read_text(encoding="utf-8"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            here = {"torch_version": torch.__version__, "device": device}
            observed = {key: metrics.get(key) for key in here}
            if observed != here:
                results["V3"].append(f"controle retreinado em {observed}, esta máquina tem {here}")
    control_stats = load_stats(stats_path)
    stats_expected = {
        "window_count": gate.le2i_train_windows,
        "frames_hash": sha256_file(inputs.le2i.frames_path),
    }
    stats_observed = {
        "window_count": control_stats.window_count,
        "frames_hash": control_stats.frames_hash,
    }
    results["V2"].extend(
        compare_expected(stats_expected, stats_observed, "estatísticas do controle ")
    )
    alarm_path = control.run_dir / "alarm_protocol.yaml"
    if not alarm_path.is_file():
        results["V2"].append(f"{alarm_path} ausente; o alarme do controle não foi conferido")
    elif load_alarm_protocol(alarm_path) != BASELINE_A_ALARM_PROTOCOL:
        results["V2"].append(f"{alarm_path} diverge de BASELINE_A_ALARM_PROTOCOL")
    return results, config, checkpoint


def treatment_config(seed: int, stats_path: Path) -> TrainConfig:
    """A receita congelada do braço A com o JSON de padronização do tratamento."""
    return _resolve_config(seed, stats_path, sha256_file(stats_path))


def config_problems(treatment: TrainConfig, control: TrainConfig) -> list[str]:
    actual = treatment.to_dict()
    reference = control.to_dict()
    differing = sorted(
        key for key in set(actual) | set(reference) if actual.get(key) != reference.get(key)
    )
    return [
        f"configuração do tratamento difere do controle em {key}: "
        f"{actual.get(key)!r} != {reference.get(key)!r}"
        for key in differing
        if key not in TREATMENT_CONFIG_FIELDS
    ]


def assess_training(inputs: TreatmentInputs, seed: int, control: ControlRun) -> Assessment:
    """Todas as condições exigidas antes de treinar uma seed de T ou S."""
    assessment = assess_data(inputs)
    results = {condition: assessment.results.get(condition, []) for condition in CONDITIONS}
    assessment.results = results
    control_checks, control_config, checkpoint = control_results(inputs, seed, control)
    for condition, problems in control_checks.items():
        results[condition].extend(problems)
    assessment.control_checkpoint_sha256 = checkpoint
    results["saídas"] = output_problems(inputs, control)
    if assessment.sources is None:
        results["padronização e pesos"] = ["não verificado: proveniência ou adapter reprovados"]
        return assessment
    results["padronização e pesos"] = preparation_problems(inputs, assessment)
    stats_path = inputs.prepared_dir / PREPARED_STATS_FILE
    if assessment.stats is not None and control_config is not None:
        assessment.config = treatment_config(seed, stats_path)
        results["V4"].extend(config_problems(assessment.config, control_config))
    return assessment


def report(title: str, results: dict[str, list[str]]) -> bool:
    print(f"\n=== {title} ===")
    for condition, problems in results.items():
        print(f"[{'FAIL' if problems else 'PASS'}] {condition}")
        for problem in problems:
            print(f"  {problem}")
    print(
        "\ntest do Le2i-CS: só o hash do índice de janelas entra em V2; nenhuma feature, "
        "predição ou métrica de test é calculada antes da decisão GO/NO-GO em val"
    )
    return all(not problems for problems in results.values())
