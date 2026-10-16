"""Resumo por dobra dos runs do braço A nas dobras por sujeito do Le2i-CS.

Lê só artefatos já gravados na raiz das dobras: `folds.json`, a proveniência
das estatísticas e os runs; frames, manifesto e features não são abertos.
Antes de relatar qualquer número, `folds.json` é reconciliado com o contrato
congelado e com a composição gravada, a proveniência de cada dobra com
`folds.json`, e cada run com a configuração determinística da dobra, seu
`fold_run.json`, a integridade do checkpoint, o suporte por classe gravado e
a igualdade das posições `val` e `test`. Cada dobra é relatada sozinha: não
há escore agregado entre dobras.
"""

import json
from pathlib import Path
from typing import cast

from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import display_path, write_text_outputs
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.artifacts import validate_training_run
from gatefall.train.internal_folds.training import FOLD_RUN_FILE, validate_fold_run_document
from gatefall.train.shared.metrics import RESTRICTED_CLASSES
from gatefall.train.subject_folds.folds import (
    ARMS,
    BASELINE_VARIANT,
    CHECKPOINT_POLICY,
    EVALUATION_SPLIT_MAPPING,
    PROTOCOL,
    RESTRICTED_LABELS,
    SEEDS,
    SELECTION_POLICY,
    STAGE,
    SUBJECT_BALANCED_VARIANT,
    fold_run_dir,
    read_folds_document,
    stats_dir,
    summary_dir,
)
from gatefall.train.subject_folds.stats import load_fold_stats
from gatefall.train.subject_folds.training import (
    detection_dropout_run_document,
    fold_run_document,
    resolve_fold_config,
    subject_balanced_run_document,
    validate_sampling_record,
)

SUMMARY_JSON_FILE = "subject_fold_summary.json"
SUMMARY_MARKDOWN_FILE = "subject_fold_summary.md"
BASELINE_SUMMARY = "baseline"

LIMITATIONS = [
    "Exploratório e não canônico: as dobras usam só o train oficial do Le2i-CS; val e test oficiais não são "
    "lidos, e nada aqui estima o desempenho no test oficial nem verifica o alvo de 0,800.",
    "São três dobras de tamanhos muito diferentes: o sujeito 0 tem cerca de metade das janelas rotuladas do "
    "train, e a dobra que o retém treina com a outra metade. As magnitudes não são comparáveis entre dobras nem "
    "com o Le2i-CS oficial, e por isso não há escore agregado.",
    "Uma seed por dobra: a variação entre seeds não é estimada. Este resumo é a linha de base para uma "
    "comparação pareada futura (mesma dobra, seed e receita), não um resultado de desempenho.",
    "As dobras são por sujeito bruto, não por cena: ambientes e câmeras aparecem nos dois lados, e a identidade "
    "global dos IDs brutos não é verificável.",
    "Contagens por janela usam janelas sobrepostas (stride 4 no treino, stride 1 no retido) e não são "
    "observações independentes.",
]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} deve ser um objeto")
    return data


def _support_by_name(split_metrics: dict) -> dict[str, int]:
    support = cast(dict[str, int], split_metrics["support"])
    return {name: int(support[name]) for name in LE2I_LABEL_NAMES}


def fold_limitations(fold: dict) -> list[str]:
    videos = cast(dict[str, int], fold["retained_class_videos_eval_stride"])
    subjects = cast(dict[str, int], fold["retained_class_subjects_eval_stride"])
    support = cast(dict[str, int], fold["retained_support_eval_stride"])
    limitations: list[str] = []
    if len(fold["retained_subjects"]) == 1:
        limitations.append("Retém um único sujeito: a variação entre sujeitos não é estimável nesta dobra.")
    single_video = [label for label in RESTRICTED_LABELS if videos[label] == 1]
    if single_video:
        limitations.append(f"Classes do retido vindas de um único vídeo: {', '.join(single_video)}.")
    if len(fold["retained_subjects"]) > 1:
        single_subject = [label for label in RESTRICTED_LABELS if subjects[label] == 1]
        if single_subject:
            limitations.append(f"Classes do retido vindas de um único sujeito: {', '.join(single_subject)}.")
    fewest = min(RESTRICTED_LABELS, key=lambda label: (videos[label], support[label]))
    limitations.append(
        f"Classe restrita do retido com menos vídeos: {fewest} ({videos[fewest]} vídeos, {support[fewest]} janelas, "
        f"{subjects[fewest]} sujeito(s)); sua F1 depende de poucos eventos."
    )
    development = cast(dict[str, int], fold["development_support_train_stride"])
    rarest = min(RESTRICTED_LABELS, key=lambda label: development[label])
    limitations.append(
        f"Classe restrita com menos janelas no treino de desenvolvimento: {rarest} ({development[rarest]} em stride 4)."
    )
    return limitations


def expected_run_document(
    root: Path, document: dict, fold: dict, arm: str, seed: int, stats_sha256: str, variant: str
) -> dict[str, object]:
    control_document = fold_run_document(
        fold,
        arm,
        seed,
        stats_sha256,
        cast(dict[str, str], document["input_sha256"]),
        cast(str, document["pose_features_sha256"]),
        cast(str, document["membership_sha256"]),
    )
    if variant == BASELINE_VARIANT:
        return control_document
    control_sha256 = sha256_file(fold_run_dir(root, fold["fold"], arm, seed) / FOLD_RUN_FILE)
    if variant == SUBJECT_BALANCED_VARIANT:
        sidecar = fold_run_dir(root, fold["fold"], arm, seed, variant) / FOLD_RUN_FILE
        sampling = _read_json(sidecar).get("sampling") if sidecar.is_file() else None
        return subject_balanced_run_document(
            control_document, control_sha256, validate_sampling_record(sampling, fold, seed, sidecar)
        )
    return detection_dropout_run_document(control_document, control_sha256)


def run_entry(
    root: Path, document: dict, fold: dict, arm: str, seed: int, variant: str = BASELINE_VARIANT
) -> dict[str, object]:
    run_dir = fold_run_dir(root, fold["fold"], arm, seed, variant)
    input_sha256 = cast(dict[str, str], document["input_sha256"])
    pose_sha256 = cast(str, document["pose_features_sha256"])
    stats = load_fold_stats(fold, stats_dir(root, fold["fold"]), input_sha256, pose_sha256)
    config = resolve_fold_config(seed, stats.path, stats.sha256, variant)
    expected_document = expected_run_document(root, document, fold, arm, seed, stats.sha256, variant)
    try:
        validate_training_run(run_dir, expected_config=config)
        validate_fold_run_document(run_dir, expected_document)
    except RuntimeError as exc:
        raise RuntimeError(f"run da dobra inválido em {run_dir}: {exc}") from exc
    final = _read_json(run_dir / "metrics.json")["final"]
    if final["val"] != final["test"]:
        raise RuntimeError(f"{run_dir}: val e test deveriam avaliar o mesmo grupo retido")
    if _support_by_name(final["train"]) != fold["development_support_train_stride"]:
        raise RuntimeError(f"{run_dir}: suporte de treino diverge de development_support_train_stride")
    retained = final["val"]
    support = _support_by_name(retained)
    if support != fold["retained_support_eval_stride"]:
        raise RuntimeError(f"{run_dir}: suporte do retido diverge de retained_support_eval_stride")
    f1_by_class = cast(dict[str, float], retained["f1_by_class"])
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": display_path(run_dir),
        "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt"),
        "standardization_stats_sha256": stats.sha256,
        "retained_macro_f1_restricted": retained["macro_f1_restricted"],
        "retained_per_class": {
            LE2I_LABEL_NAMES[class_id]: {
                "f1": f1_by_class[str(class_id)],
                "support": support[LE2I_LABEL_NAMES[class_id]],
            }
            for class_id in RESTRICTED_CLASSES
        },
        "retained_support": support,
    }


def control_fold_run_sha256(root: Path, document: dict, fold: dict, arm: str, seed: int) -> str:
    """Valida o controle da Etapa 4B da dobra e devolve o SHA-256 do seu `fold_run.json`; nunca o treina."""
    run_dir = fold_run_dir(root, fold["fold"], arm, seed)
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"controle da Etapa 4B ausente em {display_path(run_dir)}; treine-o com `train --variant "
            f"{BASELINE_VARIANT}` antes; a intervenção não treina nem substitui o controle"
        )
    try:
        run_entry(root, document, fold, arm, seed)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"controle da Etapa 4B inválido em {display_path(run_dir)}: {exc}") from exc
    return sha256_file(run_dir / FOLD_RUN_FILE)


def build_summary(root: Path) -> dict[str, object]:
    document = read_folds_document(root)
    folds = cast(list[dict], document["folds"])
    missing = [
        display_path(fold_run_dir(root, fold["fold"], arm, seed))
        for fold in folds
        for arm in ARMS
        for seed in SEEDS
        if not fold_run_dir(root, fold["fold"], arm, seed).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"runs ausentes: {missing}")
    fold_blocks = [
        {
            "fold": fold["fold"],
            "retained_subjects": fold["retained_subjects"],
            "development_subjects": fold["development_subjects"],
            "development_windows_train_stride": fold["development_windows_train_stride"],
            "retained_windows_eval_stride": fold["retained_windows_eval_stride"],
            "n_retained_videos": len(fold["retained_video_ids"]),
            "runs": [run_entry(root, document, fold, arm, seed) for arm in ARMS for seed in SEEDS],
            "limitations": fold_limitations(fold),
        }
        for fold in folds
    ]
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "canonical": False,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "input_sha256": document["input_sha256"],
        "pose_features_sha256": document["pose_features_sha256"],
        "membership_sha256": document["membership_sha256"],
        "restricted_classes": list(RESTRICTED_CLASSES),
        "evaluation_split_mapping": EVALUATION_SPLIT_MAPPING,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "selection_policy": SELECTION_POLICY,
        "pooled_score": None,
        "folds": fold_blocks,
        "limitations": LIMITATIONS,
    }


def _fmt(value: object) -> str:
    return f"{cast(float, value):.3f}"


def render_summary(summary: dict[str, object]) -> str:
    lines = [
        "# Dobras por sujeito do Le2i-CS — linha de base do braço A (Etapa 4B, não canônico)",
        "",
        "Treinos exploratórios em dobras do train oficial do Le2i-CS; val e test oficiais não são lidos. "
        "O grupo de sujeitos retido ocupa as posições val e test do engine. Checkpoint da última época.",
        "",
        str(summary["selection_policy"]),
    ]
    for block in cast(list[dict[str, object]], summary["folds"]):
        retained = ", ".join(str(subject) for subject in cast(list[int], block["retained_subjects"]))
        lines += [
            "",
            f"## Dobra `{block['fold']}` — retém os sujeitos {retained}",
            "",
            f"Treino de desenvolvimento: {block['development_windows_train_stride']} janelas rotuladas em stride 4. "
            f"Retido: {block['n_retained_videos']} vídeos, {block['retained_windows_eval_stride']} janelas "
            "rotuladas em stride 1.",
        ]
        for entry in cast(list[dict[str, object]], block["runs"]):
            per_class = cast(dict[str, dict[str, object]], entry["retained_per_class"])
            lines += [
                "",
                f"Braço {entry['arm']}, seed {entry['seed']}: macro-F1 restrita no retido "
                f"**{_fmt(entry['retained_macro_f1_restricted'])}**.",
                "",
                "| Classe | F1 | Suporte (janelas) |",
                "| --- | ---: | ---: |",
                *(f"| {label} | {_fmt(values['f1'])} | {values['support']} |" for label, values in per_class.items()),
            ]
        lines += ["", "Limitações da dobra:", ""]
        lines += [f"- {text}" for text in cast(list[str], block["limitations"])]
    lines += ["", "## Limitações gerais", ""]
    lines += [f"- {text}" for text in cast(list[str], summary["limitations"])]
    return "\n".join(lines) + "\n"


def write_summary(root: Path, force: bool) -> bool:
    summary = build_summary(root)
    directory = summary_dir(root) / BASELINE_SUMMARY
    targets = {
        directory / SUMMARY_JSON_FILE: json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        directory / SUMMARY_MARKDOWN_FILE: render_summary(summary),
    }
    return write_text_outputs(directory, targets, force)
