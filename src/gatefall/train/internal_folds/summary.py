"""Resumo das dobras internas e aplicação do critério congelado da Etapa 2A.

Lê só artefatos já gravados no diretório das dobras: `folds.json`, a
proveniência das estatísticas e os runs; frames, manifesto e features não são
abertos. Antes de relatar qualquer número, `folds.json` é reconciliado com o
contrato congelado e com a composição gravada, a proveniência de cada dobra
com `folds.json`, e cada run com a configuração determinística da dobra, seu
`fold_run.json` e as contagens de janelas registradas. A fração
predita como `other` vem da matriz de confusão do ambiente retido, com o
mesmo cálculo da consolidação multi-seed.
"""

import json
from pathlib import Path
from typing import cast

from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import display_path, write_text_outputs
from gatefall.train.internal_folds.folds import (
    ARMS,
    CHECKPOINT_POLICY,
    EVALUATION_SPLIT_MAPPING,
    PROTOCOL,
    REPRODUCTION_CRITERION,
    SEED_SETS,
    fold_run_dir,
    read_folds_document,
    stats_dir,
    summary_dir,
)
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.train.internal_folds.stats import StatsFiles, load_recorded_stats_files
from gatefall.train.internal_folds.training import (
    QualityFeatures,
    fold_run_document,
    resolve_fold_config,
    validate_fold_run,
    validate_fold_run_document,
)
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

SUMMARY_JSON_FILE = "internal_fold_summary.json"
SUMMARY_MARKDOWN_FILE = "internal_fold_summary.md"
OTHER_LABEL = "other"
COMPARED_ARMS = ("B0", "B1")
MINIMUM_REPRODUCING_FOLDS = 2

LIMITATIONS = [
    "Exploratório e não canônico: as dobras usam só o train oficial do Le2i-CV; val e test oficiais não são "
    "lidos, e nada aqui confirma ou refuta a falha observada no test oficial.",
    "Cada dobra treina com dois ambientes em vez de três, com menos janelas e outra composição de classes, "
    "sujeitos e cenas; magnitudes não são comparáveis às do Le2i-CV oficial.",
    "As frações são por janela em stride 1, sobre janelas sobrepostas, e a proporção de other muda entre "
    "ambientes retidos; o critério só compara armas dentro da mesma dobra.",
    "A F1 por classe no ambiente retido depende de poucos vídeos em algumas dobras e não deve ser lida como "
    "resultado.",
    "Todo o train oficial é 320x240 a 25 fps: as dobras não testam a hipótese de resolução ou fps do test "
    "oficial. Um resultado negativo não identifica a causa da falha no test.",
    "IDs brutos de sujeito se repetem entre ambientes; as dobras são disjuntas por ambiente, não "
    "comprovadamente por sujeito.",
]


def predicted_other_fraction(split_metrics: dict) -> tuple[float | None, int]:
    other_id = LE2I_LABEL_NAMES.index(OTHER_LABEL)
    matrix = cast(list[list[int]], split_metrics["confusion_matrix"])
    predicted_other = sum(row[other_id] for true_id, row in enumerate(matrix) if true_id != other_id)
    outside_other = sum(sum(row) for true_id, row in enumerate(matrix) if true_id != other_id)
    return (predicted_other / outside_other if outside_other else None), outside_other


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} deve ser um objeto")
    return data


def _total_support(split_metrics: dict) -> int:
    return sum(cast(dict[str, int], split_metrics["support"]).values())


def run_entry(
    root: Path, fold: dict, files: StatsFiles, quality: QualityFeatures, arm: str, seed: int
) -> dict[str, object]:
    run_dir = fold_run_dir(root, fold["fold"], arm, seed)
    config = resolve_fold_config(arm, seed, files, quality)
    try:
        validate_fold_run(run_dir, config)
        validate_fold_run_document(
            run_dir,
            fold_run_document(
                fold["fold"], fold["retained_environment"], fold["development_environments"], arm, seed, files
            ),
        )
    except RuntimeError as exc:
        raise RuntimeError(f"run da dobra inválido em {run_dir}: {exc}") from exc
    final = _read_json(run_dir / "metrics.json")["final"]
    if final["val"] != final["test"]:
        raise RuntimeError(f"{run_dir}: val e test deveriam avaliar o mesmo ambiente retido")
    if _total_support(final["train"]) != fold["development_windows_train_stride"]:
        raise RuntimeError(f"{run_dir}: janelas de treino divergem de development_windows_train_stride")
    if _total_support(final["val"]) != fold["retained_windows_eval_stride"]:
        raise RuntimeError(f"{run_dir}: janelas do retido divergem de retained_windows_eval_stride")
    retained = final["val"]
    fraction, outside_other = predicted_other_fraction(retained)
    return {
        "fold": fold["fold"],
        "arm": arm,
        "seed": seed,
        "run_dir": display_path(run_dir),
        "retained_windows_outside_other": outside_other,
        "predicted_other_fraction": fraction,
        "retained_macro_f1_restricted": retained["macro_f1_restricted"],
    }


def _fold_comparison(runs: list[dict[str, object]]) -> dict[str, object]:
    def fractions(arm: str) -> list[float]:
        values = [run["predicted_other_fraction"] for run in runs if run["arm"] == arm]
        if not values or any(value is None for value in values):
            raise ValueError(f"{arm}: fração predita como other indisponível no ambiente retido")
        return cast(list[float], values)

    a_max = max(fractions("A"))
    comparison: dict[str, object] = {"A_max": a_max}
    for arm in COMPARED_ARMS:
        arm_min = min(fractions(arm))
        comparison[f"{arm}_min"] = arm_min
        comparison[f"{arm}_above_A"] = arm_min > a_max
    return comparison


def apply_criterion(fold_blocks: list[dict]) -> dict[str, object]:
    comparisons = {block["fold"]: _fold_comparison(block["runs"]) for block in fold_blocks}
    reproducing = {
        arm: [fold for fold, comparison in comparisons.items() if comparison[f"{arm}_above_A"]]
        for arm in COMPARED_ARMS
    }
    reproduced_by_arm = {arm: len(folds) >= MINIMUM_REPRODUCING_FOLDS for arm, folds in reproducing.items()}
    return {
        "comparisons": comparisons,
        "reproducing_folds": reproducing,
        "reproduced_by_arm": reproduced_by_arm,
        "reproduced": any(reproduced_by_arm.values()),
    }


def build_summary(root: Path, seed_set: str) -> dict[str, object]:
    document = read_folds_document(root)
    seeds = SEED_SETS[seed_set]
    quality = QualityFeatures(
        root=Path(cast(str, document["quality_features_path"])),
        sha256=cast(str, document["quality_features_sha256"]),
    )
    folds = cast(list[dict], document["folds"])
    missing = [
        display_path(fold_run_dir(root, fold["fold"], arm, seed))
        for fold in folds
        for arm in ARMS
        for seed in seeds
        if not fold_run_dir(root, fold["fold"], arm, seed).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"runs ausentes para o conjunto {seed_set}: {missing}")

    input_sha256 = cast(dict[str, str], document["input_sha256"])
    fold_blocks: list[dict] = []
    for fold in folds:
        files = load_recorded_stats_files(stats_dir(root, fold["fold"]), fold, input_sha256)
        fold_blocks.append(
            {
                "fold": fold["fold"],
                "retained_environment": fold["retained_environment"],
                "development_environments": fold["development_environments"],
                "runs": [run_entry(root, fold, files, quality, arm, seed) for arm in ARMS for seed in seeds],
            }
        )

    criterion: dict[str, object] = {"text": REPRODUCTION_CRITERION, "seed_set": seed_set, "seeds": list(seeds)}
    if seed_set == "assessment":
        result = apply_criterion(fold_blocks)
        comparisons = cast(dict[str, dict[str, object]], result.pop("comparisons"))
        for block in fold_blocks:
            block["comparison"] = comparisons[block["fold"]]
        criterion.update(status="evaluated", **result)
    else:
        criterion.update(
            status="not_applicable",
            reason="O piloto tem uma seed por arma: o critério compara mínimo e máximo entre seeds e não se aplica. "
            "O piloto só valida o pipeline das dobras.",
        )
    return {
        "protocol": PROTOCOL,
        "canonical": False,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "input_sha256": input_sha256,
        "membership_sha256": document["membership_sha256"],
        "restricted_classes": list(RESTRICTED_CLASSES),
        "evaluation_split_mapping": EVALUATION_SPLIT_MAPPING,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "criterion": criterion,
        "folds": fold_blocks,
        "limitations": LIMITATIONS,
    }


def _fmt(value: object) -> str:
    return "—" if value is None else f"{cast(float, value):.3f}"


def render_summary(summary: dict[str, object]) -> str:
    criterion = cast(dict[str, object], summary["criterion"])
    lines = [
        "# Dobras internas por ambiente do Le2i-CV (não canônico)",
        "",
        "Treinos exploratórios em dobras do train oficial do Le2i-CV; val e test oficiais não são lidos. "
        "O ambiente retido ocupa as posições val e test do engine. Checkpoint da última época.",
        "",
        f"Conjunto de seeds: `{criterion['seed_set']}` "
        f"({', '.join(str(seed) for seed in cast(list, criterion['seeds']))}).",
    ]
    for block in cast(list[dict[str, object]], summary["folds"]):
        lines += [
            "",
            f"## Dobra `{block['fold']}` — retém {block['retained_environment']}",
            "",
            "| Arma | Seed | Janelas fora de other | Fração predita other | Macro-F1 restrita (retido) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for entry in cast(list[dict[str, object]], block["runs"]):
            lines.append(
                f"| {entry['arm']} | {entry['seed']} | {entry['retained_windows_outside_other']} | "
                f"{_fmt(entry['predicted_other_fraction'])} | {_fmt(entry['retained_macro_f1_restricted'])} |"
            )
        if "comparison" in block:
            comparison = cast(dict[str, object], block["comparison"])
            lines += [
                "",
                f"Maior fração de A: {_fmt(comparison['A_max'])}. Menor de B0: {_fmt(comparison['B0_min'])} "
                f"(acima de A: {'sim' if comparison['B0_above_A'] else 'não'}). Menor de B1: "
                f"{_fmt(comparison['B1_min'])} (acima de A: {'sim' if comparison['B1_above_A'] else 'não'}).",
            ]
    lines += ["", "## Critério de reprodução (congelado na Etapa 2A)", "", str(criterion["text"]), ""]
    if criterion["status"] == "evaluated":
        reproducing = cast(dict[str, list[str]], criterion["reproducing_folds"])
        for arm, folds in reproducing.items():
            lines.append(f"- {arm}: {len(folds)} de 3 dobras ({', '.join(folds) or 'nenhuma'}).")
        lines += ["", f"Deslocamento reproduzido: **{'sim' if criterion['reproduced'] else 'não'}**."]
    else:
        lines.append(f"Não aplicado: {criterion['reason']}")
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in cast(list[str], summary["limitations"])]
    return "\n".join(lines) + "\n"


def write_summary(root: Path, seed_set: str, force: bool) -> bool:
    summary = build_summary(root, seed_set)
    directory = summary_dir(root) / seed_set
    targets = {
        directory / SUMMARY_JSON_FILE: json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        directory / SUMMARY_MARKDOWN_FILE: render_summary(summary),
    }
    return write_text_outputs(directory, targets, force)
