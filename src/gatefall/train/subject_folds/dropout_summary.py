"""Resumo pareado da perda simulada de detecção contra os controles da Etapa 4B (Etapa 4C).

Lê só artefatos gravados. Cada controle passa pela mesma validação do resumo
da Etapa 4B, e cada run aumentado também é reconciliado com a intervenção
congelada e com o `fold_run.json` atual do seu controle. Os deltas são
pareados por dobra, braço e seed (aumentado menos controle) e nunca são
agregados entre dobras; o critério exploratório é o congelado em
`DECISION_RULE` antes dos treinos.
"""

import json
from pathlib import Path
from typing import cast

from gatefall.eval.analysis.classification_diagnosis import display_path, write_text_outputs
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.train.subject_folds.detection_dropout import (
    DECISION_RULE,
    MIN_FOLD_DELTA,
    MIN_POSITIVE_FOLDS,
    STAGE,
    intervention_record,
)
from gatefall.train.subject_folds.folds import (
    ARMS,
    BASELINE_VARIANT,
    CHECKPOINT_POLICY,
    DETECTION_DROPOUT_VARIANT,
    EVALUATION_SPLIT_MAPPING,
    PROTOCOL,
    RESTRICTED_LABELS,
    SEEDS,
    SELECTION_POLICY,
    fold_run_dir,
    read_folds_document,
    summary_dir,
)
from gatefall.train.subject_folds.summary import fold_limitations, run_entry

SUMMARY_JSON_FILE = "subject_fold_detection_dropout_summary.json"
SUMMARY_MARKDOWN_FILE = "subject_fold_detection_dropout_summary.md"
FOCUS_CLASSES = ("standing", "fall")

LIMITATIONS = [
    "Exploratório e não canônico: o critério é uma regra de decisão predeclarada, não um teste de significância, "
    "e nada aqui valida o alvo de 0,800 nem estima o desempenho no test oficial, que não é lido.",
    "Três dobras de tamanhos muito diferentes e uma seed: um delta pequeno pode vir só da variação entre seeds, "
    "que não é estimada. Os deltas não são agregados entre dobras.",
    "O retido é avaliado limpo, como na Etapa 4B: o resumo mede o efeito do aumento no desempenho em dados reais, "
    "não a robustez a perdas de detecção simuladas no retido.",
    "A intervenção simula só lacunas curtas de três quadros dentro da janela, com probabilidade fixa; ela não "
    "reproduz a distribuição real das perdas de detecção, que se concentram em fall e em Home_01.",
    "Janelas que terminam antes do terceiro quadro do vídeo não têm três quadros reais distintos e ficam limpas.",
    "final.train do run aumentado é medido sobre janelas com a intervenção sorteada na última passagem e não é "
    "comparável ao do controle; só o retido entra nos deltas.",
    "Contagens por janela usam janelas sobrepostas (stride 1 no retido) e não são observações independentes.",
]


def evaluate_decision(deltas: list[float]) -> dict[str, object]:
    positive = sum(1 for delta in deltas if delta > 0)
    lowest = min(deltas)
    met = positive >= MIN_POSITIVE_FOLDS and lowest >= MIN_FOLD_DELTA
    return {
        "positive_folds": positive,
        "n_folds": len(deltas),
        "min_delta": lowest,
        "criterion_met": met,
    }


def paired_entry(control: dict, treatment: dict) -> dict[str, object]:
    if treatment["retained_support"] != control["retained_support"]:
        raise RuntimeError(f"{treatment['run_dir']}: suporte do retido diverge do controle {control['run_dir']}")
    control_classes = cast(dict[str, dict], control["retained_per_class"])
    treatment_classes = cast(dict[str, dict], treatment["retained_per_class"])
    control_macro = cast(float, control["retained_macro_f1_restricted"])
    treatment_macro = cast(float, treatment["retained_macro_f1_restricted"])
    return {
        "arm": control["arm"],
        "seed": control["seed"],
        "control": control,
        "treatment": treatment,
        "delta_macro_f1_restricted": treatment_macro - control_macro,
        "per_class": {
            label: {
                "control_f1": control_classes[label]["f1"],
                "treatment_f1": treatment_classes[label]["f1"],
                "delta_f1": treatment_classes[label]["f1"] - control_classes[label]["f1"],
                "support": control_classes[label]["support"],
            }
            for label in RESTRICTED_LABELS
        },
    }


def _decisions(fold_blocks: list[dict]) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for arm in ARMS:
        for seed in SEEDS:
            pairs = {
                block["fold"]: next(p for p in block["pairs"] if p["arm"] == arm and p["seed"] == seed)
                for block in fold_blocks
            }
            deltas = {fold: cast(float, pair["delta_macro_f1_restricted"]) for fold, pair in pairs.items()}
            decisions.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "fold_deltas": deltas,
                    **evaluate_decision(list(deltas.values())),
                    "focus_classes": {
                        label: {fold: pair["per_class"][label] for fold, pair in pairs.items()}
                        for label in FOCUS_CLASSES
                    },
                }
            )
    return decisions


def build_dropout_summary(root: Path) -> dict[str, object]:
    document = read_folds_document(root)
    folds = cast(list[dict], document["folds"])
    missing = [
        display_path(fold_run_dir(root, fold["fold"], arm, seed, variant))
        for fold in folds
        for arm in ARMS
        for seed in SEEDS
        for variant in (BASELINE_VARIANT, DETECTION_DROPOUT_VARIANT)
        if not fold_run_dir(root, fold["fold"], arm, seed, variant).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"runs ausentes: {missing}")
    fold_blocks = [
        {
            "fold": fold["fold"],
            "retained_subjects": fold["retained_subjects"],
            "retained_windows_eval_stride": fold["retained_windows_eval_stride"],
            "n_retained_videos": len(fold["retained_video_ids"]),
            "pairs": [
                paired_entry(
                    run_entry(root, document, fold, arm, seed),
                    run_entry(root, document, fold, arm, seed, DETECTION_DROPOUT_VARIANT),
                )
                for arm in ARMS
                for seed in SEEDS
            ],
            "limitations": fold_limitations(fold),
        }
        for fold in folds
    ]
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "canonical": False,
        "variant": DETECTION_DROPOUT_VARIANT,
        "control_variant": BASELINE_VARIANT,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "input_sha256": document["input_sha256"],
        "pose_features_sha256": document["pose_features_sha256"],
        "membership_sha256": document["membership_sha256"],
        "evaluation_split_mapping": EVALUATION_SPLIT_MAPPING,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "selection_policy": SELECTION_POLICY,
        "interventions": {str(seed): intervention_record(seed) for seed in SEEDS},
        "decision_rule": DECISION_RULE,
        "pooled_score": None,
        "pooled_delta": None,
        "folds": fold_blocks,
        "decisions": _decisions(fold_blocks),
        "limitations": LIMITATIONS,
    }


def _fmt(value: object) -> str:
    return f"{cast(float, value):.3f}"


def _signed(value: object) -> str:
    return f"{cast(float, value):+.4f}"


def _render_decision(decision: dict[str, object]) -> list[str]:
    deltas = cast(dict[str, float], decision["fold_deltas"])
    outcome = "atendido" if decision["criterion_met"] else "não atendido"
    lines = [
        "",
        f"## Critério exploratório — braço {decision['arm']}, seed {decision['seed']}",
        "",
        "| Dobra | Δ macro-F1 restrita |",
        "| --- | ---: |",
        *(f"| `{fold}` | {_signed(delta)} |" for fold, delta in deltas.items()),
        "",
        f"Delta positivo em {decision['positive_folds']} de {decision['n_folds']} dobras; menor delta "
        f"{_signed(decision['min_delta'])}. Critério **{outcome}** (positivo em ao menos {MIN_POSITIVE_FOLDS} "
        f"dobras e nenhum delta abaixo de {MIN_FOLD_DELTA:+.2f}).",
    ]
    for label, by_fold in cast(dict[str, dict[str, dict]], decision["focus_classes"]).items():
        lines += [
            "",
            f"`{label}`:",
            "",
            "| Dobra | F1 controle | F1 aumentado | Δ F1 | Suporte (janelas) |",
            "| --- | ---: | ---: | ---: | ---: |",
            *(
                f"| `{fold}` | {_fmt(values['control_f1'])} | {_fmt(values['treatment_f1'])} | "
                f"{_signed(values['delta_f1'])} | {values['support']} |"
                for fold, values in by_fold.items()
            ),
        ]
    return lines


def render_dropout_summary(summary: dict[str, object]) -> str:
    rule = cast(dict[str, object], summary["decision_rule"])
    lines = [
        "# Dobras por sujeito do Le2i-CS — perda simulada de detecção vs. controle (Etapa 4C, não canônico)",
        "",
        "Cada run aumentado treina com a perda simulada de três quadros consecutivos de detecção em 20% das "
        "janelas sorteadas do treino de desenvolvimento e é comparado ao controle da Etapa 4B da mesma dobra, "
        "seed e receita. O retido é avaliado limpo. Os deltas são aumentado menos controle e não são agregados "
        "entre dobras.",
        "",
        str(rule["statement"]),
    ]
    for decision in cast(list[dict[str, object]], summary["decisions"]):
        lines += _render_decision(decision)
    for block in cast(list[dict[str, object]], summary["folds"]):
        retained = ", ".join(str(subject) for subject in cast(list[int], block["retained_subjects"]))
        lines += [
            "",
            f"## Dobra `{block['fold']}` — retém os sujeitos {retained}",
            "",
            f"Retido: {block['n_retained_videos']} vídeos, {block['retained_windows_eval_stride']} janelas rotuladas "
            "em stride 1.",
        ]
        for pair in cast(list[dict[str, object]], block["pairs"]):
            control = cast(dict[str, object], pair["control"])
            treatment = cast(dict[str, object], pair["treatment"])
            per_class = cast(dict[str, dict[str, object]], pair["per_class"])
            lines += [
                "",
                f"Braço {pair['arm']}, seed {pair['seed']}: macro-F1 restrita no retido "
                f"{_fmt(control['retained_macro_f1_restricted'])} no controle e "
                f"{_fmt(treatment['retained_macro_f1_restricted'])} com a intervenção "
                f"(Δ **{_signed(pair['delta_macro_f1_restricted'])}**).",
                "",
                "| Classe | F1 controle | F1 aumentado | Δ F1 | Suporte (janelas) |",
                "| --- | ---: | ---: | ---: | ---: |",
                *(
                    f"| {label} | {_fmt(values['control_f1'])} | {_fmt(values['treatment_f1'])} | "
                    f"{_signed(values['delta_f1'])} | {values['support']} |"
                    for label, values in per_class.items()
                ),
            ]
        lines += ["", "Limitações da dobra:", ""]
        lines += [f"- {text}" for text in cast(list[str], block["limitations"])]
    lines += ["", "## Limitações gerais", ""]
    lines += [f"- {text}" for text in cast(list[str], summary["limitations"])]
    return "\n".join(lines) + "\n"


def write_dropout_summary(root: Path, force: bool) -> bool:
    summary = build_dropout_summary(root)
    directory = summary_dir(root) / DETECTION_DROPOUT_VARIANT
    targets = {
        directory / SUMMARY_JSON_FILE: json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        directory / SUMMARY_MARKDOWN_FILE: render_dropout_summary(summary),
    }
    return write_text_outputs(directory, targets, force)
