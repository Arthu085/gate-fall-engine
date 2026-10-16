"""Resumo pareado da amostragem balanceada por sujeito contra os controles da Etapa 4B (Etapa 5A).

Lê só artefatos gravados. Cada controle passa pela validação do resumo da
Etapa 4B, e cada run balanceado também é reconciliado com a intervenção
congelada, com as contagens sorteadas gravadas e com o `fold_run.json` atual
do seu controle. Os deltas são pareados por dobra, braço e seed (balanceado
menos controle) e nunca são agregados entre dobras; o critério exploratório é
o congelado em `DECISION_RULE` antes dos treinos.
"""

import json
from pathlib import Path
from typing import cast

from gatefall.eval.analysis.classification_diagnosis import display_path, write_text_outputs
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.train.internal_folds.training import FOLD_RUN_FILE
from gatefall.train.subject_folds.dropout_summary import paired_entry
from gatefall.train.subject_folds.folds import (
    ARMS,
    BASELINE_VARIANT,
    CHECKPOINT_POLICY,
    EVALUATION_SPLIT_MAPPING,
    PROTOCOL,
    SEEDS,
    SELECTION_POLICY,
    SUBJECT_BALANCED_VARIANT,
    fold_run_dir,
    read_folds_document,
    summary_dir,
)
from gatefall.train.subject_folds.subject_balanced_sampling import (
    DECISION_RULE,
    MIN_FOLD_DELTA,
    MIN_IMPROVED_FOLDS,
    MIN_IMPROVEMENT,
    STAGE,
    intervention_record,
)
from gatefall.train.subject_folds.summary import fold_limitations, run_entry

SUMMARY_JSON_FILE = "subject_fold_subject_balanced_summary.json"
SUMMARY_MARKDOWN_FILE = "subject_fold_subject_balanced_summary.md"
FOCUS_CLASSES = ("standing", "other", "fall")

LIMITATIONS = [
    "Exploratório e não canônico, mesmo que o critério seja atendido: o critério é uma regra de decisão "
    "predeclarada, não um teste de significância, e nada aqui valida o alvo de 0,800 nem estima o desempenho no "
    "test oficial, que não é lido.",
    "Três dobras de tamanhos muito diferentes e uma seed de treino: a variação entre seeds não é estimada, e o "
    "sorteio por época usa um único fluxo aleatório por dobra. Os deltas não são agregados entre dobras.",
    "Desequilíbrio entre sujeitos: sujeitos com poucas janelas são repetidos muitas vezes por época, e sujeitos "
    "dominantes têm parte das janelas nunca sorteada; o efeito mistura menos domínio de poucos sujeitos com mais "
    "sobreajuste aos vídeos dos sujeitos pequenos.",
    "A amostragem por sujeito muda também a distribuição de classes vista no treino; os pesos de classe continuam "
    "os do treino limpo completo, como no controle, e não compensam essa mudança.",
    "Janelas sobrepostas: no treino (stride 4) sorteios repetidos de janelas vizinhas reforçam a redundância, e no "
    "retido (stride 1) as contagens por janela não são observações independentes.",
    "final.train é medido sobre todas as janelas limpas do treino de desenvolvimento, sem o sampler; ele não "
    "descreve a distribuição sorteada, e só o retido entra nos deltas.",
]


def evaluate_decision(deltas: list[float]) -> dict[str, object]:
    improved = sum(1 for delta in deltas if delta >= MIN_IMPROVEMENT)
    lowest = min(deltas)
    return {
        "improved_folds": improved,
        "n_folds": len(deltas),
        "min_delta": lowest,
        "criterion_met": improved >= MIN_IMPROVED_FOLDS and lowest >= MIN_FOLD_DELTA,
    }


def sampling_block(sampling: dict) -> dict[str, object]:
    subject_windows = cast(dict[str, int], sampling["subject_windows"])
    sampled = cast(dict[str, int], sampling["sampled_subject_windows_total"])
    distinct = cast(dict[str, int], sampling["distinct_windows_drawn_by_subject"])
    clean_support = cast(dict[str, int], sampling["clean_class_support"])
    sampled_classes = cast(dict[str, int], sampling["sampled_class_windows_total"])
    windows = cast(int, sampling["development_windows"])
    epochs = cast(int, sampling["epochs"])
    return {
        "epochs": epochs,
        "draws_per_epoch": sampling["draws_per_epoch"],
        "draws_sha256": sampling["draws_sha256"],
        "subjects": {
            subject: {
                "clean_windows": count,
                "clean_share": count / windows,
                "window_probability": sampling["window_probability"][subject],
                "sampled_total": sampled[subject],
                "sampled_per_epoch": sampled[subject] / epochs,
                "sampled_share": sampled[subject] / (windows * epochs),
                "distinct_windows_drawn": distinct[subject],
            }
            for subject, count in subject_windows.items()
        },
        "classes": {
            label: {
                "clean_support": clean_support[label],
                "sampled_total": sampled_classes[label],
                "sampled_per_epoch": sampled_classes[label] / epochs,
            }
            for label in clean_support
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


def _pair(root: Path, document: dict, fold: dict, arm: str, seed: int) -> dict[str, object]:
    control = run_entry(root, document, fold, arm, seed)
    treatment = run_entry(root, document, fold, arm, seed, SUBJECT_BALANCED_VARIANT)
    sidecar = json.loads(
        (fold_run_dir(root, fold["fold"], arm, seed, SUBJECT_BALANCED_VARIANT) / FOLD_RUN_FILE).read_text(
            encoding="utf-8"
        )
    )
    return {**paired_entry(control, treatment), "sampling": sampling_block(sidecar["sampling"])}


def build_sampling_summary(root: Path) -> dict[str, object]:
    document = read_folds_document(root)
    folds = cast(list[dict], document["folds"])
    missing = [
        display_path(fold_run_dir(root, fold["fold"], arm, seed, variant))
        for fold in folds
        for arm in ARMS
        for seed in SEEDS
        for variant in (BASELINE_VARIANT, SUBJECT_BALANCED_VARIANT)
        if not fold_run_dir(root, fold["fold"], arm, seed, variant).is_dir()
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
            "pairs": [_pair(root, document, fold, arm, seed) for arm in ARMS for seed in SEEDS],
            "limitations": fold_limitations(fold),
        }
        for fold in folds
    ]
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "canonical": False,
        "variant": SUBJECT_BALANCED_VARIANT,
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
        f"Delta de pelo menos {MIN_IMPROVEMENT:+.2f} em {decision['improved_folds']} de {decision['n_folds']} "
        f"dobras; menor delta {_signed(decision['min_delta'])}. Critério **{outcome}** (pelo menos "
        f"{MIN_IMPROVEMENT:+.2f} em ao menos {MIN_IMPROVED_FOLDS} dobras e nenhum delta abaixo de "
        f"{MIN_FOLD_DELTA:+.2f}). A variante segue não canônica em qualquer caso.",
    ]
    for label, by_fold in cast(dict[str, dict[str, dict]], decision["focus_classes"]).items():
        lines += [
            "",
            f"`{label}`:",
            "",
            "| Dobra | F1 controle | F1 balanceado | Δ F1 | Suporte (janelas) |",
            "| --- | ---: | ---: | ---: | ---: |",
            *(
                f"| `{fold}` | {_fmt(values['control_f1'])} | {_fmt(values['treatment_f1'])} | "
                f"{_signed(values['delta_f1'])} | {values['support']} |"
                for fold, values in by_fold.items()
            ),
        ]
    return lines


def _render_sampling(sampling: dict[str, object]) -> list[str]:
    subjects = cast(dict[str, dict[str, object]], sampling["subjects"])
    classes = cast(dict[str, dict[str, object]], sampling["classes"])
    return [
        "",
        f"Amostragem: {sampling['draws_per_epoch']} sorteios com reposição por época, {sampling['epochs']} épocas.",
        "",
        "| Sujeito | Janelas limpas | Fração limpa | Sorteadas (total) | Sorteadas por época | Fração sorteada | "
        "Janelas distintas sorteadas |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(
            f"| {subject} | {values['clean_windows']} | {_fmt(values['clean_share'])} | {values['sampled_total']} | "
            f"{cast(float, values['sampled_per_epoch']):.1f} | {_fmt(values['sampled_share'])} | "
            f"{values['distinct_windows_drawn']} |"
            for subject, values in subjects.items()
        ),
        "",
        "| Classe | Suporte limpo (janelas) | Sorteadas (total) | Sorteadas por época |",
        "| --- | ---: | ---: | ---: |",
        *(
            f"| {label} | {values['clean_support']} | {values['sampled_total']} | "
            f"{cast(float, values['sampled_per_epoch']):.1f} |"
            for label, values in classes.items()
        ),
    ]


def render_sampling_summary(summary: dict[str, object]) -> str:
    rule = cast(dict[str, object], summary["decision_rule"])
    lines = [
        "# Dobras por sujeito do Le2i-CS — amostragem balanceada por sujeito vs. controle (Etapa 5A, não canônico)",
        "",
        "Cada run balanceado sorteia, com reposição, o número original de janelas do treino de desenvolvimento em "
        "cada época, com probabilidade igual por sujeito, e é comparado ao controle da Etapa 4B da mesma dobra, "
        "seed e receita. O retido é avaliado limpo. Os deltas são balanceado menos controle e não são agregados "
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
            f"Treino de desenvolvimento: {block['development_windows_train_stride']} janelas rotuladas em stride 4. "
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
                f"{_fmt(treatment['retained_macro_f1_restricted'])} com a amostragem balanceada "
                f"(Δ **{_signed(pair['delta_macro_f1_restricted'])}**).",
                "",
                "| Classe | F1 controle | F1 balanceado | Δ F1 | Suporte (janelas) |",
                "| --- | ---: | ---: | ---: | ---: |",
                *(
                    f"| {label} | {_fmt(values['control_f1'])} | {_fmt(values['treatment_f1'])} | "
                    f"{_signed(values['delta_f1'])} | {values['support']} |"
                    for label, values in per_class.items()
                ),
            ]
            lines += _render_sampling(cast(dict[str, object], pair["sampling"]))
        lines += ["", "Limitações da dobra:", ""]
        lines += [f"- {text}" for text in cast(list[str], block["limitations"])]
    lines += ["", "## Limitações gerais", ""]
    lines += [f"- {text}" for text in cast(list[str], summary["limitations"])]
    return "\n".join(lines) + "\n"


def write_sampling_summary(root: Path, force: bool) -> bool:
    summary = build_sampling_summary(root)
    directory = summary_dir(root) / SUBJECT_BALANCED_VARIANT
    targets = {
        directory / SUMMARY_JSON_FILE: json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        directory / SUMMARY_MARKDOWN_FILE: render_sampling_summary(summary),
    }
    return write_text_outputs(directory, targets, force)
