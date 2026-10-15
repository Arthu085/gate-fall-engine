"""Diagnóstico estratificado, somente leitura, dos erros de classificação por ambiente, sujeito e vídeo.

Os relatórios de classificação persistidos não guardam a identidade de cada
janela, então este módulo reusa o caminho de inferência validado de cada
avaliador de eventos (`load_event_evaluation` de A, B0, B1, C0 e C1) sobre o
checkpoint já treinado para obter `(video_id, k_end, rótulo verdadeiro,
rótulo predito)` em `val` e `test`. Não retreina, não reextrai features, não
altera checkpoints e não chama o ciclo de publicação de `event_metrics.json`.

Cada predição é ligada à janela exata de `build_window_index` (stride de
avaliação, sem descartar `IGNORE_LABEL`) e aos metadados de ambiente e
sujeito do vídeo. Janelas `IGNORE_LABEL` ficam fora das métricas, a
Macro-F1 restrita mantém as oito classes fixas de `RESTRICTED_CLASSES` e as
matrizes somadas de cada nível de agrupamento precisam reproduzir exatamente
a `confusion_matrix` de `metrics.json`. Sujeitos são agrupados pelo par
(ambiente, sujeito), nunca pelo ID bruto. O resultado é exploratório e não
canônico; o split `test` é usado só descritivamente.

As predições analisadas são as da reexecução validada, não as históricas da
avaliação de treino, que nunca foram persistidas por janela. A reconciliação
com `metrics.json` é uma checagem de integridade necessária, mas não prova
que a reexecução atribuiu os mesmos rótulos às mesmas janelas: atribuições
diferentes podem produzir a mesma matriz global e matrizes de subgrupo
diferentes.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, NUM_CLASSES
from gatefall.data.windowing import build_window_index
from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    PROTOCOL_HELD_OUT_FACTOR,
    PROTOCOL_NAMES,
    DiagnosisInput,
    _parse_inputs,
    _read_json,
    _read_yaml,
    _validate_dataset_and_arm,
    _validate_run_identity,
    check_output_dir,
    diagnose_split,
    display_path,
    dominant_confusions,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.grouped_bootstrap import SPLITS, load_arm_evaluation
from gatefall.eval.analysis.multiseed_summary import ARMS
from gatefall.eval.baseline_a import cli as baseline_a_events
from gatefall.eval.shared.orchestration import EventEvaluation, Predictions
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT, validate_local_run_dir
from gatefall.train.shared.metrics import (
    RESTRICTED_CLASSES,
    classification_summary,
    restricted_macro_f1,
)

STRATIFIED_JSON_FILE = "stratified_error_diagnosis.json"
STRATIFIED_CSV_FILE = "stratified_error_diagnosis.csv"
STRATIFIED_SUMMARY_FILE = "stratified_error_diagnosis.md"

REQUIRED_RUN_FILES: tuple[str, ...] = ("config.yaml", "metrics.json", "checkpoint.pt")
LEVELS: tuple[str, ...] = ("environment", "environment_subject", "video")
PER_CLASS_FIELDS: tuple[str, ...] = ("support", "precision", "recall", "f1", "tp", "fp", "fn")
REPORTED_ITEMS = 3

# Limiares que apenas disparam hipóteses no resumo; não filtram nem
# selecionam nenhum resultado.
LOW_GROUP_MACRO_F1_GAP = 0.2
ERROR_CONCENTRATION_SHARE = 0.25

CSV_COLUMNS = [
    "dataset",
    "protocol",
    "arm",
    "seed",
    "source",
    "split",
    "level",
    "group",
    "env",
    "subject",
    "video_id",
    "entity",
    "metric",
    "value",
]


def load_evaluation(arm: str, dataset: str, run_dir: Path) -> EventEvaluation:
    if arm == "A":
        # `load_arm_evaluation` fixa a seed congelada de A; aqui A aceita
        # qualquer seed de treino, como B0, B1, C0 e C1.
        return baseline_a_events.load_event_evaluation(dataset, run_dir)
    return load_arm_evaluation(arm, dataset, run_dir)


def _video_metadata_conflicts(frames: pd.DataFrame) -> list[str]:
    counts = cast(pd.DataFrame, frames.groupby("video_id")[["split", "env", "subject"]].nunique())
    conflicts = counts[(counts > 1).any(axis=1)]
    return sorted(str(video_id) for video_id in conflicts.index)


def window_metadata(frames: pd.DataFrame, split: str, context: str) -> pd.DataFrame:
    conflicts = _video_metadata_conflicts(frames)
    if conflicts:
        raise ValueError(f"{context}: vídeo(s) com split, ambiente ou sujeito ambíguo: {conflicts[:5]}")
    split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
    windows = build_window_index(split_frames, stride=EVAL_STRIDE, drop_ignored=False)
    if windows.empty:
        raise ValueError(f"{context}: nenhuma janela no split")
    return pd.DataFrame(
        {
            "video_id": windows["video_id"].astype(str),
            "env": windows["env"].astype(str),
            "subject": windows["subject"].astype(np.int64),
            "k_end": windows["k_end"].astype(np.int64),
            "label": windows["label"].astype(np.int64),
        }
    )


def _identity_sample(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return [(str(video_id), int(k_end)) for video_id, k_end in zip(frame["video_id"], frame["k_end"])][:5]


def align_predictions(windows: pd.DataFrame, predictions: Predictions, context: str) -> pd.DataFrame:
    video_ids, k_ends, true_labels, pred_labels = predictions
    lengths = {len(video_ids), len(k_ends), len(true_labels), len(pred_labels)}
    if len(lengths) != 1:
        raise ValueError(f"{context}: listas de predição com tamanhos diferentes")
    predicted = pd.DataFrame(
        {
            "video_id": [str(video_id) for video_id in video_ids],
            "k_end": np.asarray(k_ends, dtype=np.int64),
            "true_label": np.asarray(true_labels, dtype=np.int64),
            "pred_label": np.asarray(pred_labels, dtype=np.int64),
        }
    )
    duplicated = cast(pd.DataFrame, predicted[predicted.duplicated(["video_id", "k_end"], keep=False)])
    if not duplicated.empty:
        raise ValueError(f"{context}: identidade de janela duplicada nas predições: {_identity_sample(duplicated)}")
    invalid = cast(
        pd.DataFrame,
        predicted[(predicted["pred_label"] < 0) | (predicted["pred_label"] >= NUM_CLASSES)],
    )
    if not invalid.empty:
        raise ValueError(f"{context}: rótulo predito fora de 0..{NUM_CLASSES - 1}: {_identity_sample(invalid)}")

    joined = windows.merge(predicted, on=["video_id", "k_end"], how="outer", indicator=True)
    missing = cast(pd.DataFrame, joined[joined["_merge"] == "left_only"])
    extra = cast(pd.DataFrame, joined[joined["_merge"] == "right_only"])
    if not missing.empty or not extra.empty:
        raise ValueError(
            f"{context}: predições não alinham com o índice de janelas; sem predição: "
            f"{len(missing)} {_identity_sample(missing)}; sem janela: {len(extra)} {_identity_sample(extra)}"
        )
    inconsistent = cast(pd.DataFrame, joined[joined["label"] != joined["true_label"]])
    if not inconsistent.empty:
        raise ValueError(
            f"{context}: rótulo verdadeiro das predições diverge do índice de janelas em "
            f"{len(inconsistent)} janela(s): {_identity_sample(inconsistent)}"
        )
    aligned = cast(pd.DataFrame, joined[["video_id", "env", "subject", "k_end", "label", "pred_label"]])
    return aligned.sort_values(["video_id", "k_end"]).reset_index(drop=True)


def subgroup_metrics(table: pd.DataFrame) -> dict:
    labeled = cast(pd.DataFrame, table[table["label"] != IGNORE_LABEL])
    y_true = labeled["label"].to_numpy(dtype=np.int64)
    y_pred = labeled["pred_label"].to_numpy(dtype=np.int64)
    summary = classification_summary(y_true, y_pred, LE2I_LABEL_NAMES, NUM_CLASSES)
    macro_f1, _ = restricted_macro_f1(y_true, y_pred, NUM_CLASSES)
    per_class = {
        label: {field: row[field] for field in PER_CLASS_FIELDS} for label, row in summary["per_class"].items()
    }
    absent = [label for label, row in per_class.items() if row["support"] == 0]
    restricted_labels = {LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES}
    matrix = summary["confusion_matrix"]
    return {
        "n_windows": len(table),
        "n_labeled_windows": len(labeled),
        "n_ignored_windows": len(table) - len(labeled),
        "n_errors": int(np.sum(y_true != y_pred)),
        "macro_f1_restricted": macro_f1 if len(labeled) > 0 else None,
        "per_class": per_class,
        "absent_labels": absent,
        "absent_restricted_labels": [label for label in absent if label in restricted_labels],
        "predicted_absent_labels": [label for label in absent if per_class[label]["fp"] > 0],
        "confusion_matrix": matrix,
        "dominant_confusions": dominant_confusions(matrix, LE2I_LABEL_NAMES),
    }


def _n_videos(group: pd.DataFrame) -> int:
    return len(set(group["video_id"].tolist()))


def stratify(table: pd.DataFrame) -> dict[str, list[dict]]:
    environments = [
        {"group": str(env), "env": str(env), "subject": None, "video_id": None,
         "n_videos": _n_videos(group), **subgroup_metrics(group)}
        for env, group in table.groupby("env", sort=True)
    ]
    environment_subjects = []
    for keys, group in table.groupby(["env", "subject"], sort=True):
        env, subject = cast(tuple[str, int], keys)
        environment_subjects.append(
            {"group": f"{env}/subject_{subject}", "env": str(env), "subject": int(subject), "video_id": None,
             "n_videos": _n_videos(group), **subgroup_metrics(group)}
        )
    videos = [
        {"group": str(video_id), "env": str(group["env"].iloc[0]), "subject": int(group["subject"].iloc[0]),
         "video_id": str(video_id), "n_videos": 1, **subgroup_metrics(group)}
        for video_id, group in table.groupby("video_id", sort=True)
    ]
    return {"environment": environments, "environment_subject": environment_subjects, "video": videos}


def reconcile(levels: dict[str, list[dict]], overall: dict, stored: dict, context: str) -> dict:
    expected = np.asarray(stored["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(np.asarray(overall["confusion_matrix"], dtype=np.int64), expected):
        raise ValueError(f"{context}: matriz recalculada das predições diverge de metrics.json")
    expected_support = {label: row["support"] for label, row in stored["per_class"].items()}
    for level, groups in levels.items():
        summed = np.sum([np.asarray(group["confusion_matrix"], dtype=np.int64) for group in groups], axis=0)
        if not np.array_equal(summed, expected):
            raise ValueError(f"{context}: soma das matrizes por {level} diverge de metrics.json")
        summed_support = {
            label: sum(group["per_class"][label]["support"] for group in groups) for label in LE2I_LABEL_NAMES
        }
        if summed_support != expected_support:
            raise ValueError(f"{context}: soma dos suportes por {level} diverge de metrics.json")
        for field in ("n_windows", "n_labeled_windows", "n_ignored_windows", "n_errors"):
            if sum(group[field] for group in groups) != overall[field]:
                raise ValueError(f"{context}: soma de {field} por {level} diverge do total do split")
    return {
        "ok": True,
        "prediction_source": "validated_rerun",
        "proves_historical_window_predictions": False,
        "levels_checked": list(levels),
        "n_labeled_windows": int(expected.sum()),
        "macro_f1_restricted_metrics_json": stored["macro_f1_restricted"],
    }


def coverage(table: pd.DataFrame, levels: dict[str, list[dict]]) -> dict:
    subject_envs = table.groupby("subject")["env"].unique()
    return {
        "n_environments": len(levels["environment"]),
        "n_environment_subject_groups": len(levels["environment_subject"]),
        "n_videos": len(levels["video"]),
        "videos_by_environment": {group["group"]: group["n_videos"] for group in levels["environment"]},
        "subject_ids_in_multiple_environments": {
            str(subject): sorted(str(env) for env in envs)
            for subject, envs in subject_envs.items()
            if len(envs) > 1
        },
        "groups_without_labeled_windows": {
            level: [group["group"] for group in groups if group["n_labeled_windows"] == 0]
            for level, groups in levels.items()
        },
    }


def analyze_split_with_table(
    frames: pd.DataFrame,
    evaluate_split: Callable[[str], tuple[int, Predictions]],
    split: str,
    stored_block: object,
    context: str,
) -> tuple[pd.DataFrame, dict]:
    stored = diagnose_split(stored_block, LE2I_LABEL_NAMES, context)
    windows = window_metadata(frames, split, context)
    usable_windows, predictions = evaluate_split(split)
    if usable_windows != len(windows):
        raise ValueError(
            f"{context}: avaliador relatou {usable_windows} janelas, índice de janelas tem {len(windows)}"
        )
    table = align_predictions(windows, predictions, context)
    overall = subgroup_metrics(table)
    levels = stratify(table)
    return table, {
        "overall": overall,
        "coverage": coverage(table, levels),
        "reconciliation": reconcile(levels, overall, stored, context),
        "levels": levels,
    }


def analyze_split(
    frames: pd.DataFrame,
    evaluate_split: Callable[[str], tuple[int, Predictions]],
    split: str,
    stored_block: object,
    context: str,
) -> dict:
    return analyze_split_with_table(frames, evaluate_split, split, stored_block, context)[1]


def _frames_path(dataset: str) -> Path:
    return get_dataset(dataset).frames_path


def missing_prerequisites(item: DiagnosisInput) -> list[str]:
    missing = [str(item.path / name) for name in REQUIRED_RUN_FILES if not (item.path / name).is_file()]
    if not _frames_path(item.dataset).is_file():
        missing.append(str(_frames_path(item.dataset)))
    return missing


def run_input_files(item: DiagnosisInput) -> list[Path]:
    return [*(item.path / name for name in REQUIRED_RUN_FILES), _frames_path(item.dataset)]


@dataclass(frozen=True)
class ValidatedRun:
    context: str
    header: dict
    final: dict


def validate_run(item: DiagnosisInput) -> ValidatedRun:
    context = f"{PROTOCOL_NAMES[item.dataset]}/{item.arm} ({item.path})"
    config = _read_yaml(item.path / "config.yaml")
    metrics = _read_json(item.path / "metrics.json")
    _validate_run_identity(config, item.dataset, item.arm, context)
    input_sha256 = {display_path(path): sha256_file(path) for path in run_input_files(item)}
    checkpoint_sha256 = input_sha256[display_path(item.path / "checkpoint.pt")]
    if metrics.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{context}: checkpoint.pt diverge do checkpoint_sha256 de metrics.json")
    final = metrics.get("final")
    if not isinstance(final, dict) or any(split not in final for split in SPLITS):
        raise ValueError(f"{context}: metrics.json sem final.{'/'.join(SPLITS)}")
    header = {
        "dataset": item.dataset,
        "protocol": PROTOCOL_NAMES[item.dataset],
        "arm": item.arm,
        "seed": config.get("seed"),
        "run_name": metrics.get("run_name"),
        "source": display_path(item.path),
        "checkpoint_sha256": checkpoint_sha256,
        "input_sha256": input_sha256,
    }
    return ValidatedRun(context=context, header=header, final=final)


def analyze_run(item: DiagnosisInput) -> dict:
    run = validate_run(item)
    frames, evaluate_split = load_evaluation(item.arm, item.dataset, item.path).prepare()
    splits = {
        split: analyze_split(frames, evaluate_split, split, run.final[split], f"{run.context} [{split}]")
        for split in SPLITS
    }
    return {**run.header, "splits": splits}


def _fmt(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.4f}"


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _entry_label(entry: dict) -> str:
    return f"{entry['arm']} (seed {entry['seed']})"


def _split_observations(split: str, analysis: dict) -> list[str]:
    overall = analysis["overall"]
    environments = [group for group in analysis["levels"]["environment"] if group["macro_f1_restricted"] is not None]
    if not environments:
        return []
    lowest = min(environments, key=lambda group: group["macro_f1_restricted"])
    highest = max(environments, key=lambda group: group["macro_f1_restricted"])
    environment_range = (
        f"ambiente único: {lowest['group']}"
        if len(environments) == 1
        else f"por ambiente, de {lowest['group']} ({_fmt(lowest['macro_f1_restricted'])}) a "
        f"{highest['group']} ({_fmt(highest['macro_f1_restricted'])})"
    )
    observations = [
        f"{split}: Macro-F1 restrita {_fmt(overall['macro_f1_restricted'])} em {overall['n_labeled_windows']} "
        f"janelas rotuladas ({overall['n_ignored_windows']} IGNORE_LABEL excluídas); {environment_range}."
    ]
    total_errors = overall["n_errors"]
    if total_errors == 0:
        return observations
    worst_env = max(analysis["levels"]["environment"], key=lambda group: (group["n_errors"], group["group"]))
    confusion = worst_env["dominant_confusions"][0]
    confusion_text = (
        f"{confusion['true_label']}→{confusion['predicted_label']} ({confusion['count']} janelas, "
        f"{_pct(confusion['share_of_true_support'])} do suporte de {confusion['true_label']})"
    )
    observations.append(
        f"{split}: confusão dominante: {confusion_text}."
        if len(analysis["levels"]["environment"]) == 1
        else f"{split}: {worst_env['group']} concentra {worst_env['n_errors']} de {total_errors} erros "
        f"({_pct(worst_env['n_errors'] / total_errors)}); confusão dominante nele: {confusion_text}."
    )
    videos = sorted(analysis["levels"]["video"], key=lambda group: (-group["n_errors"], group["group"]))
    top_videos = videos[:REPORTED_ITEMS]
    top_errors = sum(group["n_errors"] for group in top_videos)
    observations.append(
        f"{split}: os {len(top_videos)} vídeos com mais erros somam {_pct(top_errors / total_errors)} dos erros: "
        + ", ".join(f"{group['group']} ({group['n_errors']})" for group in top_videos)
        + "."
    )
    return observations


def _split_hypotheses(dataset: str, split: str, analysis: dict) -> list[str]:
    hypotheses: list[str] = []
    overall = analysis["overall"]
    overall_macro = overall["macro_f1_restricted"]
    for group in analysis["levels"]["environment"]:
        macro = group["macro_f1_restricted"]
        if macro is None or overall_macro is None or overall_macro - macro <= LOW_GROUP_MACRO_F1_GAP:
            continue
        absent = group["absent_restricted_labels"]
        if absent:
            hypotheses.append(
                f"{group['group']} em {split}: Macro-F1 {_fmt(macro)} contra {_fmt(overall_macro)} no split, com "
                f"{len(absent)} classe(s) restrita(s) ausente(s) ({', '.join(absent)}); parte da diferença pode "
                "ser mecânica (F1 0 das classes ausentes na média fixa de oito classes), não só erro do modelo."
            )
        else:
            cause = (
                "o ambiente retido em Le2i-CV também muda sujeitos, ações e suporte; não é efeito isolado de câmera ou backbone"
                if dataset == "le2i-cv"
                else "ambiente e sujeitos de test se confundem neste protocolo"
            )
            hypotheses.append(
                f"{group['group']} em {split}: Macro-F1 {_fmt(macro)} contra {_fmt(overall_macro)} no split sem "
                f"classes restritas ausentes; pode refletir dificuldade específica do grupo, mas {cause}."
            )
    total_errors = overall["n_errors"]
    if total_errors > 0:
        top_video = max(analysis["levels"]["video"], key=lambda group: (group["n_errors"], group["group"]))
        share = top_video["n_errors"] / total_errors
        if share >= ERROR_CONCENTRATION_SHARE:
            hypotheses.append(
                f"{split}: {top_video['group']} responde por {_pct(share)} dos erros; as métricas do split podem "
                "ser dominadas por poucos vídeos, e janelas sobrepostas do mesmo vídeo não são independentes."
            )
    return hypotheses


def build_findings(protocols: dict) -> tuple[list[dict], list[dict]]:
    observations: list[dict] = []
    hypotheses: list[dict] = []
    for dataset, block in protocols.items():
        for arm, entries in block["arms"].items():
            for entry in entries:
                base = {"dataset": dataset, "protocol": block["name"], "arm": arm, "seed": entry["seed"],
                        "source": entry["source"]}
                for split, analysis in entry["splits"].items():
                    observations.extend({**base, "split": split, "text": text}
                                        for text in _split_observations(split, analysis))
                    hypotheses.extend({**base, "split": split, "text": text}
                                      for text in _split_hypotheses(dataset, split, analysis))
    return observations, hypotheses


def _limitations(protocols: dict) -> list[str]:
    limitations = [
        "Análise exploratória e não canônica: nenhum resultado substitui as referências em runs/reference "
        "nem foi revisado; o split test é descritivo e nada aqui seleciona run, seed, checkpoint, limiar ou "
        "hiperparâmetro.",
        "Métricas por janela com eval_stride 1: janelas vizinhas se sobrepõem e não são observações "
        "independentes; contagens não equivalem a eventos, vídeos ou sujeitos.",
        f"A Macro-F1 restrita usa sempre as oito classes de RESTRICTED_CLASSES {RESTRICTED_CLASSES}; num "
        "subgrupo, classe restrita ausente entra com F1 0, então a Macro-F1 de subgrupos pequenos é "
        "mecanicamente menor e não é comparável à do split inteiro. As classes ausentes estão listadas por grupo.",
        "IDs brutos de sujeito não são tratados como globais: os grupos usam o par (ambiente, sujeito). "
        "IDs repetidos em mais de um ambiente aparecem em coverage.subject_ids_in_multiple_environments.",
        "Le2i-CV retém ambientes inteiros: diferenças entre ambientes misturam câmera, iluminação, cenário, "
        "sujeitos, ações e suporte por classe e não isolam efeito causal de câmera nem de backbone. Em "
        "Le2i-CS, os ambientes de test têm poucos sujeitos, então ambiente e sujeito se confundem.",
        "Cada run é uma única seed; não há variação entre seeds nem intervalo de confiança aqui (ver o "
        "sumário multi-seed e o bootstrap agrupado por sujeito).",
        "As predições por janela vêm da reexecução validada sobre o checkpoint congelado; as predições "
        "históricas da avaliação de treino não foram persistidas por janela. A igualdade exata com a "
        "confusion_matrix de metrics.json é uma checagem de integridade necessária, mas não prova que cada "
        "janela recebeu o mesmo rótulo historicamente: atribuições diferentes podem ter a mesma matriz global "
        "e matrizes por ambiente, sujeito ou vídeo diferentes. Os arquivos de features também não são "
        "hasheados um a um.",
        "Hipóteses são geradas por regras fixas sobre os números confirmados e não foram testadas.",
    ]
    ambiguous_by_split: dict[tuple[str, str], dict] = {}
    for block in protocols.values():
        for entries in block["arms"].values():
            for entry in entries:
                for split, analysis in entry["splits"].items():
                    ambiguous = analysis["coverage"]["subject_ids_in_multiple_environments"]
                    if ambiguous:
                        ambiguous_by_split.setdefault((block["name"], split), ambiguous)
    limitations.extend(
        f"{protocol} [{split}]: IDs de sujeito presentes em mais de um ambiente: {ambiguous}."
        for (protocol, split), ambiguous in ambiguous_by_split.items()
    )
    return limitations


def _empty_protocols() -> dict:
    return {
        dataset: {
            "name": name,
            "held_out_factor": PROTOCOL_HELD_OUT_FACTOR[dataset],
            "arms": {arm: [] for arm in ARMS},
        }
        for dataset, name in PROTOCOL_NAMES.items()
    }


def _not_provided(protocols: dict, missing_inputs: list[dict]) -> list[dict]:
    reported = {(item["dataset"], item["arm"]) for item in missing_inputs}
    return [
        {"dataset": dataset, "arm": arm, "source": None, "reason": "nenhum run fornecido", "missing_files": []}
        for dataset, block in protocols.items()
        for arm, entries in block["arms"].items()
        if not entries and (dataset, arm) not in reported
    ]


def build_report(runs: list[DiagnosisInput]) -> dict:
    protocols = _empty_protocols()
    missing_inputs: list[dict] = []
    for item in runs:
        missing = missing_prerequisites(item)
        if missing:
            missing_inputs.append(
                {"dataset": item.dataset, "arm": item.arm, "source": display_path(item.path),
                 "reason": "pré-requisitos ausentes", "missing_files": missing}
            )
            continue
        try:
            entry = analyze_run(item)
        except FileNotFoundError as exc:
            missing_inputs.append(
                {"dataset": item.dataset, "arm": item.arm, "source": display_path(item.path),
                 "reason": f"arquivo ausente durante a inferência: {exc}", "missing_files": []}
            )
            continue
        protocols[item.dataset]["arms"][item.arm].append(entry)
    missing_inputs.extend(_not_provided(protocols, missing_inputs))

    observations, hypotheses = build_findings(protocols)
    return {
        "analysis": "stratified_error_diagnosis",
        "canonical": False,
        "read_only": True,
        "macro_f1_definition": {
            "restricted_classes": list(RESTRICTED_CLASSES),
            "restricted_labels": [LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES],
            "source": "gatefall.train.shared.metrics.restricted_macro_f1",
        },
        "label_names": list(LE2I_LABEL_NAMES),
        "ignore_label": IGNORE_LABEL,
        "eval_stride": EVAL_STRIDE,
        "splits": list(SPLITS),
        "levels": {
            "environment": "env",
            "environment_subject": "(env, subject); IDs brutos de sujeito não são globais",
            "video": "video_id",
        },
        "protocols": protocols,
        "missing_inputs": missing_inputs,
        "observations": observations,
        "hypotheses": hypotheses,
        "limitations": _limitations(protocols),
    }


def _group_rows(base: dict, level: str, group: dict) -> list[dict]:
    location = {"level": level, "group": group["group"], "env": group["env"],
                "subject": group["subject"], "video_id": group["video_id"]}
    rows = [
        {**base, **location, "entity": "", "metric": metric, "value": group[metric]}
        for metric in ("n_windows", "n_labeled_windows", "n_ignored_windows", "n_errors", "macro_f1_restricted")
    ]
    for label, metrics in group["per_class"].items():
        rows.extend(
            {**base, **location, "entity": label, "metric": metric, "value": metrics[metric]}
            for metric in ("support", "precision", "recall", "f1")
        )
    rows.extend({**base, **location, "entity": label, "metric": "absent", "value": 1} for label in group["absent_labels"])
    matrix = group["confusion_matrix"]
    rows.extend(
        {**base, **location, "entity": f"{LE2I_LABEL_NAMES[true_id]}->{LE2I_LABEL_NAMES[pred_id]}",
         "metric": "confusion_count", "value": count}
        for true_id, row in enumerate(matrix)
        for pred_id, count in enumerate(row)
        if count > 0
    )
    return rows


def csv_rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    for dataset, block in report["protocols"].items():
        for arm, entries in block["arms"].items():
            for entry in entries:
                for split, analysis in entry["splits"].items():
                    base = {"dataset": dataset, "protocol": block["name"], "arm": arm, "seed": entry["seed"],
                            "source": entry["source"], "split": split}
                    overall = {"group": "all", "env": None, "subject": None, "video_id": None, **analysis["overall"]}
                    rows.extend(_group_rows(base, "split", overall))
                    for level in LEVELS:
                        for group in analysis["levels"][level]:
                            rows.extend(_group_rows(base, level, group))
    return rows


def _group_table(entries: list[dict], split: str, level: str, title: str) -> list[str]:
    groups = sorted({group["group"] for entry in entries for group in entry["splits"][split]["levels"][level]})
    lookup = {
        (index, group["group"]): group
        for index, entry in enumerate(entries)
        for group in entry["splits"][split]["levels"][level]
    }
    first = {group["group"]: group for group in entries[0]["splits"][split]["levels"][level]}
    lines = [
        "",
        f"{title} — Macro-F1 restrita (erros/janelas rotuladas):",
        "",
        "| Grupo | Classes restritas ausentes | " + " | ".join(_entry_label(entry) for entry in entries) + " |",
        "| --- | --- |" + " --- |" * len(entries),
    ]
    for name in groups:
        absent = ", ".join(first[name]["absent_restricted_labels"]) if name in first else "n/d"
        cells = []
        for index in range(len(entries)):
            group = lookup.get((index, name))
            cells.append(
                "n/d" if group is None
                else f"{_fmt(group['macro_f1_restricted'])} ({group['n_errors']}/{group['n_labeled_windows']})"
            )
        lines.append(f"| {name} | {absent or '—'} | " + " | ".join(cells) + " |")
    return lines


def render_summary(report: dict) -> str:
    definition = report["macro_f1_definition"]
    lines = [
        "# Diagnóstico estratificado de erros de classificação (não canônico)",
        "",
        "Análise exploratória e somente leitura sobre predições reexecutadas a partir de checkpoints "
        "congelados, sem treino, ajuste ou seleção; a reconciliação com metrics.json não prova que elas "
        "coincidem janela a janela com as predições históricas, que não foram persistidas. Macro-F1 restrita = média simples do F1 sobre "
        f"{definition['restricted_labels']}, mesmo em subgrupos sem alguma dessas classes; janelas "
        "IGNORE_LABEL ficam fora. Sujeitos são agrupados por (ambiente, sujeito). Detalhes por vídeo e por "
        "classe estão no JSON e no CSV.",
    ]
    for dataset, block in report["protocols"].items():
        entries = [entry for arm_entries in block["arms"].values() for entry in arm_entries]
        lines += ["", f"## {block['name']} (`{dataset}`, {block['held_out_factor']} retidos)"]
        if not entries:
            lines += ["", "Nenhum run disponível."]
            continue
        for split in report["splits"]:
            lines += ["", f"### {split}"]
            lines += _group_table(entries, split, "environment", "Por ambiente")
            lines += _group_table(entries, split, "environment_subject", "Por ambiente e sujeito")
            ambiguous = entries[0]["splits"][split]["coverage"]["subject_ids_in_multiple_environments"]
            if ambiguous:
                lines += ["", f"IDs de sujeito presentes em mais de um ambiente (tratados como grupos distintos): {ambiguous}."]
        for entry in entries:
            key = (dataset, entry["arm"], entry["source"])
            lines += ["", f"### {_entry_label(entry)} — `{entry['source']}`", "", "Observações confirmadas:", ""]
            lines += [f"- {item['text']}" for item in report["observations"]
                      if (item["dataset"], item["arm"], item["source"]) == key]
            lines += ["", "Hipóteses (não verificadas):", ""]
            lines += [f"- {item['text']}" for item in report["hypotheses"]
                      if (item["dataset"], item["arm"], item["source"]) == key] or ["- Nenhuma regra disparada."]
    lines += ["", "## Entradas ausentes", ""]
    lines += [
        f"- {PROTOCOL_NAMES[item['dataset']]}/{item['arm']}"
        + (f" (`{item['source']}`)" if item["source"] else "")
        + f": {item['reason']}"
        + (f" — {', '.join(item['missing_files'])}" if item["missing_files"] else "")
        for item in report["missing_inputs"]
    ] or ["- Nenhuma."]
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def protected_paths(runs: list[DiagnosisInput]) -> list[Path]:
    dataset_paths = [
        path
        for dataset in {item.dataset for item in runs}
        for path in (
            _frames_path(dataset).parent,
            get_dataset(dataset).pose_root,
            get_dataset(dataset).dinov3_root,
            get_dataset(dataset).quality_root,
            get_dataset(dataset).sam3_root,
        )
    ]
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        *dataset_paths,
        *(item.path for item in runs),
    ]


def validate_runs(runs: list[DiagnosisInput]) -> None:
    if not runs:
        raise ValueError("nenhum --run fornecido")
    seen: set[Path] = set()
    for item in runs:
        _validate_dataset_and_arm(item.dataset, item.arm)
        validate_local_run_dir(item.path, item.dataset)
        resolved = item.path.resolve()
        if resolved in seen:
            raise ValueError(f"run fornecido mais de uma vez: {item.path}")
        seen.add(resolved)


def run_stratified_diagnosis(runs: list[DiagnosisInput], output_dir: Path, force: bool = False) -> bool:
    validate_runs(runs)
    check_output_dir(output_dir, protected_paths(runs))
    available_inputs = sorted(
        {path.resolve() for item in runs if not missing_prerequisites(item) for path in run_input_files(item)}
    )
    snapshot = snapshot_inputs(available_inputs)
    report = build_report(runs)
    report["input_sha256"] = {display_path(Path(path)): digest for path, digest in snapshot.items()}
    rows = csv_rows(report)
    targets = {
        output_dir / STRATIFIED_JSON_FILE: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        output_dir / STRATIFIED_CSV_FILE: pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(index=False),
        output_dir / STRATIFIED_SUMMARY_FILE: render_summary(report),
    }
    written = write_text_outputs(output_dir, targets, force)
    verify_inputs_unchanged(snapshot)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Estratifica os erros de classificação por ambiente, sujeito e vídeo"
    )
    diagnose_parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("DATASET", "ARM", "RUN_DIR"),
        required=True,
        help="Repetível: run local treinado com config.yaml, metrics.json e checkpoint.pt",
    )
    diagnose_parser.add_argument("--output-dir", type=Path, required=True)
    diagnose_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do diagnóstico estratificado")

    args = parser.parse_args()
    if args.command == "diagnose":
        run_stratified_diagnosis(_parse_inputs(args.run), args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.stratified_error_diagnosis import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
