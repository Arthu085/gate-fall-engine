"""Viabilidade, somente leitura, de dobras internas por sujeito no `train` do Le2i-CS (Etapa 4A).

Lê só as linhas `split == train` de `frames.parquet` e do manifesto do
Le2i-CS; `val` e `test` oficiais nunca entram em memória. Antes de propor
dobras, confere se cada vídeo tem um único sujeito e um único ambiente, se
frames e manifesto concordam e como cada ID bruto de sujeito se distribui
entre ambientes. As dobras agrupam pelo ID bruto, a unidade mais grossa
disponível: um sujeito retido leva consigo todos os seus vídeos, em todos os
ambientes.

Dois esquemas são avaliados: deixar um sujeito de fora e agrupar os sujeitos
minoritários em um único grupo retido. As janelas vêm de `build_window_index`
com o contrato temporal congelado, o suporte por classe é medido em
`TRAIN_STRIDE` no treino de desenvolvimento e em `EVAL_STRIDE` no lado
retido, e a macro-F1 restrita só é comparável nas oito classes quando
nenhuma delas falta em nenhum dos lados. O relatório também avalia se as
dobras sustentam um experimento futuro de robustez de pose; nada aqui treina,
aumenta dados, ajusta estatística ou seleciona modelo.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.le2i.windows import EXPECTED_TOTAL_WINDOWS_STRIDE1, EXPECTED_USABLE_WINDOWS_STRIDE1
from gatefall.data.windowing import build_window_index
from gatefall.datasets.le2i import PROTOCOL_POSE_STATS_PATHS, LE2I_LABEL_NAMES, Le2iDatasetAdapter
from gatefall.eval.analysis.classification_diagnosis import (
    PROTOCOL_NAMES,
    check_output_dir,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.internal_fold_feasibility import (
    DEVELOPMENT_SUPPORT_STRIDE,
    EXCLUDED_CLASSES,
    OFFICIAL_TRAIN_SPLIT,
    RETAINED_SUPPORT_STRIDE,
    STRIDES,
    _video_metadata_conflicts,
    read_train_partition,
    side_block,
    validate_official_train_partition,
)
from gatefall.eval.analysis.other_walk_review_set import LOCAL_OUTPUT_ROOT
from gatefall.features.standardize import EXPECTED_VIDEOS_LOADED
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

FEASIBILITY_JSON_FILE = "subject_fold_feasibility.json"
FEASIBILITY_SUMMARY_FILE = "subject_fold_feasibility.md"

DATASET = "le2i"
PROTOCOL = PROTOCOL_NAMES[DATASET]
LOSO_SCHEME = "leave_one_subject_out"
POOLED_SCHEME = "pooled_minority_subjects"
FEASIBLE_VERDICT = "feasible_with_limitations"
NOT_FEASIBLE_VERDICT = "not_feasible"
OFFICIAL_POSE_STATS = PROTOCOL_POSE_STATS_PATHS["cs"]

EXPECTED_CS_TRAIN: dict[str, object] = {
    "environments": ("Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02", "Lecture_room", "Office"),
    "n_videos": EXPECTED_VIDEOS_LOADED[DATASET]["train"],
    "windows_total_eval_stride": EXPECTED_TOTAL_WINDOWS_STRIDE1[DATASET]["train"],
    "windows_labeled_eval_stride": EXPECTED_USABLE_WINDOWS_STRIDE1[DATASET]["train"],
}

SCHEME_RULES = {
    LOSO_SCHEME: "Cada ID bruto de sujeito do train oficial é retido sozinho em uma dobra; os demais formam o "
    "treino de desenvolvimento.",
    POOLED_SCHEME: "Um sujeito fica sozinho quando suas janelas em EVAL_STRIDE contêm as oito classes restritas e "
    "nenhuma delas vem de um único vídeo; os demais formam um grupo retido único. Enquanto esse grupo não tiver "
    "as oito classes, o sujeito isolado com menos janelas rotuladas (empate pelo menor ID) passa para ele. Menos "
    "de dois grupos não formam esquema.",
}

IDENTITY_INTERPRETATIONS = {
    "global": "O mesmo ID bruto é a mesma pessoa em todos os ambientes. É a leitura implícita no protocolo "
    "cross-subject do OmniFall, que separa train, val e test por ID bruto.",
    "per_environment": "O ID bruto só identifica a pessoa dentro do ambiente, e a identidade é o par "
    "(ambiente, sujeito). As análises do Le2i-CV usam essa leitura.",
}

STATIC_BLOCKERS = [
    "O pipeline de treino não tem modo de split por sujeito no Le2i-CS: gatefall.train.internal_folds está fixo "
    "nas dobras por ambiente do Le2i-CV. Um modo novo precisa de aprovação explícita e desloca a próxima etapa "
    "exploratória de treino.",
    "O caminho de pose não tem ponto de aumento: PoseWindowDataset guarda em cache a matriz de features de cada "
    "vídeo, montada uma vez por build_pose_features_from_arrays. Uma queda de pose simulada precisa alterar "
    "person_found e os keypoints em PoseArrays antes da imputação por forward-fill e do dt efetivo; mascarar a "
    "janela já montada ou padronizada deixa velocidade, aceleração e orientação do tronco incoerentes.",
    "Keypoint ausente não tem representação própria: as features só têm kp_conf por keypoint e person_found por "
    "quadro, e a imputação é por pessoa. Simular keypoints ausentes exige fixar antes a representação (por "
    "exemplo, confiança zero com a coordenada mantida ou zerada), sem referência observada nos dados.",
    "DINOv3 e SAM 3 não são perturbados junto com a pose: a queda simulada não traz a oclusão visual que costuma "
    "acompanhar a perda real de pose. O experimento mede robustez à falha do detector de pose, não a oclusões.",
    "Pela invariante 1, um aumento no treino precisa ser idêntico em A, B e C. Aplicá-lo a um único braço muda o "
    "protocolo e precisa ser declarado como experimento separado e não canônico.",
]

REQUIRED_SAFEGUARDS = [
    "Fixar a composição das dobras antes de treinar, em um arquivo com as listas de vídeos e sujeitos de cada "
    "lado e seu SHA-256, derivado só do train oficial, como o folds.json das dobras do Le2i-CV.",
    f"Ajustar as estatísticas de padronização de pose (e de DINOv3 ou SAM 3, se usados) por dobra, só nas janelas "
    f"do treino de desenvolvimento em TRAIN_STRIDE e sem aumento, em caminho isolado. {OFFICIAL_POSE_STATS} foi "
    "ajustado em todo o train oficial, inclusive nos sujeitos retidos, e não pode ser usado nas dobras.",
    "Aplicar o aumento só às janelas do treino de desenvolvimento. O lado retido é avaliado em EVAL_STRIDE, sem "
    "perturbação.",
    "Fixar antes de treinar a configuração do aumento (probabilidade, duração das quedas, keypoints afetados), a "
    "receita, as seeds e o critério de decisão. A comparação é pareada: mesma dobra, mesma seed e mesma receita, "
    "com e sem aumento.",
    "Registrar por run a composição da dobra, os SHA-256 das estatísticas, a configuração do aumento, o commit e "
    "os SHA-256 das entradas, e gravar os runs só em runs/local, nunca em runs/reference.",
    "Comparar a macro-F1 restrita só nas dobras em que as oito classes existem nos dois lados. A F1 de classes "
    "raras é relatada, mas não decide. Val e test oficiais não entram em nenhuma seleção, e 0,800 é um alvo de "
    "pesquisa, não um critério de seleção.",
]

GLOBAL_LEAKAGE_RISKS = [
    "Se os IDs brutos forem locais a cada ambiente, uma pessoa do lado retido pode aparecer com outro ID no "
    "treino de desenvolvimento. Os metadados não permitem verificar isso, e o mesmo risco já existe no Le2i-CS "
    "oficial.",
    f"{OFFICIAL_POSE_STATS} e as estatísticas oficiais de DINOv3 e SAM 3 do Le2i-CS foram ajustados em todo o "
    "train oficial. Reutilizá-los nas dobras vazaria média e desvio dos sujeitos retidos.",
    "Janelas nunca cruzam vídeos e todo vídeo pertence a um único sujeito, então nenhuma janela aparece nos dois "
    "lados de uma dobra.",
]

GLOBAL_LIMITATIONS = [
    "Avaliação exploratória e não canônica: nada aqui treina, aumenta dados, seleciona checkpoint, ajusta "
    "hiperparâmetro ou estatística nem altera split, configuração, limiar, checkpoint ou referência.",
    "Contagens por janela usam janelas sobrepostas (stride 4 no treino, stride 1 na avaliação) e não são "
    "observações independentes. Sujeitos e vídeos são as unidades mais próximas de independentes, e o train "
    "oficial tem poucos sujeitos brutos.",
    "As dobras são por sujeito, não por cena: todos os ambientes e câmeras aparecem nos dois lados, como no "
    "Le2i-CS oficial.",
    "As regras dos esquemas foram fixadas depois de inspecionar o suporte por classe do train oficial, sem olhar "
    "val ou test. Magnitudes das dobras não são comparáveis às do test oficial e não verificam o alvo de 0,800.",
    "Qualidade de pose, de features e de rótulos não é avaliada aqui.",
]


def validate_subject_identifiers(frames: pd.DataFrame, manifest: pd.DataFrame) -> None:
    for name, table in (("frames", frames), ("manifesto", manifest)):
        missing = sorted({str(video_id) for video_id in table.loc[table["subject"].isna(), "video_id"]})
        if missing:
            raise ValueError(f"{name}: vídeos sem sujeito: {missing}")
        values = table["subject"].to_numpy(dtype=np.float64)
        invalid_mask = (values != np.round(values)) | (values < 0)
        invalid = sorted({str(video_id) for video_id in table.loc[invalid_mask, "video_id"]})
        if invalid:
            raise ValueError(f"{name}: sujeito que não é inteiro não negativo nos vídeos {invalid}")
    conflicts = _video_metadata_conflicts(frames)
    if conflicts:
        raise ValueError(f"vídeos com ambiente ou sujeito ambíguo nos frames: {conflicts}")


def _labeled(table: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, table[table["label"] != IGNORE_LABEL])


def _subjects(table: pd.DataFrame) -> list[int]:
    return sorted({int(subject) for subject in table["subject"]})


def subject_identity_audit(manifest: pd.DataFrame, windows: dict[str, pd.DataFrame]) -> dict[str, object]:
    labeled = {name: _labeled(table) for name, table in windows.items()}
    official_labeled = len(labeled[DEVELOPMENT_SUPPORT_STRIDE])
    per_subject: list[dict[str, object]] = []
    for subject, group in manifest.groupby("subject", sort=True):
        subject_id = int(cast(int, subject))
        train_stride = int((labeled[DEVELOPMENT_SUPPORT_STRIDE]["subject"] == subject_id).sum())
        per_subject.append({
            "raw_subject": subject_id,
            "environments": sorted(str(env) for env in group["env"].unique()),
            "cameras": sorted(int(cam) for cam in group["cam"].unique()),
            "n_videos": len(group),
            "windows_labeled_train_stride": train_stride,
            "windows_labeled_eval_stride": int((labeled[RETAINED_SUPPORT_STRIDE]["subject"] == subject_id).sum()),
            "share_labeled_train_stride": train_stride / official_labeled if official_labeled else None,
        })
    spanning = [
        cast(int, entry["raw_subject"]) for entry in per_subject if len(cast(list[str], entry["environments"])) > 1
    ]
    pairs = manifest[["env", "subject"]].drop_duplicates()
    return {
        "grouping_unit": "raw_subject",
        "per_video_consistency": "verified",
        "global_identity_verifiable": False,
        "raw_subjects": [entry["raw_subject"] for entry in per_subject],
        "n_raw_subjects": len(per_subject),
        "n_environment_subject_pairs": len(pairs),
        "raw_subjects_spanning_environments": spanning,
        "per_subject": per_subject,
        "interpretations": IDENTITY_INTERPRETATIONS,
        "rejected_grouping": {
            "unit": "environment_subject",
            "raw_subjects_split_across_sides_if_global": spanning,
            "reason": "Agrupar por (ambiente, sujeito) põe nos dois lados da dobra o mesmo ID bruto quando ele "
            "aparece em mais de um ambiente. Na leitura global, isso é a mesma pessoa no treino e no lado retido.",
        },
        "decision": "As dobras agrupam pelo ID bruto. Na leitura global, o sujeito retido sai inteiro do treino. "
        "Na leitura por ambiente, o grupo retido junta pessoas possivelmente diferentes, todas fora do treino, mas "
        "não há como provar que nenhuma delas aparece no treino com outro ID.",
    }


def _covers_restricted(labeled_eval: pd.DataFrame, subjects: list[int]) -> bool:
    present = set(labeled_eval.loc[labeled_eval["subject"].isin(subjects), "label"].astype(int))
    return set(RESTRICTED_CLASSES) <= present


def _has_single_video_class(labeled_eval: pd.DataFrame, subject: int) -> bool:
    rows = cast(pd.DataFrame, labeled_eval[labeled_eval["subject"] == subject])
    counts = cast(pd.Series, rows.groupby("label")["video_id"].nunique())
    videos_per_class = {int(cast(int, label)): int(count) for label, count in counts.items()}
    return any(videos_per_class.get(class_id, 0) == 1 for class_id in RESTRICTED_CLASSES)


def leave_one_subject_out_groups(subjects: list[int]) -> list[tuple[int, ...]]:
    return [(subject,) for subject in sorted(subjects)]


def pooled_minority_groups(eval_windows: pd.DataFrame) -> tuple[list[tuple[int, ...]], list[int]]:
    labeled_eval = _labeled(eval_windows)
    subjects = _subjects(labeled_eval)
    singletons = [
        subject
        for subject in subjects
        if _covers_restricted(labeled_eval, [subject]) and not _has_single_video_class(labeled_eval, subject)
    ]
    pool = [subject for subject in subjects if subject not in singletons]
    moved: list[int] = []
    while pool and singletons and not _covers_restricted(labeled_eval, pool):
        smallest = min(singletons, key=lambda subject: (int((labeled_eval["subject"] == subject).sum()), subject))
        singletons.remove(smallest)
        pool = sorted([*pool, smallest])
        moved.append(smallest)
    groups = [(subject,) for subject in singletons] + ([tuple(pool)] if pool else [])
    return (groups if len(groups) >= 2 else []), moved


def subject_side(
    subjects: list[int], windows: dict[str, pd.DataFrame], manifest: pd.DataFrame, support_stride: str
) -> dict[str, object]:
    side_windows = {
        name: cast(pd.DataFrame, table[table["subject"].isin(subjects)]) for name, table in windows.items()
    }
    side_manifest = cast(pd.DataFrame, manifest[manifest["subject"].isin(subjects)])
    environments = sorted(str(env) for env in side_manifest["env"].unique())
    block = side_block(environments, side_windows, side_manifest, support_stride)
    support = cast(dict[str, dict[str, object]], block["restricted_class_support"])
    rows = side_windows[support_stride]
    for entry in support.values():
        class_rows = cast(pd.DataFrame, rows[rows["label"] == entry["class_id"]])
        entry["raw_subjects_at_support_stride"] = cast(pd.Series, class_rows["subject"]).nunique()
    block["single_raw_subject_classes_at_support_stride"] = [
        label for label, entry in support.items() if entry["raw_subjects_at_support_stride"] == 1
    ] if len(subjects) > 1 else []
    return {"raw_subjects": list(subjects), **block}


def check_fold_isolation(
    manifest: pd.DataFrame, development_subjects: list[int], retained_subjects: list[int]
) -> dict[str, object]:
    development = cast(pd.DataFrame, manifest[manifest["subject"].isin(development_subjects)])
    retained = cast(pd.DataFrame, manifest[manifest["subject"].isin(retained_subjects)])
    development_videos = set(development["video_id"].astype(str))
    retained_videos = set(retained["video_id"].astype(str))
    shared_videos = sorted(development_videos & retained_videos)
    shared_subjects = sorted(set(development_subjects) & set(retained_subjects))
    uncovered = sorted(set(manifest["video_id"].astype(str)) - development_videos - retained_videos)
    if shared_videos or shared_subjects or uncovered or not retained_videos or not development_videos:
        raise ValueError(
            f"dobra sem isolamento: vídeos nos dois lados={shared_videos}, sujeitos nos dois lados="
            f"{shared_subjects}, vídeos fora da dobra={uncovered}, retido vazio={not retained_videos}, "
            f"treino vazio={not development_videos}"
        )
    return {
        "shared_video_ids": shared_videos,
        "shared_raw_subjects": shared_subjects,
        "shared_environments": sorted(set(development["env"].astype(str)) & set(retained["env"].astype(str))),
        "shared_cameras": sorted({int(cam) for cam in development["cam"]} & {int(cam) for cam in retained["cam"]}),
    }


def _side_limitations(name: str, side: dict[str, object]) -> list[str]:
    stride = cast(dict[str, int], side["support_stride"])["stride"]
    support = cast(dict[str, dict[str, int]], side["restricted_class_support"])
    limitations: list[str] = []
    missing = cast(list[str], side["missing_restricted_classes_at_support_stride"])
    if missing:
        only_eval_stride = [label for label in missing if support[label]["windows_eval_stride"] > 0]
        detail = f" (com janelas só em stride {EVAL_STRIDE}: {only_eval_stride})" if only_eval_stride else ""
        limitations.append(f"{name}: classes restritas sem nenhuma janela em stride {stride}: {missing}{detail}.")
    single_video = cast(list[str], side["single_video_classes_at_support_stride"])
    if single_video:
        limitations.append(f"{name}: classes restritas presentes em um único vídeo em stride {stride}: {single_video}.")
    single_subject = [
        label for label in cast(list[str], side["single_raw_subject_classes_at_support_stride"])
        if label not in single_video
    ]
    if single_subject:
        limitations.append(
            f"{name}: classes restritas presentes em um único sujeito bruto em stride {stride}: {single_subject}."
        )
    return limitations


def fold_limitations(
    development: dict[str, object], retained: dict[str, object], official: dict[str, object]
) -> list[str]:
    limitations = _side_limitations("treino de desenvolvimento", development) + _side_limitations("retido", retained)
    if len(cast(list[int], retained["raw_subjects"])) == 1:
        limitations.append(
            "O lado retido tem um único sujeito bruto: a dobra mede a generalização para esse sujeito, sem "
            "variação entre sujeitos dentro dela."
        )
    development_labeled = cast(dict[str, dict[str, int]], development["windows"])["train_stride"]["labeled"]
    official_labeled = cast(dict[str, dict[str, int]], official["windows"])["train_stride"]["labeled"]
    if official_labeled:
        limitations.append(
            f"O treino de desenvolvimento usa {development_labeled} das {official_labeled} janelas rotuladas do "
            f"train oficial em stride {TRAIN_STRIDE} ({development_labeled / official_labeled:.0%})."
        )
    imbalance = cast(dict[str, object] | None, development["imbalance"])
    if imbalance is not None:
        limitations.append(
            f"Desbalanceamento no treino de desenvolvimento (stride {imbalance['stride']}): "
            f"{imbalance['largest_class']}/{imbalance['smallest_present_class']} = "
            f"{cast(float, imbalance['largest_to_smallest_ratio']):.1f}. Descritivo, sem limiar."
        )
    return limitations


def fold_leakage_risks(isolation: dict[str, object], retained_subjects: list[int], spanning: list[int]) -> list[str]:
    risks: list[str] = []
    environments = cast(list[str], isolation["shared_environments"])
    if environments:
        risks.append(
            f"Os dois lados compartilham os ambientes {environments} e as câmeras "
            f"{cast(list[int], isolation['shared_cameras'])}: cena e câmera vistas no treino reaparecem no lado "
            "retido."
        )
    retained_spanning = [subject for subject in retained_subjects if subject in spanning]
    if retained_spanning:
        risks.append(
            f"Os IDs brutos retidos {retained_spanning} aparecem em mais de um ambiente. Todas essas aparições "
            "ficam no lado retido; agrupar por (ambiente, sujeito) as dividiria entre os lados."
        )
    return risks


def build_fold(
    scheme: str,
    retained_subjects: tuple[int, ...],
    windows: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    official: dict[str, object],
    spanning: list[int],
) -> dict[str, object]:
    retained_ids = list(retained_subjects)
    development_ids = [subject for subject in _subjects(manifest) if subject not in retained_ids]
    isolation = check_fold_isolation(manifest, development_ids, retained_ids)
    development = subject_side(development_ids, windows, manifest, DEVELOPMENT_SUPPORT_STRIDE)
    retained = subject_side(retained_ids, windows, manifest, RETAINED_SUPPORT_STRIDE)
    missing_development = cast(list[str], development["missing_restricted_classes_at_support_stride"])
    missing_retained = cast(list[str], retained["missing_restricted_classes_at_support_stride"])
    return {
        "scheme": scheme,
        "fold": "retain_s" + "-".join(str(subject) for subject in retained_ids),
        "retained_subjects": retained_ids,
        "development_subjects": development_ids,
        "isolation": isolation,
        "development_train": development,
        "retained": retained,
        "missing_classes": {"development_train": missing_development, "retained": missing_retained},
        "rare_retained_classes": {
            "single_video": retained["single_video_classes_at_support_stride"],
            "single_raw_subject": retained["single_raw_subject_classes_at_support_stride"],
        },
        "eight_class_macro_f1_comparable": not missing_development and not missing_retained,
        "leakage_risks": fold_leakage_risks(isolation, retained_ids, spanning),
        "limitations": fold_limitations(development, retained, official),
    }


def build_scheme(
    scheme: str,
    groups: list[tuple[int, ...]],
    moved_to_pool: list[int],
    windows: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    official: dict[str, object],
    spanning: list[int],
) -> dict[str, object]:
    folds = [build_fold(scheme, group, windows, manifest, official, spanning) for group in groups]
    comparable = [cast(str, fold["fold"]) for fold in folds if fold["eight_class_macro_f1_comparable"]]
    return {
        "scheme": scheme,
        "rule": SCHEME_RULES[scheme],
        "groups": [list(group) for group in groups],
        "moved_to_pool": moved_to_pool,
        "n_folds": len(folds),
        "n_comparable_folds": len(comparable),
        "comparable_folds": comparable,
        "supports_controlled_comparison": len(folds) >= 2 and len(comparable) == len(folds),
        "folds": folds,
    }


def _candidate_scheme(schemes: list[dict[str, object]]) -> dict[str, object] | None:
    supported = [scheme for scheme in schemes if scheme["supports_controlled_comparison"]]
    return max(supported, key=lambda scheme: cast(int, scheme["n_folds"])) if supported else None


def augmentation_assessment(schemes: list[dict[str, object]], identity: dict[str, object]) -> dict[str, object]:
    candidate = _candidate_scheme(schemes)
    blockers: list[str] = []
    if candidate is None:
        blockers.append(
            "Nenhum esquema tem pelo menos duas dobras com as oito classes restritas nos dois lados: a macro-F1 "
            "restrita não é comparável em nenhuma partição completa do train."
        )
    per_subject = cast(list[dict[str, object]], identity["per_subject"])
    shares = [entry for entry in per_subject if entry["share_labeled_train_stride"] is not None]
    dominant = max(shares, key=lambda entry: cast(float, entry["share_labeled_train_stride"])) if shares else None
    observations: list[str] = []
    if dominant is not None:
        observations.append(
            f"O sujeito bruto {dominant['raw_subject']} tem {cast(float, dominant['share_labeled_train_stride']):.0%} "
            f"das janelas rotuladas do train oficial em stride {TRAIN_STRIDE}; a dobra que o retém treina com o "
            "restante."
        )
    if candidate is not None:
        for fold in cast(list[dict[str, object]], candidate["folds"]):
            rare = cast(dict[str, list[str]], fold["rare_retained_classes"])
            rare_labels = sorted(set(rare["single_video"]) | set(rare["single_raw_subject"]))
            if rare_labels:
                observations.append(
                    f"Dobra {fold['fold']}: a F1 de {rare_labels} no lado retido depende de um único vídeo ou "
                    "sujeito e não deve ser lida como resultado por classe."
                )
    return {
        "candidate_scheme": None if candidate is None else candidate["scheme"],
        "candidate_folds": [] if candidate is None else candidate["comparable_folds"],
        "data_blockers": blockers,
        "implementation_blockers": STATIC_BLOCKERS,
        "required_safeguards": REQUIRED_SAFEGUARDS,
        "observations": observations,
    }


def build_report(frames: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, object]:
    validate_subject_identifiers(frames, manifest)
    windows = {
        name: build_window_index(frames, stride=stride, drop_ignored=False) for name, stride in STRIDES.items()
    }
    environments = sorted(str(env) for env in frames["env"].unique())
    official = side_block(environments, windows, manifest, DEVELOPMENT_SUPPORT_STRIDE)
    identity = subject_identity_audit(manifest, windows)
    spanning = cast(list[int], identity["raw_subjects_spanning_environments"])
    pooled_groups, moved = pooled_minority_groups(windows[RETAINED_SUPPORT_STRIDE])
    schemes = [
        build_scheme(LOSO_SCHEME, leave_one_subject_out_groups(_subjects(manifest)), [], windows, manifest, official,
                     spanning),
        build_scheme(POOLED_SCHEME, pooled_groups, moved, windows, manifest, official, spanning),
    ]
    augmentation = augmentation_assessment(schemes, identity)
    feasible = augmentation["candidate_scheme"] is not None
    return {
        "analysis": "le2i_cs_subject_fold_feasibility",
        "stage": "4A",
        "canonical": False,
        "read_only": True,
        "status": "assessed",
        "dataset": DATASET,
        "protocol": PROTOCOL,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "label_names": list(LE2I_LABEL_NAMES),
        "ignore_label": IGNORE_LABEL,
        "window_frames": WINDOW_FRAMES,
        "strides": STRIDES,
        "support_strides": {
            "official_train": DEVELOPMENT_SUPPORT_STRIDE,
            "development_train": DEVELOPMENT_SUPPORT_STRIDE,
            "retained": RETAINED_SUPPORT_STRIDE,
        },
        "restricted_classes": list(RESTRICTED_CLASSES),
        "restricted_labels": [LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES],
        "excluded_labels": [LE2I_LABEL_NAMES[class_id] for class_id in EXCLUDED_CLASSES],
        "official_train": official,
        "subject_identity": identity,
        "schemes": schemes,
        "augmentation_feasibility": augmentation,
        "verdict": FEASIBLE_VERDICT if feasible else NOT_FEASIBLE_VERDICT,
        "missing_prerequisites": [],
        "leakage_risks": GLOBAL_LEAKAGE_RISKS,
        "limitations": GLOBAL_LIMITATIONS,
    }


def missing_prerequisites_report(missing: list[str]) -> dict[str, object]:
    return {
        "analysis": "le2i_cs_subject_fold_feasibility",
        "stage": "4A",
        "canonical": False,
        "read_only": True,
        "status": "missing_prerequisites",
        "dataset": DATASET,
        "protocol": PROTOCOL,
        "official_splits_read": [],
        "missing_prerequisites": missing,
        "schemes": [],
        "verdict": NOT_FEASIBLE_VERDICT,
        "leakage_risks": GLOBAL_LEAKAGE_RISKS,
        "limitations": GLOBAL_LIMITATIONS,
    }


def _pct(value: object) -> str:
    return "n/d" if value is None else f"{cast(float, value):.0%}"


def _bullets(items: list[str], empty: str = "- Nenhum.") -> list[str]:
    return [f"- {text}" for text in items] or [empty]


def _identity_lines(identity: dict[str, object]) -> list[str]:
    lines = [
        "",
        "## Identidade dos sujeitos",
        "",
        f"Cada vídeo tem um único sujeito e um único ambiente, e frames e manifesto concordam. O train tem "
        f"{identity['n_raw_subjects']} IDs brutos e {identity['n_environment_subject_pairs']} pares (ambiente, "
        f"sujeito). IDs brutos em mais de um ambiente: {identity['raw_subjects_spanning_environments']}.",
        "",
        f"| Sujeito bruto | Ambientes | Câmeras | Vídeos | Rotuladas stride {TRAIN_STRIDE} (fração) | "
        f"Rotuladas stride {EVAL_STRIDE} |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for entry in cast(list[dict[str, object]], identity["per_subject"]):
        lines.append(
            f"| {entry['raw_subject']} | {', '.join(cast(list[str], entry['environments']))} | "
            f"{', '.join(str(cam) for cam in cast(list[int], entry['cameras']))} | {entry['n_videos']} | "
            f"{entry['windows_labeled_train_stride']} ({_pct(entry['share_labeled_train_stride'])}) | "
            f"{entry['windows_labeled_eval_stride']} |"
        )
    interpretations = cast(dict[str, str], identity["interpretations"])
    rejected = cast(dict[str, object], identity["rejected_grouping"])
    lines += [
        "",
        f"- Leitura global: {interpretations['global']}",
        f"- Leitura por ambiente: {interpretations['per_environment']}",
        f"- Agrupamento rejeitado: {rejected['reason']}",
        f"- Decisão: {identity['decision']}",
    ]
    return lines


def _support_table(development: dict[str, object], retained: dict[str, object]) -> list[str]:
    development_support = cast(dict[str, dict[str, int]], development["restricted_class_support"])
    retained_support = cast(dict[str, dict[str, int]], retained["restricted_class_support"])
    lines = [
        "",
        f"| Classe | Treino dev. stride {TRAIN_STRIDE}: janelas | vídeos | sujeitos | "
        f"Retido stride {EVAL_STRIDE}: janelas | vídeos | sujeitos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, dev in development_support.items():
        ret = retained_support[label]
        lines.append(
            f"| {label} | {dev['windows_at_support_stride']} | {dev['videos_at_support_stride']} | "
            f"{dev['raw_subjects_at_support_stride']} | {ret['windows_at_support_stride']} | "
            f"{ret['videos_at_support_stride']} | {ret['raw_subjects_at_support_stride']} |"
        )
    return lines


def _fold_lines(fold: dict[str, object]) -> list[str]:
    development = cast(dict[str, object], fold["development_train"])
    retained = cast(dict[str, object], fold["retained"])
    rows = []
    for name, side in (("treino de desenvolvimento", development), ("retido", retained)):
        windows = cast(dict[str, dict[str, int]], side["windows"])
        rows.append(
            f"| {name} | {cast(list[int], side['raw_subjects'])} | {len(cast(list[str], side['environments']))} | "
            f"{side['n_videos']} | {windows['train_stride']['labeled']} | {windows['eval_stride']['labeled']} |"
        )
    comparable = "sim" if fold["eight_class_macro_f1_comparable"] else "não"
    return [
        "",
        f"### Dobra `{fold['fold']}`",
        "",
        f"| Lado | Sujeitos brutos | Ambientes | Vídeos | Rotuladas stride {TRAIN_STRIDE} | "
        f"Rotuladas stride {EVAL_STRIDE} |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *rows,
        *_support_table(development, retained),
        "",
        f"Macro-F1 restrita comparável nas oito classes: {comparable}.",
        "",
        "Riscos de vazamento:",
        "",
        *_bullets(cast(list[str], fold["leakage_risks"])),
        "",
        "Limitações:",
        "",
        *_bullets(cast(list[str], fold["limitations"]), "- Nenhuma."),
    ]


def _scheme_lines(scheme: dict[str, object]) -> list[str]:
    lines = ["", f"## Esquema `{scheme['scheme']}`", "", cast(str, scheme["rule"]), ""]
    if not scheme["folds"]:
        return lines + ["O esquema não forma pelo menos dois grupos; nenhuma dobra foi avaliada."]
    moved = cast(list[int], scheme["moved_to_pool"])
    lines.append(
        f"Grupos retidos: {scheme['groups']}"
        + (f"; sujeitos movidos para o grupo agregado: {moved}" if moved else "")
        + f". Dobras comparáveis nas oito classes: {scheme['n_comparable_folds']} de {scheme['n_folds']}."
    )
    for fold in cast(list[dict[str, object]], scheme["folds"]):
        lines += _fold_lines(fold)
    return lines


def _augmentation_lines(augmentation: dict[str, object]) -> list[str]:
    candidate = augmentation["candidate_scheme"]
    return [
        "",
        "## Experimento futuro de robustez de pose",
        "",
        f"Esquema candidato: `{candidate}`; dobras: {augmentation['candidate_folds']}."
        if candidate is not None else "Nenhum esquema candidato.",
        "",
        "Bloqueios nos dados:",
        "",
        *_bullets(cast(list[str], augmentation["data_blockers"])),
        "",
        "Bloqueios de implementação e protocolo (exigem aprovação antes de treinar):",
        "",
        *_bullets(cast(list[str], augmentation["implementation_blockers"])),
        "",
        "Salvaguardas obrigatórias:",
        "",
        *_bullets(cast(list[str], augmentation["required_safeguards"])),
        "",
        "Observações:",
        "",
        *_bullets(cast(list[str], augmentation["observations"]), "- Nenhuma."),
    ]


def render_summary(report: dict[str, object]) -> str:
    lines = [
        "# Viabilidade das dobras internas por sujeito do Le2i-CS (Etapa 4A, não canônico)",
        "",
        "Avaliação somente leitura sobre o split train oficial do Le2i-CS; val e test oficiais não são lidos. "
        "Nada foi treinado, aumentado, selecionado ou ajustado. Janelas pelo contrato temporal congelado; "
        "IGNORE_LABEL pelo rótulo do último quadro; classes restritas fixas de RESTRICTED_CLASSES. Suporte por "
        f"classe em stride {TRAIN_STRIDE} no treino de desenvolvimento e em stride {EVAL_STRIDE} no lado retido.",
        "",
        f"Veredito: `{report['verdict']}`.",
    ]
    if report["status"] == "missing_prerequisites":
        lines += ["", "## Pré-requisitos ausentes", ""]
        lines += [f"- `{path}`" for path in cast(list[str], report["missing_prerequisites"])]
        lines += ["", "Nenhum número foi calculado."]
    else:
        lines += _identity_lines(cast(dict[str, object], report["subject_identity"]))
        for scheme in cast(list[dict[str, object]], report["schemes"]):
            lines += _scheme_lines(scheme)
        lines += _augmentation_lines(cast(dict[str, object], report["augmentation_feasibility"]))
    lines += ["", "## Riscos de vazamento gerais", "", *_bullets(cast(list[str], report["leakage_risks"]))]
    lines += ["", "## Limitações gerais", "", *_bullets(cast(list[str], report["limitations"]))]
    return "\n".join(lines) + "\n"


def protected_paths(frames_path: Path, manifest_path: Path) -> list[Path]:
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT / "docs",
        frames_path.parent,
        manifest_path.parent,
    ]


def check_feasibility_output_dir(output_dir: Path, frames_path: Path, manifest_path: Path) -> None:
    check_output_dir(output_dir, protected_paths(frames_path, manifest_path))
    resolved = output_dir.resolve()
    local_root = LOCAL_OUTPUT_ROOT.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT) and not (resolved == local_root or local_root in resolved.parents):
        raise ValueError(f"--output-dir {output_dir} dentro do repositório precisa ficar em {LOCAL_OUTPUT_ROOT}")


def run_assessment(frames_path: Path, manifest_path: Path, output_dir: Path, force: bool = False) -> bool:
    check_feasibility_output_dir(output_dir, frames_path, manifest_path)
    missing = [display_path(path) for path in (frames_path, manifest_path) if not path.is_file()]
    if missing:
        report = missing_prerequisites_report(missing)
        print(f"pré-requisitos ausentes: {missing}; nenhum número calculado", file=sys.stderr)
        snapshot: dict[str, str] = {}
    else:
        snapshot = snapshot_inputs([frames_path, manifest_path])
        frames = read_train_partition(frames_path)
        manifest = read_train_partition(manifest_path)
        validate_subject_identifiers(frames, manifest)
        validate_official_train_partition(frames, manifest, EXPECTED_CS_TRAIN)
        report = build_report(frames, manifest)
        report["input_sha256"] = {display_path(Path(path)): digest for path, digest in snapshot.items()}
    targets = {
        output_dir / FEASIBILITY_JSON_FILE: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        output_dir / FEASIBILITY_SUMMARY_FILE: render_summary(report),
    }
    verify_inputs_unchanged(snapshot)
    return write_text_outputs(output_dir, targets, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapter = Le2iDatasetAdapter(protocol="cs")
    assess_parser = subparsers.add_parser(
        "assess", help="Avalia a viabilidade de dobras internas por sujeito no train do Le2i-CS"
    )
    assess_parser.add_argument("--frames", type=Path, default=adapter.frames_path)
    assess_parser.add_argument("--manifest", type=Path, default=adapter.manifest_path)
    assess_parser.add_argument("--output-dir", type=Path, required=True)
    assess_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da viabilidade das dobras por sujeito")

    args = parser.parse_args()
    if args.command == "assess":
        run_assessment(args.frames, args.manifest, args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.subject_fold_feasibility import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
