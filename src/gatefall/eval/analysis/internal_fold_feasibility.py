"""Viabilidade, somente leitura, das dobras internas por ambiente do Le2i-CV (candidato 1 da Etapa 2A).

Avalia as três dobras propostas em `docs/analysis/intervention-decision.md`:
cada dobra retém um dos três ambientes do `train` oficial do Le2i-CV e usa os
outros dois como treino de desenvolvimento. Lê só as linhas `split == train`
de `frames.parquet` e do manifesto; `val` e `test` oficiais nunca são
carregados. As janelas vêm de `build_window_index` com o contrato temporal
congelado (`WINDOW_FRAMES`, `TRAIN_STRIDE`, `EVAL_STRIDE`) e `IGNORE_LABEL`
mantém a semântica de sempre: a janela é ignorada pelo rótulo do seu último
quadro. As classes avaliadas são as oito fixas de `RESTRICTED_CLASSES`.

Não treina, não cria split nem estatística de padronização, não seleciona
hiperparâmetro e não introduz limiar de aceitação: as limitações listadas são
fatos estruturais (classe ausente, classe sustentada por um único vídeo ou
identidade, critério sem denominador) ou comparações descritivas com o
`train` oficial. O resultado é exploratório e não canônico.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, NUM_CLASSES, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.le2i.generalization import (
    EXPECTED_CV_SPLIT_SHAPE,
    fps_distribution,
    resolution_distribution,
)
from gatefall.data.le2i.windows import (
    EXPECTED_TOTAL_WINDOWS_STRIDE1,
    EXPECTED_USABLE_WINDOWS_STRIDE1,
)
from gatefall.data.windowing import build_window_index
from gatefall.datasets.le2i import LE2I_LABEL_NAMES, Le2iDatasetAdapter
from gatefall.eval.analysis.classification_diagnosis import (
    check_output_dir,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

FEASIBILITY_JSON_FILE = "internal_fold_feasibility.json"
FEASIBILITY_SUMMARY_FILE = "internal_fold_feasibility.md"

DATASET = "le2i-cv"
OFFICIAL_TRAIN_SPLIT = "train"
OTHER_CLASS = LE2I_LABEL_NAMES.index("other")
EXCLUDED_CLASSES = [class_id for class_id in range(NUM_CLASSES) if class_id not in RESTRICTED_CLASSES]
STRIDES = {"train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE}

EXPECTED_OFFICIAL_TRAIN: dict[str, object] = {
    "environments": ("Coffee_room_01", "Coffee_room_02", "Lecture_room"),
    "n_videos": EXPECTED_CV_SPLIT_SHAPE["train"]["n_videos"],
    "windows_total_eval_stride": EXPECTED_TOTAL_WINDOWS_STRIDE1[DATASET]["train"],
    "windows_labeled_eval_stride": EXPECTED_USABLE_WINDOWS_STRIDE1[DATASET]["train"],
}

APPROVAL_REQUIRED = [
    "Acrescentar ao pipeline de treino um modo de split por dobra interna do Le2i-CV. É um protocolo de "
    "desenvolvimento novo: não altera os splits oficiais, mas precisa de aprovação explícita e desloca a "
    "próxima etapa exploratória de treino.",
    "Ajustar estatísticas de padronização de pose, DINOv3 e SAM 3 só nos dois ambientes de treino de cada "
    "dobra, em caminhos isolados, sem sobrescrever os arquivos *_le2i_cv.json oficiais.",
    "Fixar antes de treinar o orçamento de runs (piloto de 9 com uma seed ou 45 com as cinco seeds) e as "
    "armas (A, B0, B1), com a receita congelada, checkpoint da última época e sem nova extração.",
    "Fixar antes de treinar o critério de reprodução da Etapa 2A (menor fração predita como other entre as "
    "seeds de B0 ou B1 acima da maior entre as seeds de A em pelo menos dois dos três ambientes retidos), "
    "sem ajustá-lo depois de ver as dobras.",
    "Manter RESTRICTED_CLASSES fixo em todas as dobras e relatar, sem compensar, as classes restritas "
    "ausentes ou sustentadas por um único vídeo ou identidade em algum lado da dobra.",
    "Gravar os runs das dobras em diretório local próprio, nunca em runs/reference nem nos runs locais do "
    "Le2i-CV oficial, e não usar val nem test oficiais para nenhuma seleção.",
]

GLOBAL_LIMITATIONS = [
    "Avaliação exploratória e não canônica: nada aqui treina, seleciona checkpoint, ajusta hiperparâmetro "
    "nem altera split, padronização, features ou resultados existentes.",
    "Contagens por janela usam janelas sobrepostas (stride 4 no treino, stride 1 na avaliação); elas não "
    "são observações independentes. Vídeos e identidades (ambiente, sujeito) são as unidades mais "
    "próximas de independentes.",
    "IDs brutos de sujeito não são globais no Le2i. Identidades são o par (ambiente, sujeito), então lados "
    "diferentes de uma dobra nunca compartilham identidade por construção, mas não há como confirmar que "
    "sejam pessoas diferentes.",
    "Cada dobra treina com dois ambientes em vez de três, com menos janelas e outra composição de classes, "
    "sujeitos e cenas; magnitudes das dobras não são comparáveis às do Le2i-CV oficial.",
    "Com três ambientes há só três dobras: o critério 'dois de três' tem três unidades e nenhuma variação "
    "entre ambientes pode ser estimada dentro de uma dobra.",
    "Qualidade de pose, de features e de rótulos não é avaliada aqui.",
]


def read_train_partition(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path, filters=[("split", "==", OFFICIAL_TRAIN_SPLIT)])
    unexpected = sorted(set(table["split"].astype(str)) - {OFFICIAL_TRAIN_SPLIT})
    if unexpected:
        raise ValueError(f"{path}: leitura filtrada devolveu splits fora de train: {unexpected}")
    return table


def _video_metadata_conflicts(frames: pd.DataFrame) -> list[str]:
    per_video = cast(pd.DataFrame, frames.groupby("video_id")[["env", "subject"]].nunique())
    ambiguous = cast(pd.Series, (per_video > 1).any(axis=1))
    return sorted(str(video_id) for video_id, is_ambiguous in ambiguous.items() if is_ambiguous)


def validate_official_train_partition(frames: pd.DataFrame, manifest: pd.DataFrame) -> None:
    expected = EXPECTED_OFFICIAL_TRAIN
    environments = tuple(sorted(str(env) for env in frames["env"].unique()))
    if environments != expected["environments"]:
        raise ValueError(
            f"ambientes do train oficial {environments} diferem dos esperados {expected['environments']}"
        )
    frame_videos = set(frames["video_id"].astype(str))
    manifest_videos = set(manifest["video_id"].astype(str))
    if frame_videos != manifest_videos:
        raise ValueError(
            "vídeos de train divergem entre frames e manifesto: só em frames="
            f"{sorted(frame_videos - manifest_videos)}, só no manifesto={sorted(manifest_videos - frame_videos)}"
        )
    if len(frame_videos) != expected["n_videos"]:
        raise ValueError(f"train tem {len(frame_videos)} vídeos, esperado {expected['n_videos']}")
    total = len(build_window_index(frames, stride=EVAL_STRIDE, drop_ignored=False))
    labeled = len(build_window_index(frames, stride=EVAL_STRIDE, drop_ignored=True))
    if total != expected["windows_total_eval_stride"] or labeled != expected["windows_labeled_eval_stride"]:
        raise ValueError(
            f"janelas de train em stride {EVAL_STRIDE}: total={total}, rotuladas={labeled}; esperado "
            f"total={expected['windows_total_eval_stride']}, rotuladas={expected['windows_labeled_eval_stride']}"
        )


def build_folds(environments: list[str]) -> list[dict[str, object]]:
    ordered = sorted(environments)
    return [
        {
            "fold": f"retain_{retained}",
            "retained_environment": retained,
            "development_environments": [env for env in ordered if env != retained],
        }
        for retained in ordered
    ]


def _identity_keys(table: pd.DataFrame) -> list[str]:
    pairs = table[["env", "subject"]].drop_duplicates()
    return sorted(f"{env}/{int(subject)}" for env, subject in zip(pairs["env"], pairs["subject"]))


def _class_support(windows: dict[str, pd.DataFrame], class_ids: list[int]) -> dict[str, dict[str, object]]:
    labeled_train = windows["train_stride"][windows["train_stride"]["label"] != IGNORE_LABEL]
    n_labeled_train = len(labeled_train)
    n_labeled_eval = int((windows["eval_stride"]["label"] != IGNORE_LABEL).sum())
    support: dict[str, dict[str, object]] = {}
    for class_id in class_ids:
        eval_rows = cast(pd.DataFrame, windows["eval_stride"][windows["eval_stride"]["label"] == class_id])
        n_train = int((labeled_train["label"] == class_id).sum())
        support[LE2I_LABEL_NAMES[class_id]] = {
            "class_id": class_id,
            "windows_train_stride": n_train,
            "windows_eval_stride": len(eval_rows),
            "share_train_stride": n_train / n_labeled_train if n_labeled_train else None,
            "share_eval_stride": len(eval_rows) / n_labeled_eval if n_labeled_eval else None,
            "videos": cast(pd.Series, eval_rows["video_id"]).nunique(),
            "identities": len(_identity_keys(eval_rows)),
        }
    return support


def _imbalance(support: dict[str, dict[str, object]]) -> dict[str, object] | None:
    present = {
        label: cast(int, entry["windows_train_stride"])
        for label, entry in support.items()
        if cast(int, entry["windows_train_stride"]) > 0
    }
    if not present:
        return None
    largest = max(present, key=lambda label: present[label])
    smallest = min(present, key=lambda label: present[label])
    return {
        "stride": TRAIN_STRIDE,
        "largest_class": largest,
        "largest_windows": present[largest],
        "smallest_present_class": smallest,
        "smallest_present_windows": present[smallest],
        "largest_to_smallest_ratio": present[largest] / present[smallest],
    }


def side_block(
    environments: list[str], windows: dict[str, pd.DataFrame], manifest: pd.DataFrame
) -> dict[str, object]:
    side_windows = {
        name: cast(pd.DataFrame, table[table["env"].isin(environments)]) for name, table in windows.items()
    }
    side_manifest = cast(pd.DataFrame, manifest[manifest["env"].isin(environments)])
    window_counts = {
        name: {
            "stride": STRIDES[name],
            "total": len(table),
            "labeled": int((table["label"] != IGNORE_LABEL).sum()),
            "ignored": int((table["label"] == IGNORE_LABEL).sum()),
            "labeled_outside_other": int(((table["label"] != IGNORE_LABEL) & (table["label"] != OTHER_CLASS)).sum()),
        }
        for name, table in side_windows.items()
    }
    support = _class_support(side_windows, RESTRICTED_CLASSES)
    identities = _identity_keys(side_windows["eval_stride"])
    return {
        "environments": list(environments),
        "cameras": sorted(int(cam) for cam in side_manifest["cam"].unique()),
        "resolution_distribution": resolution_distribution(side_manifest),
        "fps_distribution": fps_distribution(side_manifest),
        "n_videos": cast(pd.Series, side_windows["eval_stride"]["video_id"]).nunique(),
        "identities": identities,
        "n_identities": len(identities),
        "raw_subject_ids": sorted({int(subject) for subject in side_windows["eval_stride"]["subject"]}),
        "windows": window_counts,
        "restricted_class_support": support,
        "excluded_class_support": _class_support(side_windows, EXCLUDED_CLASSES),
        "missing_restricted_classes": [label for label, entry in support.items() if entry["windows_eval_stride"] == 0],
        "single_video_classes": [label for label, entry in support.items() if entry["videos"] == 1],
        "single_identity_classes": [label for label, entry in support.items() if entry["identities"] == 1],
        "imbalance": _imbalance(support),
    }


def criterion_blockers(development: dict[str, object], retained: dict[str, object]) -> list[str]:
    blockers: list[str] = []
    development_windows = cast(dict[str, dict[str, int]], development["windows"])
    retained_windows = cast(dict[str, dict[str, int]], retained["windows"])
    development_support = cast(dict[str, dict[str, int]], development["restricted_class_support"])
    if development_windows["train_stride"]["labeled"] == 0:
        blockers.append("o treino de desenvolvimento não tem janela rotulada")
    if development_support["other"]["windows_train_stride"] == 0:
        blockers.append(
            "o treino de desenvolvimento não tem janela other: a fração predita como other no ambiente "
            "retido deixa de medir o deslocamento"
        )
    if retained_windows["eval_stride"]["labeled_outside_other"] == 0:
        blockers.append(
            "o ambiente retido não tem janela rotulada fora de other: a fração predita como other não tem "
            "denominador"
        )
    return blockers


def _fmt_ratio(value: object) -> str:
    return f"{cast(float, value):.1f}"


def fold_limitations(
    fold: dict[str, object], development: dict[str, object], retained: dict[str, object], official: dict[str, object]
) -> list[str]:
    limitations: list[str] = []
    sides = (("treino de desenvolvimento", development), (f"ambiente retido {fold['retained_environment']}", retained))
    for name, side in sides:
        missing = cast(list[str], side["missing_restricted_classes"])
        if missing:
            limitations.append(f"{name}: classes restritas sem nenhuma janela: {missing}.")
        single_video = cast(list[str], side["single_video_classes"])
        if single_video:
            limitations.append(f"{name}: classes restritas presentes em um único vídeo: {single_video}.")
        single_identity = [
            label for label in cast(list[str], side["single_identity_classes"]) if label not in single_video
        ]
        if single_identity:
            limitations.append(
                f"{name}: classes restritas presentes em uma única identidade (ambiente, sujeito): {single_identity}."
            )
    development_windows = cast(dict[str, dict[str, int]], development["windows"])
    official_windows = cast(dict[str, dict[str, int]], official["windows"])
    official_labeled = official_windows["train_stride"]["labeled"]
    if official_labeled:
        share = development_windows["train_stride"]["labeled"] / official_labeled
        limitations.append(
            f"O treino de desenvolvimento usa {development_windows['train_stride']['labeled']} das "
            f"{official_labeled} janelas rotuladas do train oficial em stride {TRAIN_STRIDE} ({share:.0%})."
        )
    development_imbalance = cast(dict[str, object] | None, development["imbalance"])
    official_imbalance = cast(dict[str, object] | None, official["imbalance"])
    if development_imbalance is not None and official_imbalance is not None:
        limitations.append(
            "Desbalanceamento no treino de desenvolvimento (stride 4): "
            f"{development_imbalance['largest_class']}/{development_imbalance['smallest_present_class']} = "
            f"{_fmt_ratio(development_imbalance['largest_to_smallest_ratio'])}, contra "
            f"{official_imbalance['largest_class']}/{official_imbalance['smallest_present_class']} = "
            f"{_fmt_ratio(official_imbalance['largest_to_smallest_ratio'])} no train oficial. Descritivo, sem limiar."
        )
    limitations += _other_share(development, retained, official)
    shared_raw_ids = sorted(
        set(cast(list[int], development["raw_subject_ids"])) & set(cast(list[int], retained["raw_subject_ids"]))
    )
    if shared_raw_ids:
        limitations.append(
            f"IDs brutos de sujeito presentes nos dois lados: {shared_raw_ids}. Não é possível confirmar se são "
            "as mesmas pessoas; a dobra é disjunta por ambiente, não comprovadamente por sujeito."
        )
    if len(cast(list[int], retained["cameras"])) == 1:
        limitations.append(
            f"O ambiente retido tem uma única câmera ({cast(list[int], retained['cameras'])[0]}): ambiente e "
            "câmera não se separam."
        )
    return limitations


def _other_share(
    development: dict[str, object], retained: dict[str, object], official: dict[str, object]
) -> list[str]:
    def share(side: dict[str, object], key: str) -> float | None:
        support = cast(dict[str, dict[str, float | None]], side["restricted_class_support"])
        return support["other"][key]

    shares = (share(development, "share_train_stride"), share(retained, "share_eval_stride"),
              share(official, "share_train_stride"))
    if any(value is None for value in shares):
        return []
    development_share, retained_share, official_share = cast(tuple[float, float, float], shares)
    return [
        f"Fração das janelas rotuladas que são other: {development_share:.0%} no treino de desenvolvimento "
        f"(stride {TRAIN_STRIDE}), {retained_share:.0%} no ambiente retido (stride {EVAL_STRIDE}) e "
        f"{official_share:.0%} no train oficial (stride {TRAIN_STRIDE}). A fração predita como other também "
        "depende dessa proporção; o critério compara armas dentro da mesma dobra."
    ]


def _input_limitations(official: dict[str, object]) -> list[str]:
    resolutions = cast(dict[str, int], official["resolution_distribution"])
    fps_values = cast(dict[str, int], official["fps_distribution"])
    if len(resolutions) == 1 and len(fps_values) == 1:
        return [
            f"Todo o train oficial tem a mesma resolução ({next(iter(resolutions))}) e o mesmo fps "
            f"({next(iter(fps_values))}): as dobras não podem testar a hipótese de mudança de resolução ou fps "
            "específica do test oficial."
        ]
    return []


def build_report(frames: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, object]:
    conflicts = _video_metadata_conflicts(frames)
    if conflicts:
        raise ValueError(f"vídeos com ambiente ou sujeito ambíguo: {conflicts}")
    windows = {
        name: build_window_index(frames, stride=stride, drop_ignored=False) for name, stride in STRIDES.items()
    }
    environments = sorted(str(env) for env in frames["env"].unique())
    official = side_block(environments, windows, manifest)
    folds: list[dict[str, object]] = []
    for fold in build_folds(environments):
        development = side_block(cast(list[str], fold["development_environments"]), windows, manifest)
        retained = side_block([cast(str, fold["retained_environment"])], windows, manifest)
        folds.append(
            {
                **fold,
                "development_train": development,
                "retained": retained,
                "criterion_blockers": criterion_blockers(development, retained),
                "limitations": fold_limitations(fold, development, retained, official),
            }
        )
    return {
        "analysis": "le2i_cv_internal_fold_feasibility",
        "canonical": False,
        "read_only": True,
        "status": "assessed",
        "dataset": DATASET,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "label_names": list(LE2I_LABEL_NAMES),
        "ignore_label": IGNORE_LABEL,
        "window_frames": WINDOW_FRAMES,
        "strides": STRIDES,
        "restricted_classes": list(RESTRICTED_CLASSES),
        "restricted_labels": [LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES],
        "excluded_labels": [LE2I_LABEL_NAMES[class_id] for class_id in EXCLUDED_CLASSES],
        "official_train": official,
        "folds": folds,
        "missing_prerequisites": [],
        "limitations": GLOBAL_LIMITATIONS + _input_limitations(official),
        "approval_required": APPROVAL_REQUIRED,
    }


def missing_prerequisites_report(missing: list[str]) -> dict[str, object]:
    return {
        "analysis": "le2i_cv_internal_fold_feasibility",
        "canonical": False,
        "read_only": True,
        "status": "missing_prerequisites",
        "dataset": DATASET,
        "official_splits_read": [],
        "missing_prerequisites": missing,
        "folds": [],
        "limitations": GLOBAL_LIMITATIONS,
        "approval_required": APPROVAL_REQUIRED,
    }


def _side_row(name: str, side: dict[str, object]) -> str:
    windows = cast(dict[str, dict[str, int]], side["windows"])
    train, evaluation = windows["train_stride"], windows["eval_stride"]
    return (
        f"| {name} | {', '.join(cast(list[str], side['environments']))} | "
        f"{', '.join(str(cam) for cam in cast(list[int], side['cameras']))} | {side['n_videos']} | "
        f"{side['n_identities']} | {train['labeled']} / {train['ignored']} | "
        f"{evaluation['labeled']} / {evaluation['ignored']} | {evaluation['labeled_outside_other']} |"
    )


def _support_table(development: dict[str, object], retained: dict[str, object]) -> list[str]:
    development_support = cast(dict[str, dict[str, int]], development["restricted_class_support"])
    retained_support = cast(dict[str, dict[str, int]], retained["restricted_class_support"])
    lines = [
        "",
        "| Classe | Treino dev.: janelas (stride 4) | vídeos | identidades | "
        "Retido: janelas (stride 1) | vídeos | identidades |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, dev in development_support.items():
        ret = retained_support[label]
        lines.append(
            f"| {label} | {dev['windows_train_stride']} | {dev['videos']} | {dev['identities']} | "
            f"{ret['windows_eval_stride']} | {ret['videos']} | {ret['identities']} |"
        )
    return lines


SIDE_TABLE_HEADER = [
    "",
    "| Lado | Ambientes | Câmeras | Vídeos | Identidades | Janelas stride 4 (rotuladas / ignoradas) | "
    "Janelas stride 1 (rotuladas / ignoradas) | Stride 1 fora de other |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
]


def render_summary(report: dict[str, object]) -> str:
    lines = [
        "# Viabilidade das dobras internas por ambiente do Le2i-CV (não canônico)",
        "",
        "Avaliação somente leitura do candidato 1 da Etapa 2A sobre o split train oficial do Le2i-CV; val e "
        "test oficiais não são lidos. Nada foi treinado, selecionado ou ajustado. Janelas pelo contrato "
        "temporal congelado; IGNORE_LABEL pelo rótulo do último quadro; classes restritas fixas de "
        "RESTRICTED_CLASSES. Identidades = (ambiente, sujeito).",
    ]
    if report["status"] == "missing_prerequisites":
        lines += ["", "## Pré-requisitos ausentes", ""]
        lines += [f"- `{path}`" for path in cast(list[str], report["missing_prerequisites"])]
        lines += ["", "Nenhum número foi calculado."]
    else:
        official = cast(dict[str, object], report["official_train"])
        lines += ["", "## Train oficial (referência)"] + SIDE_TABLE_HEADER + [_side_row("train oficial", official)]
        for fold in cast(list[dict[str, object]], report["folds"]):
            development = cast(dict[str, object], fold["development_train"])
            retained = cast(dict[str, object], fold["retained"])
            lines += ["", f"## Dobra `{fold['fold']}` — retém {fold['retained_environment']}"]
            lines += SIDE_TABLE_HEADER
            lines += [_side_row("treino de desenvolvimento", development), _side_row("retido", retained)]
            lines += _support_table(development, retained)
            blockers = cast(list[str], fold["criterion_blockers"])
            lines += ["", "Bloqueios do critério da Etapa 2A:", ""]
            lines += [f"- {text}" for text in blockers] or ["- Nenhum."]
            lines += ["", "Limitações da dobra:", ""]
            lines += [f"- {text}" for text in cast(list[str], fold["limitations"])] or ["- Nenhuma."]
    lines += ["", "## Exige aprovação explícita antes de implementar ou treinar", ""]
    lines += [f"- {text}" for text in cast(list[str], report["approval_required"])]
    lines += ["", "## Limitações gerais", ""]
    lines += [f"- {text}" for text in cast(list[str], report["limitations"])]
    return "\n".join(lines) + "\n"


def protected_paths(frames_path: Path, manifest_path: Path) -> list[Path]:
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        frames_path.parent,
        manifest_path.parent,
    ]


def run_assessment(frames_path: Path, manifest_path: Path, output_dir: Path, force: bool = False) -> bool:
    check_output_dir(output_dir, protected_paths(frames_path, manifest_path))
    missing = [display_path(path) for path in (frames_path, manifest_path) if not path.exists()]
    if missing:
        report = missing_prerequisites_report(missing)
        print(f"pré-requisitos ausentes: {missing}; nenhum número calculado", file=sys.stderr)
        snapshot: dict[str, str] = {}
    else:
        snapshot = snapshot_inputs([frames_path, manifest_path])
        frames = read_train_partition(frames_path)
        manifest = read_train_partition(manifest_path)
        validate_official_train_partition(frames, manifest)
        report = build_report(frames, manifest)
        report["input_sha256"] = {display_path(Path(path)): digest for path, digest in snapshot.items()}
    targets = {
        output_dir / FEASIBILITY_JSON_FILE: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        output_dir / FEASIBILITY_SUMMARY_FILE: render_summary(report),
    }
    written = write_text_outputs(output_dir, targets, force)
    verify_inputs_unchanged(snapshot)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapter = Le2iDatasetAdapter(protocol="cv")
    assess_parser = subparsers.add_parser(
        "assess", help="Avalia a viabilidade das três dobras internas por ambiente do train do Le2i-CV"
    )
    assess_parser.add_argument("--frames", type=Path, default=adapter.frames_path)
    assess_parser.add_argument("--manifest", type=Path, default=adapter.manifest_path)
    assess_parser.add_argument("--output-dir", type=Path, required=True)
    assess_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da avaliação de viabilidade")

    args = parser.parse_args()
    if args.command == "assess":
        run_assessment(args.frames, args.manifest, args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.internal_fold_feasibility import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
