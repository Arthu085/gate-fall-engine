"""Padronização (z-score) das features de pose do Le2i, treinada só no split de treino."""

import argparse
import sys
from typing import Callable

import numpy as np

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.frames import read_frames
from gatefall.data.le2i.frames import FRAMES_PATH
from gatefall.data.le2i.pose_dataset import (
    EXPECTED_FEATURE_DIM,
    EXPECTED_USABLE_WINDOWS_STRIDE4,
    PoseWindowDataset,
)
from gatefall.features.standardization import (
    STATS_PATH,
    TRAIN_SPLIT,
    apply_standardization,
    compute_train_stats,
    excluded_dimension_mask,
    load_stats,
    save_stats,
    stale_stats_mismatches,
)
from gatefall.features.standardization_selftest import run_standardization_selftest
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features, feature_blocks

EVAL_SPLITS = ["val", "test"]
EXPECTED_VIDEOS_LOADED = {"train": 133, "val": 19, "test": 38}


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def run_build(force: bool) -> None:
    stats = compute_train_stats(stride=TRAIN_STRIDE)
    save_stats(stats, STATS_PATH, force=force)


def _counting_pose_loader(
    counters: dict[str, int], split: str
) -> Callable[[str], np.ndarray]:
    def loader(video_id: str) -> np.ndarray:
        counters[split] = counters.get(split, 0) + 1
        return build_pose_features(video_id)[0]

    return loader


def run_report() -> None:
    if not STATS_PATH.exists():
        print(
            f"{STATS_PATH} não existe; rode `build` antes de `report`",
            file=sys.stderr,
        )
        sys.exit(1)

    stats = load_stats(STATS_PATH)
    frames = read_frames(FRAMES_PATH)
    names = stats.feature_names
    excluded_mask = excluded_dimension_mask(names)

    checks: list[bool] = []
    videos_loaded: dict[str, int] = {}

    mismatches = stale_stats_mismatches(stats)
    checks.append(
        _check(
            "estatísticas persistidas batem com gatefall.pose.kinematics (campos "
            f"divergentes: {mismatches if mismatches else 'nenhum'})",
            not mismatches,
        )
    )

    checks.append(
        _check(
            f"hash persistido de {FRAMES_PATH} bate com o arquivo atual",
            stats.frames_hash == sha256_file(FRAMES_PATH),
        )
    )

    mean = np.asarray(stats.mean, dtype=np.float64)
    std = np.asarray(stats.std, dtype=np.float64)
    checks.append(
        _check(
            "mean e std têm shape [134] e são finitos",
            mean.shape == (EXPECTED_FEATURE_DIM,)
            and std.shape == (EXPECTED_FEATURE_DIM,)
            and bool(np.isfinite(mean).all())
            and bool(np.isfinite(std).all()),
        )
    )

    train_dataset = PoseWindowDataset(
        frames, TRAIN_SPLIT, TRAIN_STRIDE, _counting_pose_loader(videos_loaded, TRAIN_SPLIT)
    )

    checks.append(
        _check(
            f"contagem de janelas de treino em stride={TRAIN_STRIDE} == "
            f"{EXPECTED_USABLE_WINDOWS_STRIDE4[TRAIN_SPLIT]}",
            len(train_dataset) == EXPECTED_USABLE_WINDOWS_STRIDE4[TRAIN_SPLIT]
            and stats.window_count == EXPECTED_USABLE_WINDOWS_STRIDE4[TRAIN_SPLIT],
        )
    )

    train_raw_windows: list[np.ndarray] = []
    for i in range(len(train_dataset)):
        window, _, _ = train_dataset[i]
        train_raw_windows.append(window)
    train_raw = np.concatenate(train_raw_windows, axis=0)
    train_standardized = apply_standardization(train_raw, stats)

    standardized_dim_mask = ~excluded_mask & ~np.asarray(stats.guarded_mask, dtype=bool)
    per_dim_mean = train_standardized[:, standardized_dim_mask].astype(np.float64).mean(axis=0)
    per_dim_std = train_standardized[:, standardized_dim_mask].astype(np.float64).std(
        axis=0, ddof=0
    )
    checks.append(
        _check(
            "após padronizar: mean por dimensão dentro de 1e-3 de 0 e std dentro de "
            "1e-3 de 1 nas dimensões padronizadas",
            bool(np.allclose(per_dim_mean, 0.0, atol=1e-3))
            and bool(np.allclose(per_dim_std, 1.0, atol=1e-3)),
        )
    )

    kp_conf_train = train_raw[:, excluded_mask]
    kp_conf_min = float(kp_conf_train.min())
    kp_conf_max = float(kp_conf_train.max())
    # A confiança bruta do YOLO-Pose (saída de sigmoid) nunca satura em
    # exatamente 1.0 na prática — só o mínimo 0.0, produzido pelo
    # forward-fill de `impute_missing` em quadros sem detecção, é atingido
    # de forma exata. Verificamos os limites do intervalo e o mínimo exato;
    # o máximo observado é reportado, não comparado por igualdade exata.
    checks.append(
        _check(
            "kp_conf no treino: valores em [0, 1] com min == 0.0 exato",
            bool(np.all((kp_conf_train >= 0.0) & (kp_conf_train <= 1.0)))
            and kp_conf_min == 0.0,
        )
    )
    print(f"\nkp_conf no treino: min={kp_conf_min:.6f}, max={kp_conf_max:.6f}")

    eval_non_finite: dict[str, int] = {}
    for split in EVAL_SPLITS:
        dataset = PoseWindowDataset(
            frames, split, EVAL_STRIDE, _counting_pose_loader(videos_loaded, split)
        )
        non_finite = 0
        for i in range(len(dataset)):
            window, _, _ = dataset[i]
            standardized = apply_standardization(window, stats)
            non_finite += int(np.sum(~np.isfinite(standardized)))
        eval_non_finite[split] = non_finite
        checks.append(
            _check(f"split={split}: nenhum valor não finito após padronizar", non_finite == 0)
        )

    print("\n=== estatísticas por bloco após padronização (treino, |valor|) ===")
    for name, start, end in feature_blocks():
        block = train_standardized[:, start:end]
        abs_block = np.abs(block)
        p50, p99, p999, p_max = (
            float(np.percentile(abs_block, 50)),
            float(np.percentile(abs_block, 99)),
            float(np.percentile(abs_block, 99.9)),
            float(abs_block.max()),
        )
        print(
            f"  {name}: p50={p50:.4f}, p99={p99:.4f}, p99.9={p999:.4f}, max={p_max:.4f}"
        )

    guarded_names = [
        name for name, guarded in zip(names, stats.guarded_mask) if guarded
    ]
    print(f"\ndimensões guardadas ({stats.guarded_count}): {guarded_names}")

    print("\n=== vídeos carregados por split ===")
    total_videos = 0
    for split, expected in EXPECTED_VIDEOS_LOADED.items():
        loaded = videos_loaded.get(split, 0)
        total_videos += loaded
        print(f"  {split}: {loaded} (esperado {expected})")
    print(f"  total: {total_videos} (esperado {sum(EXPECTED_VIDEOS_LOADED.values())})")

    if not all(checks):
        print("\nstandardize report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nstandardize report OK: todas as checagens passaram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Calcula e persiste as estatísticas de padronização do treino"
    )
    build_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o arquivo já existente"
    )
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas da padronização"
    )
    subparsers.add_parser(
        "report",
        help="Roda a padronização sobre o dataset real do Le2i e reporta estatísticas",
    )

    args = parser.parse_args()
    if args.command == "build":
        run_build(force=args.force)
    elif args.command == "selftest":
        run_standardization_selftest()
    elif args.command == "report":
        run_report()


if __name__ == "__main__":
    main()
