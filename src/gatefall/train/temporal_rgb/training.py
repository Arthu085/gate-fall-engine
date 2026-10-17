"""Sonda linear com padronização ajustada só no treino de desenvolvimento."""

import json
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gatefall.config import NUM_CLASSES
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.hashing import sha256_array, sha256_file
from gatefall.train.imbalance_experiment.cli import OUTPUT_ROOT as BASELINE_ROOT, _sources
from gatefall.train.imbalance_experiment.report import _read_run
from gatefall.train.baseline_a.config import load_config
from gatefall.train.internal_folds import folds as cv_folds, stats as cv_stats, summary as cv_summary
from gatefall.train.internal_folds.training import QualityFeatures
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT, RETAINED_SPLIT
from gatefall.train.subject_folds import folds as cs_folds, summary as cs_summary
from gatefall.train.shared.imbalance import RECIPES, class_weights
from gatefall.train.shared.metrics import classification_summary, restricted_macro_f1
from gatefall.train.temporal_rgb.contract import FEATURE_DIM, MODES, SEED, contract, digest, guard_destination, setup
from gatefall.train.temporal_rgb.data import Context, Partition, context_record, fold_windows, window_identity
from gatefall.train.temporal_rgb.extraction import load_features
from gatefall.train.temporal_rgb.storage import cache_path, write_json

ARTIFACTS = ("checkpoint.pt", "metrics.json", "predictions.npz")


def summary(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("predições e rótulos incompatíveis")
    if np.any((labels < 0) | (labels >= NUM_CLASSES)) or np.any((predictions < 0) | (predictions >= NUM_CLASSES)):
        raise ValueError("rótulos ou predições fora do contrato")
    macro, _ = restricted_macro_f1(labels, predictions)
    return {"restricted_macro_f1": macro, **classification_summary(labels, predictions, LE2I_LABEL_NAMES)}


def normalize(train: np.ndarray, retained: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    return (train - mean) / std, (retained - mean) / std, mean, std


def fit_head(
    train_x: np.ndarray, train_y: np.ndarray, retained_x: np.ndarray, device: str, epochs: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[float]]:
    setup(device)
    train, retained, mean, std = normalize(train_x, retained_x)
    model = nn.Linear(FEATURE_DIM, NUM_CLASSES).to(device)
    recipe = contract()
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train), torch.from_numpy(train_y.copy())),
        batch_size=recipe["batch_size"], shuffle=True, num_workers=0, generator=generator,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_y, NUM_CLASSES, RECIPES["baseline"]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("perda não finita")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip_norm"])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
        scheduler.step()
        history.append(loss_sum / len(train_y))
    model.eval()

    def predict(features: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return np.concatenate([
                model(batch.to(device)).argmax(1).cpu().numpy()
                for batch in torch.from_numpy(features).split(recipe["batch_size"])
            ])

    checkpoint = {
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "mean": torch.from_numpy(mean), "std": torch.from_numpy(std), "epochs": epochs,
    }
    return checkpoint, predict(train), predict(retained), history


def checkpoint_digest(checkpoint: dict[str, Any]) -> str:
    return digest({
        name: sha256_array(tensor.numpy())
        for name, tensor in {**checkpoint["state_dict"], "mean": checkpoint["mean"], "std": checkpoint["std"]}.items()
    })


def _native_control(context: Context, partition: Partition, fold_root: Path) -> Path:
    if context.protocol == "cs":
        document = cs_folds.read_folds_document(fold_root)
        cs_summary.run_entry(fold_root, document, partition.membership_record, "A", SEED)
        return cs_folds.fold_run_dir(fold_root, partition.fold, "A", SEED)
    document = cv_folds.read_folds_document(fold_root)
    files = cv_stats.load_recorded_stats_files(
        cv_folds.stats_dir(fold_root, partition.fold), partition.membership_record, context.input_sha256,
    )
    quality = QualityFeatures(
        root=Path(cast(str, document["quality_features_path"])),
        sha256=cast(str, document["quality_features_sha256"]),
    )
    cv_summary.run_entry(fold_root, partition.membership_record, files, quality, "A", SEED)
    return cv_folds.fold_run_dir(fold_root, partition.fold, "A", SEED)


def _control_record(path: Path, sidecar: str, windows: dict[str, Any]) -> dict[str, Any]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    for split, index in windows.items():
        labels = index["label"].to_numpy(dtype=np.int64)
        expected = {name: int(np.sum(labels == class_id)) for class_id, name in enumerate(LE2I_LABEL_NAMES)}
        actual = metrics["final"][split]
        if actual["support"] != expected or any(actual["per_class"][name]["support"] != expected[name] for name in expected):
            raise ValueError(f"suporte do controle A diverge das janelas atuais: {path}/{split}")
    retained = metrics["final"]["val"]
    return {
        "restricted_macro_f1": retained["macro_f1_restricted"], "per_class": retained["per_class"],
        "confusion_matrix": retained["confusion_matrix"], "path": str(path),
        "metrics": metrics, "config": load_config(path / "config.yaml").to_dict(),
        "fold_provenance": json.loads((path / sidecar).read_text(encoding="utf-8")),
        "window_sha256": {split: window_identity(index) for split, index in windows.items()},
        "source": "native" if sidecar == "fold_run.json" else "imbalance",
        "artifact_sha256": {name: sha256_file(path / name)
                            for name in ("config.yaml", "metrics.json", "checkpoint.pt", sidecar)},
    }


def _verify_equivalent_controls(native: dict[str, Any], imbalance: dict[str, Any]) -> None:
    left_config, right_config = native["config"], imbalance["config"]
    if ({key: value for key, value in left_config.items() if key != "run_name"}
            != {key: value for key, value in right_config.items() if key != "run_name"}
            or native["metrics"]["final"] != imbalance["metrics"]["final"]
            or native["window_sha256"] != imbalance["window_sha256"]):
        raise ValueError("controles nativo e imbalance discordam na receita, nas janelas ou nas métricas")
    left = torch.load(Path(native["path"]) / "checkpoint.pt", map_location="cpu", weights_only=True)
    right = torch.load(Path(imbalance["path"]) / "checkpoint.pt", map_location="cpu", weights_only=True)
    if set(left) != set(right) or any(left[key].dtype != right[key].dtype or not torch.equal(left[key], right[key]) for key in left):
        raise ValueError("controles nativo e imbalance discordam nos tensores do checkpoint")


def baseline_control(context: Context, partition: Partition, fold_root: Path) -> dict[str, Any]:
    if context.protocol not in ("cs", "cv"):
        raise ValueError("protocolo deve ser cs ou cv")
    folds = cs_folds if context.protocol == "cs" else cv_folds
    native = folds.fold_run_dir(fold_root, partition.fold, "A", SEED)
    imbalance = BASELINE_ROOT / context.protocol / partition.fold / "baseline" / f"seed_{SEED}"
    if not native.exists() and not imbalance.exists():
        raise FileNotFoundError(f"controle A ausente: {native} e {imbalance}; nenhum controle será treinado")
    development, retained, _, source = _sources(context.protocol, fold_root, partition.fold)
    provenance = cast(dict[str, Any], source["provenance"])
    if provenance["input_sha256"] != context.input_sha256 or provenance["membership_sha256"] != context.membership_sha256:
        raise ValueError("controle A diverge do contexto RGB")
    if provenance["development_videos"] != list(partition.development_video_ids) or provenance["retained_videos"] != list(partition.retained_video_ids):
        raise ValueError("vídeos do controle A divergem da partição RGB")
    windows = {}
    for baseline, side in ((development, DEVELOPMENT_SPLIT), (retained, RETAINED_SPLIT)):
        index = fold_windows(partition, side)
        if window_identity(baseline._windows) != window_identity(index):
            raise ValueError("janelas de A e RGB não são idênticas")
        windows["train" if side == DEVELOPMENT_SPLIT else "val"] = index
    windows["test"] = windows["val"]
    controls = []
    if native.exists():
        _native_control(context, partition, fold_root)
        controls.append(_control_record(native, "fold_run.json", windows))
    if imbalance.exists():
        row = _read_run(imbalance, context.protocol, partition.fold, "baseline", SEED)
        if any(cast(dict[str, Any], row["provenance"]).get(key) != value for key, value in provenance.items()):
            raise ValueError("controle imbalance diverge das entradas e estatísticas atuais da dobra")
        controls.append(_control_record(imbalance, "experiment.json", windows))
    if len(controls) == 2:
        _verify_equivalent_controls(controls[0], controls[1])
    return {**controls[0], "equivalent_controls": controls[1:]}


def verify_run(
    path: Path, expected: dict[str, Any], train_x: np.ndarray, train_y: np.ndarray,
    retained_x: np.ndarray, retained_y: np.ndarray,
) -> dict[str, Any]:
    record = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if record["provenance"] != expected:
        raise ValueError(f"run incompatível ou antigo: {path}")
    if record["artifact_sha256"] != {name: sha256_file(path / name) for name in ARTIFACTS}:
        raise ValueError(f"hash dos artefatos diverge: {path}")
    checkpoint = torch.load(path / "checkpoint.pt", weights_only=True, map_location="cpu")
    if checkpoint["epochs"] != contract()["epochs"] or checkpoint_digest(checkpoint) != record["checkpoint_state_sha256"]:
        raise ValueError(f"checkpoint não corresponde à última época: {path}")
    model = nn.Linear(FEATURE_DIM, NUM_CLASSES)
    model.load_state_dict(checkpoint["state_dict"])
    if checkpoint["mean"].shape != (FEATURE_DIM,) or checkpoint["std"].shape != (FEATURE_DIM,):
        raise ValueError("padronização incompatível")
    normalized_train, normalized_retained, mean, std = normalize(train_x, retained_x)
    if not np.array_equal(checkpoint["mean"].numpy(), mean) or not np.array_equal(checkpoint["std"].numpy(), std):
        raise ValueError("estatística do checkpoint não foi ajustada só no desenvolvimento")
    device = expected["settings"]["device"]
    model.eval().to(device)
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if len(metrics["train_loss_history"]) != contract()["epochs"]:
        raise ValueError("histórico de épocas incompleto")
    with np.load(path / "predictions.npz", allow_pickle=False) as predictions:
        for side, labels, features in (("development", train_y, normalized_train), ("retained", retained_y, normalized_retained)):
            if not np.array_equal(predictions[f"{side}_true"], labels):
                raise ValueError("rótulos do run divergem das janelas")
            if metrics[side] != summary(labels, predictions[f"{side}_pred"]):
                raise ValueError("métricas não reproduzem as predições")
            with torch.inference_mode():
                actual = np.concatenate([
                    model(batch.to(device)).argmax(1).cpu().numpy()
                    for batch in torch.from_numpy(features).split(contract()["batch_size"])
                ])
            if not np.array_equal(actual, predictions[f"{side}_pred"]):
                raise ValueError("checkpoint não reproduz as predições registradas")
    return {"path": str(path), "provenance": expected, "checkpoint_state_sha256": record["checkpoint_state_sha256"],
            **metrics["retained"]}


def run_probe(
    context: Context, partition: Partition, fold_root: Path, root: Path, mode: str, device: str,
    *, verify_only: bool = False,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("modo inválido")
    settings = setup(device)
    development = fold_windows(partition, DEVELOPMENT_SPLIT)
    retained = fold_windows(partition, RETAINED_SPLIT)
    if set(partition.development_video_ids) & set(partition.retained_video_ids):
        raise ValueError("vazamento de vídeos")
    control = baseline_control(context, partition, fold_root)
    train_x, train_hashes = load_features(context, root, development, mode, expected_settings=settings)
    retained_x, retained_hashes = load_features(context, root, retained, mode, expected_settings=settings)
    train_y = development["label"].to_numpy(dtype=np.int64)
    retained_y = retained["label"].to_numpy(dtype=np.int64)
    expected = {
        **context_record(context), "fold": partition.fold, "mode": mode, "contract": contract(),
        "fold_membership": partition.membership_record, "settings": settings,
        "development_window_sha256": window_identity(development),
        "retained_window_sha256": window_identity(retained),
        "feature_sha256": {**train_hashes, **retained_hashes}, "baseline_a": control,
    }
    destination = root / context.protocol / "runs" / partition.fold / mode / f"seed_{SEED}"
    guard_destination(destination)
    if destination.exists() or verify_only:
        return verify_run(destination, expected, train_x, train_y, retained_x, retained_y)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".partial-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        checkpoint, train_pred, retained_pred, history = fit_head(train_x, train_y, retained_x, device, contract()["epochs"])
        torch.save(checkpoint, staging / "checkpoint.pt")
        np.savez_compressed(staging / "predictions.npz", development_true=train_y, development_pred=train_pred,
                            retained_true=retained_y, retained_pred=retained_pred)
        write_json(staging / "metrics.json", {
            "development": summary(train_y, train_pred), "retained": summary(retained_y, retained_pred),
            "train_loss_history": history,
        })
        write_json(staging / "run.json", {
            "provenance": expected, "artifact_sha256": {name: sha256_file(staging / name) for name in ARTIFACTS},
            "checkpoint_state_sha256": checkpoint_digest(checkpoint),
        })
        verify_run(staging, expected, train_x, train_y, retained_x, retained_y)
        for name, digest_value in context.input_sha256.items():
            if sha256_file(getattr(context.adapter, f"{name}_path")) != digest_value:
                raise RuntimeError("entradas mudaram durante o treino")
        for video_id, digest_value in expected["feature_sha256"].items():
            if sha256_file(cache_path(root / context.protocol / "features", video_id)) != digest_value:
                raise RuntimeError("features mudaram durante o treino")
        staging.rename(destination)
    return verify_run(destination, expected, train_x, train_y, retained_x, retained_y)
