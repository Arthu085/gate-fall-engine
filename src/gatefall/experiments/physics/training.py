"""Treino e avaliação do experimento, sem modificar os runs de referência."""

import json
import hashlib
import math
import os
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gatefall.config import NUM_CLASSES, TARGET_FPS, TRAIN_STRIDE
from gatefall.data.windowing import window_frame_indices
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.shared.events import detect_alarms_for_video, split_event_report
from gatefall.experiments.physics import FEATURE_SCHEMA, SCHEMA_VERSION
from gatefall.experiments.physics.data import code_provenance, load_bundle, write_json
from gatefall.experiments.physics.model import FUSIONS, TRAINABLE_GATES, FusionClassifier, ordering_loss
from gatefall.experiments.tcnte import ARM as TCNTE_ARM
from gatefall.experiments.tcnte import FUSION as TCNTE_FUSION
from gatefall.experiments.tcnte.model import FocalLoss, TCNTEClassifier
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import feature_blocks
from gatefall.runs import validate_local_run_dir
from gatefall.train.shared.determinism import configure_determinism
from gatefall.train.shared.metrics import classification_summary, restricted_macro_f1

STUDY_ARMS = ("A", "B0", "B1", "C0", "C1", TCNTE_ARM)
STUDY_FUSIONS = (*FUSIONS, TCNTE_FUSION)
POSE_ONLY_ARMS = ("A", TCNTE_ARM)
DEFAULT_OPTIMIZATION = {"optimizer": "adamw", "lr": .001, "weight_decay": .01,
                        "scheduler": "cosine", "loss": "weighted_cross_entropy"}
# Adam lr=1e-4, sem decaimento nem scheduler, e perda focal γ=3 seguem o treino
# publicado do TCNTE; pesos de classe passam a ser o inverso da frequência de treino.
TCNTE_OPTIMIZATION = {"optimizer": "adam", "lr": 1e-4, "weight_decay": 0.0,
                      "scheduler": None, "loss": "focal_gamma3"}
EXTERNAL_COMPATIBILITY_KEYS = ("schema_version", "feature_schema", "visual_kind", "visual_dim",
                               "label_names", "evidence_names", "target_fps")


def optimization_recipe(fusion: str) -> dict:
    return dict(TCNTE_OPTIMIZATION if fusion == TCNTE_FUSION else DEFAULT_OPTIMIZATION)


def build_classifier(visual_dim: int, fusion: str, horizons: list[int] | None = None) -> nn.Module:
    if fusion == TCNTE_FUSION:
        if horizons is not None:
            raise ValueError("TCNTE não participa da matriz de aprendizado de movimento")
        return TCNTEClassifier()
    if horizons is None:
        return FusionClassifier(visual_dim, fusion)
    from gatefall.experiments.motion.learning import MotionClassifier
    return MotionClassifier(visual_dim, fusion, horizons)


def peak_memory_bytes(device: str) -> int:
    if device.startswith("cuda"):
        return int(torch.cuda.max_memory_allocated())
    # ru_maxrss é reportado em KiB no Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def pose_validity(values: np.ndarray) -> np.ndarray:
    mask = np.broadcast_to(values[:, 134:135] != 0, values.shape).copy()
    for name, start, end in feature_blocks():
        if name.endswith("velocity"):
            mask[:, start:end] &= values[:, 135:136] != 0
        elif name.endswith("acceleration"):
            mask[:, start:end] &= values[:, 136:137] != 0
        elif name == "trunk":
            mask[:, end - 1] &= values[:, 135] != 0
    mask[:, 134:] = True
    return mask


def fit_stats(manifest: dict, videos: dict[str, dict[str, np.ndarray]]) -> dict:
    result: dict = {"feature_schema": FEATURE_SCHEMA, "fit_split": "train"}
    entries = [e for e in manifest["videos"] if e["split"] == "train"]
    result["fit_video_ids"] = [e["video_id"] for e in entries]
    for name, dimension in (("pose", 138), ("visual", manifest["visual_dim"])):
        total = np.zeros(dimension); squares = np.zeros(dimension); count = np.zeros(dimension)
        for entry in entries:
            data = videos[entry["video_id"]]; values = data[name].astype(np.float64)
            valid = (pose_validity(values) if name == "pose"
                     else np.broadcast_to(data["evidence"][:, 7:8] != 0, values.shape))
            total += np.where(valid, values, 0).sum(axis=0)
            squares += np.where(valid, values**2, 0).sum(axis=0)
            count += valid.sum(axis=0)
        mean = total / np.maximum(count, 1)
        std = np.sqrt(np.maximum(0, squares / np.maximum(count, 1) - mean**2))
        std[std < 1e-6] = 1
        if name == "pose":
            mean[134:] = 0; std[134:] = 1
        elif manifest["visual_kind"] == "sam3":
            mean[0] = 0; std[0] = 1
        result[name] = {"mean": mean.tolist(), "std": std.tolist(), "count": count.astype(int).tolist()}
    return result


def standardize(values: np.ndarray, stats: dict, name: str, evidence: np.ndarray) -> np.ndarray:
    mean = np.asarray(stats[name]["mean"], dtype=np.float64)
    std = np.asarray(stats[name]["std"], dtype=np.float64)
    if mean.shape != (values.shape[-1],) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("estatísticas incompatíveis")
    result = ((values.astype(np.float64) - mean) / std).astype(np.float32)
    valid = (pose_validity(values) if name == "pose"
             else np.broadcast_to(evidence[:, 7:8] != 0, values.shape))
    result[~valid] = 0
    if not np.isfinite(result).all():
        raise ValueError("padronização não finita")
    return result


class Windows(Dataset):
    def __init__(self, manifest: dict, videos: dict[str, dict[str, np.ndarray]], stats: dict,
                 split: str, stride: int, labeled_only: bool = True,
                 motion_targets: dict | None = None, raw_pose: bool = False) -> None:
        self.motion_targets = motion_targets
        self.index: list[tuple[str, int]] = []
        self.videos: dict[str, dict[str, np.ndarray]] = {}
        for entry in manifest["videos"]:
            if entry["split"] != split:
                continue
            identity = entry["video_id"]
            raw = videos[identity]
            data = dict(raw)
            data["pose"] = standardize(raw["pose"], stats, "pose", raw["evidence"])
            data["visual"] = standardize(raw["visual"], stats, "visual", raw["evidence"])
            for modality in ("pose", "visual"):
                variant = modality + "_corrupt"
                if variant in raw:
                    data[variant] = standardize(raw[variant], stats, modality, raw["evidence_" + variant])
            if raw_pose:
                # O TCNTE normaliza por quadro (min–max); estatísticas de treino não se aplicam.
                data["pose"] = raw["pose"].astype(np.float32)
            self.videos[identity] = data
            self.index.extend((identity, k) for k in range(0, entry["k"], stride)
                              if not labeled_only or data["labels"][k] != -1)
        if not self.index:
            raise ValueError(f"nenhuma janela usável em {split}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        identity, end = self.index[index]
        data = self.videos[identity]
        indices = window_frame_indices(end, len(data["labels"]))
        result = {name: torch.from_numpy(value[indices].copy()) for name, value in data.items() if name != "labels"}
        result["label"] = torch.tensor(int(data["labels"][end]), dtype=torch.long)
        if self.motion_targets is not None:
            for name, value in self.motion_targets[identity].items():
                result["motion_" + name] = torch.from_numpy(value[end].copy())
        return result


def checkpoint_is_better(score: float, best: float, policy: str) -> bool:
    if policy not in ("last", "best_val") or not math.isfinite(score):
        raise ValueError("política ou escore de checkpoint inválido")
    return policy == "last" or score > best


@torch.no_grad()
def predict(model: nn.Module, source: Windows, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true: list[np.ndarray] = []; predictions: list[np.ndarray] = []
    for batch in DataLoader(source, batch_size=batch_size, shuffle=False):
        logits = model(batch["pose"].to(device), batch["visual"].to(device), batch["evidence"].to(device))
        true.append(batch["label"].numpy()); predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(true), np.concatenate(predictions)


def classification(true: np.ndarray, predicted: np.ndarray, labels: tuple[str, ...]) -> dict:
    keep = true != -1
    summary = classification_summary(true[keep], predicted[keep], labels, NUM_CLASSES)
    summary["macro_f1_restricted"] = restricted_macro_f1(true[keep], predicted[keep])[0]
    return summary


def validate_recipe(arm: str, fusion: str, visual_kind: str, epochs: int, batch_size: int,
                    policy: str, augment: bool, order_weight: float, order_margin: float) -> None:
    if arm not in STUDY_ARMS or fusion not in STUDY_FUSIONS:
        raise ValueError("braço ou fusão desconhecidos")
    if (arm == TCNTE_ARM) != (fusion == TCNTE_FUSION):
        raise ValueError("TCNTE exige --fusion tcnte, e tcnte pertence só ao TCNTE")
    if arm == TCNTE_ARM and (augment or order_weight > 0):
        raise ValueError("TCNTE não usa falhas pareadas nem perda de ordenação")
    if arm == "A" and fusion != "pose":
        raise ValueError("A exige --fusion pose")
    if arm not in POSE_ONLY_ARMS and (visual_kind != ("dinov3" if arm.startswith("B") else "sam3") or fusion == "pose"):
        raise ValueError("braço incompatível com a fonte visual/fusão")
    if arm.endswith("0") and fusion != "concat":
        raise ValueError("B0/C0 exigem concat")
    if arm.endswith("1") and fusion == "concat":
        raise ValueError("concat pertence ao controle B0/C0")
    if epochs < 1 or batch_size < 1 or policy not in ("last", "best_val"):
        raise ValueError("épocas, batch ou política inválidos")
    if not math.isfinite(order_weight) or order_weight < 0 or not math.isfinite(order_margin) or not 0 < order_margin < 1:
        raise ValueError("peso deve ser >=0 e margem deve estar em (0,1)")
    if order_weight > 0 and (not augment or fusion not in TRAINABLE_GATES):
        raise ValueError("regularização exige --augment e gate treinável")


def train(
    bundle: Path, run_dir: Path, *, arm: str, fusion: str, seed: int = 42,
    epochs: int = 30, batch_size: int = 64, checkpoint_policy: str = "best_val",
    augment: bool = False, order_weight: float = 0, order_margin: float = .05,
    motion_recipe: str | None = None, targets: Path | None = None,
    motion_weight: float = .1, physics_weight: float = .1,
) -> dict:
    bundle = bundle.resolve(); run_dir = run_dir.resolve()
    validate_local_run_dir(run_dir)
    if run_dir.exists() or bundle in run_dir.parents:
        raise ValueError("run deve ser novo e externo ao pacote de entrada")
    manifest, videos = load_bundle(bundle)
    motion_config = None
    motion_targets = None
    if motion_recipe is not None:
        from gatefall.experiments.motion import RECIPES, SCHEMA, SPATIAL_SCALE
        from gatefall.experiments.motion.data import load_targets
        if (motion_recipe not in RECIPES or targets is None
                or any(not math.isfinite(w) or w <= 0 for w in (motion_weight, physics_weight))):
            raise ValueError("receita motion exige targets e pesos finitos positivos")
        meta, motion_targets, coverage = load_targets(targets, bundle, manifest, videos)
        motion_config = {
            "schema": SCHEMA, "recipe": motion_recipe, "horizons": meta["horizons"],
            "motion_weight": motion_weight, "physics_weight": physics_weight,
            "spatial_scale": SPATIAL_SCALE, "coverage": coverage,
            "targets_sha256": sha256_file(targets / "motion-manifest.json"),
            "supervision_on": "clean_train_only",
        }
    elif targets is not None:
        raise ValueError("targets exige motion_recipe")
    validate_local_run_dir(run_dir, manifest["dataset"])
    validate_recipe(arm, fusion, manifest["visual_kind"], epochs, batch_size,
                    checkpoint_policy, augment, order_weight, order_margin)
    stats = fit_stats(manifest, videos)
    raw_pose = fusion == TCNTE_FUSION
    train_data = Windows(manifest, videos, stats, "train", TRAIN_STRIDE, motion_targets=motion_targets,
                         raw_pose=raw_pose)
    val_data = Windows(manifest, videos, stats, "val", 1, raw_pose=raw_pose)
    device = configure_determinism(seed)
    horizons = None if motion_config is None else motion_config["horizons"]
    model = build_classifier(manifest["visual_dim"], fusion, horizons).to(device)
    if motion_config is not None:
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        motion_config["initial_state_sha256"] = digest.hexdigest()
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    counts = np.bincount([int(train_data.videos[v]["labels"][k]) for v, k in train_data.index], minlength=NUM_CLASSES)
    weights[counts > 0] = 1 / counts[counts > 0]
    optimization = optimization_recipe(fusion)
    criterion: nn.Module
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    if fusion == TCNTE_FUSION:
        criterion = FocalLoss(torch.tensor(weights, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=optimization["lr"])
        scheduler = None
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    config = {
        "schema_version": SCHEMA_VERSION, "feature_schema": FEATURE_SCHEMA,
        "dataset": manifest["dataset"], "visual_kind": manifest["visual_kind"],
        "visual_dim": manifest["visual_dim"], "arm": arm, "fusion": fusion, "seed": seed,
        "epochs": epochs, "batch_size": batch_size, "checkpoint_policy": checkpoint_policy,
        "augment": augment, "order_weight": order_weight, "order_margin": order_margin,
        "lr": optimization["lr"], "weight_decay": optimization["weight_decay"], "window_frames": 24, "train_stride": TRAIN_STRIDE,
        "eval_stride": 1, "channels": [32, 32, 32], "dilations": [1, 2, 4], "dropout": .3,
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "code_provenance": code_provenance(),
        "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if fusion == TCNTE_FUSION:
        config.update(optimization=optimization, input="pose138_kp_xy_body12_minmax_per_frame",
                      channels=[8, 16, 32], dilations=[1, 2, 4], dropout=.2, encoder="1x(d32,h4,ff32)")
    if motion_config is not None:
        config["motion"] = motion_config
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".physics-train-", dir=run_dir.parent) as temporary:
        stage = Path(temporary) / "run"; stage.mkdir()
        write_json(stage / "config.json", config); write_json(stage / "stats.json", stats)
        history = []; best = -math.inf; selected_epoch = 0
        started = time.perf_counter(); epoch_seconds: list[float] = []
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train(); total_loss = total_order = 0.0; seen = 0
            total_classification = total_motion = total_physics = 0.0
            for batch_cpu in loader:
                b = {name: value.to(device) for name, value in batch_cpu.items()}
                optimizer.zero_grad()
                if motion_config is not None and motion_recipe != "baseline":
                    from gatefall.experiments.motion.learning import MotionClassifier, motion_losses
                    assert isinstance(model, MotionClassifier)
                    clean_logits, position, velocity = model.forward_with_motion(b["pose"], b["visual"], b["evidence"])
                    supervised, consistency = motion_losses(
                        position, velocity, b,
                        torch.tensor(motion_config["horizons"], device=device) / TARGET_FPS)
                else:
                    clean_logits = model(b["pose"], b["visual"], b["evidence"])
                    supervised = consistency = clean_logits.sum() * 0
                loss = criterion(clean_logits, b["label"])
                order = clean_logits.sum() * 0
                if augment:
                    lp = criterion(model(b["pose_corrupt"], b["visual"], b["evidence_pose_corrupt"]), b["label"])
                    if fusion == "pose":
                        loss = .5 * (loss + lp)
                    else:
                        lv = criterion(model(b["pose"], b["visual_corrupt"], b["evidence_visual_corrupt"]), b["label"])
                        loss = .5 * loss + .25 * (lp + lv)
                    if order_weight > 0:
                        assert isinstance(model, FusionClassifier) and model.gate is not None
                        order = ordering_loss(
                            model.gate(b["evidence"]), model.gate(b["evidence_pose_corrupt"]),
                            model.gate(b["evidence_visual_corrupt"]),
                            b["pose_changed"] & (b["evidence"][..., 7] > 0),
                            b["visual_changed"] & (b["evidence"][..., 6] > 0), order_margin,
                        )
                classification_loss = loss
                loss = loss + order_weight * order
                if motion_recipe in ("motion", "motion_physics"):
                    loss = loss + motion_weight * supervised
                if motion_recipe == "motion_physics":
                    loss = loss + physics_weight * consistency
                if not torch.isfinite(loss):
                    raise RuntimeError("perda não finita")
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                count = len(b["label"]); seen += count
                total_loss += loss.item() * count; total_order += order.item() * count
                total_classification += classification_loss.item() * count
                total_motion += supervised.item() * count
                total_physics += consistency.item() * count
            if scheduler is not None:
                scheduler.step()
            true, pred = predict(model, val_data, device, batch_size)
            score = restricted_macro_f1(true, pred)[0]
            if checkpoint_is_better(score, best, checkpoint_policy):
                best = score; selected_epoch = epoch
                torch.save(model.state_dict(), stage / "checkpoint.pt")
            history.append({"epoch": epoch, "train_loss": total_loss / seen,
                            "order_loss": total_order / seen, "val_macro_f1_restricted": score})
            if motion_config is not None:
                history[-1].update(classification_loss=total_classification / seen,
                                   motion_loss=total_motion / seen, consistency_loss=total_physics / seen)
            epoch_seconds.append(time.perf_counter() - epoch_started)
            print(f"{arm}/{fusion} epoch {epoch}/{epochs}: val_macro_f1={score:.4f}")
        train_seconds = time.perf_counter() - started
        torch.save(model.state_dict(), stage / "last.pt")
        model.load_state_dict(torch.load(stage / "checkpoint.pt", map_location=device, weights_only=True))
        final = {}
        for split, data in (("train", train_data), ("val", val_data)):
            true, pred = predict(model, data, device, batch_size)
            final[split] = classification(true, pred, tuple(manifest["label_names"]))
        metrics = {"history": history, "final": final, "selected_epoch": selected_epoch,
                   "checkpoint_policy": checkpoint_policy, "device": device,
                   "torch_version": torch.__version__, "numpy_version": np.__version__,
                   "config_sha256": sha256_file(stage / "config.json"),
                   "stats_sha256": sha256_file(stage / "stats.json"),
                   "checkpoint_sha256": sha256_file(stage / "checkpoint.pt"),
                   "last_checkpoint_sha256": sha256_file(stage / "last.pt"),
                   "timing": {"train_seconds": train_seconds, "epoch_seconds": epoch_seconds,
                              "peak_memory_bytes": peak_memory_bytes(device)}}
        write_json(stage / "metrics.json", metrics)
        if run_dir.exists():
            raise FileExistsError(run_dir)
        os.replace(stage, run_dir)
    return metrics


def load_verified_run(bundle: Path, run_dir: Path, equivalence: Path | None = None
                      ) -> tuple[dict, dict, dict, dict, dict, dict | None]:
    """Confere hashes do run e a origem das estatísticas (só treino deste pacote).

    Um run treinado sobre outro pacote só é aceito com `equivalence` apontando para um atestado
    que o liga ao pacote atual; o último item é o resumo desse atestado, ou None.
    """
    validate_local_run_dir(run_dir)
    manifest, videos = load_bundle(bundle)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION or config.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("run incompatível com o pacote/schema")
    attested = None
    if config["manifest_sha256"] != sha256_file(bundle / "manifest.json"):
        if equivalence is None:
            raise ValueError("run incompatível com o pacote/schema")
        from gatefall.experiments.motion.equivalence import attested_run
        attested = attested_run(config, equivalence, bundle)
    for name, field in (("config.json", "config_sha256"), ("stats.json", "stats_sha256"),
                        ("checkpoint.pt", "checkpoint_sha256"), ("last.pt", "last_checkpoint_sha256")):
        if sha256_file(run_dir / name) != metrics[field]:
            raise ValueError(f"hash incompatível: {name}")
    if stats != fit_stats(manifest, videos):
        raise ValueError("estatísticas não correspondem exclusivamente ao treino deste pacote")
    return manifest, videos, config, metrics, stats, attested


def _load_checkpoint(run_dir: Path, config: dict, visual_dim: int) -> nn.Module:
    horizons = config["motion"]["horizons"] if "motion" in config else None
    model = build_classifier(visual_dim, config["fusion"], horizons)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True))
    return model


def _evaluation_report(model: nn.Module, manifest: dict, videos: dict, stats: dict, config: dict,
                       metrics: dict, split: str) -> dict:
    data = Windows(manifest, videos, stats, split, 1, labeled_only=False,
                   raw_pose=config["fusion"] == TCNTE_FUSION)
    started = time.perf_counter()
    true, pred = predict(model, data, "cpu", config["batch_size"])
    classification_seconds = time.perf_counter() - started
    ids = [v for v, _ in data.index]; ends = [k for _, k in data.index]
    events = split_event_report(ids, ends, true.tolist(), pred.tolist(), BASELINE_A_ALARM_PROTOCOL,
                                len(data), len(data), int(np.sum(true != -1)), round_latency=False)
    return {"split": split, "classification": classification(true, pred, tuple(manifest["label_names"])),
            "events": events, "alarms": alarm_records(ids, ends, pred),
            "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
            "latency_precision": "full", "selection_performed": False,
            "code_provenance": code_provenance(),
            "predictions": [{"video_id": identity, "k_end": end, "label": int(label),
                             "prediction": int(prediction)}
                            for (identity, end), label, prediction in zip(data.index, true, pred)],
            "checkpoint_sha256": metrics["checkpoint_sha256"], "config_sha256": metrics["config_sha256"],
            "manifest_sha256": config["manifest_sha256"],
            "timing": {"device": "cpu", "windows": len(data), "classification_seconds": classification_seconds,
                       "seconds_per_window": classification_seconds / len(data),
                       "peak_memory_bytes": peak_memory_bytes("cpu")}}


def alarm_records(ids: list[str], ends: list[int], predictions: np.ndarray) -> list[dict]:
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(ids):
        grouped.setdefault(identity, []).append(index)
    alarms = []
    for identity, indices in grouped.items():
        order = sorted(indices, key=lambda index: ends[index])
        found = detect_alarms_for_video(identity, np.array([ends[i] for i in order], dtype=np.int64),
                                        predictions[order], BASELINE_A_ALARM_PROTOCOL)
        alarms += [{"video_id": a.video_id, "trigger_k": a.trigger_k, "trigger_time_s": a.trigger_time_s}
                   for a in found]
    return alarms


def _publish(output: Path, report: dict) -> None:
    # Escrita exclusiva: nunca substituir uma avaliação já publicada neste run.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def evaluate(bundle: Path, run_dir: Path, split: str = "test", equivalence: Path | None = None) -> dict:
    if split not in ("train", "val", "test"):
        raise ValueError("split desconhecido")
    validate_local_run_dir(run_dir)
    output = run_dir / f"evaluation-{split}.json"
    if output.exists():
        raise FileExistsError(f"avaliação já existe: {output}")
    manifest, videos, config, metrics, stats, attested = load_verified_run(bundle, run_dir, equivalence)
    model = _load_checkpoint(run_dir, config, manifest["visual_dim"])
    report = _evaluation_report(model, manifest, videos, stats, config, metrics, split)
    if attested is not None:
        report["equivalence"] = attested
    _publish(output, report)
    return report


def evaluate_external(bundle: Path, run_dir: Path, target_bundle: Path) -> dict:
    """Aplica checkpoint e estatísticas congelados do treino de origem ao teste externo, sem adaptação."""
    source_manifest, _, config, metrics, stats, _ = load_verified_run(bundle, run_dir)
    target_manifest, target_videos = load_bundle(target_bundle)
    divergent = [key for key in EXTERNAL_COMPATIBILITY_KEYS
                 if source_manifest.get(key) != target_manifest.get(key)]
    if divergent:
        raise ValueError(f"pacote externo incompatível com o run: {divergent}")
    if target_manifest["dataset"] == source_manifest["dataset"]:
        raise ValueError("avaliação externa exige dataset diferente do treino")
    if set(target_manifest.get("splits", ())) != {"test"}:
        raise ValueError("pacote externo deve conter somente o split test")
    output = run_dir / f"evaluation-external-{target_manifest['dataset']}.json"
    if output.exists():
        raise FileExistsError(f"avaliação externa já existe: {output}")
    model = _load_checkpoint(run_dir, config, source_manifest["visual_dim"])
    report = _evaluation_report(model, target_manifest, target_videos, stats, config, metrics, "test")
    report.update(adaptation="none", source_dataset=source_manifest["dataset"],
                  target_dataset=target_manifest["dataset"],
                  target_manifest_sha256=sha256_file(target_bundle.resolve() / "manifest.json"))
    _publish(output, report)
    return report
