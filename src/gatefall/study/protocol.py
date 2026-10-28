"""Contrato experimental congelado do estudo; qualquer mudança altera o hash."""

import hashlib
import json

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, NUM_CLASSES, TARGET_FPS, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.le2i.annotations import OMNIFALL_REVISION
from gatefall.data.staged.specs import GMDCSA24_SPLIT, UPFALL_SPLIT
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.experiments.physics import EVIDENCE_NAMES, FEATURE_SCHEMA, SCHEMA_VERSION
from gatefall.experiments.tcnte import SCHEMA as TCNTE_SCHEMA
from gatefall.study import STUDY_ID, STUDY_SCHEMA
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 30
BATCH_SIZE = 64
CHECKPOINT_POLICY = "best_val"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260
CONFIDENCE_LEVEL = .95

FROZEN_PROTOCOL: dict = {
    "study": STUDY_ID,
    "schema": STUDY_SCHEMA,
    "input": {
        "modality": "monocular_rgb",
        "target_fps": TARGET_FPS,
        "window_frames": WINDOW_FRAMES,
        "train_stride": TRAIN_STRIDE,
        "eval_stride": EVAL_STRIDE,
        "causal_windows": True,
        "depth_descriptors": False,
    },
    "labels": {
        "taxonomy": "omnifall",
        "omnifall_revision": OMNIFALL_REVISION,
        "names": list(LE2I_LABEL_NAMES),
        "num_classes": NUM_CLASSES,
        "ignore_label": IGNORE_LABEL,
        "omnifall_extended_classes": "ignored",
        "restricted_macro_f1_classes": list(RESTRICTED_CLASSES),
    },
    "backbones": {"pose": "yolo-pose", "dinov3": "dinov3_vitb16", "sam3": "facebook/sam3",
                  "trainable": False, "features": "offline_per_video_hdf5"},
    "features": {"schema": FEATURE_SCHEMA, "schema_version": SCHEMA_VERSION,
                 "evidence": list(EVIDENCE_NAMES), "normalization": "train_split_only",
                 "tcnte_input": TCNTE_SCHEMA},
    "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "checkpoint_policy": CHECKPOINT_POLICY,
                 "augment": False, "order_weight": 0.0, "selection_split": "val",
                 "test_consulted_during_training": False},
    "seeds": list(SEEDS),
    "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
    "splits": {"upfall": UPFALL_SPLIT.to_dict(), "gmdcsa24": GMDCSA24_SPLIT.to_dict(),
               "le2i": "omnifall-le2i-cs", "le2i-cv": "omnifall-le2i-cv"},
    "transfer": {"source": "upfall", "target": "gmdcsa24", "adaptation": "none",
                 "normalization": "source_train_stats"},
    "statistics": {"bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                   "confidence_level": CONFIDENCE_LEVEL, "cluster_unit": "subject_or_recording",
                   "seed_pairing": "same_seed_same_bundle"},
}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def protocol_sha256(protocol: dict = FROZEN_PROTOCOL) -> str:
    return hashlib.sha256(canonical_json(protocol)).hexdigest()
