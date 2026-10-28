"""Cinemática v2: derivadas entre observações, sem diferenças contra ausência."""

import math

import numpy as np

from gatefall.config import TARGET_FPS
from gatefall.pose.kinematics import build_pose_features_from_arrays, feature_blocks
from gatefall.pose.loading import PoseArrays

POSE_DIM = 138


def observed_derivatives(
    values: np.ndarray, observed: np.ndarray, times: np.ndarray, *, max_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if x.ndim != 2 or observed.shape != (len(x),) or times.shape != (len(x),):
        raise ValueError("valores [K,D], presença [K] e tempos [K] devem estar alinhados")
    if (not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(x[observed]).all()):
        raise ValueError("tempos crescentes, valores observados finitos e gap positivo são obrigatórios")
    velocity = np.zeros_like(x)
    acceleration = np.zeros_like(x)
    velocity_valid = np.zeros(len(x), dtype=bool)
    acceleration_valid = np.zeros(len(x), dtype=bool)
    previous: int | None = None
    previous_velocity: np.ndarray | None = None
    previous_midpoint = 0.0
    for i in np.flatnonzero(observed):
        if previous is not None:
            dt = times[i] - times[previous]
            if dt <= max_gap_seconds + 1e-9:
                velocity[i] = (x[i] - x[previous]) / dt
                velocity_valid[i] = True
                midpoint = (times[i] + times[previous]) / 2
                if previous_velocity is not None:
                    acceleration[i] = (velocity[i] - previous_velocity) / (midpoint - previous_midpoint)
                    acceleration_valid[i] = True
                previous_velocity = velocity[i].copy()
                previous_midpoint = midpoint
            else:
                previous_velocity = None
        previous = int(i)
    return (velocity.astype(np.float32), acceleration.astype(np.float32),
            velocity_valid, acceleration_valid)


def corrected_pose_features(pose: PoseArrays, max_gap_seconds: float = .5) -> np.ndarray:
    if pose.k < 1 or pose.keypoints.shape != (pose.k, 17, 3) or pose.bbox.shape != (pose.k, 4):
        raise ValueError("pose deve conter K>0, keypoints [K,17,3] e bbox [K,4]")
    found = np.asarray(pose.person_found, dtype=bool)
    if found.shape != (pose.k,) or pose.width <= 0 or pose.height <= 0:
        raise ValueError("presença ou dimensões de pose inválidas")
    if (not np.isfinite(pose.keypoints).all() or not np.isfinite(pose.bbox).all()
            or np.any(pose.bbox[found, 2:] <= pose.bbox[found, :2])):
        raise ValueError("pose contém valores não finitos ou bbox observada degenerada")
    matrix, _ = build_pose_features_from_arrays(pose)
    blocks = {name: slice(start, end) for name, start, end in feature_blocks()}
    times = np.arange(pose.k, dtype=np.float64) / TARGET_FPS
    vv = av = np.zeros(pose.k, dtype=bool)
    for position, velocity, acceleration in (
        ("kp_xy", "kp_velocity", "kp_acceleration"),
        ("bbox_pos", "bbox_velocity", "bbox_acceleration"),
    ):
        v, a, vv, av = observed_derivatives(
            matrix[:, blocks[position]], found, times, max_gap_seconds=max_gap_seconds,
        )
        matrix[:, blocks[velocity]] = v
        matrix[:, blocks[acceleration]] = a
    trunk = matrix[:, blocks["trunk"]]
    theta = np.zeros((pose.k, 1), dtype=np.float64)
    theta[found, 0] = np.unwrap(np.arctan2(trunk[found, 0], trunk[found, 1]))
    angular_velocity, _, _, _ = observed_derivatives(
        theta, found, times, max_gap_seconds=max_gap_seconds,
    )
    trunk[:, 2] = angular_velocity[:, 0]
    age = np.ones(pose.k, dtype=np.float32)
    previous: int | None = None
    for i in range(pose.k):
        if found[i]:
            previous = i
        if previous is not None:
            age[i] = min(1.0, (times[i] - times[previous]) / max_gap_seconds)
    return np.column_stack((matrix, found, vv, av, age)).astype(np.float32)
