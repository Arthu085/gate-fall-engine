"""Proxies causais de consistência; não são probabilidades nem dinâmica 3D."""

import numpy as np

from gatefall.config import TARGET_FPS
from gatefall.pose.loading import PoseArrays
from gatefall.pose.quality import pose_quality_from_arrays


def _similarity_error(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    keep = np.ones(len(source), dtype=bool)
    residual = np.zeros(len(source))
    for _ in range(2):
        left_center = source[keep].mean(axis=0)
        right_center = target[keep].mean(axis=0)
        left = source - left_center
        right = target - right_center
        denominator = np.square(left[keep]).sum()
        if denominator < 1e-10:
            raise ValueError("geometria degenerada")
        u, singular, vt = np.linalg.svd(left[keep].T @ right[keep])
        sign = np.array([1.0, np.linalg.det(u @ vt)])
        rotation = u @ np.diag(sign) @ vt
        scale = float(np.sum(singular * sign) / denominator)
        residual = np.linalg.norm(scale * left @ rotation + right_center - target, axis=1)
        keep = residual <= np.quantile(residual, .8)
        if np.count_nonzero(keep) < 4:
            break
    return residual


def pose_residual(pose: PoseArrays, max_gap_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros(pose.k, dtype=np.float32)
    valid = np.zeros(pose.k, dtype=bool)
    previous: int | None = None
    for i in range(pose.k):
        if not pose.person_found[i]:
            continue
        if previous is not None and (i - previous) / TARGET_FPS <= max_gap_seconds + 1e-9:
            common = (pose.keypoints[i, :, 2] >= .5) & (pose.keypoints[previous, :, 2] >= .5)
            scale = float(np.linalg.norm(pose.bbox[i, 2:] - pose.bbox[i, :2]))
            if np.count_nonzero(common) >= 6 and scale > 1e-6:
                try:
                    errors = _similarity_error(
                        pose.keypoints[previous, common, :2].astype(np.float64),
                        pose.keypoints[i, common, :2].astype(np.float64),
                    )
                except (ValueError, np.linalg.LinAlgError):
                    previous = i
                    continue
                # Média dos três maiores erros preserva evidência de articulação isolada.
                error = float(np.sort(errors)[-3:].mean() / scale)
                residual[i] = error / (1 + error)
                valid[i] = True
        previous = i
    return residual, valid


def sam3_residual(visual: np.ndarray, max_gap_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros(len(visual), dtype=np.float32)
    valid = np.zeros(len(visual), dtype=bool)
    previous: int | None = None
    for i in range(len(visual)):
        if visual[i, 0] == 0:
            continue
        if previous is not None and (i - previous) / TARGET_FPS <= max_gap_seconds + 1e-9:
            before = visual[previous, [1, 6]].astype(np.float64)
            now = visual[i, [1, 6]].astype(np.float64)
            if np.all(before > 0) and np.all(now > 0):
                error = float(np.abs(np.log(now / before)).mean() / (i - previous))
                residual[i] = error / (1 + error)
                valid[i] = True
        previous = i
    return residual, valid


def evidence_for_video(
    pose: PoseArrays, visual: np.ndarray, visual_quality: np.ndarray,
    visual_present: np.ndarray, visual_kind: str, max_gap_seconds: float,
) -> np.ndarray:
    qp = pose_quality_from_arrays(pose.keypoints, pose.bbox, pose.person_found).q_pose
    rp, rp_valid = pose_residual(pose, max_gap_seconds)
    rv, rv_valid = (sam3_residual(visual, max_gap_seconds) if visual_kind == "sam3"
                    else (np.zeros(pose.k), np.zeros(pose.k, dtype=bool)))
    evidence = np.column_stack((qp, visual_quality, rp, rv, rp_valid, rv_valid,
                                pose.person_found, visual_present)).astype(np.float32)
    if evidence.shape != (pose.k, 8) or not np.isfinite(evidence).all():
        raise ValueError("evidência deve ser finita com shape [K,8]")
    if np.any(evidence < 0) or np.any(evidence > 1):
        raise ValueError("evidência fora de [0,1]")
    return evidence
