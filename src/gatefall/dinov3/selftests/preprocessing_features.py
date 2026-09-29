import numpy as np
import torch

from gatefall.dinov3.backbone import FEATURE_DIM, RESIZE_SIZE
from gatefall.dinov3.features import compute_features
from gatefall.dinov3.preprocessing import preprocess_frames
from gatefall.dinov3.selftests.fixtures import _check, _FakeBackbone


def _check_preprocess_frames_shape_and_dtype() -> bool:
    ok = True
    for height, width in [(64, 64), (100, 200), (300, 150)]:
        frames = [
            np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        out = preprocess_frames(frames)
        ok = ok and (
            out.shape == (3, 3, RESIZE_SIZE, RESIZE_SIZE)
            and out.dtype == torch.float32
        )
    return _check(
        "preprocess_frames: shape [K,3,224,224] float32 para H,W variados", ok
    )


def _check_compute_features_formula() -> bool:
    batch_size = 2
    cls_token = torch.arange(batch_size * 768, dtype=torch.float32).reshape(
        batch_size, 768
    )
    patch_tokens = torch.zeros((batch_size, 4, 768), dtype=torch.float32)
    patch_tokens[0] = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1).expand(4, 768)
    patch_tokens[1] = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(4, 1).expand(4, 768)
    expected_mean_patch = torch.tensor(
        [[2.5] * 768, [6.5] * 768], dtype=torch.float32
    )

    backbone = _FakeBackbone(cls_token, patch_tokens)
    features = compute_features(backbone, torch.zeros((batch_size, 3, 8, 8)))

    ok = (
        features.shape == (batch_size, FEATURE_DIM)
        and features.dtype == np.float16
        and np.allclose(
            features[:, :768], cls_token.numpy(), atol=1e-2
        )
        and np.allclose(
            features[:, 768:], expected_mean_patch.numpy(), atol=1e-2
        )
    )
    return _check(
        "compute_features: [B,1536] float16 com CLS nas primeiras 768 colunas "
        "e média dos patch tokens nas últimas 768",
        ok,
    )
