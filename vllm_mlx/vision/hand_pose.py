# SPDX-License-Identifier: Apache-2.0
"""Hand pose estimation adapter backed by wilor-mlx (optional).

wilor_mlx is imported lazily at load() time, not at module import time,
so this module can be imported even when the dependency is absent.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(
    r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE
)


def _decode_image_to_numpy(image_data: str):
    """Decode a base64 data URI or raw base64 string to a numpy RGB array."""
    import numpy as np
    from PIL import Image

    raw_b64 = _DATA_URI_RE.sub("", image_data)
    img_bytes = base64.b64decode(raw_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


@dataclass
class HandPoseDetection:
    """A detected hand with pose estimation results."""

    hand_side: str  # "left" or "right"
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in pixel coords
    keypoints_2d: list[list[float]]
    keypoints_3d: list[list[float]] | None = None
    vertices: list[list[float]] | None = None


class HandPoseEngine:
    """Lazy-loading wrapper around the wilor-mlx hand pose pipeline.

    Uses HandPosePipeline for end-to-end detection + pose estimation:
    full image in, detected hands with 3D pose out.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._pipeline = None
        self.backend_name = "wilor-mlx"
        self.model_name = "wilor/hand-pose"

    def load(self) -> None:
        """Load the WiLoR-MLX pipeline. Raises ImportError if wilor_mlx absent."""
        from wilor_mlx import HandPosePipeline

        self._pipeline = HandPosePipeline.from_pretrained()
        self._loaded = True
        logger.info("HandPoseEngine loaded (backend=%s)", self.backend_name)

    def predict(
        self,
        image_data: str,
        *,
        include_3d: bool = False,
        include_vertices: bool = False,
    ) -> list[HandPoseDetection]:
        """Run hand detection + pose estimation on a full image.

        Args:
            image_data: base64 data URI or raw base64 string of any image.
            include_3d: Include 3D keypoints from WiLoR.
            include_vertices: Include MANO mesh vertices.

        Returns:
            List of HandPoseDetection, one per detected hand.
        """
        if not self._loaded:
            self.load()

        image_np = _decode_image_to_numpy(image_data)

        hands = self._pipeline(
            image_np,
            include_3d=include_3d,
            include_vertices=include_vertices,
        )

        return [
            HandPoseDetection(
                hand_side=h.hand_side,
                confidence=h.confidence,
                bbox=h.bbox,
                keypoints_2d=h.keypoints_2d,
                keypoints_3d=h.keypoints_3d,
                vertices=h.vertices,
            )
            for h in hands
        ]
