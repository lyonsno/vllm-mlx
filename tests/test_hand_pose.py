# SPDX-License-Identifier: Apache-2.0
"""
Tests for optional /v1/vision/hand_pose endpoint.

This endpoint is backed by wilor-mlx (optional dependency).
Tests verify:
1. Import isolation: missing wilor_mlx does not break server startup
2. Schema stability: response matches declared Pydantic models
3. Route isolation: hand_pose route does not pollute LLM/VLM/audio/embedding
4. Backend identity: response metadata names the effective backend, not a fallback
5. Output-heavy options: vertices/MANO are opt-in, not default
"""

import base64
import io
import platform
import sys
import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# ---------------------------------------------------------------------------
# 1. Pydantic model tests (schema stability)
# ---------------------------------------------------------------------------


class TestHandPoseModels:
    """Test hand pose request/response Pydantic models."""

    def test_request_model_minimal(self):
        """Minimal request requires only image."""
        from vllm_mlx.api.models import HandPoseRequest

        req = HandPoseRequest(image="data:image/jpeg;base64,/9j/4AAQ")
        assert req.image is not None
        assert req.include_3d is False
        assert req.include_vertices is False
        assert req.include_faces is False

    def test_request_model_with_options(self):
        """Request with explicit 3D and vertices flags."""
        from vllm_mlx.api.models import HandPoseRequest

        req = HandPoseRequest(
            image="data:image/jpeg;base64,/9j/4AAQ",
            include_3d=True,
            include_vertices=True,
        )
        assert req.include_3d is True
        assert req.include_vertices is True

    def test_response_model_shape(self):
        """Response has expected fields and types."""
        from vllm_mlx.api.models import HandPoseResponse, HandPoseResult

        result = HandPoseResult(
            hand_side="left",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.95,
            keypoints_2d=[[0.1, 0.2]] * 21,
        )
        resp = HandPoseResponse(
            hands=[result],
            backend="wilor-mlx",
            model="wilor/hand-pose",
        )
        assert len(resp.hands) == 1
        assert resp.hands[0].hand_side == "left"
        assert resp.hands[0].keypoints_3d is None
        assert resp.hands[0].vertices is None
        assert resp.faces is None
        assert resp.backend == "wilor-mlx"

    def test_response_model_with_faces(self):
        """Faces at response level contains triangle indices."""
        from vllm_mlx.api.models import HandPoseResponse, HandPoseResult

        result = HandPoseResult(
            hand_side="right",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.9,
            keypoints_2d=[[0.1, 0.2]] * 21,
        )
        resp = HandPoseResponse(
            hands=[result],
            faces=[[0, 1, 2], [2, 3, 0]],
            backend="wilor-mlx",
            model="wilor/hand-pose",
        )
        assert resp.faces is not None
        assert len(resp.faces) == 2
        assert all(len(f) == 3 for f in resp.faces)

    def test_response_model_with_3d(self):
        """Response with 3D keypoints populated."""
        from vllm_mlx.api.models import HandPoseResult

        result = HandPoseResult(
            hand_side="right",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.88,
            keypoints_2d=[[0.1, 0.2]] * 21,
            keypoints_3d=[[0.1, 0.2, 0.3]] * 21,
        )
        assert result.keypoints_3d is not None
        assert len(result.keypoints_3d) == 21

    def test_response_model_with_vertices(self):
        """Vertices are present only when explicitly populated."""
        from vllm_mlx.api.models import HandPoseResult

        result = HandPoseResult(
            hand_side="right",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.88,
            keypoints_2d=[[0.1, 0.2]] * 21,
            vertices=[[0.0, 0.0, 0.0]] * 778,
        )
        assert result.vertices is not None

    def test_request_rejects_missing_image(self):
        """Request without image field fails validation."""
        from vllm_mlx.api.models import HandPoseRequest

        with pytest.raises(ValidationError):
            HandPoseRequest()

    def test_request_rejects_oversized_image(self):
        """Image exceeding 30 MB base64 cap is rejected."""
        from vllm_mlx.api.models import HandPoseRequest

        oversized = "x" * (30 * 1024 * 1024 + 1)
        with pytest.raises(ValidationError, match="string_too_long"):
            HandPoseRequest(image=oversized)

    def test_result_rejects_invalid_hand_side(self):
        """hand_side must be 'left' or 'right'."""
        from vllm_mlx.api.models import HandPoseResult

        with pytest.raises(ValidationError, match="hand_side"):
            HandPoseResult(
                hand_side="middle",
                bbox=[10.0, 20.0, 100.0, 150.0],
                confidence=0.9,
                keypoints_2d=[[0.1, 0.2]] * 21,
            )

    def test_result_rejects_confidence_out_of_range(self):
        """confidence must be between 0.0 and 1.0."""
        from vllm_mlx.api.models import HandPoseResult

        with pytest.raises(ValidationError):
            HandPoseResult(
                hand_side="left",
                bbox=[10.0, 20.0, 100.0, 150.0],
                confidence=1.5,
                keypoints_2d=[[0.1, 0.2]] * 21,
            )

    def test_response_schema_always_includes_optional_keys(self):
        """model_dump() includes keypoints_3d and vertices as null, not missing."""
        from vllm_mlx.api.models import HandPoseResponse, HandPoseResult

        result = HandPoseResult(
            hand_side="left",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.9,
            keypoints_2d=[[0.1, 0.2]] * 21,
        )
        resp = HandPoseResponse(
            hands=[result], backend="wilor-mlx", model="wilor/hand-pose"
        )
        dumped = resp.model_dump()
        assert "keypoints_3d" in dumped["hands"][0]
        assert dumped["hands"][0]["keypoints_3d"] is None
        assert "vertices" in dumped["hands"][0]
        assert dumped["hands"][0]["vertices"] is None
        assert "faces" in dumped
        assert dumped["faces"] is None



# ---------------------------------------------------------------------------
# 2. Import isolation: missing wilor_mlx must not break server import
# ---------------------------------------------------------------------------


class TestImportIsolation:
    """Verify that the hand pose backend is lazy-loaded."""

    def test_server_imports_without_wilor(self):
        """Core server module loads without importing wilor_mlx at module level.

        The hand_pose route must be registered without wilor_mlx being
        imported by the server module itself. The server uses lazy import
        inside the endpoint handler, not at module scope.
        """
        import vllm_mlx.server as srv

        # Route must be registered regardless of wilor_mlx availability
        route_paths = [r.path for r in srv.app.routes if hasattr(r, "path")]
        assert "/v1/vision/hand_pose" in route_paths

        # The vision adapter module itself must not import wilor_mlx at module level
        import vllm_mlx.vision.hand_pose as hp_mod
        # wilor_mlx import only happens inside load(), not at module scope
        assert "wilor_mlx" not in dir(hp_mod)

    def test_hand_pose_adapter_import_errors_gracefully(self):
        """Importing the adapter module itself succeeds; calling run fails."""
        from vllm_mlx.vision.hand_pose import HandPoseEngine

        # Engine can be constructed (no wilor import at construction)
        engine = HandPoseEngine()
        assert engine._loaded is False


# ---------------------------------------------------------------------------
# 3. Endpoint integration via TestClient
# ---------------------------------------------------------------------------


class TestHandPoseEndpoint:
    """Integration tests using FastAPI TestClient with mocked backend."""

    @pytest.fixture
    def client(self):
        """TestClient with server app."""
        from vllm_mlx.server import app
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_missing_dependency_returns_503(self, client):
        """When wilor_mlx is not installed, endpoint returns 503."""
        with patch("vllm_mlx.server._hand_pose_engine", None), \
             patch.dict(sys.modules, {"wilor_mlx": None}):
            resp = client.post(
                "/v1/vision/hand_pose",
                json={"image": "data:image/jpeg;base64,/9j/4AAQ"},
            )
            assert resp.status_code == 503
            body = resp.json()
            assert "wilor" in body["detail"].lower()

    def _make_mock_engine(self, results, faces=None):
        """Create a mock HandPoseEngine with string attributes."""
        engine = MagicMock()
        engine.predict.return_value = results
        engine.get_faces.return_value = faces
        engine._loaded = True
        engine.backend_name = "wilor-mlx"
        engine.model_name = "wilor/hand-pose-v1"
        return engine

    def test_mocked_backend_returns_stable_schema(self, client):
        """Mocked WiLoR backend returns response matching schema."""
        mock_result = SimpleNamespace(
            hand_side="left",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.95,
            keypoints_2d=[[0.1, 0.2]] * 21,
            keypoints_3d=None,
            vertices=None,
        )
        mock_engine = self._make_mock_engine([mock_result])

        with patch("vllm_mlx.server._hand_pose_engine", mock_engine):
            resp = client.post(
                "/v1/vision/hand_pose",
                json={"image": "data:image/jpeg;base64,/9j/4AAQ"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "hands" in body
        assert "backend" in body
        assert "model" in body
        hand = body["hands"][0]
        assert hand["hand_side"] == "left"
        assert "keypoints_2d" in hand
        assert hand.get("keypoints_3d") is None
        assert hand.get("vertices") is None
        assert body.get("faces") is None

    def test_include_faces_flag_forwarded(self, client):
        """include_faces=True returns MANO triangle indices at response level."""
        mock_result = SimpleNamespace(
            hand_side="right",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.9,
            keypoints_2d=[[0.1, 0.2]] * 21,
            keypoints_3d=None,
            vertices=None,
        )
        mock_faces = [[0, 1, 2], [2, 3, 0], [4, 5, 6]]
        mock_engine = self._make_mock_engine([mock_result], faces=mock_faces)

        with patch("vllm_mlx.server._hand_pose_engine", mock_engine):
            resp = client.post(
                "/v1/vision/hand_pose",
                json={
                    "image": "data:image/jpeg;base64,/9j/4AAQ",
                    "include_faces": True,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["faces"] is not None
        assert len(body["faces"]) == 3
        assert body["faces"][0] == [0, 1, 2]
        mock_engine.get_faces.assert_called_once()

    def test_include_3d_flag_forwarded(self, client):
        """include_3d=True is forwarded to predict() and 3D keypoints appear."""
        mock_result = SimpleNamespace(
            hand_side="right",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.9,
            keypoints_2d=[[0.1, 0.2]] * 21,
            keypoints_3d=[[0.1, 0.2, 0.3]] * 21,
            vertices=None,
        )
        mock_engine = self._make_mock_engine([mock_result])

        with patch("vllm_mlx.server._hand_pose_engine", mock_engine):
            resp = client.post(
                "/v1/vision/hand_pose",
                json={
                    "image": "data:image/jpeg;base64,/9j/4AAQ",
                    "include_3d": True,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["hands"][0]["keypoints_3d"] is not None
        # Verify the flag was actually forwarded to the engine
        mock_engine.predict.assert_called_once_with(
            ANY, include_3d=True, include_vertices=False,
        )

    def test_vertices_not_in_default_response(self, client):
        """Vertices are absent from default (include_vertices=False) response."""
        mock_result = SimpleNamespace(
            hand_side="left",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.95,
            keypoints_2d=[[0.1, 0.2]] * 21,
            keypoints_3d=None,
            vertices=None,
        )
        mock_engine = self._make_mock_engine([mock_result])

        with patch("vllm_mlx.server._hand_pose_engine", mock_engine):
            resp = client.post(
                "/v1/vision/hand_pose",
                json={"image": "data:image/jpeg;base64,/9j/4AAQ"},
            )
        body = resp.json()
        assert body["hands"][0].get("vertices") is None

    def test_malformed_request_returns_422(self, client):
        """Missing required 'image' field returns 422, not 500."""
        resp = client.post("/v1/vision/hand_pose", json={})
        assert resp.status_code == 422

    def test_backend_identity_in_response(self, client):
        """Response metadata names the actual backend, not a generic fallback."""
        mock_result = SimpleNamespace(
            hand_side="left",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.95,
            keypoints_2d=[[0.1, 0.2]] * 21,
            keypoints_3d=None,
            vertices=None,
        )
        mock_engine = self._make_mock_engine([mock_result])

        with patch("vllm_mlx.server._hand_pose_engine", mock_engine):
            resp = client.post(
                "/v1/vision/hand_pose",
                json={"image": "data:image/jpeg;base64,/9j/4AAQ"},
            )
        body = resp.json()
        assert body["backend"] == "wilor-mlx"
        assert body["model"] == "wilor/hand-pose-v1"

    def test_corrupt_image_returns_500_not_crash(self, client):
        """Corrupt/truncated image data returns a server error, not a crash."""
        mock_engine = self._make_mock_engine([])
        mock_engine.predict.side_effect = Exception("Truncated File Read")

        with patch("vllm_mlx.server._hand_pose_engine", mock_engine):
            resp = client.post(
                "/v1/vision/hand_pose",
                json={"image": "data:image/jpeg;base64,/9j/4AAQ"},
            )
        assert resp.status_code == 500

    @pytest.mark.anyio
    async def test_predict_does_not_block_health_request(self):
        """Synchronous predict work must not serialize the ASGI event loop."""
        entered_predict = threading.Event()

        def blocking_predict(*args, **kwargs):
            entered_predict.set()
            time.sleep(0.4)
            return []

        from vllm_mlx.server import app

        mock_engine = self._make_mock_engine([])
        mock_engine.predict.side_effect = blocking_predict

        transport = httpx.ASGITransport(app=app)
        with patch("vllm_mlx.server._hand_pose_engine", mock_engine):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as async_client:
                start = time.monotonic()
                hand_task = asyncio.create_task(
                    async_client.post(
                        "/v1/vision/hand_pose",
                        json={"image": "data:image/jpeg;base64,/9j/4AAQ"},
                    )
                )
                assert await asyncio.to_thread(entered_predict.wait, 1.0)
                entered_elapsed = time.monotonic() - start
                health_resp = await async_client.get("/health")
                hand_resp = await hand_task

        assert hand_resp.status_code == 200
        assert health_resp.status_code == 200
        assert entered_elapsed < 0.2


# ---------------------------------------------------------------------------
# 4. Route isolation: hand_pose does not pollute existing routes
# ---------------------------------------------------------------------------


class TestRouteIsolation:
    """Verify hand_pose endpoint does not break or shadow existing routes."""

    @pytest.fixture
    def client(self):
        from vllm_mlx.server import app
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_health_still_works(self, client):
        """Health endpoint unaffected."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_models_still_works(self, client):
        """Models endpoint unaffected."""
        resp = client.get("/v1/models")
        # May be 200 or 500 depending on engine state, but not 404
        assert resp.status_code != 404

    def test_hand_pose_route_exists(self, client):
        """The hand_pose route is registered."""
        resp = client.post(
            "/v1/vision/hand_pose",
            json={"image": "data:image/jpeg;base64,/9j/4AAQ"},
        )
        # Should be 503 (no wilor) or 200 (mocked), never 404
        assert resp.status_code != 404


# ---------------------------------------------------------------------------
# 5. Integration test: real wilor_mlx inference (skipped when absent)
# ---------------------------------------------------------------------------

_wilor_available = False
try:
    import wilor_mlx  # noqa: F401

    _wilor_available = True
except ImportError:
    pass


@pytest.mark.skipif(not _wilor_available, reason="wilor_mlx not installed")
class TestRealInference:
    """Integration tests that run real WiLoR-MLX inference."""

    @pytest.fixture(scope="class")
    def engine(self):
        from vllm_mlx.vision.hand_pose import HandPoseEngine

        eng = HandPoseEngine()
        eng.load()
        return eng

    def _make_image_b64(self, img_np):
        """Encode a numpy RGB image as a base64 data URI."""
        from PIL import Image

        img = Image.fromarray(img_np, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def test_predict_returns_list(self, engine):
        """Real pipeline on a synthetic image returns a list (possibly empty)."""
        import numpy as np

        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        results = engine.predict(self._make_image_b64(arr))
        assert isinstance(results, list)
        # May be empty (no hand detected) or have detections
        for det in results:
            assert det.hand_side in ("left", "right")
            assert 0.0 <= det.confidence <= 1.0
            assert len(det.bbox) == 4
            assert len(det.keypoints_2d) == 21

    def test_predict_with_real_hand(self, engine):
        """Real hand image returns detection with 3D keypoints."""
        import os
        import numpy as np
        from PIL import Image

        hand_path = "/private/tmp/vllm-mlx-palm-daddy-handpose-overlay.jpg"
        if not os.path.exists(hand_path):
            pytest.skip("Real hand image not available")
        img_np = np.array(Image.open(hand_path).convert("RGB"))
        results = engine.predict(
            self._make_image_b64(img_np), include_3d=True, include_vertices=True,
        )
        assert len(results) >= 1
        det = results[0]
        assert det.hand_side in ("left", "right")
        assert det.keypoints_3d is not None
        assert len(det.keypoints_3d) == 21
        assert det.vertices is not None
        assert len(det.vertices) == 778

    def test_get_faces_returns_triangle_indices(self, engine):
        """get_faces() returns MANO mesh triangle indices."""
        faces = engine.get_faces()
        assert faces is not None
        assert len(faces) == 1538
        assert all(len(f) == 3 for f in faces)
        assert all(isinstance(f[0], int) for f in faces)
