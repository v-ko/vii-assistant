from types import SimpleNamespace

from assistant import facade as real_facade
from assistant.image_ops import scale_qwen_bbox_xyxy
from assistant.inference.context import ContextManager, ImageMessage, TextMessage
from assistant.services.hybrid_segment_service import HybridSegmentService


class _OverlayStub:
    """Minimal stand-in for overlay_VS used by update_overlay_from_text."""

    def __init__(self):
        self.shapes = []
        self.mode = None
        self.resize_meta = None
        self.display_transform = None


def test_scale_qwen_bbox_xyxy_basic():
    sx, sy, ex, ey = scale_qwen_bbox_xyxy(
        (0, 0, 1000, 1000), input_w=500, input_h=300, orig_w=1000, orig_h=600
    )
    assert (sx, sy, ex, ey) == (0, 0, 1000, 600)


def test_hybrid_segment_uses_image_size_metadata(monkeypatch):
    ctx_mgr = ContextManager()
    img_item = ImageMessage()
    img_item.position = 0
    img_item.size = 0
    img_item.width = 500
    img_item.height = 300
    ctx_mgr.insert(img_item)

    text_item = TextMessage()
    text_item.position = 100
    text_item.size = 0
    text_item.text = "output.bbox = bbox(0,0,1000,1000)"
    text_item.request = {"model": "qwen"}
    ctx_mgr.insert(text_item)

    overlay_stub = _OverlayStub()
    app_stub = SimpleNamespace(
        view_state=SimpleNamespace(
            overlay_VS=overlay_stub,
            capture_screen_info=None,
        )
    )
    pm_stub = SimpleNamespace(context_manager=ctx_mgr)
    monkeypatch.setattr(real_facade.vii, "_project_manager", pm_stub, raising=False)
    monkeypatch.setattr(real_facade.vii, "_app", app_stub, raising=False)

    svc = HybridSegmentService()
    # Pin the output resolution so the test is deterministic and headless-safe
    # (the real path reads QGuiApplication.primaryScreen()). The model-input
    # resolution still comes from the ImageMessage width/height above.
    monkeypatch.setattr(svc, "_resolve_screen_geometry", lambda: (0, 0, 1000, 600))

    svc.update_overlay_from_text("output.bbox = bbox(0,0,1000,1000)")
    shapes = overlay_stub.shapes
    assert shapes, "Expected shapes to be set"
    assert shapes[0]["type"] == "rect"
    assert "color" in shapes[0]
