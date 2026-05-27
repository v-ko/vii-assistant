from assistant import facade as real_facade
from assistant.image_ops import scale_qwen_bbox_xyxy
from assistant.inference.context import ContextItem, ContextManager
from assistant.services.hybrid_segment_service import HybridSegmentService


class _OverlayStub:
    def __init__(self):
        self.last_shapes = []

    def set_shapes(self, shapes):
        self.last_shapes = shapes


class _QtAppStub:
    def __init__(self):
        self.overlay = _OverlayStub()


def test_scale_qwen_bbox_xyxy_basic():
    sx, sy, ex, ey = scale_qwen_bbox_xyxy(
        (0, 0, 1000, 1000), input_w=500, input_h=300, orig_w=1000, orig_h=600
    )
    assert (sx, sy, ex, ey) == (0, 0, 1000, 600)


def test_hybrid_segment_uses_image_size_metadata(monkeypatch):
    ctx_mgr = ContextManager()
    img_item = ContextItem()
    img_item.position = 0
    img_item.size = 0
    img_item.content = {
        "image": (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBgWcJXssAAAAASUVORK5CYII="
        )
    }
    img_item.metadata = {"image_size": {"width": 500, "height": 300}}
    ctx_mgr.insert(img_item)

    text_item = ContextItem()
    text_item.position = 100
    text_item.size = 0
    text_item.content = {"text": "output.bbox = bbox(0,0,1000,1000)"}
    text_item.request = {"model": "qwen"}
    ctx_mgr.insert(text_item)

    monkeypatch.setattr(real_facade, "context_manager", ctx_mgr, raising=False)
    qt_stub = _QtAppStub()
    monkeypatch.setattr(real_facade, "qt_app", qt_stub, raising=False)

    svc = HybridSegmentService()
    svc.update_overlay_from_text("output.bbox = bbox(0,0,1000,1000)")
    shapes = qt_stub.overlay.last_shapes
    assert shapes, "Expected shapes to be set"
    assert shapes[0]["type"] == "rect"
    assert "color" in shapes[0]
