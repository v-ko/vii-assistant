"""Actions for experiment state mutations.

All store and view-state changes happen through @action so they fire
as batched deltas and follow the sivkit architecture.
"""

from __future__ import annotations

import logging
from base64 import b64encode
from io import BytesIO

from PIL import Image
from PySide6.QtGui import QImage
from sivkit.libs.action import action

from assistant.components.vision_overlay.view_state import DisplayTransform, OverlayMode
from assistant.facade import vii
from assistant.inference.context import ImageMessage, TextMessage
from assistant.model.experiment_config import ExperimentConfig
from assistant.model_configs import get_resolution_for_model
from assistant.utils.image_ops import ResizeMetadata, resize_to_target
from assistant.utils.misc import Shape

log = logging.getLogger(__name__)


@action("experiment.update_overlay")
def update_experiment_overlay(
    *,
    sample_image: QImage,
    display_transform: DisplayTransform,
    gt_shapes: list[Shape],
    resize_meta: ResizeMetadata,
) -> None:
    """Batch-update all overlay properties for the current experiment step."""
    overlay_vs = vii.app.view_state.overlay_VS
    overlay_vs.shapes = []
    overlay_vs.dimmed = False
    overlay_vs.sample_image = sample_image
    overlay_vs.display_transform = display_transform
    overlay_vs.gt_shapes = gt_shapes
    overlay_vs.resize_meta = resize_meta


@action("experiment.submit_context")
def submit_experiment_context(
    *,
    image: Image.Image,
    prompt: str,
    config: ExperimentConfig,
) -> None:
    """Clear context and insert system prompts + prompt + image.

    The inference request item is inserted separately by the hybrid service's
    ``run_chain`` so the agent chain owns its own lifecycle.
    """
    ctx = vii.project_manager.context_manager
    ctx.clear()

    # Insert system prompts for ALL focus modes that have prompt files
    from assistant.inference.focus_modes import FOCUS_MODES, PERCEPTION_MODES

    for mode_name, mode_config in FOCUS_MODES.items():
        if mode_config.has_prompt_file:
            try:
                prompt_text = mode_config.load_system_prompt(vii.active_agent)
            except (FileNotFoundError, TypeError) as e:
                if mode_name == config.focus_mode:
                    raise RuntimeError(
                        f"System prompt for target focus mode '{mode_name}' "
                        f"not found (agent='{vii.active_agent}'): {e}"
                    ) from e
                log.debug("Skipping system prompt for mode '%s': %s", mode_name, e)
                continue
            if prompt_text.strip():
                system_item = TextMessage()
                system_item.position = ctx.next_position()
                system_item.text = prompt_text.strip()
                system_item.origin = "system"
                system_item.metadata = {"focus_mode": mode_name}
                ctx.insert(system_item)

    # Resolve target resolution (model default or experiment override)
    res_override = None
    if config.resolution:
        res_override = tuple(config.resolution)

    model_key = vii.get_config().selected_model
    target_w, target_h = get_resolution_for_model(model_key, override=res_override)
    processed_img, meta = resize_to_target(image, target_w, target_h)

    # Store resize metadata so model output coords can be reverse-mapped
    vii.app.view_state.overlay_VS.resize_meta = meta

    buf = BytesIO()
    processed_img.save(buf, format="PNG")
    encoded = b64encode(buf.getvalue()).decode("ascii")

    # Add prompt BEFORE image (per U-Ground finding)
    text_item = TextMessage()
    text_item.position = ctx.next_position()
    text_item.text = prompt
    text_item.origin = "user"
    text_item.metadata = {"focus_mode": config.focus_mode}
    ctx.insert(text_item)

    # Add image
    img_item = ImageMessage()
    img_item.position = ctx.next_position()
    img_item.image_b64 = encoded
    img_item.width = meta["width"]
    img_item.height = meta["height"]
    img_item.size = meta["width"] * meta["height"]
    img_item.origin = "experiment"
    img_item.metadata = {"visible_to": list(PERCEPTION_MODES)}
    ctx.insert(img_item)


@action("experiment.clear_context")
def clear_experiment_context() -> None:
    """Clear all context items."""
    vii.project_manager.context_manager.clear()


@action("experiment.set_mode")
def set_experiment_mode() -> None:
    """Set overlay to experiment mode."""
    vii.app.view_state.overlay_VS.mode = OverlayMode.EXPERIMENT


@action("experiment.mark_finished")
def mark_experiment_finished() -> None:
    """Dim the overlay to indicate experiment is done."""
    vii.app.view_state.overlay_VS.dimmed = True
