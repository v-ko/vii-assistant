"""Projectors: map store state → view state.

Includes:
- Screen layout projector (system screens → ScreenInfoVS)
- Config projector (AppConfig → AssistantSettingsViewState)
- Context delta projector (context store changes → context view state)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sivkit.libs.action import action
from sivkit.storage.change import Change
from sivkit.storage.delta import Delta

from assistant.inference.context import ContextItem
from assistant.inference.context_store import OP_SEP
from assistant.model.app_config import ViiConfig
from assistant.snippet_actions import hide_snippet_overlays
from assistant.view_states.screen_info import ScreenInfoData, ScreenInfoVS

if TYPE_CHECKING:
    from assistant.app_state import AppViewState
    from assistant.inference.context import ContextManager

log = logging.getLogger(__name__)


@action("screen.project_layout", issuer="service")
def project_screen_layout(
    screen_data: list[ScreenInfoData], app_state: AppViewState
) -> None:
    """Update app_state.screens from compiled screen data.

    Diffs against existing list: updates in place where possible,
    adds/removes as needed. Emits screens_changed when done.

    Also cancels stale snippet overlays when screen list changes.
    """
    existing = app_state.screens
    existing_by_name = {s.name: s for s in existing}
    incoming_names = {d.name for d in screen_data}

    new_list: list[ScreenInfoVS] = []
    changed = False

    for data in screen_data:
        if data.name in existing_by_name:
            vs = existing_by_name[data.name]
            if vs.update_from(data):
                changed = True
            new_list.append(vs)
        else:
            # New screen appeared
            vs = ScreenInfoVS(data, parent=app_state)
            new_list.append(vs)
            changed = True

    # Detect removed screens
    if set(existing_by_name.keys()) != incoming_names:
        changed = True

    if not changed and len(new_list) == len(existing):
        return

    app_state.screens = new_list

    # Cancel active snippet overlays (screenshots are stale)
    if app_state.snippet_overlays:
        log.info("Screen layout changed — cancelling active snippet overlays")
        hide_snippet_overlays(app_state)


# ── Config projector ─────────────────────────────────────────────────────────


def project_config(cfg: ViiConfig, settings_vs, transcription_vs) -> None:
    """Project AppConfig changes onto view states.

    Called via closure from the config store's on_changes_callback.
    """
    settings_vs._set_selected_model(cfg.selected_model)
    settings_vs._set_max_new_tokens(cfg.max_new_tokens)
    settings_vs._set_capture_screen(cfg.capture_screen)

    # Transcription settings
    transcription_vs.selected_input_device = cfg.transcription.get("input_device", "")


# ── Context store → view state projector ─────────────────────────────────────


def project_context_delta_to_view(
    delta: Delta, origin: str | None, ctx_mgr: ContextManager, context_vs
) -> None:
    """Project context store changes onto the context view state."""
    for key, change_data in delta.asdict().items():
        if OP_SEP in key:
            entity_id = key.split(OP_SEP, 1)[0]
            entity = ctx_mgr._store.find_one(id=entity_id)
            if entity and isinstance(entity, ContextItem):
                context_vs.apply_entity(entity)
        else:
            eid, reverse, forward = change_data
            change = Change(eid, reverse, forward)
            if change.is_delete():
                context_vs.remove_entity(eid)
            else:
                entity = ctx_mgr._store.find_one(id=eid)
                if entity and isinstance(entity, ContextItem):
                    context_vs.apply_entity(entity)
