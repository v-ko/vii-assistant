"""CorrectionViewModel — QML ViewModel for the supervised correction window.

Shows/hides based on overlay mode. Routes decisions to the SupervisedGate.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Property, QObject, Signal, Slot

from assistant.facade import vii
from assistant.view_states.overlay import OverlayMode, OverlayViewState

log = logging.getLogger(__name__)


class CorrectionViewModel(QObject):
    """Drives the CorrectionWindow.qml visibility and routes user actions."""

    visible_changed = Signal(bool)
    review_text_changed = Signal(str)

    def __init__(self, view_state, parent: QObject | None = None):
        super().__init__(parent)
        self._visible = False
        self._review_text = ""
        overlay_vs: OverlayViewState = view_state.overlay_VS
        overlay_vs.mode_changed.connect(self._on_mode_changed)
        overlay_vs.review_text_changed.connect(self._on_review_text_changed)

    def _on_mode_changed(self, mode: OverlayMode) -> None:
        new_visible = mode == OverlayMode.SUPERVISED_REVIEW
        if new_visible != self._visible:
            self._visible = new_visible
            self.visible_changed.emit(new_visible)

    def _on_review_text_changed(self, text: str) -> None:
        if text != self._review_text:
            self._review_text = text
            self.review_text_changed.emit(text)

    @Property(bool, notify=visible_changed)
    def visible(self) -> bool:
        return self._visible

    @Property(str, notify=review_text_changed)
    def reviewText(self) -> str:
        return self._review_text

    @Slot()
    def submitCorrect(self) -> None:
        gate = vii.project_manager.hybrid_segment_service.supervised_gate
        gate.submit_correct()

    @Slot()
    def submitPass(self) -> None:
        gate = vii.project_manager.hybrid_segment_service.supervised_gate
        gate.submit_pass()

    @Slot(str)
    def submitError(self, correction_text: str) -> None:
        gate = vii.project_manager.hybrid_segment_service.supervised_gate
        gate.submit_error(correction_text)
