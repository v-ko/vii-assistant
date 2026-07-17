"""Minimal Qt app for headless experiment runs — no QML, widgets, or overlay."""

import sys

from PySide6.QtWidgets import QApplication
from sivkit.loop import set_main_loop
from sivkit.platform.qt_widgets.qt_main_loop import QtMainLoop

from assistant.app_state import AppViewState
from assistant.app_view_model import AppViewModel
from assistant.view_states.terminal import TerminalViewState


class ViiHeadlessApp(QApplication):
    """Qt app exposing only what the experiment runner needs.

    Installs the Qt-integrated asyncio loop (required for @procedure/@action)
    and constructs the GUI-free view-state objects. No QML engine, services,
    or projectors are created.
    """

    def __init__(self, view_state: AppViewState):
        super().__init__(sys.argv)
        set_main_loop(QtMainLoop(self))
        self.setQuitOnLastWindowClosed(False)

        self.view_state = view_state
        self.terminal_state = TerminalViewState(self.view_state)
        self.app_view_model = AppViewModel(parent=self)
