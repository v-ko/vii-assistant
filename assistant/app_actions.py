from assistant.facade import facade
from assistant.registries.actions import action


@action
def toggle_terminal():
    """Toggle the terminal window."""
    if not facade.qt_app.terminal_window.isVisible():
        print("Showing terminal window")
        facade.qt_app.terminal_window.show()
        # Show overlay when terminal is shown
        facade.qt_app.overlay.show()
    else:
        print("Hiding terminal window")
        facade.qt_app.terminal_window.hide()
        # Hide overlay when terminal is hidden
        facade.qt_app.overlay.hide()
