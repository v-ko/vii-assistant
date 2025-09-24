from PySide6.QtCore import QObject
from PySide6.QtCore import Signal, Slot

_actions = {}


def action(func):
    """Decorator to register a command."""
    _actions[func.__name__] = func
    return func


class FunctionExecutor(QObject):
    """Helper class to execute functions on the main thread."""

    function_requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.function_requested.connect(self._execute_function)

    @Slot(object)
    def _execute_function(self, func_data):
        func, args, kwargs = func_data
        func(*args, **kwargs)

    def execute(self, func, *args, **kwargs):
        """Queue a function to execute on the main thread."""
        self.function_requested.emit((func, args, kwargs))


# Create a global instance of the function executor
executor = FunctionExecutor()


def execute_on_main_thread(func, *args, **kwargs):
    """Execute a function on the main thread."""
    executor.execute(func, *args, **kwargs)


def execute_action(action_name):
    """Execute a command."""
    if action_name in _actions:
        _actions[action_name]()
    else:
        print(f"Command not found: {action_name}")
