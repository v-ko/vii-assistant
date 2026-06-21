"""Central logging configuration.

Call :func:`configure_logging` once from each process entry point
(GUI app, headless experiment runner, inference server). Keeping this
explicit avoids surprising import-time side effects.
"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging and silence noisy third-party loggers."""
    logging.basicConfig(level=level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
