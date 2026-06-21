import logging
import os
import signal
import sys
from pathlib import Path

# Set logging level before importing anything else (especially sivkit)
os.environ.setdefault("LOGLEVEL", "INFO")
# Run without a display server by default (overridable from the environment)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import click

from assistant.facade import vii
from assistant.inference.context import ContextManager
from assistant.init_app import init_headless
from assistant.logging_config import configure_logging
from assistant.model_configs import AVAILABLE_MODELS
from assistant.procedures import run_experiment_headless
from assistant.services.config_file_adapter import CONFIG_DIR
from assistant.services.project_manager import ViiProjectManager
from assistant.terminal_actions import set_model

log = logging.getLogger(__name__)


@click.command()
@click.argument("experiment")
@click.option("--agent", help="Agent to use (e.g. 'baseline-direct')")
@click.option(
    "--model",
    "model_key",
    type=click.Choice(list(AVAILABLE_MODELS)),
    default=None,
    help="Model to load on the inference server (default: config's selected_model).",
)
@click.option(
    "--resume",
    "resume_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Resume an existing run directory instead of starting a fresh run.",
)
def main(experiment, agent, model_key, resume_dir):
    """Run a headless experiment and exit.

    EXPERIMENT is the config name (e.g. 'synthetic_rectangle').
    """
    configure_logging()
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    ctx_manager = ContextManager()
    vii.set_project_manager(ViiProjectManager(CONFIG_DIR, context_manager=ctx_manager))

    app = init_headless(vii)
    # Explicit CLI model wins; otherwise fall back to the persisted config.
    model_key = model_key or vii.get_config().selected_model
    log.info(f"CLI: loading model '{model_key}'")
    # Kick off the model load (a procedure now — drives inference_status_VS).
    set_model(model_key).catch(lambda exc: log.error("Model load failed: %s", exc))

    exit_code = 0

    def _on_done(task):
        nonlocal exit_code
        try:
            task.result()
        except Exception:
            log.exception("Experiment run failed")
            exit_code = 1
        finally:
            sync_client = vii.project_manager._sync_client
            if sync_client:
                sync_client.stop()
            app.quit()

    task = run_experiment_headless(experiment, agent, resume_dir)
    task.add_done_callback(_on_done)

    app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
