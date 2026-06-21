from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox
from sivkit.libs.procedure import procedure
from sivkit.storage.delta import Delta

if TYPE_CHECKING:
    from pathlib import Path

    from assistant.services.hybrid_segment_service import HybridSegmentService

from assistant.constants import EXPERIMENT_RESULTS_DIR, EXPERIMENTS_DIR
from assistant.experiments.stats import compute_stats, format_stats, load_results
from assistant.facade import vii
from assistant.inference.context import TextMessage
from assistant.terminal_actions import show_context_debug

log = logging.getLogger(__name__)


@procedure
async def handle_hybrid_context_delta(
    hybrid_segment_service: HybridSegmentService,
    delta: Delta,
    origin: str | None = None,
) -> None:
    try:
        changed_items = hybrid_segment_service.changed_context_items(delta)

        for item in changed_items:
            if isinstance(item, TextMessage) and item.request is not None:
                hybrid_segment_service.update_overlay_from_text(item.text)

        # In supervised mode, every generation turn is born "pending": the
        # server withholds its continuation until the client writes a verdict.
        # review_completed_turn auto-passes non-input turns and prompts on
        # input actions. Client-side executions are post-gate (already approved)
        # so they run identically in both modes.
        settings = vii.app.view_state.settings_VS
        is_supervised = settings.execution_mode == "supervised"

        text_requests: list[tuple[TextMessage, dict]] = []
        for item in changed_items:
            if not isinstance(item, TextMessage):
                continue
            request = item.request
            if not isinstance(request, dict):
                continue
            text_requests.append((item, request))

        if is_supervised:
            for item, request in text_requests:
                if request.get("execution") == "client":
                    await hybrid_segment_service.process_client_execution_request(item)
                elif request.get("completed"):
                    await hybrid_segment_service.review_completed_turn(item)
        else:
            for item, request in text_requests:
                # Client-side tool execution (python, click_at, scroll)
                if request.get("execution") == "client":
                    await hybrid_segment_service.process_client_execution_request(item)
                    continue

                # Completed assistant messages — detect chain settle (no tool call)
                if request.get("completed"):
                    hybrid_segment_service.handle_completed_message(item)
    except Exception:
        log.error("Hybrid segment context handling failed", exc_info=True)


@procedure
async def fetch_raw_context_and_present(focus_mode: str) -> None:
    """Fetch formatted context from the inference server and show it."""

    try:
        text = await vii.inference_client.get_context_debug(focus_mode)
    except Exception as exc:
        log.error("Failed to fetch context debug for %r", focus_mode, exc_info=True)
        QMessageBox.warning(None, "Context fetch failed", str(exc))
        return
    show_context_debug(focus_mode, text)


def _safe_record_error(em, error: str) -> None:
    """Record an error step, catching save failures to avoid killing the loop."""
    try:
        em._record_error_step(error)
    except Exception:
        log.exception("Failed to record error step (save_sample_image crashed)")


@procedure
async def run_all_experiment(resume_dir: "Path | None" = None) -> None:
    """Orchestrate a full experiment run.

    Ensures the model is loaded, auto-starts the experiment if needed,
    then runs all samples as an async loop with reconnection and error handling.

    When ``resume_dir`` is given, the run reuses that existing output directory,
    reloads its prior results, and continues from the next un-run step instead
    of creating a fresh timestamped directory.
    """
    from datetime import datetime

    from assistant.experiments.actions import mark_experiment_finished
    from assistant.experiments_manager import (
        INFERENCE_TIMEOUT_S,
        MAX_CONSECUTIVE_FAILURES,
        ExperimentState,
    )
    from assistant.terminal_actions import (
        _ensure_model_loaded,
        ensure_experiment_started,
    )

    _ensure_model_loaded()

    em = vii.experiments_manager
    if not em.running:
        ensure_experiment_started()  # auto-start without stepping

    if not em.running:
        raise RuntimeError("Failed to start experiment")

    assert em._config is not None

    from assistant.experiments.utils import save_run_params

    if resume_dir is not None:
        # Resume into an existing run directory: reuse its output, reload prior
        # results, and continue from the next un-run step.
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        em._output_dir = resume_dir
        (em._output_dir / "correct").mkdir(exist_ok=True)
        (em._output_dir / "incorrect").mkdir(exist_ok=True)
        prior = load_results(em._output_dir)
        em._results = prior
        if prior:
            resume_step = max(r["step"] for r in prior) + 1
        else:
            resume_step = em._config.start_index or 0
        em._current_step = resume_step
        log.info(
            f"Resuming run from step {resume_step} "
            f"({len(prior)} prior results). Output: {em._output_dir}"
        )
    else:
        # Create a fresh timestamped output directory.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{em._config.name}_{timestamp}"
        em._output_dir = EXPERIMENT_RESULTS_DIR / run_name
        em._output_dir.mkdir(parents=True, exist_ok=True)
        (em._output_dir / "correct").mkdir(exist_ok=True)
        (em._output_dir / "incorrect").mkdir(exist_ok=True)
        save_run_params(em._output_dir, em._run_params_snapshot())
        # Apply optional start index from config
        if em._config.start_index is not None:
            em._current_step = em._config.start_index
        log.info(f"Run-all started. Output: {em._output_dir}")

    em._run_all_active = True
    em._cancelled = False
    consecutive_failures = 0

    assert em._data_loader is not None
    total = len(em._data_loader)

    end = em._config.end_index if em._config.end_index is not None else total
    end = min(end, total)

    try:
        while em._current_step < end and not em._cancelled:
            try:
                await em._run_one_step()
                consecutive_failures = 0
            except asyncio.TimeoutError:
                log.error(
                    f"Step {em._current_step} timed out after "
                    f"{INFERENCE_TIMEOUT_S}s — skipping"
                )
                em._waiting_for_response = False
                _safe_record_error(em, "timeout")
                em._current_step += 1
                consecutive_failures += 1
                await em._reconnect()
            except ConnectionError as exc:
                log.error(
                    f"Step {em._current_step} connection error: {exc} — " "reconnecting"
                )
                em._waiting_for_response = False
                _safe_record_error(em, str(exc))
                em._current_step += 1
                consecutive_failures += 1
                await em._reconnect()
            except Exception as exc:
                log.exception(f"Step {em._current_step} unexpected error")
                em._waiting_for_response = False
                _safe_record_error(em, str(exc))
                em._current_step += 1
                consecutive_failures += 1

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.error(
                    f"{consecutive_failures} consecutive failures — "
                    "stopping experiment"
                )
                break

            # Single-line progress tracker (overwritten each step)
            done = len(em._results)
            correct = sum(1 for r in em._results if r.get("correct"))
            acc = correct / done if done else 0.0
            cib = sum(1 for r in em._results if r.get("center_in_gt"))
            cib_acc = cib / done if done else 0.0
            print(
                f"\rSample {em._current_step}/{end} | "
                f"correct {correct}/{done} ({acc:.0%}) | "
                f"center-in-box {cib}/{done} ({cib_acc:.0%})",
                end="",
                flush=True,
            )

            # Small yield to let UI update
            await asyncio.sleep(0.1)
    finally:
        em._run_all_active = False
        em._run_all_task = None
        em._state = ExperimentState.FINISHED
        em._save_results_json()
        mark_experiment_finished()
        status = "cancelled" if em._cancelled else "complete"
        print()  # finish the \r progress line
        log.info(f"Run-all {status}. Results saved.")

        # Compute, save, and print summary stats
        if em._output_dir is not None:
            results = load_results(em._output_dir)
            stats = compute_stats(results)
            report = format_stats(em._output_dir, stats)
            (em._output_dir / "stats.txt").write_text(report + "\n")
            print(report)


@procedure
async def run_experiment_headless(
    config_name: str, agent_name: str | None, resume_dir: "Path | None" = None
) -> None:
    """Headless experiment run: set agent, load config, run all steps.

    Does NOT handle app lifecycle — that's the caller's responsibility.
    Raises on failure (config not found, model timeout).

    When ``resume_dir`` is given, the run continues an existing run directory
    instead of starting a fresh one.
    """
    # Set agent if specified
    if agent_name:
        vii.set_active_agent(agent_name)
        log.info(f"CLI: agent set to '{agent_name}'")

    # Resolve experiment config
    config_path = EXPERIMENTS_DIR / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    # Set the config path on the experiments manager
    vii.experiments_manager.config_path = config_path

    # Wait for model to load (poll inference status)
    status_vs = vii.app.view_state.inference_status_VS
    for _ in range(60):  # up to 60s
        if status_vs.model_state == "loaded":
            break
        await asyncio.sleep(1.0)
    else:
        raise TimeoutError(
            f"Model not loaded after 60s (state={status_vs.model_state})"
        )

    # Start a session to connect the inference WebSocket (experiments submit
    # context over it and await results back). Headless has no GUI policy to
    # auto-start, so the runner does it explicitly.
    capture = vii.app.view_state.capture_screen_info
    screen_name = capture.name if capture else vii.get_config().capture_screen
    await vii.project_manager.start_session(screen_name=screen_name)

    log.info(f"CLI: starting experiment '{config_name}'")

    # Run the experiment
    task = run_all_experiment(resume_dir)
    vii.experiments_manager._run_all_task = task
    await task

    log.info("CLI: experiment complete")
