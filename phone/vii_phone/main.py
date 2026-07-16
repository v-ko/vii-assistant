"""vii-phone CLI entrypoint."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

# Load .env from phone/ directory
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _val = _line.partition("=")
        os.environ.setdefault(_key.strip(), _val.strip())


@click.command()
@click.option(
    "--discord-token",
    envvar="VII_DISCORD_TOKEN",
    required=True,
    help="Discord bot token",
)
@click.option(
    "--channel-id",
    envvar="VII_DISCORD_CHANNEL",
    type=int,
    required=True,
    help="Discord voice channel ID to listen in",
)
@click.option(
    "--inference-url",
    envvar="VII_INFERENCE_URL",
    default="http://localhost:8008",
    help="VII inference server URL",
)
@click.option(
    "--vscode-bridge-url",
    envvar="VII_VSCODE_BRIDGE_URL",
    default="http://localhost:51178",
    help="VS Code bridge extension URL",
)
@click.option(
    "--log-level",
    envvar="LOGLEVEL",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(
    discord_token: str,
    channel_id: int,
    inference_url: str,
    vscode_bridge_url: str,
    log_level: str,
) -> None:
    """vii-phone — Discord voice bridge to VII assistant."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from vii_phone.service import ViiPhoneService

    service = ViiPhoneService(
        discord_token=discord_token,
        channel_id=channel_id,
        inference_url=inference_url,
        vscode_bridge_url=vscode_bridge_url,
    )

    try:
        service.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
