#!/usr/bin/env python3
"""Email alert for weather-bot.service stop events."""

import json
import os
import socket
import sys
import time
import uuid

from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

CONFIG_PATH = Path("/etc/weather-bot-stop-email.json")
API_URL = "https://api.resend.com/emails"


def make_message(sender, recipient, test=False):
    now = datetime.now(ZoneInfo("America/New_York"))
    result = os.getenv("SERVICE_RESULT", "unknown")
    code = os.getenv("EXIT_CODE", "unknown")
    status = os.getenv("EXIT_STATUS", "unknown")

    mode = "TEST" if test else "STOPPED"
    host = socket.gethostname()

    return {
        "from": sender,
        "to": [recipient],
        "subject": f"[Weather Bot {mode}] {host}",
        "text": "\n".join([
            f"Kalshi weather-bot.service - {mode}",
            f"Server: {host}",
            f"Time: {now:%Y-%m-%d %I:%M:%S %p %Z}",
            f"Systemd result: {result}",
            f"Exit code: {code}",
            f"Exit status: {status}",
            "",
            "If this was not a scheduled stop,",
            "review unresolved orders and positions",
            "before restarting live trading.",
            "",
            "Service status:",
            "systemctl status weather-bot.service --no-pager",
            "",
            "Recent logs:",
            "journalctl -u weather-bot.service -n 80 --no-pager",
            "",
            "Unwind recovery:",
            "systemctl status weather-unwind-watch.timer --no-pager",
        ]),
    }


def main():
    options = sys.argv[1:]

    if (
        len(options) > 1
        or (
            options
            and options[0] not in ("--test", "--dry-run")
        )
    ):
        print(
            "Usage: weather-bot-stop-email [--test|--dry-run]",
            file=sys.stderr,
        )
        return 2

    if options == ["--dry-run"]:
        message = make_message(
            "example@example.com",
            "example@example.com",
        )
        print(json.dumps(message, indent=2))
        print("DRY RUN: No email sent")
        return 0

    config = json.loads(CONFIG_PATH.read_text())

    api_key = config["api_key"].strip()

    if not all((
        api_key,
        config["to"],
        config["from"],
    )):
        raise ValueError("Incomplete email configuration")

    message = make_message(
        config["from"],
        config["to"],
        test=(options == ["--test"]),
    )

    payload = json.dumps(message).encode("utf-8")

    request = Request(
        API_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Idempotency-Key": (
                "weather-bot-stop-" + str(uuid.uuid4())
            ),
        },
    )

    for attempt in (1, 2):
        try:
            with urlopen(request, timeout=8) as response:
                data = json.load(response)

            email_id = data.get("id")

            if not email_id:
                raise ValueError(
                    "Email API returned no message ID"
                )

            print(
                f"EMAIL ALERT ACCEPTED: "
                f"message_id={email_id}",
                flush=True,
            )
            return 0

        except HTTPError as exc:
            print(
                f"EMAIL ALERT FAILED: HTTP {exc.code}",
                file=sys.stderr,
                flush=True,
            )

            if exc.code != 429 and exc.code < 500:
                return 1

        except (URLError, TimeoutError, ValueError, OSError) as exc:
            print(
                f"EMAIL ALERT FAILED: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        if attempt == 1:
            time.sleep(1)

    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError) as exc:
        print(
            f"EMAIL ALERT CONFIG ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
