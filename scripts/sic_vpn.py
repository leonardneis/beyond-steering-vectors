"""Human-in-the-loop VPN preflight for the SIC cluster on Windows (Cisco Secure Client).

The helper only *reads* the Cisco connection state (``vpncli stats``). It never
requests, reads, stores, logs or passes a VPN credential: when the tunnel is
down it opens the Cisco Secure Client window and waits until the operator has
authenticated there manually. It never connects or disconnects a tunnel itself.

Commands (exit code 0 means the expected tunnel is up):

    python scripts/sic_vpn.py status
    python scripts/sic_vpn.py ensure [--timeout 300] [--poll-interval 3]
    python scripts/sic_vpn.py ssh [--fallback] [-- <ssh arguments>]

``status`` and ``ensure`` print one status word on stdout; human-readable
messages go to stderr. Configuration is read from the git-ignored ``.env``
(see ``ENV_KEYS``); process environment variables override it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
ENV_KEYS = (
    "VPN_CLI",
    "VPN_HOST",
    "VPN_GROUP",
    "VPN_USER",
    "VPN_EXPECTED_SERVER_ADDRESS",
    "SSH_ALIAS",
    "SSH_ALIAS_FALLBACK",
)
REQUIRED_KEYS = ("VPN_CLI", "VPN_HOST", "VPN_EXPECTED_SERVER_ADDRESS")
VPNCLI_TIMEOUT_SECONDS = 30
MAX_CONSECUTIVE_PROBE_ERRORS = 3


class Status(str, Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    WRONG_TUNNEL = "WRONG_TUNNEL"
    ERROR = "ERROR"
    AUTH_FAILED = "AUTH_FAILED"
    TIMEOUT = "TIMEOUT"


EXIT_CODES = {
    Status.CONNECTED: 0,
    Status.DISCONNECTED: 10,
    Status.CONNECTING: 11,
    Status.WRONG_TUNNEL: 12,
    Status.ERROR: 13,
    Status.AUTH_FAILED: 14,
    Status.TIMEOUT: 15,
}

# Cisco "Connection State" values; transitional states count as their destination.
_CISCO_STATES = {
    "Connected": Status.CONNECTED,
    "Disconnected": Status.DISCONNECTED,
    "Disconnecting": Status.DISCONNECTED,
    "Connecting": Status.CONNECTING,
    "Reconnecting": Status.CONNECTING,
}


@dataclass(frozen=True)
class VpnConfig:
    cli: Path
    host: str
    group: str
    user: str
    expected_server: str
    ssh_alias: str = ""
    ssh_alias_fallback: str = ""


@dataclass(frozen=True)
class Probe:
    status: Status
    detail: str


def read_env_file(path: Path) -> dict[str, str]:
    """Return only the ``ENV_KEYS`` entries of a dotenv file; other keys are never retained."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        if not sep or key not in ENV_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_config(env_file: Path = ROOT / ".env", environ: Mapping[str, str] = os.environ) -> VpnConfig:
    values = read_env_file(env_file)
    values.update({key: environ[key] for key in ENV_KEYS if environ.get(key)})
    missing = [key for key in REQUIRED_KEYS if not values.get(key)]
    if missing:
        raise ValueError(f"missing VPN settings in {env_file} or environment: {', '.join(missing)}")
    return VpnConfig(
        cli=Path(values["VPN_CLI"]),
        host=values["VPN_HOST"],
        group=values.get("VPN_GROUP", ""),
        user=values.get("VPN_USER", ""),
        expected_server=values["VPN_EXPECTED_SERVER_ADDRESS"],
        ssh_alias=values.get("SSH_ALIAS", ""),
        ssh_alias_fallback=values.get("SSH_ALIAS_FALLBACK", ""),
    )


def parse_stats(text: str, expected_server: str) -> Probe:
    """Classify ``vpncli stats`` output by its Connection State and Server Address fields."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition(":")
        if sep:
            fields.setdefault(key.strip(), value.strip())
    state = fields.get("Connection State")
    if state is None:
        return Probe(Status.ERROR, "vpncli stats output has no 'Connection State' field")
    status = _CISCO_STATES.get(state)
    if status is None:
        return Probe(Status.ERROR, f"unrecognized Cisco connection state {state!r}")
    if status is not Status.CONNECTED:
        return Probe(status, f"Cisco reports {state}")
    server = fields.get("Server Address", "")
    if server == expected_server:
        return Probe(Status.CONNECTED, f"connected to expected server {server}")
    if not server or server == "Not Available":
        return Probe(Status.ERROR, "Cisco reports Connected but no server address; cannot verify tunnel")
    return Probe(Status.WRONG_TUNNEL, f"connected to {server}, expected {expected_server}")


def probe(config: VpnConfig, runner: Callable[..., subprocess.CompletedProcess] | None = None) -> Probe:
    """Read the current tunnel state. Read-only: runs ``vpncli stats`` with no stdin."""
    try:
        completed = (runner or subprocess.run)(
            [str(config.cli), "stats"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=VPNCLI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Probe(Status.ERROR, f"cannot run vpncli at {config.cli}: {exc}")
    if completed.returncode != 0:
        return Probe(Status.ERROR, f"vpncli stats exited with code {completed.returncode}")
    return parse_stats(completed.stdout, config.expected_server)


def cisco_ui_path(config: VpnConfig) -> Path:
    return config.cli.parent / "UI" / "csc_ui.exe"


def launch_cisco_ui(config: VpnConfig, popen: Callable[..., object] | None = None) -> None:
    """Open (or bring to front) the Cisco Secure Client window, detached from this process."""
    (popen or subprocess.Popen)(
        [str(cisco_ui_path(config))],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )


def cisco_defaults(preferences: Path | None = None) -> dict[str, str]:
    """Read the non-secret host/group/user defaults the Cisco window pre-fills from its own preferences."""
    if preferences is None:
        preferences = (Path(os.environ.get("LOCALAPPDATA", "")) / "Cisco" / "Cisco Secure Client" / "VPN"
                       / "preferences.xml")
    try:
        root = ET.parse(preferences).getroot()
    except (OSError, ET.ParseError):
        return {}
    tags = {"host": "DefaultHostName", "group": "DefaultGroup", "user": "DefaultUser"}
    return {name: (root.findtext(tag) or "").strip() for name, tag in tags.items()}


def _say(message: str) -> None:
    print(f"[vpn] {message}", file=sys.stderr, flush=True)


def _announce_manual_authentication(config: VpnConfig, defaults: Mapping[str, str], timeout: float) -> None:
    _say("Cisco Secure Client is waiting for manual authentication.")
    expected = {"host": config.host, "group": config.group, "user": config.user}
    for name, label in (("host", "Server"), ("group", "Group"), ("user", "Username")):
        if not expected[name]:
            continue
        remembered = defaults.get(name)
        note = ""
        if remembered not in (None, expected[name]):
            note = f"  (Cisco currently pre-fills {remembered!r}; change it)"
        _say(f"  {label}: {expected[name]}{note}")
    _say("  Click Connect and type your password in the Cisco window only. This helper never sees it.")
    _say(f"Waiting up to {timeout:.0f} s for the tunnel to {config.expected_server} (Ctrl+C aborts).")


def ensure_vpn(
    config: VpnConfig,
    *,
    timeout: float = 300.0,
    poll_interval: float = 3.0,
    probe_fn: Callable[[VpnConfig], Probe] | None = None,
    launch_fn: Callable[[VpnConfig], None] | None = None,
    defaults_fn: Callable[[], Mapping[str, str]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Probe:
    """Return CONNECTED once the expected tunnel is up; otherwise the terminal failure status.

    Opens the Cisco window only when the VPN is disconnected, then polls the
    connection state until success, timeout, or a Connecting -> Disconnected
    transition (authentication cancelled or failed). An existing tunnel, expected
    or not, is never touched.
    """
    probe_fn = probe_fn or probe
    launch_fn = launch_fn or launch_cisco_ui
    defaults_fn = defaults_fn or cisco_defaults

    current = probe_fn(config)
    if current.status in (Status.CONNECTED, Status.WRONG_TUNNEL, Status.ERROR):
        return current
    if current.status is Status.DISCONNECTED:
        try:
            launch_fn(config)
        except OSError as exc:
            return Probe(Status.ERROR, f"cannot open Cisco Secure Client at {cisco_ui_path(config)}: {exc}")
    _announce_manual_authentication(config, defaults_fn(), timeout)

    deadline = clock() + timeout
    seen_connecting = current.status is Status.CONNECTING
    consecutive_errors = 0
    while clock() < deadline:
        sleep(poll_interval)
        previous, current = current, probe_fn(config)
        if current.status is Status.ERROR:
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_PROBE_ERRORS:
                return current
            continue
        consecutive_errors = 0
        if current.status is not previous.status:
            _say(current.detail)
        if current.status in (Status.CONNECTED, Status.WRONG_TUNNEL):
            return current
        if current.status is Status.CONNECTING:
            seen_connecting = True
        elif seen_connecting:
            return Probe(Status.AUTH_FAILED, "Cisco returned to Disconnected: authentication was cancelled or failed")
    return Probe(Status.TIMEOUT, f"expected tunnel not connected within {timeout:.0f} s")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="report the tunnel state without side effects")
    for name, help_text in (("ensure", "open Cisco if needed and wait for manual authentication"),
                            ("ssh", "run ensure, then ssh to SSH_ALIAS with the remaining arguments")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--timeout", type=float, default=300.0, help="seconds to wait (default 300)")
        command.add_argument("--poll-interval", type=float, default=3.0, help="seconds between polls (default 3)")
        if name == "ssh":
            command.add_argument("--fallback", action="store_true", help="use SSH_ALIAS_FALLBACK")
            command.add_argument("ssh_args", nargs=argparse.REMAINDER, help="arguments passed to ssh after '--'")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_config(args.env_file)
    except ValueError as exc:
        result = Probe(Status.ERROR, str(exc))
    else:
        try:
            result = probe(config) if args.command == "status" else ensure_vpn(
                config, timeout=args.timeout, poll_interval=args.poll_interval)
        except KeyboardInterrupt:
            _say("aborted by user; the Cisco window and any tunnel are left untouched")
            return 130

    if args.command == "ssh" and result.status is Status.CONNECTED:
        alias = config.ssh_alias_fallback if args.fallback else config.ssh_alias
        if not alias:
            _say("SSH_ALIAS_FALLBACK is not set" if args.fallback else "SSH_ALIAS is not set")
            return EXIT_CODES[Status.ERROR]
        ssh_args = args.ssh_args[1:] if args.ssh_args[:1] == ["--"] else args.ssh_args
        return subprocess.call(["ssh", alias, *ssh_args])

    _say(f"{result.status.value}: {result.detail}")
    if args.command != "ssh":
        print(result.status.value)
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    sys.exit(main())
