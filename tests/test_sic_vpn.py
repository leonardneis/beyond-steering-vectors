"""Tests for the VPN preflight. Cisco/vpncli are always mocked; no test touches a real tunnel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sic_vpn  # noqa: E402
from sic_vpn import EXIT_CODES, Probe, Status, VpnConfig  # noqa: E402

CLI = Path("C:/Program Files (x86)/Cisco/Cisco Secure Client/vpncli.exe")
CONFIG = VpnConfig(cli=CLI, host="vpn.example.org", group="Grp", user="user01", expected_server="192.0.2.10",
                   ssh_alias="cluster", ssh_alias_fallback="cluster2")


def stats(state: str, server: str = "Not Available") -> str:
    # Shape of real `vpncli stats` output (CRLF, banner, state notices, sections).
    return (
        "Cisco Secure Client (version 5.1.2.42) .\r\n\r\n"
        f"\r  >> state: {state}\r\n\rVPN> \r  >> registered with local VPN subsystem.\r\n\r\n"
        "[ Connection Information ]\r\n\r\n"
        f"    Connection State:            {state}\r\n"
        "    Duration:                    00:00:00\r\n"
        "    Management Connection State: Disconnected (disabled)\r\n\r\n"
        "[ Address Information ]\r\n\r\n"
        "    Client Address (IPv4):       Not Available\r\n"
        f"    Server Address:              {server}\r\n"
    )


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(f"test attempted a real subprocess: {args}")

    for name in ("run", "Popen", "call"):
        monkeypatch.setattr(sic_vpn.subprocess, name, forbidden)


# --- state parsing -------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "status"),
    [
        (stats("Connected", "192.0.2.10"), Status.CONNECTED),
        (stats("Connected", "198.51.100.7"), Status.WRONG_TUNNEL),
        (stats("Connected", "Not Available"), Status.ERROR),
        (stats("Disconnected"), Status.DISCONNECTED),
        (stats("Disconnecting"), Status.DISCONNECTED),
        (stats("Connecting"), Status.CONNECTING),
        (stats("Reconnecting", "192.0.2.10"), Status.CONNECTING),
        (stats("Unknown"), Status.ERROR),
        ("VPN> \r\n  >> error: Connect not available.\r\n", Status.ERROR),
        ("", Status.ERROR),
    ],
)
def test_parse_stats(text, status):
    assert sic_vpn.parse_stats(text, "192.0.2.10").status is status


def test_parse_stats_ignores_management_tunnel_state():
    text = stats("Disconnected").replace("Disconnected (disabled)", "Connected")
    assert sic_vpn.parse_stats(text, "192.0.2.10").status is Status.DISCONNECTED


def test_wrong_tunnel_detail_names_both_servers():
    detail = sic_vpn.parse_stats(stats("Connected", "198.51.100.7"), "192.0.2.10").detail
    assert "198.51.100.7" in detail and "192.0.2.10" in detail


# --- configuration -------------------------------------------------------------------------------

def test_load_config_quoted_path_with_spaces_and_only_known_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        'VPN_CLI="C:/Program Files (x86)/Cisco/Cisco Secure Client/vpncli.exe"\n'
        "VPN_HOST=vpn.example.org\nVPN_GROUP=Grp\nexport VPN_USER='user01'\n"
        "VPN_EXPECTED_SERVER_ADDRESS=192.0.2.10\nSSH_ALIAS=cluster\n"
        "VPN_PASSWORD=must-never-be-kept\nUNRELATED_TOKEN=also-ignored\n",
        encoding="utf-8",
    )
    config = sic_vpn.load_config(env, environ={})
    assert config.cli == CLI
    assert (config.host, config.group, config.user, config.ssh_alias) == ("vpn.example.org", "Grp", "user01", "cluster")
    assert "must-never-be-kept" not in repr(config)
    assert set(sic_vpn.read_env_file(env)) <= set(sic_vpn.ENV_KEYS)


def test_environment_overrides_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("VPN_CLI=a.exe\nVPN_HOST=file.host\nVPN_EXPECTED_SERVER_ADDRESS=192.0.2.10\n", encoding="utf-8")
    assert sic_vpn.load_config(env, environ={"VPN_HOST": "env.host"}).host == "env.host"


def test_missing_required_settings(tmp_path):
    with pytest.raises(ValueError, match="VPN_EXPECTED_SERVER_ADDRESS"):
        sic_vpn.load_config(tmp_path / "absent.env", environ={"VPN_CLI": "a.exe", "VPN_HOST": "h"})


def test_cisco_defaults_reads_only_host_group_user(tmp_path):
    prefs = tmp_path / "preferences.xml"
    prefs.write_text(
        "<AnyConnectPreferences><DefaultUser>user01</DefaultUser>"
        "<ClientCertificateThumbprint>ABC</ClientCertificateThumbprint>"
        "<DefaultHostName>vpn.example.org</DefaultHostName><DefaultGroup>Grp</DefaultGroup>"
        "</AnyConnectPreferences>",
        encoding="utf-8",
    )
    assert sic_vpn.cisco_defaults(prefs) == {"host": "vpn.example.org", "group": "Grp", "user": "user01"}
    assert sic_vpn.cisco_defaults(tmp_path / "missing.xml") == {}


# --- vpncli invocation ---------------------------------------------------------------------------

def test_probe_runs_read_only_stats_with_list_argv_and_no_stdin():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=stats("Connected", "192.0.2.10"), stderr="")

    assert sic_vpn.probe(CONFIG, runner=runner).status is Status.CONNECTED
    [(argv, kwargs)] = calls
    assert argv == [str(CLI), "stats"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert "input" not in kwargs and not kwargs.get("shell")


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("missing"), subprocess.TimeoutExpired("vpncli", 30), "nonzero"],
)
def test_probe_failures_are_errors(failure):
    def runner(argv, **kwargs):
        if failure == "nonzero":
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        raise failure

    assert sic_vpn.probe(CONFIG, runner=runner).status is Status.ERROR


def test_launch_opens_ui_next_to_vpncli_detached():
    calls = []
    sic_vpn.launch_cisco_ui(CONFIG, popen=lambda argv, **kwargs: calls.append((argv, kwargs)))
    [(argv, kwargs)] = calls
    assert argv == [str(CLI.parent / "UI" / "csc_ui.exe")]
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["stdout"] is subprocess.DEVNULL


# --- ensure control flow -------------------------------------------------------------------------

class FakeCisco:
    def __init__(self, *statuses: Status, launch_error: OSError | None = None):
        self.statuses = list(statuses)
        self.launches = 0
        self.launch_error = launch_error
        self.now = 0.0

    def probe(self, config):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return Probe(status, f"fake {status.value}")

    def launch(self, config):
        if self.launch_error:
            raise self.launch_error
        self.launches += 1

    def sleep(self, seconds):
        self.now += seconds

    def ensure(self, timeout=30.0):
        return sic_vpn.ensure_vpn(CONFIG, timeout=timeout, poll_interval=3.0, probe_fn=self.probe,
                                  launch_fn=self.launch, defaults_fn=dict, sleep=self.sleep, clock=lambda: self.now)


def test_already_connected_returns_immediately():
    cisco = FakeCisco(Status.CONNECTED)
    assert cisco.ensure().status is Status.CONNECTED
    assert cisco.launches == 0 and cisco.now == 0.0


def test_disconnected_opens_cisco_once_and_waits_for_manual_login(capsys):
    cisco = FakeCisco(Status.DISCONNECTED, Status.DISCONNECTED, Status.CONNECTING, Status.CONNECTING, Status.CONNECTED)
    assert cisco.ensure().status is Status.CONNECTED
    assert cisco.launches == 1
    err = capsys.readouterr().err
    assert "waiting for manual authentication" in err
    assert "vpn.example.org" in err and "Grp" in err and "user01" in err


def test_cancelled_or_failed_authentication():
    cisco = FakeCisco(Status.DISCONNECTED, Status.CONNECTING, Status.DISCONNECTED)
    assert cisco.ensure().status is Status.AUTH_FAILED


def test_initial_connecting_waits_without_reopening_cisco():
    cisco = FakeCisco(Status.CONNECTING, Status.CONNECTED)
    assert cisco.ensure().status is Status.CONNECTED
    assert cisco.launches == 0


def test_timeout_when_nobody_authenticates():
    cisco = FakeCisco(Status.DISCONNECTED)
    result = cisco.ensure(timeout=30.0)
    assert result.status is Status.TIMEOUT
    assert cisco.launches == 1 and cisco.now >= 30.0


@pytest.mark.parametrize("initial", [Status.WRONG_TUNNEL, Status.ERROR])
def test_wrong_tunnel_or_error_is_reported_without_opening_cisco(initial):
    cisco = FakeCisco(initial)
    assert cisco.ensure().status is initial
    assert cisco.launches == 0


def test_wrong_tunnel_after_login_is_reported():
    cisco = FakeCisco(Status.DISCONNECTED, Status.CONNECTING, Status.WRONG_TUNNEL)
    assert cisco.ensure().status is Status.WRONG_TUNNEL


def test_transient_probe_error_is_tolerated():
    cisco = FakeCisco(Status.DISCONNECTED, Status.ERROR, Status.CONNECTING, Status.ERROR, Status.CONNECTED)
    assert cisco.ensure().status is Status.CONNECTED


def test_repeated_probe_errors_abort():
    cisco = FakeCisco(Status.DISCONNECTED, Status.ERROR)
    assert cisco.ensure().status is Status.ERROR


def test_cisco_ui_launch_failure_is_error():
    cisco = FakeCisco(Status.DISCONNECTED, launch_error=FileNotFoundError("csc_ui.exe"))
    assert cisco.ensure().status is Status.ERROR


def test_prompt_flags_mismatching_cisco_defaults(capsys):
    cisco = FakeCisco(Status.DISCONNECTED, Status.CONNECTED)
    sic_vpn.ensure_vpn(CONFIG, timeout=30.0, probe_fn=cisco.probe, launch_fn=cisco.launch,
                       defaults_fn=lambda: {"host": "other.example.org", "group": "Grp", "user": "user01"},
                       sleep=cisco.sleep, clock=lambda: cisco.now)
    err = capsys.readouterr().err
    assert "other.example.org" in err and err.count("change it") == 1


# --- command line --------------------------------------------------------------------------------

@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(f'VPN_CLI="{CLI}"\nVPN_HOST=vpn.example.org\nVPN_EXPECTED_SERVER_ADDRESS=192.0.2.10\n'
                    "SSH_ALIAS=cluster\nSSH_ALIAS_FALLBACK=cluster2\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("status", list(Status))
def test_status_command_prints_word_and_exit_code(monkeypatch, capsys, env_file, status):
    monkeypatch.setattr(sic_vpn, "probe", lambda config: Probe(status, "fake"))
    assert sic_vpn.main(["--env-file", str(env_file), "status"]) == EXIT_CODES[status]
    assert capsys.readouterr().out.strip() == status.value


def test_exit_codes_are_distinct_and_zero_only_when_connected():
    assert len(set(EXIT_CODES.values())) == len(Status)
    assert [s for s, code in EXIT_CODES.items() if code == 0] == [Status.CONNECTED]
    assert 2 not in EXIT_CODES.values()  # reserved for argparse usage errors


def test_status_command_with_missing_config_is_error(tmp_path, capsys, monkeypatch):
    for key in sic_vpn.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert sic_vpn.main(["--env-file", str(tmp_path / "absent.env"), "status"]) == EXIT_CODES[Status.ERROR]
    assert capsys.readouterr().out.strip() == "ERROR"


def test_ssh_is_not_started_when_preflight_fails(monkeypatch, env_file):
    monkeypatch.setattr(sic_vpn, "ensure_vpn", lambda config, **kwargs: Probe(Status.TIMEOUT, "fake"))
    assert sic_vpn.main(["--env-file", str(env_file), "ssh", "--", "hostname"]) == EXIT_CODES[Status.TIMEOUT]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["ssh"], ["ssh", "cluster"]),
        (["ssh", "--", "condor_q", "-nobatch"], ["ssh", "cluster", "condor_q", "-nobatch"]),
        (["ssh", "--fallback", "--", "hostname"], ["ssh", "cluster2", "hostname"]),
    ],
)
def test_ssh_continues_after_expected_tunnel(monkeypatch, env_file, argv, expected):
    calls = []
    monkeypatch.setattr(sic_vpn, "ensure_vpn", lambda config, **kwargs: Probe(Status.CONNECTED, "fake"))
    monkeypatch.setattr(sic_vpn.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert sic_vpn.main(["--env-file", str(env_file), *argv]) == 0
    assert calls == [expected]


def test_source_never_issues_connect_or_disconnect():
    source = (ROOT / "scripts" / "sic_vpn.py").read_text(encoding="utf-8")
    assert '"connect"' not in source and '"disconnect"' not in source
    assert "input=" not in source and "shell=True" not in source
