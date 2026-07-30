#!/usr/bin/env python3
"""Tests for final CLI metric flushing."""

import io
import threading

import run_cli
from utils.data_persistence import backup_manager, run_data_manager
from utils.agent_infrastructure.cli_agent_backends import CliSession


class _Process:
    def __init__(self):
        self.running = True

    def poll(self):
        return None if self.running else 0


class _Thread:
    def __init__(self, events):
        self.events = events

    def join(self, timeout=None):
        self.events.append(("join", timeout))

    def is_alive(self):
        return False


class _Backend:
    def __init__(self, events):
        self.events = events

    def log_cli_interaction(
        self,
        agent_memory_dir,
        processed_hashes,
        last_cli_step,
        server_url=None,
    ):
        self.events.append(("sync", server_url))
        return processed_hashes | {"event-2"}, last_cli_step + 1


def test_final_flush_stops_agent_before_syncing_metrics(tmp_path, monkeypatch):
    events = []
    process = _Process()
    session = CliSession(
        process=process,
        stop_event=threading.Event(),
        stream_thread=_Thread(events),
    )
    state = run_cli.CliRunState(
        cli_session=session,
        cli_log_file=io.StringIO(),
        processed_hashes={"event-1"},
        last_cli_step=3,
    )

    def stop_process(process, graceful_timeout, label, use_process_group):
        events.append(("stop", graceful_timeout))
        process.running = False

    monkeypatch.setattr(run_cli, "_terminate_process", stop_process)

    run_cli._flush_cli_metrics_before_shutdown(
        state,
        _Backend(events),
        tmp_path,
        "http://localhost:8100",
        graceful_timeout=7,
    )

    assert events == [
        ("stop", 7),
        ("join", 7),
        ("sync", "http://localhost:8100"),
    ]
    assert state.processed_hashes == {"event-1", "event-2"}
    assert state.last_cli_step == 4


def test_backup_restore_defaults_to_paper_accounting(monkeypatch, tmp_path):
    calls = []

    def restore_cache_from_backup(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(
        backup_manager,
        "restore_cache_from_backup",
        restore_cache_from_backup,
    )
    monkeypatch.setattr(
        run_data_manager,
        "get_cache_directory",
        lambda: tmp_path,
    )

    assert run_cli._restore_from_backup("red-init.zip")
    assert calls[-1]["restore_metrics"] is False
    assert calls[-1]["preserve_metric_context"] is True

    assert run_cli._restore_from_backup(
        "termination-backup.zip",
        continue_metrics=True,
    )
    assert calls[-1]["restore_metrics"] is True
    assert calls[-1]["preserve_metric_context"] is False
