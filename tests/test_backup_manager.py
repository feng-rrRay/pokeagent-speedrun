#!/usr/bin/env python3
"""Tests for restoring gameplay checkpoints with fresh or resumed metrics."""

import json
import zipfile
from pathlib import Path

from utils.data_persistence.backup_manager import restore_cache_from_backup


def _write_backup(tmp_path: Path) -> Path:
    source = tmp_path / "source_run"
    source.mkdir()
    (source / "checkpoint.state").write_bytes(b"game-state")
    (source / "cumulative_metrics.json").write_text(
        json.dumps(
            {
                "total_tokens": 1234,
                "prompt_tokens": 1100,
                "completion_tokens": 134,
                "cached_tokens": 500,
                "cache_write_tokens": 10,
                "total_cost": 1.25,
                "total_actions": 40,
                "total_llm_calls": 12,
                "total_run_time": 50,
                "last_update_time": 100,
                "metadata": {"run_id": "source_run"},
                "steps": [{"step": 1, "total_tokens": 1234}],
                "milestones": [{"milestone_id": "starter"}],
                "objectives": [{"objective_id": "get_starter"}],
            }
        ),
        encoding="utf-8",
    )
    (source / "checkpoint_llm.txt").write_text(
        '{"agent_step_count": 42}',
        encoding="utf-8",
    )
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.iterdir():
            handle.write(path, arcname=f"{source.name}/{path.name}")
    return archive


def test_fresh_restore_keeps_game_state_and_skips_metric_artifacts(tmp_path):
    archive = _write_backup(tmp_path)
    destination = tmp_path / "restored"

    assert restore_cache_from_backup(
        str(archive),
        cache_dir=str(destination),
        create_backup_of_current=False,
        restore_metrics=False,
    )

    assert (destination / "checkpoint.state").read_bytes() == b"game-state"
    assert not (destination / "cumulative_metrics.json").exists()
    assert not (destination / "checkpoint_llm.txt").exists()


def test_fresh_restore_can_preserve_metric_context_without_phase_totals(tmp_path):
    archive = _write_backup(tmp_path)
    destination = tmp_path / "restored"

    assert restore_cache_from_backup(
        str(archive),
        cache_dir=str(destination),
        create_backup_of_current=False,
        restore_metrics=False,
        preserve_metric_context=True,
    )

    metrics = json.loads((destination / "cumulative_metrics.json").read_text())
    assert metrics["total_tokens"] == 0
    assert metrics["prompt_tokens"] == 0
    assert metrics["completion_tokens"] == 0
    assert metrics["total_actions"] == 0
    assert metrics["total_llm_calls"] == 0
    assert metrics["steps"] == []
    assert metrics["milestones"] == [{"milestone_id": "starter"}]
    assert metrics["objectives"] == [{"objective_id": "get_starter"}]
    assert metrics["restored_metric_context"] == {
        "milestone_count": 1,
        "objective_count": 1,
        "source_run_id": "source_run",
    }
    assert not (destination / "checkpoint_llm.txt").exists()


def test_fresh_context_uses_newer_source_run_metrics(tmp_path):
    archive = _write_backup(tmp_path)
    source_metrics = tmp_path / "source_run" / "cumulative_metrics.json"
    latest = json.loads(source_metrics.read_text())
    latest["last_update_time"] = 200
    latest["objectives"].append({"objective_id": "rival_battle_1"})
    source_metrics.write_text(json.dumps(latest), encoding="utf-8")
    destination = tmp_path / "restored"

    assert restore_cache_from_backup(
        str(archive),
        cache_dir=str(destination),
        create_backup_of_current=False,
        restore_metrics=False,
        preserve_metric_context=True,
    )

    metrics = json.loads((destination / "cumulative_metrics.json").read_text())
    assert [row["objective_id"] for row in metrics["objectives"]] == [
        "get_starter",
        "rival_battle_1",
    ]
    assert metrics["restored_metric_context"]["objective_count"] == 2


def test_resume_restore_keeps_metric_artifacts(tmp_path):
    archive = _write_backup(tmp_path)
    destination = tmp_path / "restored"

    assert restore_cache_from_backup(
        str(archive),
        cache_dir=str(destination),
        create_backup_of_current=False,
        restore_metrics=True,
    )

    assert (destination / "cumulative_metrics.json").exists()
    assert (destination / "checkpoint_llm.txt").exists()
