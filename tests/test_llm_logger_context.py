#!/usr/bin/env python3
"""Tests for phase-local deltas with restored metric context."""

import json

from utils.data_persistence.llm_logger import LLMLogger


def test_restored_context_does_not_seed_fresh_objective_deltas(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_METRICS_WRITE_ENABLED", "false")
    metrics_path = tmp_path / "fresh_context.json"
    metrics_path.write_text(
        json.dumps(
            {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "total_actions": 0,
                "steps": [],
                "milestones": [],
                "objectives": [
                    {
                        "objective_id": "get_starter",
                        "cumulative_steps": 78,
                        "cumulative_prompt_tokens": 990356,
                        "cumulative_completion_tokens": 46069,
                        "cumulative_cached_tokens": 12169,
                        "cumulative_total_tokens": 1036425,
                        "cumulative_actions": 255,
                        "timestamp": 1000,
                    }
                ],
                "restored_metric_context": {
                    "milestone_count": 0,
                    "objective_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    logger = LLMLogger(log_dir=str(tmp_path / "logs"), session_id="fresh")
    assert logger.load_cumulative_metrics(str(metrics_path))
    logger.cumulative_metrics.update(
        {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "cached_tokens": 20,
            "total_tokens": 125,
            "total_actions": 3,
        }
    )

    logger.log_objective_completion(
        "exit_oaks_lab",
        "story",
        4,
        step_number=2,
        timestamp=1010,
    )

    new_objective = logger.cumulative_metrics["objectives"][-1]
    assert new_objective["split_steps"] == 2
    assert new_objective["split_prompt_tokens"] == 100
    assert new_objective["split_completion_tokens"] == 25
    assert new_objective["split_total_tokens"] == 125
    assert new_objective["split_actions"] == 3
