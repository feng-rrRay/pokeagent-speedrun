#!/usr/bin/env python3
"""Tests for Hermes response usage normalization."""

from types import SimpleNamespace

from utils.agent_infrastructure.hermes_wrapper import _extract_usage_snapshot


def test_extract_usage_snapshot_folds_hidden_output_into_completion():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=135,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
        )
    )

    usage = _extract_usage_snapshot(response)

    assert usage["prompt_tokens"] == 100
    assert usage["reported_completion_tokens"] == 10
    assert usage["completion_adjustment_tokens"] == 25
    assert usage["reasoning_tokens"] == 25
    assert usage["completion_tokens"] == 35
    assert usage["total_tokens"] == 135
