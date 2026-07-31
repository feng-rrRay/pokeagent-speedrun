"""Tests for the ACE controller window loop (agents/ace/controller.py).

All LLM calls are mocked; nothing here touches the network or the emulator.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.ace.controller import AceController
from agents.ace.playbook import empty_playbook, playbook_bullet_ids

REFLECTION = json.dumps(
    {
        "reasoning": "the agent walked into a wall repeatedly",
        "error_identification": "pressed UP against a ledge",
        "root_cause_analysis": "misread the tile as walkable",
        "correct_approach": "step around the obstruction",
        "key_insight": "if coordinates do not change after a directional press, the tile is blocked",
        "bullet_tags": [{"id": "nav-00001", "tag": "helpful"}],
    }
)

CURATION = json.dumps(
    {
        "reasoning": "the blocked-tile lesson is missing",
        "operations": [
            {
                "type": "ADD",
                "section": "stuck_state_recovery",
                "content": "If coordinates do not change after a directional press, try a perpendicular direction.",
            }
        ],
    }
)


@pytest.fixture
def ace_dir(tmp_path, monkeypatch):
    """Point the controller's cache dir at a tmp dir."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(
        "utils.data_persistence.run_data_manager.get_cache_path",
        lambda name: cache / name,
    )
    return cache / "ace"


def make_controller(ace_dir, responses=None, config=None, adapter=None):
    """Build a controller whose reflector/curator VLM returns canned strings."""
    text_vlm = MagicMock()
    if responses is None:
        responses = [REFLECTION, CURATION]
    text_vlm.get_text_query.side_effect = list(responses) * 50

    outer_vlm = MagicMock()
    outer_vlm.backend_type = "gemini"
    outer_vlm.model_name = "test-model"

    cfg = {"window_steps": 3, "warmup_steps": 0}
    cfg.update(config or {})

    with patch("utils.agent_infrastructure.vlm_backends.VLM", return_value=text_vlm):
        controller = AceController(
            vlm=outer_vlm,
            mcp_adapter=adapter or MagicMock(),
            run_data_manager=None,
            config=cfg,
        )
    controller._text_vlm_mock = text_vlm
    return controller


def drive(controller, n_steps, response_text="PLAYBOOK_USED: none", start=1):
    for step in range(start, start + n_steps):
        controller.on_step_complete(step=step, response_text=response_text, game_state_json={})


# ---------------------------------------------------------------------------
# Window scheduling
# ---------------------------------------------------------------------------


def test_window_fires_once_at_k_steps(ace_dir):
    controller = make_controller(ace_dir, config={"window_steps": 5})
    drive(controller, 4)
    assert controller.window_index == 0

    drive(controller, 1, start=5)
    assert controller.window_index == 1

    drive(controller, 5, start=6)
    assert controller.window_index == 2


def test_warmup_delays_the_first_window(ace_dir):
    controller = make_controller(ace_dir, config={"window_steps": 3, "warmup_steps": 6})
    drive(controller, 6)
    assert controller.window_index == 0
    drive(controller, 3, start=7)
    assert controller.window_index == 1


def test_repeated_step_number_is_idempotent(ace_dir):
    """The black-frame path returns success without claiming a step, so the same
    step number can arrive twice."""
    controller = make_controller(ace_dir, config={"window_steps": 3})
    controller.on_step_complete(step=1, response_text="x", game_state_json={})
    controller.on_step_complete(step=1, response_text="x", game_state_json={})
    controller.on_step_complete(step=2, response_text="x", game_state_json={})
    controller.on_step_complete(step=2, response_text="x", game_state_json={})
    assert controller.window_index == 0
    controller.on_step_complete(step=3, response_text="x", game_state_json={})
    assert controller.window_index == 1


def test_disabled_controller_is_inert(ace_dir):
    controller = make_controller(ace_dir, config={"enabled": False, "window_steps": 1})
    drive(controller, 5)
    assert controller.window_index == 0
    assert controller.get_playbook_block() == ""


# ---------------------------------------------------------------------------
# Playbook injection
# ---------------------------------------------------------------------------


def test_playbook_block_is_delimited(ace_dir):
    controller = make_controller(ace_dir)
    block = controller.get_playbook_block()
    assert block.startswith("PLAYBOOK_BEGIN")
    assert "PLAYBOOK_END" in block
    assert "## NAVIGATION AND MOVEMENT" in block


# ---------------------------------------------------------------------------
# The ACE cycle
# ---------------------------------------------------------------------------


def test_curator_operations_are_applied_and_persisted(ace_dir):
    controller = make_controller(ace_dir, config={"window_steps": 3})
    drive(controller, 3)

    ids = playbook_bullet_ids(controller.playbook)
    assert ids == ["stuck-00001"]
    assert "perpendicular direction" in controller.playbook

    assert (ace_dir / "playbook.md").exists()
    assert (ace_dir / "ace_state.json").exists()
    assert (ace_dir / "reflections.jsonl").exists()
    assert (ace_dir / "curations.jsonl").exists()
    assert (ace_dir / "ace_log.jsonl").exists()
    assert (ace_dir / "playbook_history" / "step_00003.md").exists()

    log = json.loads((ace_dir / "ace_log.jsonl").read_text().strip())
    assert log["bullets_added"] == 1
    assert log["reflected"] is True
    assert log["curated"] is True


def test_bullet_tags_update_counters_for_known_ids_only(ace_dir):
    controller = make_controller(ace_dir, config={"window_steps": 3})
    # Seed a playbook containing nav-00001 so the reflection's tag applies.
    controller.playbook = "## NAVIGATION AND MOVEMENT\n[nav-00001] helpful=0 harmful=0 :: seed\n\n## OTHERS"
    controller.next_global_id = 2

    drive(controller, 3, response_text="PLAYBOOK_USED: [nav-00001]")

    assert "[nav-00001] helpful=1 harmful=0 :: seed" in controller.playbook

    record = json.loads((ace_dir / "reflections.jsonl").read_text().strip())
    assert record["cited_bullets"] == ["nav-00001"]
    assert record["bullet_tags"] == [{"id": "nav-00001", "tag": "helpful"}]
    assert record["citation_rate"] == 1.0


def test_hallucinated_bullet_ids_are_discarded(ace_dir):
    reflection = json.dumps(
        {
            "reasoning": "r", "error_identification": "e", "root_cause_analysis": "rc",
            "correct_approach": "ca", "key_insight": "ki",
            "bullet_tags": [{"id": "zzz-99999", "tag": "harmful"}],
        }
    )
    controller = make_controller(ace_dir, responses=[reflection, CURATION], config={"window_steps": 3})
    before = controller.playbook
    drive(controller, 3)

    record = json.loads((ace_dir / "reflections.jsonl").read_text().strip())
    assert record["bullet_tags"] == []
    # The curator still ran and added its bullet.
    assert before != controller.playbook


def test_absent_citations_are_a_graceful_noop(ace_dir):
    controller = make_controller(ace_dir, config={"window_steps": 3})
    drive(controller, 3, response_text="I press UP because it looks open.")

    record = json.loads((ace_dir / "reflections.jsonl").read_text().strip())
    assert record["cited_bullets"] == []
    assert record["citation_rate"] == 0.0
    # Curation is unaffected by the missing citation.
    assert playbook_bullet_ids(controller.playbook) == ["stuck-00001"]


def test_reflector_failure_leaves_playbook_unchanged_and_run_continues(ace_dir):
    controller = make_controller(ace_dir, responses=["not json at all"], config={"window_steps": 3})
    before = controller.playbook
    drive(controller, 3)

    # A non-JSON reflection yields no tags and no parsed insight, but the raw
    # text is still handed to the curator, so we only assert the run survived.
    assert controller.window_index == 1
    assert isinstance(controller.playbook, str)
    assert "## NAVIGATION AND MOVEMENT" in controller.playbook
    assert before.count("##") == controller.playbook.count("##")


def test_reflector_api_error_is_swallowed(ace_dir):
    controller = make_controller(ace_dir, config={"window_steps": 3})
    controller._text_vlm_mock.get_text_query.side_effect = RuntimeError("429")
    before = controller.playbook
    drive(controller, 3)
    assert controller.playbook == before
    assert controller.window_index == 1


def test_curator_malformed_json_skips_curation(ace_dir):
    controller = make_controller(
        ace_dir, responses=[REFLECTION, '{"operations": "not a list"}'], config={"window_steps": 3}
    )
    before = controller.playbook
    drive(controller, 3)
    assert controller.playbook == before
    assert controller.window_index == 1


def test_freeze_playbook_skips_curator_but_still_injects(ace_dir):
    controller = make_controller(
        ace_dir, config={"window_steps": 3, "freeze_playbook": True}
    )
    controller.playbook = "## OTHERS\n[misc-00001] helpful=0 harmful=0 :: frozen advice"
    drive(controller, 3)

    assert playbook_bullet_ids(controller.playbook) == ["misc-00001"]
    assert "frozen advice" in controller.get_playbook_block()
    log = json.loads((ace_dir / "ace_log.jsonl").read_text().strip())
    assert log["curated"] is False
    assert log["reflected"] is True  # the reflector still runs


def test_freeze_playbook_also_freezes_helpful_harmful_counters(ace_dir):
    """Counters are rendered into the injected bullets and explicitly steer the
    policy ('prefer high helpful, low harmful'), so a static-context control
    that lets them drift is not static."""
    controller = make_controller(ace_dir, config={"window_steps": 3, "freeze_playbook": True})
    frozen = "## NAVIGATION AND MOVEMENT\n[nav-00001] helpful=0 harmful=0 :: seed\n\n## OTHERS"
    controller.playbook = frozen
    controller.next_global_id = 2

    drive(controller, 3, response_text="PLAYBOOK_USED: [nav-00001]")

    assert controller.playbook == frozen
    assert "[nav-00001] helpful=0 harmful=0 :: seed" in controller.get_playbook_block()

    # The reflector did produce a helpful tag for that bullet; it was not applied.
    record = json.loads((ace_dir / "reflections.jsonl").read_text().strip())
    assert record["bullet_tags"] == [{"id": "nav-00001", "tag": "helpful"}]
    log = json.loads((ace_dir / "ace_log.jsonl").read_text().strip())
    assert log["tags_applied"] == 0


def test_budget_zero_disables_pruning_reference_default_mode(ace_dir):
    """The reference's 80000-token budget is advisory (interpolated into the
    curator prompt, never enforced), so budget 0 reproduces it."""
    controller = make_controller(
        ace_dir,
        config={"window_steps": 2, "budget_tokens": 0, "min_bullets": 1,
                "unused_grace_windows": 0},
    )
    bloated = "\n".join(
        ["## OTHERS"] + [f"[misc-0000{i}] helpful=0 harmful=9 :: {'x' * 200}" for i in range(1, 5)]
    )
    controller.playbook = bloated
    controller.next_global_id = 5

    drive(controller, 2)

    # All four survive despite being far over any sane budget and flagged harmful.
    assert len(playbook_bullet_ids(controller.playbook)) == 5  # 4 + the curator's addition
    assert not (ace_dir / "pruned.jsonl").exists()


def test_window_open_location_can_still_be_novel(ace_dir):
    """Regression: seen_locations used to be updated from the window-open state
    on every step, before the novelty check snapshotted it. A map entered at the
    end of window N-1 was therefore already 'seen' when window N opened, so a
    boundary transition was invisible in both windows."""
    controller = make_controller(ace_dir, config={"window_steps": 2})
    payload = {
        "raw_state": {
            "player": {"location": "Route3", "position": {"x": 1, "y": 1}, "facing": "UP", "party": []},
            "game": {"badges": [], "money": 0, "game_state": "overworld"},
            "map": {"id": 9},
        }
    }
    for step in (1, 2):
        controller.on_step_complete(step=step, response_text="x", game_state_json=payload)

    record = json.loads((ace_dir / "reflections.jsonl").read_text().strip())
    assert "Route3" in record["facts"]["novel_locations"]
    assert record["verdict"] == "PROGRESS"
    # And it is remembered for next time.
    assert "Route3" in controller.seen_locations


def test_trajectory_rows_are_filtered_to_the_window(ace_dir, monkeypatch):
    """Black-frame steps produce no trajectory row, so asking for the last N
    lines can reach back into the previous window and reflect on it twice."""
    controller = make_controller(ace_dir, config={"window_steps": 3})

    rows = [
        {"step": s, "reasoning": f"s{s}", "location": "PewterCity", "player_coords": [1, 1],
         "pre_state": {"location": "PewterCity", "player_coords": [1, 1], "context": "overworld"}}
        for s in range(1, 7)
    ]
    path = ace_dir.parent / "trajectory_history.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(
        "agents.subagents.utils.trajectory_window.resolve_trajectory_path",
        lambda *_a, **_k: path,
    )

    fetched = controller._recent_trajectories(num_steps=3, start_step=4, end_step=6)
    assert [r["step"] for r in fetched] == [4, 5, 6]

    # Even when more lines are pulled than the window contains.
    fetched_wide = controller._recent_trajectories(num_steps=6, start_step=4, end_step=6)
    assert [r["step"] for r in fetched_wide] == [4, 5, 6]


def test_curator_frequency_skips_windows(ace_dir):
    controller = make_controller(
        ace_dir, config={"window_steps": 2, "curator_frequency": 2}
    )
    drive(controller, 4)
    assert controller.window_index == 2
    # Window 0 curates (0 % 2 == 0), window 1 does not.
    assert len(playbook_bullet_ids(controller.playbook)) == 1


def test_previous_reflection_is_passed_to_the_next_window(ace_dir):
    controller = make_controller(ace_dir, config={"window_steps": 2})
    drive(controller, 2)
    assert "coordinates do not change" in controller.previous_reflection

    drive(controller, 2, start=3)
    prompts = [c.args[0] for c in controller._text_vlm_mock.get_text_query.call_args_list]
    reflector_prompts = [p for p in prompts if "Previous Reflection" in p]
    assert len(reflector_prompts) >= 2
    assert "coordinates do not change" in reflector_prompts[1]
    assert "unverified hypothesis" in reflector_prompts[1]


def test_reflector_and_curator_use_distinct_metrics_module_names(ace_dir):
    controller = make_controller(ace_dir, config={"window_steps": 2})
    drive(controller, 2)
    modules = [c.args[1] for c in controller._text_vlm_mock.get_text_query.call_args_list]
    assert modules == ["ace_reflector", "ace_curator"]


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_pruning_fires_when_over_budget(ace_dir):
    controller = make_controller(
        ace_dir,
        config={"window_steps": 2, "budget_tokens": 20, "min_bullets": 1,
                "unused_grace_windows": 0},
    )
    controller.playbook = "\n".join(
        ["## OTHERS"] + [f"[misc-0000{i}] helpful=0 harmful=3 :: {'x' * 80}" for i in range(1, 5)]
    )
    controller.next_global_id = 5
    drive(controller, 2)

    assert len(playbook_bullet_ids(controller.playbook)) < 4
    assert (ace_dir / "pruned.jsonl").exists()
    assert "## OTHERS" in controller.playbook


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_restores_playbook_and_ids_without_collision(ace_dir):
    first = make_controller(ace_dir, config={"window_steps": 2})
    drive(first, 2)
    assert playbook_bullet_ids(first.playbook) == ["stuck-00001"]
    assert first.window_index == 1

    # A new process: step counter restarts at 1, window index must not.
    second = make_controller(ace_dir, config={"window_steps": 2})
    assert playbook_bullet_ids(second.playbook) == ["stuck-00001"]
    assert second.window_index == 1
    assert second.next_global_id == 2

    drive(second, 2)
    ids = playbook_bullet_ids(second.playbook)
    assert ids == ["stuck-00001", "stuck-00002"]
    assert len(set(ids)) == len(ids)


def test_warm_start_from_a_playbook_file(tmp_path, ace_dir):
    warm = tmp_path / "warm_playbook.md"
    warm.write_text(
        "## NAVIGATION AND MOVEMENT\n[nav-00042] helpful=7 harmful=0 :: warm advice\n\n## OTHERS"
    )
    controller = make_controller(ace_dir, config={"window_steps": 2, "playbook_path": str(warm)})
    assert "warm advice" in controller.playbook
    assert controller.next_global_id == 43


def test_existing_run_playbook_wins_over_warm_start(tmp_path, ace_dir):
    first = make_controller(ace_dir, config={"window_steps": 2})
    drive(first, 2)

    warm = tmp_path / "warm_playbook.md"
    warm.write_text("## OTHERS\n[misc-00099] helpful=0 harmful=0 :: should not be used")

    second = make_controller(ace_dir, config={"window_steps": 2, "playbook_path": str(warm)})
    assert "should not be used" not in second.playbook
    assert playbook_bullet_ids(second.playbook) == ["stuck-00001"]


def test_default_window_matches_the_continualharness_comparison_arm(ace_dir):
    """The ACE window is deliberately pinned to the ContinualHarness arm's
    --optimization-window-length (100) and its STABLE_FREQUENCY, so the two
    methods adapt on the same cadence over the same span of play."""
    from agents.utils.harness_evolver import STABLE_FREQUENCY

    outer_vlm = MagicMock()
    outer_vlm.backend_type, outer_vlm.model_name = "gemini", "test-model"
    with patch("utils.agent_infrastructure.vlm_backends.VLM", return_value=MagicMock()):
        controller = AceController(vlm=outer_vlm, mcp_adapter=MagicMock(), config=None)

    assert controller.window_steps == 100
    assert controller.window_steps == STABLE_FREQUENCY

    # And the CLI default agrees with the controller default.
    import run

    assert run.build_ace_config(type("Args", (), {})())["window_steps"] == 100


def test_fresh_controller_starts_from_the_empty_playbook(ace_dir):
    controller = make_controller(ace_dir)
    assert controller.playbook == empty_playbook()
    assert controller.next_global_id == 1
    assert controller.window_index == 0
