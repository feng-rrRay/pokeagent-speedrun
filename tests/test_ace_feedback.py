"""Tests for the ACE grounded environment feedback (agents/ace/feedback.py).

This module is the substitute for ACE's ground-truth signal, so the tests here
are as much about what it must NOT claim (blocked movement that never happened,
an LLM call, the hardcoded trajectory `outcome`) as about what it computes.
"""

import json
from unittest.mock import MagicMock

from agents.ace.feedback import (
    BATTLE_ENDED,
    BLOCKED,
    CAUGHT,
    CONTEXT_CHANGED,
    DAMAGE_DEALT,
    DAMAGE_TAKEN,
    DAMAGE_TRADED,
    DIALOGUE_ADVANCED,
    DIALOGUE_UNVERIFIED,
    ENEMY_FAINTED,
    INDETERMINATE,
    INTERACTION,
    MOVED,
    MOVEMENT,
    NO_OBSERVABLE_CHANGE,
    STATE_CHANGED,
    TRIGGERED,
    TURNED,
    UI_NAVIGATION,
    UNKNOWN_ACTION,
    WindowObservation,
    build_environment_feedback,
    classify_steps,
    format_actions_summary,
    format_ace_trace,
    read_milestones,
    summarize_game_state,
)


def _row(step, location, x, y, map_id=1, context="overworld", buttons=("UP",)):
    return {
        "step": step,
        "reasoning": f"thinking at step {step}",
        "action": {
            "type": "tool_calls",
            "tool_calls": [{"name": "press_buttons", "args": {"buttons": list(buttons)}}],
        },
        "location": location,
        "player_coords": [x, y],
        # _log_trajectory_for_step hardcodes this; the reflector must never see it.
        "outcome": {"success": True, "objectives_completed": []},
        "pre_state": {
            "location": location,
            "player_coords": [x, y],
            "map_id": map_id,
            "context": context,
            "is_in_battle": context == "battle",
            "dialog_active": context == "dialogue",
        },
    }


def _payload(location="PewterCity", x=14, y=6, map_id=2, badges=1, money=3000,
             hp=(20, 20), party=1, facing="UP", context="overworld"):
    game = {"badges": ["BOULDER"] * badges, "money": money, "game_state": context}
    if context == "battle":
        game["is_in_battle"] = True
    elif context == "dialogue":
        game["dialog_text"] = "Hello!"
    elif context == "menu":
        game["menu_active"] = True
    return {
        "raw_state": {
            "player": {
                "location": location,
                "position": {"x": x, "y": y},
                "facing": facing,
                "party": [{"current_hp": hp[0], "max_hp": hp[1]} for _ in range(party)],
            },
            "game": game,
            "map": {"id": map_id},
        }
    }


def _obs(rows, step_states, close, milestones_open=None, milestones_close=None, seen=None):
    return WindowObservation(
        start_step=min(step_states) if step_states else 1,
        end_step=max(s["step"] for s in rows) if rows else 1,
        trajectory_rows=rows,
        history_entries=[],
        step_states={k: summarize_game_state(v) for k, v in step_states.items()},
        state_close=summarize_game_state(close),
        milestones_open=milestones_open or {},
        milestones_close=milestones_close if milestones_close is not None else (milestones_open or {}),
        locations_seen_before=seen if seen is not None else {"PewterCity"},
    )


# ---------------------------------------------------------------------------
# State summarization
# ---------------------------------------------------------------------------


def test_summarize_game_state_reads_grounded_scalars():
    summary = summarize_game_state(_payload(badges=2, money=1500, hp=(5, 20), party=3))
    assert summary["location"] == "PewterCity"
    assert summary["coords"] == (14, 6)
    assert summary["badge_count"] == 2
    assert summary["money"] == 1500
    assert summary["party_size"] == 3
    assert summary["lowest_hp_fraction"] == 0.25
    assert summary["all_fainted"] is False
    assert summary["facing"] == "UP"


def test_summarize_game_state_accepts_json_string_and_bare_raw_state():
    assert summarize_game_state(json.dumps(_payload()))["location"] == "PewterCity"
    assert summarize_game_state(_payload()["raw_state"])["location"] == "PewterCity"


def test_summarize_game_state_tolerates_garbage():
    for payload in (None, "", "not json", {}, 42):
        summary = summarize_game_state(payload)
        assert summary["location"] == "Unknown"
        assert summary["badge_count"] is None


def test_summarize_detects_context_variants():
    assert summarize_game_state(_payload(context="battle"))["context"] == "battle"
    assert summarize_game_state(_payload(context="dialogue"))["context"] == "dialogue"
    assert summarize_game_state(_payload(context="menu"))["context"] == "menu"


# ---------------------------------------------------------------------------
# Step classification -- the fix for false movement judgments
# ---------------------------------------------------------------------------


def test_directional_in_dialogue_is_ui_navigation_not_blocked_movement():
    """Regression: a dialogue-only window used to report blocked moves, loops
    and stalls, all presented to the reflector as verified fact."""
    rows = [_row(i, "PewterCity", 14, 6, context="dialogue", buttons=("A",)) for i in range(1, 4)]
    states = {i: _payload(context="dialogue") for i in range(1, 4)}
    obs = _obs(rows, states, _payload(context="dialogue"))

    classified = classify_steps(obs)
    assert {r["class"] for r in classified} == {INTERACTION}
    assert all(r["outcome"] is None for r in classified)

    _, _, facts = build_environment_feedback(obs)
    assert facts["blocked"] == 0
    assert facts["loops"] == 0
    assert facts["longest_blocked_run"] == 0
    assert facts["movement_attempts"] == 0
    assert facts["interaction_steps"] == 3


def test_directional_in_menu_is_cursor_movement_not_locomotion():
    rows = [_row(i, "PewterCity", 14, 6, context="menu", buttons=("DOWN",)) for i in range(1, 4)]
    states = {i: _payload(context="menu") for i in range(1, 4)}
    obs = _obs(rows, states, _payload(context="menu"))

    classified = classify_steps(obs)
    assert {r["class"] for r in classified} == {UI_NAVIGATION}

    _, _, facts = build_environment_feedback(obs)
    assert facts["movement_attempts"] == 0
    assert facts["blocked"] == 0
    assert facts["ui_navigation_steps"] == 3


def test_directional_in_battle_is_target_selection():
    rows = [_row(1, "Route3", 5, 5, context="battle", buttons=("DOWN",))]
    obs = _obs(rows, {1: _payload(context="battle")}, _payload(context="battle"))
    assert classify_steps(obs)[0]["class"] == UI_NAVIGATION


def test_genuine_blocked_move_is_still_reported():
    rows = [_row(i, "PewterCity", 14, 6, buttons=("UP",)) for i in range(1, 4)]
    states = {i: _payload(facing="UP") for i in range(1, 4)}
    obs = _obs(rows, states, _payload(facing="UP"))

    classified = classify_steps(obs)
    assert {r["class"] for r in classified} == {MOVEMENT}
    assert [r["outcome"] for r in classified] == [BLOCKED] * 3

    feedback, verdict, facts = build_environment_feedback(obs)
    assert facts["blocked"] == 3
    assert facts["longest_blocked_run"] == 3
    assert verdict == "NO_PROGRESS"
    assert "3 blocked" in feedback


def test_turn_in_place_is_not_blocked():
    rows = [_row(1, "PewterCity", 14, 6, buttons=("LEFT",))]
    obs = _obs(rows, {1: _payload(facing="UP")}, _payload(facing="LEFT"))
    assert classify_steps(obs)[0]["outcome"] == TURNED
    _, _, facts = build_environment_feedback(obs)
    assert facts["blocked"] == 0
    assert facts["turned"] == 1


def test_wild_encounter_with_unchanged_position_is_triggered_not_blocked():
    """Pressing UP into tall grass starts a battle with the position unchanged.
    Calling that a blocked move would be actively misleading."""
    rows = [_row(1, "Route3", 8, 11, buttons=("UP",))]
    obs = _obs(
        rows,
        {1: _payload(location="Route3", x=8, y=11, facing="UP")},
        _payload(location="Route3", x=8, y=11, facing="UP", context="battle"),
    )
    assert classify_steps(obs)[0]["outcome"] == TRIGGERED
    _, _, facts = build_environment_feedback(obs)
    assert facts["blocked"] == 0
    assert facts["triggered"] == 1


def test_successful_move_is_detected():
    rows = [_row(1, "PewterCity", 14, 6, buttons=("UP",))]
    obs = _obs(rows, {1: _payload(y=6)}, _payload(y=5))
    assert classify_steps(obs)[0]["outcome"] == MOVED


def test_missing_facing_yields_indeterminate_not_blocked():
    """Without a facing reading we cannot separate a turn from a bump, and
    asserting "blocked" would break the verified-fact guarantee."""
    rows = [_row(1, "PewterCity", 14, 6, buttons=("UP",))]
    obs = _obs(rows, {1: _payload(facing=None)}, _payload(facing=None))
    assert classify_steps(obs)[0]["outcome"] == INDETERMINATE
    feedback, _, facts = build_environment_feedback(obs)
    assert facts["blocked"] == 0
    assert facts["indeterminate"] == 1
    assert "indeterminate" in feedback


def test_non_directional_press_is_never_a_movement_attempt():
    rows = [_row(1, "PewterCity", 14, 6, buttons=("A",)), _row(2, "PewterCity", 14, 6, buttons=("WAIT",))]
    obs = _obs(rows, {1: _payload(), 2: _payload()}, _payload())
    assert {r["class"] for r in classify_steps(obs)} == {INTERACTION}


def test_steps_with_no_button_record_are_not_claimed_as_interactions():
    """If the trajectory row is missing we never saw the buttons. Reporting
    those as 'interaction or wait' would be exactly the kind of unverified
    assertion this module exists to prevent."""
    obs = WindowObservation(
        start_step=1,
        end_step=2,
        trajectory_rows=[],  # no rows -> no button record
        history_entries=[],
        step_states={1: summarize_game_state(_payload()), 2: summarize_game_state(_payload())},
        state_close=summarize_game_state(_payload()),
        locations_seen_before={"PewterCity"},
    )
    assert {r["class"] for r in classify_steps(obs)} == {UNKNOWN_ACTION}

    feedback, _, facts = build_environment_feedback(obs)
    assert facts["unclassified_steps"] == 2
    assert facts["interaction_steps"] == 0
    assert facts["movement_attempts"] == 0
    assert "no button record (not classified)" in feedback


def test_loops_are_counted_over_overworld_steps_only():
    """Advancing three dialogue boxes at one tile is not a loop."""
    rows = [_row(i, "PewterCity", 14, 6, context="dialogue", buttons=("A",)) for i in range(1, 6)]
    states = {i: _payload(context="dialogue") for i in range(1, 6)}
    _, _, facts = build_environment_feedback(_obs(rows, states, _payload(context="dialogue")))
    assert facts["loops"] == 0

    # Genuine overworld oscillation between two tiles does count.
    rows2 = [_row(i, "PewterCity", 14, 6 + (i % 2), buttons=("UP",)) for i in range(1, 6)]
    states2 = {i: _payload(y=6 + (i % 2)) for i in range(1, 6)}
    _, _, facts2 = build_environment_feedback(_obs(rows2, states2, _payload(y=6)))
    assert facts2["loops"] > 0


# ---------------------------------------------------------------------------
# Boundary progress
# ---------------------------------------------------------------------------


def test_map_entered_on_the_final_step_counts_as_progress():
    """Trajectory rows hold pre-action states, so a transition on the last step
    is invisible unless the post-window state is folded into the sequence."""
    rows = [_row(i, "PewterCity", 14, 6, buttons=("UP",)) for i in range(1, 4)]
    states = {i: _payload() for i in range(1, 4)}
    close = _payload(location="Route3", x=1, y=1, map_id=9)

    _, verdict, facts = build_environment_feedback(_obs(rows, states, close, seen={"PewterCity"}))
    assert verdict == "PROGRESS"
    assert facts["novel_locations"] == ["Route3"]
    assert facts["map_transitions"] == 1


def test_verdict_progress_on_new_milestone():
    rows = [_row(i, "PewterCity", 14, 6) for i in range(1, 4)]
    states = {i: _payload() for i in range(1, 4)}
    obs = _obs(rows, states, _payload(),
               milestones_open={"A": {"completed": True}},
               milestones_close={"A": {"completed": True}, "B": {"completed": True}})
    feedback, verdict, facts = build_environment_feedback(obs)
    assert verdict == "PROGRESS"
    assert facts["new_milestones"] == ["B"]
    assert "Newly completed during this segment: B" in feedback


def test_verdict_partial_on_displacement_without_milestone():
    rows = [_row(i, "PewterCity", 14, 6 + i, buttons=("DOWN",)) for i in range(1, 6)]
    states = {i: _payload(y=6 + i) for i in range(1, 6)}
    _, verdict, facts = build_environment_feedback(_obs(rows, states, _payload(y=12)))
    assert verdict == "PARTIAL"
    assert facts["displacement"] >= 3


def test_verdict_no_progress_when_stuck():
    rows = [_row(i, "PewterCity", 14, 6, buttons=("UP",)) for i in range(1, 11)]
    states = {i: _payload(facing="UP") for i in range(1, 11)}
    feedback, verdict, facts = build_environment_feedback(_obs(rows, states, _payload(facing="UP")))
    assert verdict == "NO_PROGRESS"
    assert facts["blocked"] == 10
    assert "**Verdict:** NO_PROGRESS" in feedback


# ---------------------------------------------------------------------------
# Battle grounding -- non-movement steps used to reach the reflector with
# nothing but the agent's own narration.
# ---------------------------------------------------------------------------


def _battle_payload(opp_hp=(20, 20), own_hp=(24, 24), species="RATTATA", level=3,
                    kind="wild", money=3000, party=1):
    payload = _payload(context="battle", money=money, party=party, hp=own_hp)
    payload["raw_state"]["game"]["battle_info"] = {
        "in_battle": True,
        "battle_type": kind,
        "player_pokemon": {"current_hp": own_hp[0], "max_hp": own_hp[1]},
        "opponent_pokemon": {
            "species": species, "level": level,
            "current_hp": opp_hp[0], "max_hp": opp_hp[1],
            "is_fainted": opp_hp[0] == 0,
        },
    }
    return payload


def test_battle_press_reports_verified_hp_change():
    """An A press in battle used to produce no verification line at all."""
    rows = [_row(1, "Route3", 5, 5, context="battle", buttons=("A",))]
    obs = _obs(
        rows,
        {1: _battle_payload(opp_hp=(20, 20), own_hp=(24, 24))},
        _battle_payload(opp_hp=(12, 20), own_hp=(21, 24)),
    )
    rec = classify_steps(obs)[0]
    assert rec["class"] == INTERACTION
    assert rec["effect"] == DAMAGE_TRADED

    trace = format_ace_trace(rows, [], classify_steps(obs))
    assert "VERIFIED:" in trace
    assert "RATTATA L3 HP 100% -> 60%" in trace
    assert "own active HP 100% -> 88%" in trace


def test_damage_dealt_and_taken_are_distinguished():
    rows = [_row(1, "Route3", 5, 5, context="battle", buttons=("A",))]

    dealt = _obs(rows, {1: _battle_payload(opp_hp=(20, 20))}, _battle_payload(opp_hp=(9, 20)))
    assert classify_steps(dealt)[0]["effect"] == DAMAGE_DEALT

    taken = _obs(rows, {1: _battle_payload(own_hp=(24, 24))}, _battle_payload(own_hp=(11, 24)))
    assert classify_steps(taken)[0]["effect"] == DAMAGE_TAKEN


def test_enemy_faint_and_battle_end_are_reported():
    rows = [
        _row(1, "Route3", 5, 5, context="battle", buttons=("A",)),
        _row(2, "Route3", 5, 5, context="battle", buttons=("A",)),
    ]
    obs = _obs(
        rows,
        {1: _battle_payload(opp_hp=(4, 20)), 2: _battle_payload(opp_hp=(0, 20))},
        _payload(location="Route3", x=5, y=5),  # back to the overworld
        seen={"PewterCity", "Route3"},  # isolate the battle-driven verdict
    )
    effects = [r["effect"] for r in classify_steps(obs)]
    assert effects == [ENEMY_FAINTED, BATTLE_ENDED]

    feedback, verdict, facts = build_environment_feedback(obs)
    assert facts["battle"]["enemy_fainted"] is True
    assert facts["battle"]["ended_at_step"] == 2
    assert "**Battle (emulator memory):**" in feedback
    assert "vs RATTATA L3" in feedback
    assert verdict == "PARTIAL"


def test_battle_section_summarises_the_fight():
    rows = [_row(i, "Route3", 5, 5, context="battle", buttons=("A",)) for i in range(1, 4)]
    obs = _obs(
        rows,
        {
            1: _battle_payload(opp_hp=(20, 20), own_hp=(24, 24)),
            2: _battle_payload(opp_hp=(13, 20), own_hp=(20, 24)),
            3: _battle_payload(opp_hp=(6, 20), own_hp=(15, 24)),
        },
        _battle_payload(opp_hp=(2, 20), own_hp=(9, 24)),
    )
    feedback, _, facts = build_environment_feedback(obs)
    battle = facts["battle"]
    assert battle["steps"] == 3
    assert battle["battle_type"] == "wild"
    assert battle["damage_dealt_steps"] == 3
    assert battle["damage_taken_steps"] == 3
    assert battle["ended_at_step"] is None
    assert "Opponent HP: 100% -> 10%" in feedback
    assert "Own active HP: 100% -> 38%" in feedback
    assert "Battle was still in progress at the end of the segment" in feedback


def test_sitting_in_a_battle_doing_nothing_is_not_progress():
    """Being *in* a battle is not engagement; the old survived_battle rule
    scored a completely idle battle screen as PARTIAL."""
    rows = [_row(i, "Route3", 5, 5, context="battle", buttons=("A",)) for i in range(1, 6)]
    states = {i: _battle_payload(opp_hp=(20, 20), own_hp=(24, 24)) for i in range(1, 6)}
    feedback, verdict, facts = build_environment_feedback(
        _obs(rows, states, _battle_payload(opp_hp=(20, 20), own_hp=(24, 24)))
    )
    assert verdict == "NO_PROGRESS"
    assert facts["battle"]["idle_steps"] == 5
    assert facts["longest_idle_run"] == 5
    assert "no observable effect: 5" in feedback


def test_catching_a_pokemon_is_detected():
    rows = [_row(1, "Route3", 5, 5, context="battle", buttons=("A",))]
    obs = _obs(rows, {1: _battle_payload(party=1)}, _battle_payload(party=2))
    assert classify_steps(obs)[0]["effect"] == CAUGHT
    _, verdict, facts = build_environment_feedback(obs)
    assert facts["battle"]["caught"] is True
    assert verdict == "PARTIAL"


def test_no_battle_means_no_battle_section():
    rows = [_row(1, "PewterCity", 14, 6, buttons=("UP",))]
    feedback, _, facts = build_environment_feedback(_obs(rows, {1: _payload()}, _payload(y=5)))
    assert "**Battle" not in feedback
    assert facts["battle"]["steps"] == 0


# ---------------------------------------------------------------------------
# The context-independent stuck detector
# ---------------------------------------------------------------------------


def test_repeated_presses_with_no_effect_are_reported():
    """The generalisation of a blocked move: an agent lost in a menu, or
    re-reading one dialogue box, is otherwise invisible."""
    rows = [_row(i, "PewterCity", 14, 6, context="menu", buttons=("A",)) for i in range(1, 8)]
    states = {i: _payload(context="menu") for i in range(1, 8)}
    feedback, verdict, facts = build_environment_feedback(
        _obs(rows, states, _payload(context="menu"))
    )
    assert facts["longest_idle_run"] == 7
    assert facts["idle_context"] == "menu"
    assert verdict == "NO_PROGRESS"
    assert "Longest run of steps with NO observable state change: 7 (context: menu)" in feedback

    trace = format_ace_trace(rows, [], classify_steps(_obs(rows, states, _payload(context="menu"))))
    assert "VERIFIED: no observable change in emulator state after this press" in trace


def test_dialogue_that_advances_is_not_idle():
    rows = [_row(i, "PewterCity", 14, 6, context="dialogue", buttons=("A",)) for i in range(1, 3)]
    obs = _obs(
        rows,
        {1: _payload(context="dialogue"), 2: _payload(context="dialogue")},
        _payload(context="overworld"),  # text box closed
    )
    effects = [r["effect"] for r in classify_steps(obs)]
    assert effects[-1] == CONTEXT_CHANGED
    _, _, facts = build_environment_feedback(obs)
    assert facts["longest_idle_run"] < 2


# ---------------------------------------------------------------------------
# Dialogue grounding
# ---------------------------------------------------------------------------


def _dialogue_payload(text):
    payload = _payload(context="dialogue")
    payload["raw_state"]["game"]["dialog_text"] = text
    return payload


OAK_LINES = [
    "PROF.OAK: Hello there!",
    "PROF.OAK: Welcome to the world of POKEMON!",
    "PROF.OAK: My name is OAK.",
    "PROF.OAK: People call me the POKEMON PROF.",
]


def test_advancing_a_conversation_is_not_reported_as_stuck():
    """Regression: mashing A through a normal conversation used to report
    no_observable_change on every step, i.e. a false stall on the single most
    common productive action in the game."""
    rows = [_row(i, "PewterCity", 14, 6, context="dialogue", buttons=("A",)) for i in range(1, 5)]
    states = {i + 1: _dialogue_payload(line) for i, line in enumerate(OAK_LINES)}
    obs = _obs(rows, states, _dialogue_payload("PROF.OAK: This world is inhabited by POKEMON."))

    assert [r["effect"] for r in classify_steps(obs)] == [DIALOGUE_ADVANCED] * 4

    feedback, _, facts = build_environment_feedback(obs)
    assert facts["longest_idle_run"] == 0
    assert facts["dialogue"] == {
        "steps": 4, "advanced": 4, "closed": 0, "idle": 0, "unverified": 0
    }
    assert "4 advanced the text" in feedback


def test_stuck_on_one_text_box_is_still_detected():
    """The complement: a prompt the agent cannot dismiss must still read as idle."""
    prompt = "PROF.OAK: Are you a boy or a girl?"
    rows = [_row(i, "PewterCity", 14, 6, context="dialogue", buttons=("A",)) for i in range(1, 5)]
    states = {i: _dialogue_payload(prompt) for i in range(1, 5)}
    obs = _obs(rows, states, _dialogue_payload(prompt))

    assert [r["effect"] for r in classify_steps(obs)] == [NO_OBSERVABLE_CHANGE] * 4
    feedback, verdict, facts = build_environment_feedback(obs)
    assert facts["longest_idle_run"] == 4
    assert facts["dialogue"]["idle"] == 4
    assert verdict == "NO_PROGRESS"
    assert "0 advanced the text" in feedback


def test_screen_text_reaches_the_reflector():
    """The reflector never sees the screenshot, so the on-screen text is the
    only way it can know what a conversation actually said."""
    rows = [_row(1, "PewterCity", 14, 6, context="dialogue", buttons=("A",))]
    obs = _obs(rows, {1: _dialogue_payload(OAK_LINES[0])}, _dialogue_payload(OAK_LINES[1]))
    trace = format_ace_trace(rows, [], classify_steps(obs))
    assert 'SCREEN TEXT (emulator memory): "PROF.OAK: Hello there!"' in trace
    assert "VERIFIED: dialogue advanced (on-screen text changed)" in trace


def test_dialogue_without_captured_text_is_unverified_not_idle():
    """Regression from run 20260730_205220_ace.

    Runs use --no-ocr (matching the paper runs), which suppresses dialog_text
    while leaving dialogue *context* detection intact. Every dialogue step then
    landed in `idle`, the reflector read 12 idle steps as evidence of
    button-mashing, and the curator wrote three timing bullets out of a
    measurement artifact -- the FiNER context-pollution failure mode.
    """
    rows = [_row(i, "PewterCity", 14, 6, context="dialogue", buttons=("A",)) for i in range(1, 5)]
    # context "dialogue" via game_state, but no dialog_text (OCR disabled)
    no_text = _payload(context="dialogue")
    no_text["raw_state"]["game"].pop("dialog_text", None)
    no_text["raw_state"]["game"]["game_state"] = "dialog"
    states = {i: no_text for i in range(1, 5)}
    obs = _obs(rows, states, no_text)

    assert [r["effect"] for r in classify_steps(obs)] == [DIALOGUE_UNVERIFIED] * 4

    feedback, _, facts = build_environment_feedback(obs)
    assert facts["dialogue"]["unverified"] == 4
    assert facts["dialogue"]["idle"] == 0
    # Crucially, unverifiable steps must not feed the stuck detector.
    assert facts["longest_idle_run"] == 0
    assert "it is UNKNOWN whether the press advanced the conversation" in feedback
    assert "Do NOT read these as wasted or mistimed inputs" in feedback


def test_unverified_dialogue_is_labelled_unverified_in_the_trace():
    rows = [_row(1, "PewterCity", 14, 6, context="dialogue", buttons=("A",))]
    no_text = _payload(context="dialogue")
    no_text["raw_state"]["game"].pop("dialog_text", None)
    no_text["raw_state"]["game"]["game_state"] = "dialog"
    trace = format_ace_trace(rows, [], classify_steps(_obs(rows, {1: no_text}, no_text)))
    assert "UNVERIFIED: in a text box, on-screen text was not captured" in trace
    assert "VERIFIED: no observable change" not in trace


def test_unresolved_opponent_species_is_skipped():
    """The species byte reads 0 on the battle-entry frame, rendering as
    "Species_0"; the reflector should be shown the resolved name instead."""
    rows = [_row(i, "Route3", 5, 5, context="battle", buttons=("A",)) for i in range(1, 3)]
    entering = _battle_payload(opp_hp=(20, 20), species="Species", level=None)
    entering["raw_state"]["game"]["battle_info"]["opponent_pokemon"]["species"] = "Species_0"
    entering["raw_state"]["game"]["battle_info"]["opponent_pokemon"]["level"] = None
    obs = _obs(
        rows,
        {1: entering, 2: _battle_payload(opp_hp=(14, 20), species="PIDGEY", level=5)},
        _battle_payload(opp_hp=(9, 20), species="PIDGEY", level=5),
    )
    feedback, _, facts = build_environment_feedback(obs)
    assert facts["battle"]["opponent"] == "PIDGEY L5"
    assert "Species_0" not in feedback


def test_dialogue_text_whitespace_jitter_is_not_an_advance():
    """read_screen_text pads the tilemap; redraw jitter must not look like new text."""
    rows = [_row(1, "PewterCity", 14, 6, context="dialogue", buttons=("A",))]
    obs = _obs(
        rows,
        {1: _dialogue_payload("PROF.OAK:  Hello   there!")},
        _dialogue_payload("PROF.OAK: Hello there!\n"),
    )
    assert classify_steps(obs)[0]["effect"] == NO_OBSERVABLE_CHANGE


def test_red_dialog_context_spelling_is_normalised():
    """Red's memory reader says "dialog"; the classifier must not treat that as
    a different context from "dialogue"."""
    raw = {"raw_state": {"player": {"location": "PewterCity", "position": {"x": 1, "y": 1}, "party": []},
                         "game": {"game_state": "dialog"}, "map": {"id": 1}}}
    assert summarize_game_state(raw)["context"] == "dialogue"


def test_inventory_and_pokedex_changes_count_as_observable():
    rows = [_row(1, "PewterCity", 14, 6, buttons=("A",))]
    before, after = _payload(), _payload()
    before["raw_state"]["game"]["item_count"] = 3
    after["raw_state"]["game"]["item_count"] = 4
    assert classify_steps(_obs(rows, {1: before}, after))[0]["effect"] == STATE_CHANGED


# ---------------------------------------------------------------------------
# Mid-window events
# ---------------------------------------------------------------------------


def test_party_wipe_is_detected_mid_window_not_only_at_close():
    """After a blackout the game revives the party at a Pokemon Center, so a
    close-only reading would report 'no wipe' on the very window it happened."""
    rows = [_row(i, "Route3", 5, 5, context="battle", buttons=("A",)) for i in range(1, 4)]
    states = {
        1: _payload(location="Route3", hp=(8, 20), context="battle"),
        2: _payload(location="Route3", hp=(0, 20), context="battle"),  # wiped here
        3: _payload(location="PewterCity", hp=(20, 20)),
    }
    close = _payload(location="PewterCity", hp=(20, 20))  # healed by the Center

    feedback, _, facts = build_environment_feedback(_obs(rows, states, close))
    assert facts["party_wiped"] is True
    assert facts["min_hp_fraction"] == 0.0
    assert "Party wipe (blackout) during segment: YES" in feedback


def test_min_hp_is_tracked_across_the_window():
    rows = [_row(i, "Route3", 5, 5) for i in range(1, 4)]
    states = {1: _payload(hp=(20, 20)), 2: _payload(hp=(3, 20)), 3: _payload(hp=(15, 20))}
    _, _, facts = build_environment_feedback(_obs(rows, states, _payload(hp=(15, 20))))
    assert facts["min_hp_fraction"] == 0.15


# ---------------------------------------------------------------------------
# The two things feedback must never do
# ---------------------------------------------------------------------------


def test_hardcoded_outcome_field_is_never_shown_to_the_reflector():
    rows = [_row(i, "PewterCity", 14, 6) for i in range(1, 6)]
    states = {i: _payload() for i in range(1, 6)}
    obs = _obs(rows, states, _payload())
    feedback, _, _ = build_environment_feedback(obs)
    trace = format_ace_trace(rows, [], classify_steps(obs))

    for text in (feedback, trace):
        assert "objectives_completed" not in text
        assert "'success': True" not in text
        assert '"success": true' not in text.lower()


def test_feedback_construction_makes_no_llm_calls():
    vlm = MagicMock()
    rows = [_row(i, "PewterCity", 14, 6) for i in range(1, 6)]
    states = {i: _payload() for i in range(1, 6)}
    obs = _obs(rows, states, _payload())
    build_environment_feedback(obs)
    format_ace_trace(rows, [], classify_steps(obs))
    format_actions_summary(rows, [], classify_steps(obs))
    assert vlm.get_text_query.call_count == 0
    assert vlm.get_query.call_count == 0


def test_feedback_states_that_it_contains_no_model_judgement():
    rows = [_row(1, "PewterCity", 14, 6)]
    feedback, _, _ = build_environment_feedback(_obs(rows, {1: _payload()}, _payload()))
    assert "no model judgement" in feedback


# ---------------------------------------------------------------------------
# Milestones
# ---------------------------------------------------------------------------


def test_read_milestones_from_cache_file(tmp_path, monkeypatch):
    payload = {
        "milestones": {
            "GOT_STARTER": {"completed": True, "timestamp": 100},
            "NOT_YET": {"completed": False},
        }
    }
    (tmp_path / "milestones_progress.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        "utils.data_persistence.run_data_manager.get_cache_path",
        lambda name: tmp_path / name,
    )
    assert set(read_milestones()) == {"GOT_STARTER"}


def test_read_milestones_fallback_matches_the_real_endpoint_schema(tmp_path, monkeypatch):
    """server/app.py's get_progress_summary nests this under "progress".
    The previous test mocked a flat schema and hid a dead fallback."""
    monkeypatch.setattr(
        "utils.data_persistence.run_data_manager.get_cache_path",
        lambda name: tmp_path / name,  # missing file
    )
    adapter = MagicMock()
    adapter.call_tool.return_value = {
        "success": True,
        "progress": {
            "milestones_completed": ["GOT_STARTER", "PEWTER_CITY"],
            "total_milestones_completed": 2,
        },
    }
    assert set(read_milestones(adapter)) == {"GOT_STARTER", "PEWTER_CITY"}
    adapter.call_tool.assert_called_once_with("get_progress_summary", {"compact": True})


def test_read_milestones_returns_empty_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "utils.data_persistence.run_data_manager.get_cache_path",
        lambda name: tmp_path / name,
    )
    adapter = MagicMock()
    adapter.call_tool.side_effect = RuntimeError("server down")
    assert read_milestones(adapter) == {}


# ---------------------------------------------------------------------------
# Trace formatting
# ---------------------------------------------------------------------------


def test_trace_includes_press_buttons_and_tool_results():
    """PromptOptimizer's formatter drops press_buttons entirely; ours must not,
    since it is this scaffold's whole action space."""
    rows = [_row(1, "PewterCity", 14, 6)]
    history = [
        {
            "step": 1,
            "llm_response": "walking north",
            "tool_calls": [
                {
                    "name": "press_buttons",
                    "args": {"buttons": ["UP"], "reasoning": "go north"},
                    "result": '{"success": true, "frames": 12}',
                }
            ],
            "start_coords": (14, 6),
            "end_coords": (14, 5),
        }
    ]
    trace = format_ace_trace(rows, history)
    assert "press_buttons" in trace
    assert '"UP"' in trace
    assert "frames" in trace  # the tool result survived


def test_trace_only_asserts_blocked_when_verified():
    rows = [_row(i, "PewterCity", 14, 6, buttons=("UP",)) for i in range(1, 4)]
    states = {i: _payload(facing="UP") for i in range(1, 4)}
    trace = format_ace_trace(rows, [], classify_steps(_obs(rows, states, _payload(facing="UP"))))
    assert "position/map/facing all unchanged (blocked)" in trace

    dlg_rows = [_row(i, "PewterCity", 14, 6, context="dialogue", buttons=("DOWN",)) for i in range(1, 4)]
    dlg_states = {i: _payload(context="dialogue") for i in range(1, 4)}
    dlg_trace = format_ace_trace(
        dlg_rows, [], classify_steps(_obs(dlg_rows, dlg_states, _payload(context="dialogue")))
    )
    assert "blocked" not in dlg_trace
    assert "menu/dialogue/battle navigation" in dlg_trace


def test_trace_labels_agent_reasoning_as_unverified():
    trace = format_ace_trace([_row(1, "PewterCity", 14, 6)], [])
    assert "the agent's own claim, not verified" in trace


def test_trace_and_actions_handle_empty_input():
    assert "No trajectory recorded" in format_ace_trace([], [])
    assert "No actions recorded" in format_actions_summary([], [])


def test_actions_summary_one_line_per_step_with_verified_outcome():
    rows = [_row(1, "PewterCity", 14, 6, buttons=("UP",)), _row(2, "PewterCity", 14, 5, buttons=("UP",))]
    states = {1: _payload(y=6), 2: _payload(y=5)}
    classified = classify_steps(_obs(rows, states, _payload(y=4)))
    summary = format_actions_summary(rows, [], classified)
    assert len(summary.split("\n")) == 2
    assert "press_buttons(['UP'])" in summary
    assert "moved" in summary
