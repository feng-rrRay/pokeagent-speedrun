"""Grounded environment feedback for the ACE reflector.

ACE's reflector normally sees a ground-truth answer (or a unit-test report).  A
reset-free Pokemon run has neither, so this module builds the substitute the
paper licenses in section 4.3: "signals naturally available during execution".

Every line emitted here is a deterministic read of emulator memory or of the
recorded trajectory.  There is **no model judgement anywhere in this file** --
that is the whole point.  The paper's own FiNER counterexample (label-free
adaptation dropping *below* baseline when there is no environment to check
against) is the failure mode this avoids.

Because the prompt tells the reflector this block is *verified fact*, every
claim here has to be one we can actually check from two adjacent state samples
plus the recorded buttons.  In particular a "blocked move" is only asserted
when the agent pressed a direction **in the overworld** and the position, map
and facing all stayed put; a direction pressed in a menu, dialogue or battle is
cursor movement, and a position that held still while the context flipped is a
triggered event (a wild encounter, say), not a bump.

Two things are deliberately withheld from the reflector:

* ``outcome`` from ``trajectory_history.jsonl``.  ``PokeAgent._log_trajectory_for_step``
  hardcodes ``{"success": True}`` on every step, so surfacing it would be fake
  ground truth handed to the diagnostician.
* Anything derived from an LLM (the agent's own reasoning is shown as a trace,
  clearly labelled as the agent's claim, never as fact).

Known limitation, stated rather than papered over: one ``press_buttons`` call
may contain several buttons, and we only sample state at step boundaries.  A
context change part-way through such a sequence is attributed to the whole
step, so classification is at step granularity and claims nothing finer.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_REASONING_CHARS = 400
_MAX_TOOL_RESULT_CHARS = 200
_MAX_SCREEN_TEXT_CHARS = 220

DIRECTIONAL_BUTTONS = {"UP", "DOWN", "LEFT", "RIGHT"}
OVERWORLD = "overworld"

# Step classes
UI_NAVIGATION = "ui_navigation"
INTERACTION = "interaction"
MOVEMENT = "movement"
# No button record survived for the step (e.g. the trajectory row was missing).
# Distinct from INTERACTION on purpose: claiming "interaction or wait" when we
# never saw the buttons would be the same unverified assertion this module
# exists to avoid.
UNKNOWN_ACTION = "unknown_action"

# Movement outcomes
MOVED = "moved"
TURNED = "turned"
TRIGGERED = "triggered"
BLOCKED = "blocked"
INDETERMINATE = "indeterminate"

# Effects of a non-movement step (battle, dialogue, menu, wait). Movement gets
# an outcome; everything else gets one of these, so that battle and dialogue
# steps are grounded too rather than leaving the reflector with nothing but the
# agent's own narration.
DAMAGE_DEALT = "damage_dealt"
DAMAGE_TAKEN = "damage_taken"
DAMAGE_TRADED = "damage_traded"
ENEMY_FAINTED = "enemy_fainted"
BATTLE_STARTED = "battle_started"
BATTLE_ENDED = "battle_ended"
CAUGHT = "caught"
DIALOGUE_ADVANCED = "dialogue_advanced"
# In a text box the on-screen text is the only change-signal we have. Runs use
# --no-ocr (to match the paper runs), which suppresses `dialog_text`, so we
# usually cannot tell an advanced conversation from a stalled one. Saying
# "nothing happened" in that case would be the same unverified assertion this
# module exists to prevent -- and it demonstrably misleads the Reflector, which
# reads it as evidence of button-mashing and curates timing advice from it.
DIALOGUE_UNVERIFIED = "dialogue_unverified"
CONTEXT_CHANGED = "context_changed"
STATE_CHANGED = "state_changed"
NO_OBSERVABLE_CHANGE = "no_observable_change"
EFFECT_UNKNOWN = "effect_unknown"

_DAMAGE_EFFECTS = {DAMAGE_DEALT, DAMAGE_TAKEN, DAMAGE_TRADED, ENEMY_FAINTED}


# ---------------------------------------------------------------------------
# Reading state
# ---------------------------------------------------------------------------


def _as_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def summarize_game_state(payload: Any) -> Dict[str, Any]:
    """Pull the grounded scalars out of a ``get_game_state`` payload.

    Accepts either the MCP wrapper (which carries ``raw_state``) or a bare raw
    state.  Everything is read straight from emulator memory by the server.
    """
    wrapper = _as_dict(payload)
    raw = wrapper.get("raw_state") if isinstance(wrapper.get("raw_state"), dict) else wrapper

    player = raw.get("player") or {}
    game = raw.get("game") or {}
    map_info = raw.get("map") or {}

    position = player.get("position") or {}
    if isinstance(position, dict):
        coords = (position.get("x"), position.get("y"))
    elif isinstance(position, (list, tuple)) and len(position) >= 2:
        coords = (position[0], position[1])
    else:
        coords = (None, None)

    badges = game.get("badges")
    if isinstance(badges, list):
        badge_count = len(badges)
    elif isinstance(badges, int):
        badge_count = badges
    else:
        badge_count = None

    party = player.get("party") or game.get("party") or []
    if not isinstance(party, list):
        party = []

    hp_fractions = []
    for mon in party:
        if not isinstance(mon, dict):
            continue
        max_hp = mon.get("max_hp")
        cur_hp = mon.get("current_hp")
        if isinstance(max_hp, int) and max_hp > 0 and isinstance(cur_hp, int):
            hp_fractions.append(max(0.0, min(1.0, cur_hp / max_hp)))

    battle_raw = game.get("battle_info")
    battle = battle_raw if isinstance(battle_raw, dict) else {}
    opponent = battle.get("opponent_pokemon") if isinstance(battle.get("opponent_pokemon"), dict) else {}
    active = battle.get("player_pokemon") if isinstance(battle.get("player_pokemon"), dict) else {}

    opponent_label = None
    if opponent.get("species"):
        level = opponent.get("level")
        opponent_label = f"{opponent['species']}" + (f" L{level}" if level else "")

    return {
        "location": player.get("location") or "Unknown",
        "coords": coords,
        "map_id": map_info.get("id"),
        "facing": player.get("facing"),
        "badge_count": badge_count,
        "party_size": len(party),
        "lowest_hp_fraction": min(hp_fractions) if hp_fractions else None,
        "all_fainted": bool(hp_fractions) and all(f <= 0.0 for f in hp_fractions),
        "money": game.get("money"),
        "context": _context_from_raw(game),
        # On-screen text while a text box is up (read_screen_text). This is the
        # only way to tell "the A press advanced the conversation" from "the A
        # press did nothing", and it is the one thing on screen a text-only
        # reflector would otherwise never see.
        "dialog_text": _normalize_text(game.get("dialog_text")),
        "item_count": game.get("item_count"),
        "pokedex_seen": game.get("pokedex_seen"),
        "pokedex_caught": game.get("pokedex_caught"),
        # Battle scalars -- everything the reflector needs to verify what a
        # press did inside a fight (red_memory_reader.read_battle_details).
        "in_battle": bool(game.get("is_in_battle")) or bool(battle.get("in_battle")),
        "battle_type": battle.get("battle_type"),
        "opponent": opponent_label,
        "opponent_hp_fraction": _hp_fraction(opponent),
        "opponent_fainted": bool(opponent.get("is_fainted")) if opponent else False,
        "active_hp_fraction": _hp_fraction(active),
        "_present": bool(raw),
    }


def _normalize_text(text: Any) -> Optional[str]:
    """Collapse whitespace so redraw jitter is not mistaken for new text."""
    if not text:
        return None
    collapsed = " ".join(str(text).split())
    return collapsed or None


def _hp_fraction(mon: Dict[str, Any]) -> Optional[float]:
    if not isinstance(mon, dict):
        return None
    cur, mx = mon.get("current_hp"), mon.get("max_hp")
    if isinstance(cur, int) and isinstance(mx, int) and mx > 0:
        return max(0.0, min(1.0, cur / mx))
    return None


def _context_from_raw(game: Dict[str, Any]) -> str:
    if game.get("is_in_battle"):
        return "battle"
    if game.get("dialog_text"):
        return "dialogue"
    if game.get("menu_active"):
        return "menu"
    state = game.get("game_state")
    if isinstance(state, str) and state:
        # Red's memory reader reports "dialog"; normalise so the two spellings
        # do not look like two different contexts.
        return "dialogue" if state == "dialog" else state
    return OVERWORLD


def read_milestones(mcp_adapter=None) -> Dict[str, Dict[str, Any]]:
    """Completed milestones, read from the server-written cache file.

    The server and the agent share ``.pokeagent_cache/{run_id}/``, so this needs
    no HTTP call.  Falls back to the ``get_progress_summary`` MCP endpoint, which
    is already in ``MCPToolAdapter.endpoint_map`` -- note that this exposes no
    new tool to the policy model, it is an out-of-band read by the reflector.
    """
    try:
        from utils.data_persistence.run_data_manager import get_cache_path

        path = get_cache_path("milestones_progress.json")
        if path and path.exists():
            data = json.loads(path.read_text())
            milestones = data.get("milestones", {})
            if isinstance(milestones, dict):
                return {
                    mid: info
                    for mid, info in milestones.items()
                    if isinstance(info, dict) and info.get("completed")
                }
    except Exception as exc:
        logger.debug("Could not read milestones file: %s", exc)

    if mcp_adapter is not None:
        try:
            result = _as_dict(mcp_adapter.call_tool("get_progress_summary", {"compact": True}))
            # The endpoint nests this under "progress" (server/app.py get_progress_summary).
            progress = result.get("progress")
            completed = None
            if isinstance(progress, dict):
                completed = progress.get("milestones_completed")
            if completed is None:
                completed = result.get("milestones_completed") or result.get("completed_milestones")
            if isinstance(completed, list):
                return {str(m): {"completed": True} for m in completed}
        except Exception as exc:
            logger.debug("Could not read progress summary: %s", exc)

    return {}


# ---------------------------------------------------------------------------
# Window observation
# ---------------------------------------------------------------------------


@dataclass
class WindowObservation:
    """Everything the reflector's grounded feedback is computed from.

    ``step_states[N]`` is the emulator state sampled *before* step N acted, so
    ``step_states[N + 1]`` doubles as step N's post-action state.  ``state_close``
    supplies the post-action state for the final step.  This is what lets us
    verify movement rather than infer it from consecutive pre-states.
    """

    start_step: int
    end_step: int
    trajectory_rows: List[Dict[str, Any]] = field(default_factory=list)
    history_entries: List[Dict[str, Any]] = field(default_factory=list)
    step_states: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    state_close: Dict[str, Any] = field(default_factory=dict)
    milestones_open: Dict[str, Any] = field(default_factory=dict)
    milestones_close: Dict[str, Any] = field(default_factory=dict)
    locations_seen_before: set = field(default_factory=set)

    @property
    def state_open(self) -> Dict[str, Any]:
        if self.start_step in self.step_states:
            return self.step_states[self.start_step]
        if self.step_states:
            return self.step_states[min(self.step_states)]
        return {}


def _coords_of(row: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
    coords = row.get("player_coords") or (row.get("pre_state") or {}).get("player_coords")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return (coords[0], coords[1])
    return None


def _location_of(row: Dict[str, Any]) -> str:
    return row.get("location") or (row.get("pre_state") or {}).get("location") or "Unknown"


def _state_from_trajectory_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback pre-state when the per-step payload was not captured."""
    pre = row.get("pre_state") or {}
    return {
        "location": _location_of(row),
        "coords": _coords_of(row),
        "map_id": pre.get("map_id"),
        "facing": None,
        "badge_count": None,
        "party_size": None,
        "lowest_hp_fraction": None,
        "all_fainted": False,
        "money": None,
        "context": pre.get("context") or "unknown",
        "_present": bool(pre),
    }


def _buttons_for_step(traj_row: Dict[str, Any], hist_entry: Dict[str, Any]) -> List[str]:
    """Buttons pressed on a step, from the trajectory row or history entry.

    ``_log_trajectory_for_step`` only strips ``screenshot_base64`` from tool
    args, so ``buttons`` survives into the JSONL.
    """
    for source in (traj_row, hist_entry):
        if not isinstance(source, dict):
            continue
        calls = source.get("tool_calls")
        if not calls:
            action = source.get("action")
            if isinstance(action, dict):
                calls = action.get("tool_calls")
        for call in calls or []:
            if not isinstance(call, dict) or call.get("name") != "press_buttons":
                continue
            buttons = (call.get("args") or {}).get("buttons")
            if isinstance(buttons, str):
                return [buttons.upper()]
            if isinstance(buttons, (list, tuple)):
                return [str(b).upper() for b in buttons]
    return []


# ---------------------------------------------------------------------------
# Step classification -- the core of the "verified" claim
# ---------------------------------------------------------------------------


def classify_steps(obs: WindowObservation) -> List[Dict[str, Any]]:
    """Classify every step in the window as UI nav, interaction, or movement.

    Movement outcomes are decided from the pre/post state pair, never from the
    agent's own account of what it did.
    """
    traj_by_step = {
        row["step"]: row
        for row in obs.trajectory_rows or []
        if isinstance(row.get("step"), int)
    }
    hist_by_step = {
        entry["step"]: entry
        for entry in obs.history_entries or []
        if isinstance(entry.get("step"), int)
    }

    steps = sorted(set(traj_by_step) | set(obs.step_states))
    steps = [s for s in steps if obs.start_step <= s <= obs.end_step]

    def state_at(step: int) -> Dict[str, Any]:
        if step in obs.step_states:
            return obs.step_states[step]
        if step in traj_by_step:
            return _state_from_trajectory_row(traj_by_step[step])
        return {}

    classified: List[Dict[str, Any]] = []
    for step in steps:
        pre = state_at(step)
        if step + 1 <= obs.end_step:
            post = state_at(step + 1)
        else:
            post = obs.state_close or state_at(step + 1)

        buttons = _buttons_for_step(traj_by_step.get(step, {}), hist_by_step.get(step, {}))
        has_directional = any(b in DIRECTIONAL_BUTTONS for b in buttons)
        pre_context = pre.get("context") or "unknown"

        record = {
            "step": step,
            "buttons": buttons,
            "pre": pre,
            "post": post,
            "context": pre_context,
            "outcome": None,
            "effect": None,
        }

        if not buttons:
            record["class"] = UNKNOWN_ACTION
            record["effect"] = _nonmovement_effect(pre, post)
        elif not has_directional:
            record["class"] = INTERACTION
            record["effect"] = _nonmovement_effect(pre, post)
        elif pre_context != OVERWORLD:
            # A direction inside a menu, dialogue or battle is cursor/target
            # selection, not locomotion.
            record["class"] = UI_NAVIGATION
            record["effect"] = _nonmovement_effect(pre, post)
        else:
            record["class"] = MOVEMENT
            record["outcome"] = _movement_outcome(pre, post)

        classified.append(record)

    return classified


def _movement_outcome(pre: Dict[str, Any], post: Dict[str, Any]) -> str:
    if not post or not post.get("_present", True):
        return INDETERMINATE

    pre_xy, post_xy = pre.get("coords"), post.get("coords")
    known = (
        isinstance(pre_xy, tuple) and isinstance(post_xy, tuple)
        and pre_xy[0] is not None and post_xy[0] is not None
    )

    if post.get("map_id") is not None and pre.get("map_id") is not None:
        if post["map_id"] != pre["map_id"]:
            return MOVED
    if post.get("location") != pre.get("location"):
        return MOVED
    if known and pre_xy != post_xy:
        return MOVED
    if not known:
        return INDETERMINATE

    # Position held. Decide why.
    if (post.get("context") or OVERWORLD) != OVERWORLD:
        # Walked into grass / a trainer's line of sight / a sign: a real event.
        return TRIGGERED

    pre_facing, post_facing = pre.get("facing"), post.get("facing")
    if pre_facing is not None and post_facing is not None:
        if pre_facing != post_facing:
            return TURNED
        return BLOCKED

    # Without a facing reading we cannot separate a turn-in-place from a bump,
    # and asserting "blocked" would break the verified-fact guarantee.
    return INDETERMINATE


_OBSERVABLE_KEYS = (
    "location", "coords", "map_id", "context", "badge_count", "party_size",
    "money", "lowest_hp_fraction", "in_battle", "opponent", "opponent_hp_fraction",
    "active_hp_fraction", "dialog_text", "item_count", "pokedex_seen", "pokedex_caught",
)


def _dropped(before: Optional[float], after: Optional[float]) -> bool:
    return before is not None and after is not None and after < before


def _nonmovement_effect(pre: Dict[str, Any], post: Dict[str, Any]) -> str:
    """What a non-movement press verifiably did.

    Battle and dialogue steps make up a large share of play, and without this
    the reflector receives nothing but the agent's own account of them.
    ``NO_OBSERVABLE_CHANGE`` is the context-independent generalisation of a
    blocked move: it is how an agent lost in a menu, or re-reading one dialogue
    box, becomes visible at all.
    """
    if not pre or not post or not post.get("_present", True):
        return EFFECT_UNKNOWN

    was_battling, is_battling = bool(pre.get("in_battle")), bool(post.get("in_battle"))

    if was_battling and not is_battling:
        return BATTLE_ENDED
    if not was_battling and is_battling:
        return BATTLE_STARTED

    if was_battling and is_battling:
        if post.get("opponent_fainted") and not pre.get("opponent_fainted"):
            return ENEMY_FAINTED
        dealt = _dropped(pre.get("opponent_hp_fraction"), post.get("opponent_hp_fraction"))
        taken = _dropped(pre.get("active_hp_fraction"), post.get("active_hp_fraction"))
        if dealt and taken:
            return DAMAGE_TRADED
        if dealt:
            return DAMAGE_DEALT
        if taken:
            return DAMAGE_TAKEN

    pre_party, post_party = pre.get("party_size"), post.get("party_size")
    if isinstance(pre_party, int) and isinstance(post_party, int) and post_party > pre_party:
        return CAUGHT

    if post.get("context") != pre.get("context"):
        return CONTEXT_CHANGED

    # Advancing a text box is the single most common productive action in the
    # game. Without comparing the on-screen text it looks identical to being
    # stuck, which would be a false stall report on ordinary conversation.
    pre_text, post_text = pre.get("dialog_text"), post.get("dialog_text")
    if pre_text and post_text and pre_text != post_text:
        return DIALOGUE_ADVANCED

    if any(pre.get(key) != post.get(key) for key in _OBSERVABLE_KEYS):
        return STATE_CHANGED

    # Nothing we track moved -- but inside a text box with no captured text,
    # "nothing we track" is not the same as "nothing happened".
    if pre.get("context") == "dialogue" and not pre.get("dialog_text") and not post.get("dialog_text"):
        return DIALOGUE_UNVERIFIED

    return NO_OBSERVABLE_CHANGE


def _pct(value: Optional[float]) -> Optional[str]:
    return f"{value * 100:.0f}%" if value is not None else None


def _hp_phrase(before: Optional[float], after: Optional[float]) -> Optional[str]:
    if before is None or after is None or before == after:
        return None
    return f"{_pct(before)} -> {_pct(after)}"


# ---------------------------------------------------------------------------
# Window statistics
# ---------------------------------------------------------------------------


def _window_stats(obs: WindowObservation, classified: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {MOVED: 0, TURNED: 0, TRIGGERED: 0, BLOCKED: 0, INDETERMINATE: 0}
    classes = {MOVEMENT: 0, UI_NAVIGATION: 0, INTERACTION: 0, UNKNOWN_ACTION: 0}
    effects: Dict[str, int] = {}
    for rec in classified:
        classes[rec["class"]] = classes.get(rec["class"], 0) + 1
        if rec["outcome"]:
            counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
        if rec.get("effect"):
            effects[rec["effect"]] = effects.get(rec["effect"], 0) + 1

    # Longest run of consecutive blocked movement attempts.
    longest_blocked, run, stall_at = 0, 0, None
    for rec in classified:
        if rec["outcome"] == BLOCKED:
            run += 1
            if run > longest_blocked:
                longest_blocked = run
                stall_at = (rec["pre"].get("location"), rec["pre"].get("coords"), rec["pre"].get("facing"))
        else:
            run = 0

    # Loops are measured over the overworld subsequence only: cycling A through
    # three dialogue boxes at one tile is not a loop.
    overworld_positions = [
        (rec["pre"].get("location"), rec["pre"].get("coords"))
        for rec in classified
        if rec["context"] == OVERWORLD and rec["pre"].get("coords")
    ]
    loops = sum(
        1
        for i in range(2, len(overworld_positions))
        if overworld_positions[i] == overworld_positions[i - 2]
    )

    # The full location sequence includes post-action states, so a map entered
    # on the final step of the window is visible.
    sequence = [obs.step_states[s] for s in sorted(obs.step_states)]
    if obs.state_close:
        sequence.append(obs.state_close)
    if not sequence:
        sequence = [_state_from_trajectory_row(r) for r in obs.trajectory_rows or []]

    locations, map_transitions = [], 0
    for idx, state in enumerate(sequence):
        loc = state.get("location") or "Unknown"
        if idx and loc != (sequence[idx - 1].get("location") or "Unknown"):
            map_transitions += 1
        if loc not in locations:
            locations.append(loc)

    displacement = None
    first_xy = next((s.get("coords") for s in sequence if s.get("coords") and s["coords"][0] is not None), None)
    last_xy = next(
        (s.get("coords") for s in reversed(sequence) if s.get("coords") and s["coords"][0] is not None), None
    )
    same_map = bool(sequence) and (sequence[0].get("location") == sequence[-1].get("location"))
    if first_xy and last_xy and same_map and all(isinstance(v, int) for v in (*first_xy, *last_xy)):
        displacement = abs(last_xy[0] - first_xy[0]) + abs(last_xy[1] - first_xy[1])

    contexts: Dict[str, int] = {}
    for rec in classified:
        contexts[rec["context"]] = contexts.get(rec["context"], 0) + 1

    # Resource extremes across the whole window, not just its endpoints -- a
    # blackout heals the party at a Pokemon Center, so sampling only at close
    # would make the most consequential event in the game invisible.
    hp_values = [s.get("lowest_hp_fraction") for s in sequence if s.get("lowest_hp_fraction") is not None]
    money_values = [s.get("money") for s in sequence if isinstance(s.get("money"), int)]
    wiped = any(s.get("all_fainted") for s in sequence)

    # Context-independent stuck detector: the generalisation of "blocked" that
    # also catches an agent lost in a menu or re-reading one dialogue box.
    longest_idle, idle_run, idle_context = 0, 0, None
    for rec in classified:
        stuck = rec.get("outcome") == BLOCKED or rec.get("effect") == NO_OBSERVABLE_CHANGE
        if stuck:
            idle_run += 1
            if idle_run > longest_idle:
                longest_idle, idle_context = idle_run, rec["context"]
        else:
            idle_run = 0

    dialogue_steps = [rec for rec in classified if rec["context"] == "dialogue"]

    return {
        "classes": classes,
        "outcomes": counts,
        "effects": effects,
        "battle": _battle_stats(classified, effects),
        "dialogue": {
            "steps": len(dialogue_steps),
            "advanced": sum(1 for rec in dialogue_steps if rec.get("effect") == DIALOGUE_ADVANCED),
            "closed": sum(1 for rec in dialogue_steps if rec.get("effect") == CONTEXT_CHANGED),
            "idle": sum(1 for rec in dialogue_steps if rec.get("effect") == NO_OBSERVABLE_CHANGE),
            "unverified": sum(1 for rec in dialogue_steps if rec.get("effect") == DIALOGUE_UNVERIFIED),
        },
        "longest_idle_run": longest_idle,
        "idle_context": idle_context,
        "longest_blocked_run": longest_blocked,
        "stall_at": stall_at,
        "loops": loops,
        "locations": locations,
        "map_transitions": map_transitions,
        "displacement": displacement,
        "contexts": contexts,
        "min_hp_fraction": min(hp_values) if hp_values else None,
        "min_money": min(money_values) if money_values else None,
        "party_wiped": wiped,
        "steps": len(classified),
    }


def _first_resolved_opponent(labels: List[Optional[str]]) -> Optional[str]:
    """First opponent label whose species actually resolved.

    ``read_battle_details`` renders an unknown species id as ``Species_<id>``,
    and the id reads as 0 on the frame the battle starts. Showing that to the
    reflector is noise it cannot act on.
    """
    for label in labels:
        if label and not label.startswith("Species_"):
            return label
    return next((label for label in labels if label), None)


def _opponent_label(pre: Dict[str, Any], post: Dict[str, Any]) -> str:
    return _first_resolved_opponent([pre.get("opponent"), post.get("opponent")]) or "opponent"


def _battle_stats(classified: List[Dict[str, Any]], effects: Dict[str, int]) -> Dict[str, Any]:
    """Summarise any fighting that happened in the window."""
    in_battle = [
        rec for rec in classified
        if rec["pre"].get("in_battle") or rec["context"] == "battle"
    ]
    if not in_battle:
        return {"steps": 0}

    # The species byte is not populated on the battle-entry frame, so the first
    # sample often reads "Species_0". Prefer the first label that resolved.
    opponents = []
    for rec in in_battle:
        opponents.append(rec["pre"].get("opponent"))
        opponents.append((rec.get("post") or {}).get("opponent"))
    opp_hp = [rec["pre"].get("opponent_hp_fraction") for rec in in_battle
              if rec["pre"].get("opponent_hp_fraction") is not None]
    own_hp = [rec["pre"].get("active_hp_fraction") for rec in in_battle
              if rec["pre"].get("active_hp_fraction") is not None]

    # Fold in the post-state of the final battle step so the last exchange counts.
    last = in_battle[-1].get("post") or {}
    if last.get("opponent_hp_fraction") is not None:
        opp_hp.append(last["opponent_hp_fraction"])
    if last.get("active_hp_fraction") is not None:
        own_hp.append(last["active_hp_fraction"])

    ended = next((rec["step"] for rec in classified if rec.get("effect") == BATTLE_ENDED), None)

    return {
        "steps": len(in_battle),
        "battle_type": next((rec["pre"].get("battle_type") for rec in in_battle
                             if rec["pre"].get("battle_type")), None),
        "opponent": _first_resolved_opponent(opponents),
        "opponent_hp_first": opp_hp[0] if opp_hp else None,
        "opponent_hp_last": opp_hp[-1] if opp_hp else None,
        "own_hp_first": own_hp[0] if own_hp else None,
        "own_hp_last": own_hp[-1] if own_hp else None,
        # Counted from the HP deltas directly, not from the primary-effect
        # histogram: a press that faints the opponent is labelled ENEMY_FAINTED
        # but still dealt damage.
        "damage_dealt_steps": sum(
            1 for rec in in_battle
            if _dropped(rec["pre"].get("opponent_hp_fraction"),
                        (rec.get("post") or {}).get("opponent_hp_fraction"))
        ),
        "damage_taken_steps": sum(
            1 for rec in in_battle
            if _dropped(rec["pre"].get("active_hp_fraction"),
                        (rec.get("post") or {}).get("active_hp_fraction"))
        ),
        "enemy_fainted": effects.get(ENEMY_FAINTED, 0) > 0,
        "caught": effects.get(CAUGHT, 0) > 0,
        "ended_at_step": ended,
        "idle_steps": sum(
            1 for rec in in_battle if rec.get("effect") == NO_OBSERVABLE_CHANGE
        ),
    }


def _fmt_coords(coords) -> str:
    if not coords or coords[0] is None or coords[1] is None:
        return "(?,?)"
    return f"({coords[0]},{coords[1]})"


def _fmt_delta(before, after, label: str) -> str:
    if before is None and after is None:
        return f"{label}: unknown"
    if before == after:
        return f"{label}: {after} (unchanged)"
    return f"{label}: {after} (was {before})"


def build_environment_feedback(obs: WindowObservation) -> Tuple[str, str, Dict[str, Any]]:
    """Build the grounded feedback string, a verdict, and the raw facts.

    Verdict:
      ``PROGRESS``    -- a new milestone completed, or a location never visited
                         before in this run was reached.
      ``PARTIAL``     -- no new milestone, but the agent changed maps, ended up
                         a meaningful distance away, or fought and survived.
      ``NO_PROGRESS`` -- none of the above.
    """
    classified = classify_steps(obs)
    stats = _window_stats(obs, classified)

    new_milestones = sorted(set(obs.milestones_close) - set(obs.milestones_open))
    total_before = len(obs.milestones_open)
    latest_before = _latest_milestone(obs.milestones_open)

    distinct_locations = [loc for loc in stats["locations"] if loc and loc != "Unknown"]
    novel_locations = [loc for loc in distinct_locations if loc not in obs.locations_seen_before]

    open_state = obs.state_open or {}
    close_state = obs.state_close or {}

    battle = stats["battle"]
    # Being *in* a battle is not engagement -- an agent can sit on a battle
    # screen for twenty steps achieving nothing. Require verified damage or a
    # resolved fight.
    engaged_in_battle = bool(
        battle.get("damage_dealt_steps")
        or battle.get("damage_taken_steps")
        or battle.get("enemy_fainted")
        or battle.get("caught")
        or battle.get("ended_at_step") is not None
    )

    if new_milestones or novel_locations:
        verdict = "PROGRESS"
    elif stats["map_transitions"] > 0 or (stats["displacement"] or 0) >= 3 or engaged_in_battle:
        verdict = "PARTIAL"
    else:
        verdict = "NO_PROGRESS"

    outcomes = stats["outcomes"]
    classes = stats["classes"]
    move_attempts = classes.get(MOVEMENT, 0)

    move_line = (
        f"- {move_attempts} movement attempts (directional press while in the overworld)"
        f" -> {outcomes.get(MOVED, 0)} moved, {outcomes.get(TURNED, 0)} turned in place,"
        f" {outcomes.get(BLOCKED, 0)} blocked, {outcomes.get(TRIGGERED, 0)} triggered an event"
    )
    if outcomes.get(INDETERMINATE):
        move_line += (
            f", {outcomes[INDETERMINATE]} position unchanged but turn-vs-bump indeterminate"
            " (no facing reading)"
        )

    stall_at = stats["stall_at"]
    stall_line = f"- Longest consecutive blocked run: {stats['longest_blocked_run']}"
    if stall_at and stall_at[0]:
        stall_line += f" (at {stall_at[0]} {_fmt_coords(stall_at[1])}, facing {stall_at[2]})"

    hp_line = (
        f"Lowest party HP fraction seen during the segment: {stats['min_hp_fraction']:.2f}"
        if stats["min_hp_fraction"] is not None
        else "Lowest party HP fraction: unknown"
    )

    context_line = ", ".join(f"{k} {v}" for k, v in sorted(stats["contexts"].items())) or "unknown"
    num_steps = max(1, stats["steps"])

    lines = [
        f"**Verdict:** {verdict}",
        "",
        "**Verified milestones (emulator memory):**",
        f"- Completed before this segment: {total_before}"
        + (f" (latest: {latest_before})" if latest_before else ""),
        f"- Newly completed during this segment: {', '.join(new_milestones) if new_milestones else 'none'}",
        f"- {_fmt_delta(open_state.get('badge_count'), close_state.get('badge_count'), 'Badges')}",
        "",
        "**Location (emulator memory, sampled before and after every step):**",
        f"- Start: {open_state.get('location', 'Unknown')} {_fmt_coords(open_state.get('coords'))}"
        f" [map {open_state.get('map_id')}]  ->  End: {close_state.get('location', 'Unknown')}"
        f" {_fmt_coords(close_state.get('coords'))} [map {close_state.get('map_id')}]",
        f"- Distinct locations visited: {', '.join(distinct_locations) if distinct_locations else 'unknown'}",
        f"- Locations never visited before in this run: {', '.join(novel_locations) if novel_locations else 'none'}",
        f"- Map transitions: {stats['map_transitions']}"
        f"  |  Net tile displacement: {stats['displacement'] if stats['displacement'] is not None else 'n/a (changed map)'}",
        "",
        "**What the button presses actually did (verified against the state before and after each step):**",
        move_line,
        stall_line,
        f"- Overworld revisits (same tile two overworld steps apart): {stats['loops']}",
        f"- Non-movement steps: {classes.get(UI_NAVIGATION, 0)} menu/dialogue/battle navigation,"
        f" {classes.get(INTERACTION, 0)} interaction or wait"
        + (
            f", {classes.get(UNKNOWN_ACTION, 0)} with no button record (not classified)"
            if classes.get(UNKNOWN_ACTION)
            else ""
        ),
        f"- Longest run of steps with NO observable state change: {stats['longest_idle_run']}"
        + (f" (context: {stats['idle_context']})" if stats["idle_context"] else ""),
        *_dialogue_lines(stats["dialogue"]),
        *_battle_lines(battle),
        "",
        "**Party and resources (emulator memory):**",
        f"- {_fmt_delta(open_state.get('party_size'), close_state.get('party_size'), 'Party size')}  |  {hp_line}",
        f"- {_fmt_delta(open_state.get('money'), close_state.get('money'), 'Money')}"
        f"  |  Party wipe (blackout) during segment: {'YES' if stats['party_wiped'] else 'no'}",
        "",
        f"**Game context over the {num_steps} steps:** {context_line}",
        "",
        "**Note:** every line above is computed deterministically from emulator memory reads and the",
        "recorded button presses. It contains no model judgement and no hidden ground-truth answer.",
    ]

    facts = {
        "verdict": verdict,
        "new_milestones": new_milestones,
        "milestones_total": len(obs.milestones_close),
        "novel_locations": novel_locations,
        "distinct_locations": distinct_locations,
        "badge_count": close_state.get("badge_count"),
        "movement_attempts": move_attempts,
        "moved": outcomes.get(MOVED, 0),
        "turned": outcomes.get(TURNED, 0),
        "blocked": outcomes.get(BLOCKED, 0),
        "triggered": outcomes.get(TRIGGERED, 0),
        "indeterminate": outcomes.get(INDETERMINATE, 0),
        "ui_navigation_steps": classes.get(UI_NAVIGATION, 0),
        "interaction_steps": classes.get(INTERACTION, 0),
        "unclassified_steps": classes.get(UNKNOWN_ACTION, 0),
        "effects": stats["effects"],
        "battle": battle,
        "dialogue": stats["dialogue"],
        "longest_idle_run": stats["longest_idle_run"],
        "idle_context": stats["idle_context"],
        "longest_blocked_run": stats["longest_blocked_run"],
        "loops": stats["loops"],
        "map_transitions": stats["map_transitions"],
        "displacement": stats["displacement"],
        "party_wiped": stats["party_wiped"],
        "min_hp_fraction": stats["min_hp_fraction"],
        "contexts": stats["contexts"],
        "steps": stats["steps"],
    }

    return "\n".join(lines), verdict, facts


def _dialogue_lines(dialogue: Dict[str, Any]) -> List[str]:
    """One line summarising text-box handling, omitted when there was none."""
    if not dialogue.get("steps"):
        return []
    line = (
        f"- Dialogue: {dialogue['steps']} steps in a text box -> {dialogue['advanced']} advanced"
        f" the text, {dialogue['closed']} closed the box, {dialogue['idle']} changed nothing"
    )
    # The buckets do not partition the steps: a press can also start a battle or
    # otherwise leave the text box. Say so rather than leaving the reader to
    # wonder where the remainder went.
    unverified = dialogue.get("unverified", 0)
    other = (
        dialogue["steps"] - dialogue["advanced"] - dialogue["closed"]
        - dialogue["idle"] - unverified
    )
    if other > 0:
        line += f", {other} led elsewhere (e.g. started a battle)"
    lines = [line]
    if unverified:
        lines.append(
            f"  NOTE: on-screen text was not captured this run, so for {unverified} of those steps"
            " it is UNKNOWN whether the press advanced the conversation. Do NOT read these as"
            " wasted or mistimed inputs -- there is no evidence either way."
        )
    return lines


def _battle_lines(battle: Dict[str, Any]) -> List[str]:
    """Battle section of the feedback block, omitted when no fighting happened."""
    if not battle.get("steps"):
        return []

    kind = battle.get("battle_type") or "unknown-type"
    opponent = battle.get("opponent") or "an unidentified opponent"
    lines = ["", "**Battle (emulator memory):**",
             f"- {battle['steps']} steps in a {kind} battle vs {opponent}"]

    opp = _hp_phrase(battle.get("opponent_hp_first"), battle.get("opponent_hp_last"))
    if opp:
        lines.append(f"- Opponent HP: {opp}" + (" (fainted)" if battle.get("enemy_fainted") else ""))
    elif battle.get("opponent_hp_first") is not None:
        lines.append(f"- Opponent HP: {_pct(battle['opponent_hp_first'])} (unchanged)")

    own = _hp_phrase(battle.get("own_hp_first"), battle.get("own_hp_last"))
    if own:
        lines.append(f"- Own active HP: {own}")
    elif battle.get("own_hp_first") is not None:
        lines.append(f"- Own active HP: {_pct(battle['own_hp_first'])} (unchanged)")

    lines.append(
        f"- Presses that dealt damage: {battle.get('damage_dealt_steps', 0)}"
        f"  |  that took damage: {battle.get('damage_taken_steps', 0)}"
        f"  |  with no observable effect: {battle.get('idle_steps', 0)}"
    )
    if battle.get("caught"):
        lines.append("- A Pokemon was added to the party during this battle")
    if battle.get("ended_at_step") is not None:
        lines.append(f"- Battle ended at step {battle['ended_at_step']}")
    else:
        lines.append("- Battle was still in progress at the end of the segment")
    return lines


def _latest_milestone(milestones: Dict[str, Dict[str, Any]]) -> Optional[str]:
    latest, latest_ts = None, -1.0
    for mid, info in milestones.items():
        ts = info.get("timestamp", 0) if isinstance(info, dict) else 0
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = 0.0
        if ts >= latest_ts:
            latest, latest_ts = mid, ts
    return latest


# ---------------------------------------------------------------------------
# Trace formatting
# ---------------------------------------------------------------------------


_OUTCOME_NOTE = {
    BLOCKED: "VERIFIED: directional press in the overworld, position/map/facing all unchanged (blocked)",
    TURNED: "VERIFIED: turned in place without moving",
    TRIGGERED: "VERIFIED: position unchanged but the game context changed (triggered an event)",
    MOVED: None,
    INDETERMINATE: "position unchanged; turn-vs-bump could not be determined (no facing reading)",
}


def _effect_note(rec: Dict[str, Any]) -> Optional[str]:
    """Verified description of what a non-movement press did.

    Built from the pre/post state pair rather than a static table so battle
    steps carry the actual HP numbers.
    """
    effect = rec.get("effect")
    if not effect or effect == EFFECT_UNKNOWN:
        return None

    pre = rec.get("pre") or {}
    post = rec.get("post") or {}
    parts: List[str] = []

    if effect == NO_OBSERVABLE_CHANGE:
        return "VERIFIED: no observable change in emulator state after this press"

    if effect == DIALOGUE_UNVERIFIED:
        return (
            "UNVERIFIED: in a text box, on-screen text was not captured this run, so whether "
            "this press advanced the conversation cannot be determined either way"
        )

    if effect in _DAMAGE_EFFECTS:
        opponent = _opponent_label(pre, post)
        dealt = _hp_phrase(pre.get("opponent_hp_fraction"), post.get("opponent_hp_fraction"))
        taken = _hp_phrase(pre.get("active_hp_fraction"), post.get("active_hp_fraction"))
        if dealt:
            parts.append(f"{opponent} HP {dealt}")
        if taken:
            parts.append(f"own active HP {taken}")
        if effect == ENEMY_FAINTED:
            parts.append(f"{opponent} fainted")
    elif effect == BATTLE_STARTED:
        parts.append(f"battle started vs {_opponent_label(pre, post)}")
    elif effect == BATTLE_ENDED:
        parts.append(f"{pre.get('battle_type') or 'unknown'} battle ended")
        if isinstance(pre.get("money"), int) and isinstance(post.get("money"), int):
            if pre["money"] != post["money"]:
                parts.append(f"money {pre['money']} -> {post['money']}")
    elif effect == CAUGHT:
        parts.append(f"party size {pre.get('party_size')} -> {post.get('party_size')}")
    elif effect == DIALOGUE_ADVANCED:
        parts.append("dialogue advanced (on-screen text changed)")
    elif effect == CONTEXT_CHANGED:
        parts.append(f"game context {pre.get('context')} -> {post.get('context')}")
    elif effect == STATE_CHANGED:
        parts.append("emulator state changed")

    return "VERIFIED: " + "; ".join(parts) if parts else None


def format_ace_trace(
    trajectory_rows: List[Dict[str, Any]],
    history_entries: List[Dict[str, Any]],
    classified: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render the segment as a step-by-step trace for the reflector.

    Joins the persisted trajectory rows (grounded: location, coords, map id,
    game context) with the in-memory conversation history (which is the only
    place tool *results* survive -- ``_log_trajectory_for_step`` writes only
    ``{name, args}``).

    This is adapted rather than reusing ``PromptOptimizer._format_trajectories_for_analysis``:
    that formatter whitelists skill/subagent/memory tool args and drops
    ``press_buttons`` entirely, which is this scaffold's whole action space.

    When ``classified`` is supplied, per-step outcome notes come from the
    verified pre/post state comparison rather than from a guess.
    """
    by_step: Dict[int, Dict[str, Any]] = {}
    for row in trajectory_rows or []:
        step = row.get("step")
        if isinstance(step, int):
            by_step.setdefault(step, {})["traj"] = row
    for entry in history_entries or []:
        step = entry.get("step")
        if isinstance(step, int):
            by_step.setdefault(step, {})["hist"] = entry

    if not by_step:
        return "(No trajectory recorded for this segment)"

    outcome_by_step = {rec["step"]: rec for rec in classified or []}

    blocks: List[str] = []
    for step, parts in sorted(by_step.items()):
        traj = parts.get("traj") or {}
        hist = parts.get("hist") or {}

        pre = traj.get("pre_state") or {}
        loc = _location_of(traj)
        coords = _coords_of(traj)
        ctx = pre.get("context") or "unknown"

        block = [f"### Step {step} | {loc} {_fmt_coords(coords)} | map {pre.get('map_id')} | {ctx}"]

        # The reflector is text-only and never sees the screenshot, so the
        # on-screen text is the only way it can know what a conversation said.
        rec_for_text = (outcome_by_step.get(step) or {}).get("pre") or {}
        screen_text = rec_for_text.get("dialog_text")
        if screen_text:
            if len(screen_text) > _MAX_SCREEN_TEXT_CHARS:
                screen_text = screen_text[:_MAX_SCREEN_TEXT_CHARS] + "..."
            block.append(f'SCREEN TEXT (emulator memory): "{screen_text}"')

        reasoning = (traj.get("reasoning") or hist.get("llm_response") or "").strip()
        if len(reasoning) > _MAX_REASONING_CHARS:
            reasoning = reasoning[:_MAX_REASONING_CHARS] + "..."
        if reasoning:
            block.append(f"AGENT REASONING (the agent's own claim, not verified): {reasoning}")

        tool_lines = []
        for call in hist.get("tool_calls") or traj.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name", "?")
            args = {k: v for k, v in (call.get("args") or {}).items() if k != "screenshot_base64"}
            line = f"- {name}({json.dumps(args, default=str)})"
            result = call.get("result")
            if result is not None:
                text = result if isinstance(result, str) else json.dumps(result, default=str)
                if len(text) > _MAX_TOOL_RESULT_CHARS:
                    text = text[:_MAX_TOOL_RESULT_CHARS] + "..."
                line += f" -> {text}"
            tool_lines.append(line)
        if tool_lines:
            block.append("TOOLS:")
            block.extend(tool_lines)

        rec = outcome_by_step.get(step)
        if rec:
            if rec["class"] == MOVEMENT:
                note = _OUTCOME_NOTE.get(rec.get("outcome"))
                if note:
                    block.append(note)
            else:
                if rec["class"] == UI_NAVIGATION:
                    block.append("(directional press used for menu/dialogue/battle navigation, not movement)")
                note = _effect_note(rec)
                if note:
                    block.append(note)

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


def format_actions_summary(
    trajectory_rows: List[Dict[str, Any]],
    history_entries: List[Dict[str, Any]],
    classified: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """One line per step: what was pressed and what the emulator did about it."""
    by_step: Dict[int, Dict[str, Any]] = {}
    for row in trajectory_rows or []:
        step = row.get("step")
        if isinstance(step, int):
            by_step.setdefault(step, {})["traj"] = row
    for entry in history_entries or []:
        step = entry.get("step")
        if isinstance(step, int):
            by_step.setdefault(step, {})["hist"] = entry

    if not by_step:
        return "(No actions recorded for this segment)"

    outcome_by_step = {rec["step"]: rec for rec in classified or []}

    lines = []
    for step, parts in sorted(by_step.items()):
        traj = parts.get("traj") or {}
        hist = parts.get("hist") or {}

        buttons = _buttons_for_step(traj, hist)
        label = f"press_buttons({buttons})" if buttons else str(hist.get("action") or traj.get("action") or "?")

        rec = outcome_by_step.get(step)
        if rec:
            start = rec["pre"].get("coords")
            end = rec["post"].get("coords") if rec.get("post") else None
            move = f"{_fmt_coords(start)}->{_fmt_coords(end)}" if end else _fmt_coords(start)
            tag = rec.get("outcome") or rec["class"]
        else:
            start = hist.get("start_coords") or _coords_of(traj)
            end = hist.get("end_coords")
            move = f"{_fmt_coords(start)}->{_fmt_coords(end)}" if end else _fmt_coords(start)
            tag = ""

        lines.append(f"step {step}: {label} {move}" + (f" [{tag}]" if tag else ""))

    return "\n".join(lines)
