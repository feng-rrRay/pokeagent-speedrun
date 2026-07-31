"""ACE controller: the window loop, persistence, and the one object PokeAgent holds.

ACE's unit of adaptation is one episode: the Generator runs a task to
completion, the Reflector diagnoses the whole trajectory, the Curator emits
deltas.  A Pokemon Red run is reset-free with no episode boundary, so the
pseudo-episode here is a fixed, non-overlapping window of ``window_steps``
agent steps (default 100).

Why 100: it matches the ContinualHarness arm this is compared against, which
runs with ``--optimization-window-length 100`` and settles to an evolution
every ``STABLE_FREQUENCY = 100`` steps (``agents/utils/harness_evolver.py``).
Holding the adaptation cadence and the analysed trajectory span equal across
the two arms keeps the comparison about *what* each method does with a segment
rather than how often it gets to look at one.

The trade-off, stated rather than hidden: AppWorld episodes -- the paper's own
agentic benchmark -- average ~19.9 generator steps (Appendix A.3, Table 12), so
a 100-step window puts roughly 5x more play between adaptations than the regime
ACE was tuned in.  ``--ace-window-steps 20`` recovers that regime if the
comparison ever needs it.

One consequence of the larger window: ``PokeAgent.conversation_history`` is
capped at ``ACTION_HISTORY_WINDOW`` (20) entries, so raw tool *results* survive
only for the tail of the window.  Verification is unaffected -- step
classification reads ``_step_states`` (kept for the whole window) and the
buttons recorded in the trajectory rows -- and the truncated part is the
near-constant ``{"success": true, "frames_advanced": N}`` echo.

The windows are non-overlapping on purpose: overlapping ones would double-count
the same evidence into the helpful/harmful counters that drive both curation and
pruning.

Everything in this module is best-effort.  A failure anywhere leaves the
playbook untouched and the game loop running.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.ace.curator import AceCurator
from agents.ace.feedback import (
    WindowObservation,
    build_environment_feedback,
    classify_steps,
    format_actions_summary,
    format_ace_trace,
    read_milestones,
    summarize_game_state,
)
from agents.ace.playbook import (
    apply_curator_operations,  # noqa: F401  (re-exported for tests)
    count_tokens,
    empty_playbook,
    extract_cited_bullet_ids,
    extract_playbook_bullets,
    get_next_global_id,
    get_playbook_stats,
    playbook_bullet_ids,
    prune_to_budget,
    update_bullet_counts,
)
from agents.ace.prompts import ACE_PLAYBOOK_BLOCK, ACE_TASK_STATEMENT
from agents.ace.reflector import AceReflector, filter_tags_to_playbook

logger = logging.getLogger(__name__)

DEFAULT_ACE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "window_steps": 100,
    "playbook_path": None,
    "budget_tokens": 6000,
    "min_bullets": 10,
    "unused_grace_windows": 5,
    "curator_frequency": 1,
    "warmup_steps": 0,
    "freeze_playbook": False,
    "max_cited_bullets": 40,
    "max_steps": None,
}


class AceController:
    """Owns the playbook and drives the reflect -> curate -> prune cycle."""

    def __init__(self, vlm, mcp_adapter=None, run_data_manager=None, config: Optional[Dict] = None,
                 game_name: str = "Pokemon Red"):
        cfg = dict(DEFAULT_ACE_CONFIG)
        cfg.update(config or {})
        self.config = cfg

        self.enabled = bool(cfg["enabled"])
        self.window_steps = max(1, int(cfg["window_steps"]))
        self.budget_tokens = int(cfg["budget_tokens"])
        self.min_bullets = int(cfg["min_bullets"])
        self.unused_grace_windows = int(cfg["unused_grace_windows"])
        self.curator_frequency = max(1, int(cfg["curator_frequency"]))
        self.warmup_steps = max(0, int(cfg["warmup_steps"]))
        self.freeze_playbook = bool(cfg["freeze_playbook"])
        self.max_cited_bullets = int(cfg["max_cited_bullets"])
        self.game_name = game_name

        self.mcp_adapter = mcp_adapter
        self.run_manager = run_data_manager

        # A separate, tool-free VLM so get_text_query returns a plain string.
        # Same construction PromptOptimizer uses.
        from utils.agent_infrastructure.vlm_backends import VLM

        self.text_vlm = VLM(
            backend=vlm.backend_type,
            model_name=vlm.model_name,
            tools=None,
            system_instruction=None,
        )
        self.reflector = AceReflector(self.text_vlm, game_name=game_name)
        self.curator = AceCurator(self.text_vlm, game_name=game_name)

        self.ace_dir = self._resolve_ace_dir()

        # --- persistent state -------------------------------------------------
        self.playbook = empty_playbook()
        self.next_global_id = 1
        self.window_index = 0
        self.seen_locations: set = set()
        self.bullet_added_window: Dict[str, int] = {}
        self.previous_reflection = "(none)"
        self._load_state()

        # --- per-window buffers ----------------------------------------------
        # last_window_end restarts at 0 every process because PokeAgent's step
        # counter does; window_index is what survives a resume.
        self.last_window_end = 0
        self._last_seen_step = 0
        self._cited: Dict[int, List[str]] = {}
        # step -> emulator state sampled BEFORE that step acted. run() fetches
        # get_game_state before each step, so step N+1's sample doubles as
        # step N's post-action state. Keeping the whole sequence is what lets
        # movement be verified rather than inferred, and what makes mid-window
        # events (blackouts, boundary map transitions) observable at all.
        self._step_states: Dict[int, Dict[str, Any]] = {}
        self._window_open_milestones: Optional[Dict[str, Any]] = None

        logger.info(
            "ACE controller ready: window=%d steps, budget=%d tokens, curator every %d window(s), "
            "%d bullets in playbook%s",
            self.window_steps,
            self.budget_tokens,
            self.curator_frequency,
            len(playbook_bullet_ids(self.playbook)),
            " [FROZEN]" if self.freeze_playbook else "",
        )

    # ------------------------------------------------------------------
    # Paths and persistence
    # ------------------------------------------------------------------

    def _resolve_ace_dir(self) -> Optional[Path]:
        try:
            from utils.data_persistence.run_data_manager import get_cache_path

            path = get_cache_path("ace")
            path.mkdir(parents=True, exist_ok=True)
            (path / "playbook_history").mkdir(exist_ok=True)
            return path
        except Exception as exc:
            logger.warning("ACE: could not resolve cache directory (%s); running without persistence", exc)
            return None

    def _path(self, name: str) -> Optional[Path]:
        return self.ace_dir / name if self.ace_dir else None

    def _load_state(self) -> None:
        """Restore the playbook and counters, or warm-start from a file."""
        playbook_file = self._path("playbook.md")
        loaded_from = None

        if playbook_file and playbook_file.exists():
            try:
                self.playbook = playbook_file.read_text()
                loaded_from = str(playbook_file)
            except Exception as exc:
                logger.warning("ACE: failed to read %s (%s); starting fresh", playbook_file, exc)

        elif self.config.get("playbook_path"):
            warm = Path(self.config["playbook_path"])
            if warm.exists():
                try:
                    self.playbook = warm.read_text()
                    loaded_from = f"{warm} (warm start)"
                except Exception as exc:
                    logger.warning("ACE: failed to read warm-start playbook %s (%s)", warm, exc)
            else:
                logger.warning("ACE: warm-start playbook not found at %s", warm)

        self.next_global_id = get_next_global_id(self.playbook)

        state_file = self._path("ace_state.json")
        if state_file and state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                self.window_index = int(state.get("window_index", 0))
                self.seen_locations = set(state.get("seen_locations", []))
                self.bullet_added_window = {
                    str(k): int(v) for k, v in (state.get("bullet_added_window") or {}).items()
                }
                self.previous_reflection = state.get("previous_reflection") or "(none)"
                self.next_global_id = max(self.next_global_id, int(state.get("next_global_id", 1)))
            except Exception as exc:
                logger.warning("ACE: failed to restore state (%s); starting fresh", exc)

        if loaded_from:
            logger.info(
                "ACE: loaded playbook from %s (%d bullets, next id %d, window %d)",
                loaded_from,
                len(playbook_bullet_ids(self.playbook)),
                self.next_global_id,
                self.window_index,
            )

    def _save_state(self) -> None:
        if not self.ace_dir:
            return
        try:
            self._path("playbook.md").write_text(self.playbook)
            self._path("ace_state.json").write_text(
                json.dumps(
                    {
                        "window_index": self.window_index,
                        "next_global_id": self.next_global_id,
                        "seen_locations": sorted(self.seen_locations),
                        "bullet_added_window": self.bullet_added_window,
                        "previous_reflection": self.previous_reflection,
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            logger.warning("ACE: failed to persist state: %s", exc)

    def _append_jsonl(self, name: str, record: Dict[str, Any]) -> None:
        path = self._path(name)
        if not path:
            return
        try:
            with open(path, "a") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning("ACE: failed to append to %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Generator-side injection
    # ------------------------------------------------------------------

    def get_playbook_block(self) -> str:
        """The playbook block injected at the top of every step's prompt."""
        if not self.enabled:
            return ""
        return ACE_PLAYBOOK_BLOCK.format(playbook=self.playbook)

    # ------------------------------------------------------------------
    # Step hook
    # ------------------------------------------------------------------

    def on_step_complete(self, step: int, response_text: str, game_state_json: Any = None) -> None:
        """Called once per completed agent step from ``PokeAgent.run``."""
        if not self.enabled:
            return

        # The black-frame path returns success without claiming a step, so the
        # same step number can arrive twice. Ignore replays.
        if step <= self._last_seen_step:
            return
        self._last_seen_step = step

        if self._window_open_milestones is None:
            self._window_open_milestones = read_milestones(self.mcp_adapter)

        # Sample the pre-action state for this step. Deliberately NOT folded
        # into seen_locations here: doing so would mark the window's opening
        # location as "already seen" before the novelty check runs, which would
        # make a map entered at the end of the previous window invisible in
        # both windows. seen_locations is updated in _run_window instead.
        self._step_states[step] = summarize_game_state(game_state_json)

        cited = extract_cited_bullet_ids(response_text or "")
        if cited:
            self._cited[step] = cited

        if step < self.warmup_steps:
            self.last_window_end = step
            self._step_states = {step: self._step_states[step]}
            return

        if step - self.last_window_end >= self.window_steps:
            try:
                self._run_window(step)
            except Exception as exc:
                logger.error("ACE window at step %d failed: %s", step, exc, exc_info=True)
            finally:
                self.last_window_end = step
                self._cited = {}
                self._step_states = {}
                self._window_open_milestones = None

    # ------------------------------------------------------------------
    # The ACE cycle
    # ------------------------------------------------------------------

    def _run_window(self, end_step: int) -> None:
        start_step = self.last_window_end + 1
        num_steps = end_step - self.last_window_end
        logger.info("ACE: reflecting on steps %d-%d (window %d)", start_step, end_step, self.window_index)

        obs = self._collect_observation(start_step, end_step, num_steps)
        classified = classify_steps(obs)
        feedback, verdict, facts = build_environment_feedback(obs)

        # Only now, after the novelty check has read locations_seen_before.
        for loc in facts.get("distinct_locations", []):
            if loc and loc != "Unknown":
                self.seen_locations.add(loc)

        known_ids = playbook_bullet_ids(self.playbook)
        cited = self._collect_cited(known_ids)
        citation_rate = len(self._cited) / max(1, num_steps)
        bullets_used = extract_playbook_bullets(self.playbook, cited)

        trace = format_ace_trace(obs.trajectory_rows, obs.history_entries, classified)
        actions = format_actions_summary(obs.trajectory_rows, obs.history_entries, classified)
        task = ACE_TASK_STATEMENT.format(game_name=self.game_name)

        # --- Reflector -----------------------------------------------------
        raw_reflection, reflection_json, bullet_tags = self.reflector.reflect(
            task=task,
            trace=trace,
            actions=actions,
            environment_feedback=feedback,
            bullets_used=bullets_used,
            window_start=start_step,
            window_end=end_step,
            previous_reflection=self.previous_reflection,
        )

        # --- Counter layer (deterministic) ---------------------------------
        # Counters are rendered into the injected bullets, so they influence the
        # policy: a frozen playbook has to freeze these too, or the "static
        # context" control condition is not actually static.
        tags = filter_tags_to_playbook(bullet_tags, known_ids)
        if tags and not self.freeze_playbook:
            self.playbook = update_bullet_counts(self.playbook, tags)

        self._append_jsonl(
            "reflections.jsonl",
            {
                "window": self.window_index,
                "start_step": start_step,
                "end_step": end_step,
                "verdict": verdict,
                "facts": facts,
                "citation_rate": round(citation_rate, 3),
                "cited_bullets": cited,
                "bullet_tags": tags,
                "reflection": reflection_json or raw_reflection,
            },
        )

        # --- Curator (delta operations) ------------------------------------
        added: List[Dict[str, Any]] = []
        operations: List[Dict[str, Any]] = []
        should_curate = (
            not self.freeze_playbook
            and raw_reflection
            and self.window_index % self.curator_frequency == 0
        )
        if should_curate:
            playbook_before = self.playbook
            self.playbook, self.next_global_id, operations, added = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=raw_reflection,
                question_context=self._segment_context(start_step, end_step, obs, facts),
                current_step=self.window_index + 1,
                total_samples=self._total_segments(),
                token_budget=self.budget_tokens,
                playbook_stats=get_playbook_stats(self.playbook),
                next_global_id=self.next_global_id,
            )
            for entry in added:
                self.bullet_added_window[entry["id"]] = self.window_index
            if added:
                self._append_jsonl(
                    "curations.jsonl",
                    {
                        "window": self.window_index,
                        "end_step": end_step,
                        "operations": operations,
                        "added": added,
                        "playbook_tokens_before": count_tokens(playbook_before),
                    },
                )

        # --- Budget enforcement (deterministic pruning) ---------------------
        # budget_tokens <= 0 disables enforcement entirely, reproducing the
        # reference implementation, whose 80000-token budget is advisory (it is
        # interpolated into the curator prompt and never enforced in code).
        pruned: List[Dict[str, Any]] = []
        if not self.freeze_playbook and self.budget_tokens > 0:
            self.playbook, pruned = prune_to_budget(
                self.playbook,
                budget_tokens=self.budget_tokens,
                min_bullets=self.min_bullets,
                age_by_id=self.bullet_added_window,
                current_window=self.window_index,
                unused_grace_windows=self.unused_grace_windows,
            )
            for entry in pruned:
                self.bullet_added_window.pop(entry["id"], None)
                self._append_jsonl("pruned.jsonl", {"window": self.window_index, **entry})

        # --- Carry the reflection forward as an unverified hypothesis -------
        if reflection_json:
            self.previous_reflection = " ".join(
                part
                for part in (
                    reflection_json.get("key_insight"),
                    reflection_json.get("correct_approach"),
                )
                if isinstance(part, str) and part.strip()
            ) or "(none)"

        # --- Persist --------------------------------------------------------
        self.window_index += 1
        self._save_state()
        if self.ace_dir:
            try:
                (self.ace_dir / "playbook_history" / f"step_{end_step:05d}.md").write_text(self.playbook)
            except Exception as exc:
                logger.debug("ACE: failed to snapshot playbook: %s", exc)

        self._append_jsonl(
            "ace_log.jsonl",
            {
                "window": self.window_index - 1,
                "start_step": start_step,
                "end_step": end_step,
                "verdict": verdict,
                "citation_rate": round(citation_rate, 3),
                "cited_count": len(cited),
                "tags_applied": len(tags) if not self.freeze_playbook else 0,
                "bullets_added": len(added),
                "bullets_pruned": len(pruned),
                "movement": {
                    "attempts": facts.get("movement_attempts"),
                    "moved": facts.get("moved"),
                    "turned": facts.get("turned"),
                    "blocked": facts.get("blocked"),
                    "triggered": facts.get("triggered"),
                    "indeterminate": facts.get("indeterminate"),
                    "ui_navigation": facts.get("ui_navigation_steps"),
                    "interaction": facts.get("interaction_steps"),
                },
                "party_wiped": facts.get("party_wiped"),
                "new_milestones": facts.get("new_milestones"),
                "novel_locations": facts.get("novel_locations"),
                "total_bullets": len(playbook_bullet_ids(self.playbook)),
                "playbook_tokens": count_tokens(self.playbook),
                "reflected": bool(raw_reflection),
                "curated": bool(should_curate),
            },
        )

        logger.info(
            "ACE window %d done: verdict=%s cited=%d tagged=%d added=%d pruned=%d total=%d (%d tok)",
            self.window_index - 1,
            verdict,
            len(cited),
            len(tags),
            len(added),
            len(pruned),
            len(playbook_bullet_ids(self.playbook)),
            count_tokens(self.playbook),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_observation(self, start_step: int, end_step: int, num_steps: int) -> WindowObservation:
        return WindowObservation(
            start_step=start_step,
            end_step=end_step,
            trajectory_rows=self._recent_trajectories(num_steps, start_step, end_step),
            history_entries=self._recent_history(start_step, end_step),
            step_states=dict(self._step_states),
            state_close=summarize_game_state(self._current_state()),
            milestones_open=self._window_open_milestones or {},
            milestones_close=read_milestones(self.mcp_adapter),
            locations_seen_before=set(self.seen_locations),
        )

    def _recent_trajectories(self, num_steps: int, start_step: int, end_step: int) -> List[Dict[str, Any]]:
        """Trajectory rows for this window only.

        Steps that produce no row (the black-frame path) mean a request for the
        last ``num_steps`` lines can reach back into the previous window, which
        would make those steps get reflected on twice. Filter by step range.
        """
        try:
            from agents.subagents.utils.trajectory_window import (
                read_last_jsonl_lines,
                resolve_trajectory_path,
            )

            path = resolve_trajectory_path(self.run_manager)
            if not path:
                return []
            rows = []
            for line in read_last_jsonl_lines(path, num_steps):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = row.get("step")
                if isinstance(step, int) and start_step <= step <= end_step:
                    rows.append(row)
            return rows
        except Exception as exc:
            logger.debug("ACE: could not read trajectories: %s", exc)
            return []

    def set_history_source(self, history_getter) -> None:
        """Register a callable returning the agent's conversation history."""
        self._history_getter = history_getter

    def _recent_history(self, start_step: int, end_step: int) -> List[Dict[str, Any]]:
        getter = getattr(self, "_history_getter", None)
        if getter is None:
            return []
        try:
            entries = getter() or []
        except Exception:
            return []
        return [
            entry
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("step"), int)
            and start_step <= entry["step"] <= end_step
        ]

    def _current_state(self) -> Any:
        if self.mcp_adapter is None:
            return {}
        try:
            return self.mcp_adapter.call_tool("get_game_state", {})
        except Exception as exc:
            logger.debug("ACE: could not read current state: %s", exc)
            return {}

    def _collect_cited(self, known_ids: List[str]) -> List[str]:
        known = set(known_ids)
        ordered: List[str] = []
        seen = set()
        for step in sorted(self._cited):
            for cid in self._cited[step]:
                if cid in known and cid not in seen:
                    seen.add(cid)
                    ordered.append(cid)
        return ordered[: self.max_cited_bullets]

    def _total_segments(self) -> Any:
        max_steps = self.config.get("max_steps")
        if isinstance(max_steps, int) and max_steps > 0:
            return max(1, max_steps // self.window_steps)
        return "unknown (continuous reset-free run)"

    def _segment_context(
        self, start_step: int, end_step: int, obs: WindowObservation, facts: Dict[str, Any]
    ) -> str:
        open_state = obs.state_open or {}
        close_state = obs.state_close or {}
        latest = facts.get("new_milestones") or []
        return (
            f"Segment steps {start_step}-{end_step}. "
            f"Location at start: {open_state.get('location', 'Unknown')} "
            f"{open_state.get('coords', ('?', '?'))} -> end: {close_state.get('location', 'Unknown')} "
            f"{close_state.get('coords', ('?', '?'))}. "
            f"Game context at end: {close_state.get('context', 'unknown')}. "
            f"Badges: {close_state.get('badge_count')}. "
            f"Milestones completed so far: {facts.get('milestones_total')}"
            + (f" (new this segment: {', '.join(latest)})" if latest else "")
            + f". Verdict for this segment: {facts.get('verdict')}."
        )


def create_ace_controller(vlm, mcp_adapter=None, run_data_manager=None, config=None,
                          game_name: str = "Pokemon Red") -> AceController:
    """Factory mirroring ``create_prompt_optimizer`` / ``create_harness_evolver``."""
    return AceController(
        vlm=vlm,
        mcp_adapter=mcp_adapter,
        run_data_manager=run_data_manager,
        config=config,
        game_name=game_name,
    )
