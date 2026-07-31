# ACE scaffold — fidelity notes

A port of **ACE** (*Agentic Context Engineering*, [arXiv 2510.04618](https://arxiv.org/abs/2510.04618),
ICLR 2026) to this repo's reset-free Pokémon setting, exposed as `--scaffold ace`.

The reference implementation is vendored read-only at the repository root under `ace/`
(git-ignored). Nothing in this package imports from it; the reusable parts were ported and
are marked as such in `playbook.py`.

The reference targets **single-turn QA with ground-truth labels**: every task is coerced into a
`{context, question, target}` triple and the training loop branches on
`data_processor.answer_is_correct(answer, target)`. Its `no_ground_truth` flag only hides the
target string from the Reflector's prompt — the loop still uses the label, and
`environment_feedback` is one of two hardcoded strings. Pokémon Red is the opposite regime: one
continuous trajectory of thousands of steps, no labels, no episode boundary, no way to re-answer.

---

## Faithful to ACE

| Element | Where |
|---|---|
| Three roles, same division of labour: Generator acts and cites; Reflector diagnoses and tags but **never writes the playbook**; Curator emits **ADD-only deltas** | `reflector.py`, `curator.py` |
| Deltas merged by **deterministic non-LLM logic** — the component the paper's ablation shows is load-bearing (−11.7 TGC / −27.8 SGC without it) | `playbook.apply_curator_operations` |
| Bullet grammar `[slug-#####] helpful=N harmful=M :: content`, `##` sections, one global monotone ID counter | `playbook.py` |
| Counters incremented **deterministically** from the Reflector's tags; an LLM never rewrites the numbers | `playbook.update_bullet_counts` |
| Generator cites the bullets it used; the Reflector sees **only** those bullets | `playbook.extract_cited_bullet_ids`, `extract_playbook_bullets` |
| No-ground-truth prompt variants, reframed around `**Environment Feedback:**` | `prompts.py` |
| Curator sees the full playbook + reflection + budget/progress/stats, and is told "Do NOT regenerate the entire playbook" | `prompts.ACE_CURATOR_PROMPT_NO_GT` |
| Reflection is **per-episode over the full trajectory**, not per step (AppWorld: Reflector ≈1.8×/episode of ≈19.9 generator steps, Appendix A.3 Table 12) | `controller.py`, 20-step window |
| The deployed Generator prompt contains the playbook but **not** the reflection — matching `ace/ace/ace.py:604,612`, which pass `reflection="(empty)"` outside the refinement loop | `controller.get_playbook_block` |
| `curator_frequency=1`, temperature 0 | reference defaults |

Functions ported essentially verbatim from `ace/playbook_utils.py`: `parse_playbook_line`,
`get_next_global_id`, `format_playbook_line`, `update_bullet_counts`, `apply_curator_operations`,
`get_playbook_stats`, `extract_json_from_text`, `extract_playbook_bullets`.

---

## Deviations, and why

### D1 — Iterative reflection refinement: not used

The paper describes it as optional: the Reflector critiques traces to extract lessons,
*"optionally refining them across multiple iterations."*

The reference implements that option as **reflect → regenerate → recheck**
(`ace/ace/ace.py:501-545`): each round re-answers the *same question* with the reflection
injected, so round *n*'s Reflector sees a newly produced trace, and only the final round's
reflection reaches the Curator (`ace.py:584`). The mechanism is that the curated reflection has
been **validated by re-execution**.

Regeneration requires replaying the environment. Reset-free, that would mean rewinding the
emulator and giving ACE a retry budget the `simplest` baseline does not get — which would
invalidate the comparison rather than strengthen it. Re-running the Reflector on identical input
at temperature 0 reproduces the call count without the mechanism, so we do not do that either.

Cost, from the authors' own ablations: 1 round retains **+8.0** of the +14.3 available at 5 rounds
(Table 19); dropping the Reflector *and* multi-epoch entirely still retains **+12.7** of +17.0
(Table 3).

### D2 — Semantic de-duplication: off, matching the reference default

The paper's grow-and-refine specifies embedding-based de-duplication, implemented in
`ace/ace/core/bulletpoint_analyzer.py` (`all-mpnet-base-v2` + faiss). It is **`store_true`,
default `False`** in every reference eval runner (`ace/eval/*/run.py`) and in `ace/README.md:126`
— the paper's headline numbers were produced without it, and Table 20 shows performance is flat
across 50–90% thresholds. Running with it off is therefore the reference configuration.

What does need disclosing is that we *added* a different mechanism (utility-based pruning, below),
not that we omitted this one. `--ace-playbook-budget-tokens 0` turns the addition off.

### D3 — Token budget: enforced by default, with a reference-default mode

The reference's `playbook_token_budget` (default 80000) is **advisory**: it is interpolated into
the Curator prompt as text and never enforced — there is no truncation or pruning code anywhere
in the reference.

Here the playbook is re-sent on **every one of thousands of steps** rather than once per sample,
so unbounded growth is a first-order cost and context-dilution driver. Default is 6000 tokens with
deterministic, metadata-driven pruning (harmful → long-unused → lowest net utility, oldest first;
`playbook.prune_to_budget`). Pruning is never an LLM rewrite — that would be exactly the context
collapse ACE was designed to prevent. Pruned bullets are archived to `pruned.jsonl`.

**`--ace-playbook-budget-tokens 0` disables enforcement entirely**, reproducing the reference.
Report which mode each arm ran in.

### D4 — Pseudo-episode = fixed non-overlapping 100-step window

Reset-free play has no task boundary and no terminal verdict, so the pseudo-episode is a fixed
window. **K = 100 matches the ContinualHarness arm** this is compared against, which runs with
`--optimization-window-length 100` (`scripts/test_scaffolds_red.sh`) and settles to an evolution
every `STABLE_FREQUENCY = 100` steps (`agents/utils/harness_evolver.py`). Equalising both the
adaptation cadence and the analysed trajectory span keeps the comparison about *what* each method
does with a segment rather than how often it gets to look at one.

The cost, stated rather than hidden: AppWorld episodes average ~19.9 generator steps (Appendix A.3,
Table 12), so a 100-step window puts ~5× more play between adaptations than the regime ACE was
tuned in, and the paper gives no evidence about behaviour at that spacing. `--ace-window-steps 20`
recovers the ACE-native regime and is worth running as a sensitivity check if a reviewer presses on
it. Note the two arms are not identical early on either: ContinualHarness evolves every 25 steps
until step 200 (`EARLY_FREQUENCY`), so it adapts 8 times before ACE adapts twice.

Non-overlapping, because overlapping windows would double-count evidence into the counters that
drive both curation and pruning. Milestone-triggered segmentation was considered and rejected:
inter-milestone gaps range from ~3 to 1500+ steps, which would curate constantly early and then
not adapt for a thousand steps mid-game.

Because `PokeAgent.conversation_history` is capped at 20 entries, raw tool *results* survive only
for the tail of a 100-step window. Verification is unaffected — classification reads the per-step
state samples and the buttons in the trajectory rows — and the truncated part is the near-constant
`{"success": true, "frames_advanced": N}` echo.

### D5 — Ground truth replaced by ROM-verified emulator feedback

There is no label for "what should the agent have pressed". Paper §4.3 licenses label-free ACE
*when the substitute is a grounded execution signal* — on AppWorld, dropping ground truth costs
2.2 points (59.4→57.2, still +14.8 over base) and the online label-free variant reaches 59.5.
The paper's FiNER counterexample (label-free dropping *below* base when there is no environment to
check against) is the failure mode to avoid, so `feedback.py` contains **no model judgement at
all**: milestones from emulator memory, badges/party/money/coords/facing from state reads, and
movement outcomes verified against the state sampled before and after each step.

A "blocked move" is asserted only when a **directional button was pressed in the overworld** and
position, map and facing all held. A direction in a menu/dialogue/battle is cursor selection; a
held position with a changed context is a triggered event (wild encounter); a held position with
no facing reading is reported as indeterminate rather than blocked. Steps with no recorded buttons
are counted separately rather than assumed to be interactions.

**Non-movement steps are grounded too.** Battles and dialogue are a large share of play and of
failure modes, so an `effect` is derived from the same pre/post pair for every step that is not a
movement attempt: `damage_dealt` / `damage_taken` / `damage_traded` / `enemy_fainted` /
`battle_started` / `battle_ended` / `caught` / `dialogue_advanced` / `context_changed` /
`state_changed` / `no_observable_change`. Battle scalars come from `game.battle_info`
(`red_memory_reader.read_battle_details`), so a press inside a fight reports the real numbers:

```
VERIFIED: PIDGEY L5 HP 55% -> 0%; own active HP 79% -> 62%; PIDGEY L5 fainted
```

Dialogue is decided by comparing `game.dialog_text` (`read_screen_text`) across the step, which
is the only way to separate "the A press advanced the conversation" from "the A press did
nothing". Without it, mashing A through an ordinary conversation — the most common productive
action in the game — reads as a stall. The on-screen text is also surfaced in the trace, since
the reflector is text-only and never sees the screenshot:

```
SCREEN TEXT (emulator memory): "PROF.OAK: Hello there!"
VERIFIED: dialogue advanced (on-screen text changed)
```

Whitespace is collapsed before comparison so tilemap redraw jitter is not mistaken for new text,
and Red's `"dialog"` context string is normalised to `"dialogue"` so the two spellings do not look
like two different contexts.

**Dialogue is a declared blind spot under `--no-ocr`.** Runs match the paper configuration, where
`--no-ocr` suppresses `dialog_text` while leaving dialogue *context* detection (a VRAM border
check) intact. With no text to compare, such steps are labelled `dialogue_unverified`, excluded
from the stall detector, and reported as an explicit unknown:

```
- Dialogue: 12 steps in a text box -> 0 advanced the text, 0 closed the box, 0 changed nothing
  NOTE: on-screen text was not captured this run, so for 12 of those steps it is UNKNOWN whether
  the press advanced the conversation. Do NOT read these as wasted or mistimed inputs -- there is
  no evidence either way.
```

This is not cosmetic. In the first pilot run (`20260730_205220_ace`) those steps were reported as
`idle`; the Reflector read "12 idle dialogue steps" as evidence of button-mashing, and the Curator
turned it into three timing bullets — a playbook written from a measurement artifact, which is
exactly the context pollution §4.3 warns about when the substitute signal is unreliable.

`no_observable_change` is the context-independent generalisation of a blocked move, and is what
makes an agent stuck on a prompt it cannot dismiss visible at all:

```
- Longest run of steps with NO observable state change: 4 (context: dialogue)
- Dialogue: 4 steps in a text box -> 0 advanced the text, 0 closed the box, 4 changed nothing
```

Consequently, *being* in a battle is not treated as progress. `PARTIAL` requires verified damage,
a faint, a capture, or a resolved battle — an agent parked on a battle screen for twenty steps
scores `NO_PROGRESS`, which is the diagnosis the reflector needs.

### D6 — Playbook injected in the per-step user prompt, not the system prompt

`VLM` is constructed once with `system_instruction` (`PokeAgent.py:249`) and Gemini keys its
context cache on it, so mutating it every window would invalidate the cache. The playbook sits at
the very top of the step prompt with the same `PLAYBOOK_BEGIN`/`PLAYBOOK_END` sentinels; the
static usage and citation instructions stay in the system prompt (`ACE_RED.md`).

### D7 — Citation via `PLAYBOOK_USED:` rather than a JSON `bullet_ids` field

The Generator here is a function-calling agent; forcing JSON output would change the base scaffold
and destroy the comparison. The reference itself ships a non-JSON path using
`r'\[([a-z]{3,}-\d{5})\]'` over free text (`ace/ace/core/generator.py:115`), used here as a
fallback. Missing citations degrade to the reference's own no-op. `citation_rate` is logged per
window — **check it on the first live run**; if the model never cites, the credit-assignment
channel is dead and ACE reduces to append-only prompt growth.

### D8 — Pokémon-specific sections, slugs ≥3 lowercase chars

The reference's `get_section_slug` falls back to first-letters-of-words, producing two-character
slugs for several of its own sections (`problem-solving heuristics` → `ph`). Its Generator regex
requires `[a-z]{3,}`, so those bullets can never be cited and their counters can never move. Our
slug map is a total lookup with a module-level assertion and a regression test.

### D9 — `press_buttons` only

`ace` has no `process_memory`, so the curated playbook is its only persistent-knowledge channel.
This makes `ace` vs `simplest` a comparison of knowledge mechanisms rather than a pure `+ACE`
ablation; `--ace-freeze-playbook` with an empty playbook gives a press-buttons-only control arm.

---

## Experimental arms

| Arm | Command delta |
|---|---|
| H_min baseline | `--scaffold simplest` |
| ACE | `--scaffold ace` |
| ACE, reference-default budget | `--scaffold ace --ace-playbook-budget-tokens 0` |
| ACE, static context (control) | `--scaffold ace --ace-freeze-playbook --ace-playbook <file>` |
| press_buttons only (control) | `--scaffold ace --ace-freeze-playbook` (empty playbook) |
| ACE, offline warm start | run once → `--ace-playbook <prior final playbook>` |

`--ace-freeze-playbook` freezes curation, pruning **and** the helpful/harmful counters — the
counters are rendered into the injected bullets and the prompt tells the model to prefer high-
helpful/low-harmful ones, so letting them drift would make the "static context" arm non-static.

## Artifacts

Written to `.pokeagent_cache/{run_id}/ace/`, so `--backup-state` restore carries them for free:

```
playbook.md                  current playbook
playbook_history/step_*.md   per-window snapshots
reflections.jsonl            reflection JSON + grounded facts + citation rate per window
curations.jsonl              curator operations and the bullets they added
pruned.jsonl                 bullets dropped by budget enforcement
ace_log.jsonl                one row per window: verdict, movement breakdown, counts
ace_state.json              window index, next bullet id, seen locations, bullet ages
```

Reflector and Curator calls are logged under the module names `ace_reflector` and `ace_curator`,
so their token cost can be separated from the orchestrator's (`ace_orchestrator`) in
`cumulative_metrics.json`.
