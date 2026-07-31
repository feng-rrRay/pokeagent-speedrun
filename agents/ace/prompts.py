"""Prompt templates for the ACE scaffold.

These mirror ``ace/ace/prompts/reflector.py`` and ``ace/ace/prompts/curator.py``
in the vendored reference clone, using the **no-ground-truth** variants
(``REFLECTOR_PROMPT_NO_GT`` / ``CURATOR_PROMPT_NO_GT``) because a reset-free
Pokemon run has no labels.  The instruction lists, the JSON output contract and
the ADD-only operation schema are kept as close to the originals as the domain
remap allows, so that the method under comparison is recognisably ACE.

Domain remap of the reference's QA slots:

    Question              -> the fixed task statement (ACE_TASK_STATEMENT)
    Model's Reasoning Trace -> the agent's step-by-step trace for the segment
    Model's Predicted Answer -> the actions actually taken in the segment
    Environment Feedback  -> deterministic emulator-memory readout (feedback.py)

Deliberate additions over the reference, each with a reason:

* Reflector: a paragraph making explicit that the environment feedback is
  verified fact unavailable to the agent at play time, an instruction to prefer
  generalisable insights over narration, an instruction to confirm/refute the
  previous segment's reflection (the surrogate for ACE's reflect->regenerate
  loop, which a non-resettable emulator cannot support), and a ban on inventing
  bullet ids.
* Curator: the explicit list of valid section names (ours are new, and the
  reference's implicit inference is what produces its "Section not found"
  fallback), a hard-limit framing of the token budget (we enforce it; the
  reference only prints it), and two scoping clauses that keep the playbook from
  filling with one-off coordinates or with strategy-guide knowledge the scaffold
  is not allowed to have.
"""

# ---------------------------------------------------------------------------
# Generator-side injection
# ---------------------------------------------------------------------------

# Only the mutable playbook content is injected per step; the static "how to use
# the playbook" instructions live in the system prompt (ACE.md / ACE_RED.md) so
# they stay inside the cached prefix.
ACE_PLAYBOOK_BLOCK = """PLAYBOOK_BEGIN
{playbook}
PLAYBOOK_END
"""


ACE_TASK_STATEMENT = (
    "Play {game_name} from wherever the game currently is and make as much verifiable "
    "progress as possible (new milestones, new locations, new badges). The agent's only "
    "tool is press_buttons (Game Boy hardware buttons). There is no walkthrough, no "
    "pathfinding, no wiki, no note-taking tool, and no reset -- the run is one "
    "continuous trajectory, so a mistake cannot be undone by restarting."
)


# ---------------------------------------------------------------------------
# Reflector (no ground truth)
# ---------------------------------------------------------------------------

ACE_REFLECTOR_PROMPT_NO_GT = """You are an expert analyst and educator. Your job is to diagnose why an agent's play went wrong during one segment of a continuous {game_name} playthrough.

**Instructions:**
- Carefully analyze the agent's step-by-step trace to identify where it went wrong
- Take the environment feedback into account. It is computed by reading the emulator's memory directly (milestones, badges, party, money, coordinates, map ids, game context) together with the recorded trajectory. It is verified fact about WHAT HAPPENED, not an opinion about what the agent should have done, and it was NOT available to the agent while it was playing.
- Identify specific perceptual errors, control errors, or misapplied strategies
- Provide actionable insights that could help the agent avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the agent should have done differently
- Prefer insights that generalize to future situations in this game over narration of this particular segment
- If a previous reflection is shown, treat it as an UNVERIFIED hypothesis from the previous segment. State explicitly whether this segment's environment feedback confirms or refutes it.
- You will receive bulletpoints that are part of the playbook that's used by the agent to choose its actions.
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the agent to make verifiable progress in the game)
- Do NOT invent bullet ids. Only tag ids that appear in the bulletpoints listed below.

Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis of the segment
  - error_identification: what specifically went wrong in this segment?
  - root_cause_analysis: why did this error occur? What was misunderstood about the game, the screen, or the controls?
  - correct_approach: what should the agent have done instead?
  - key_insight: what strategy, rule, or principle should be remembered to avoid this error?
  - bullet_tags: a list of json objects with id and tag for each bulletpoint used by the agent

**Task:**
{task}

**Agent's Step-by-Step Trace (steps {window_start}-{window_end}):**
{trace}

**Actions Taken:**
{actions}

**Previous Reflection (unverified hypothesis from the previous segment):**
{previous_reflection}

**Environment Feedback:**
{environment_feedback}

**Part of Playbook that's used by the agent to choose its actions:**
{bullets_used}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis]",
  "error_identification": "[What specifically went wrong in this segment?]",
  "root_cause_analysis": "[Why did this error occur? What was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[What strategy, rule, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "nav-00001", "tag": "helpful"}},
    {{"id": "menu-00002", "tag": "harmful"}}
  ]
}}

---
"""


# ---------------------------------------------------------------------------
# Curator (no ground truth)
# ---------------------------------------------------------------------------

ACE_CURATOR_PROMPT_NO_GT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous segment of play.

**Context:**
- The playbook you create is injected into the agent's prompt at EVERY step of a continuous, reset-free {game_name} playthrough. It must stay compact, general, and immediately actionable.
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used. So you need to come up with content that helps the agent act correctly using ONLY what it can see on screen and in its prompt.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous segment
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Prefer durable, reusable tactics over facts about one specific moment. Do NOT add anything that is only true at one coordinate at one point in time.
- Do NOT add walkthrough knowledge or game facts you were not shown. Only generalize from the reflection and the existing playbook.
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable and should fit on a single line

**Training Context:**
- Total token budget: {token_budget} tokens. This is a HARD limit: when the playbook exceeds it the system deterministically prunes bullets (harmful first, then long-unused, then lowest net utility). Low-value additions therefore push out high-value ones.
- Training progress: Segment {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Segment Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Sections (use one of these exact names):**
{section_names}

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[nav-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis here]",
  "operations": [
    {{
      "type": "ADD",
      "section": "navigation_and_movement",
      "content": "[New tactic...]"
    }}
  ]
}}

---
"""
