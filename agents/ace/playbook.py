"""Playbook data model for the ACE scaffold.

Ported from the ACE reference implementation (``ace/playbook_utils.py`` in the
vendored clone at the repository root, https://github.com/ace-agent/ace).  The
parsing / merging functions are kept behaviourally identical to the reference so
that the algorithm we are comparing against is the published one:

    parse_playbook_line, get_next_global_id, format_playbook_line,
    update_bullet_counts, apply_curator_operations, get_playbook_stats,
    extract_json_from_text, extract_playbook_bullets

Three things are *not* ported verbatim, each for a stated reason:

1. ``get_section_slug`` is a total lookup over :data:`ACE_SLUG_MAP` rather than
   the reference's hand-rolled fallback.  The reference falls back to the first
   letter of each word, which produces two-character slugs for several of its
   own sections (e.g. ``PROBLEM-SOLVING HEURISTICS`` -> ``ph``).  The generator's
   own bullet-id regex requires three or more lowercase letters, so those
   bullets can never be cited and their counters can never move.  Every slug
   here is asserted to match ``^[a-z]{3,}$``.

2. ``count_tokens`` is a ``len // 4`` estimate.  ``tiktoken`` is not a
   dependency of this repository and the count is only used for budgeting.

3. ``prune_to_budget`` is new.  ACE injects its playbook once per sample; this
   scaffold injects it on every one of thousands of steps, so the token budget
   is a real cost driver rather than the reference's advisory prompt string.
   Pruning is deterministic and metadata-driven (never an LLM rewrite, which
   would be exactly the "context collapse" failure ACE was designed to avoid).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

# (header, normalized key, slug).  The normalized key is what
# ``apply_curator_operations`` derives from the "## HEADER" line, and is the
# name the curator is told to emit in its ADD operations.
ACE_SECTIONS: List[Tuple[str, str, str]] = [
    ("NAVIGATION AND MOVEMENT", "navigation_and_movement", "nav"),
    ("MENUS AND DIALOGUE", "menus_and_dialogue", "menu"),
    ("BATTLE TACTICS", "battle_tactics", "btl"),
    ("PROGRESSION AND STORY GATES", "progression_and_story_gates", "prog"),
    ("PARTY AND ITEM MANAGEMENT", "party_and_item_management", "party"),
    ("COMMON MISTAKES TO AVOID", "common_mistakes_to_avoid", "err"),
    ("STUCK STATE RECOVERY", "stuck_state_recovery", "stuck"),
    ("SCREEN READING CUES", "screen_reading_cues", "cue"),
    ("OTHERS", "others", "misc"),
]

ACE_SLUG_MAP: Dict[str, str] = {key: slug for _, key, slug in ACE_SECTIONS}
ACE_SECTION_KEYS: List[str] = [key for _, key, _ in ACE_SECTIONS]

_SLUG_PATTERN = re.compile(r"^[a-z]{3,}$")
_BULLET_ID_PATTERN = re.compile(r"[a-z]{3,}-\d{5}")

# The reference's latent bug is a hard error here rather than a silent one.
assert all(_SLUG_PATTERN.match(slug) for slug in ACE_SLUG_MAP.values()), (
    "every ACE section slug must match ^[a-z]{3,}$ or the generator's bullet-id "
    "regex can never match its citations"
)
assert len(set(ACE_SLUG_MAP.values())) == len(ACE_SLUG_MAP), "section slugs must be unique"


def normalize_section(name: str) -> str:
    """Normalize a section name the way ``apply_curator_operations`` does."""
    return name.strip().lower().replace(" ", "_").replace("&", "and")


def get_section_slug(section_key: str) -> str:
    """Map a normalized section key to its id slug.

    Total: anything unrecognised lands in ``misc`` (the OTHERS slug).  Unlike
    the reference there is no first-letter fallback, which could emit slugs too
    short for the bullet-id regex to ever match.
    """
    return ACE_SLUG_MAP.get(normalize_section(section_key), "misc")


def empty_playbook() -> str:
    """An empty playbook containing only the section headers.

    The headers must exist up front: ``apply_curator_operations`` builds its
    section map from the ``##`` lines already present, so a bullet added to a
    section with no header would fall through to OTHERS.
    """
    return "\n\n".join(f"## {header}" for header, _, _ in ACE_SECTIONS)


# ---------------------------------------------------------------------------
# Bullet parsing / formatting  (ported verbatim)
# ---------------------------------------------------------------------------

_LINE_PATTERN = r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"


def parse_playbook_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single playbook line to extract components."""
    match = re.match(_LINE_PATTERN, line.strip())
    if match:
        return {
            "id": match.group(1),
            "helpful": int(match.group(2)),
            "harmful": int(match.group(3)),
            "content": match.group(4),
            "raw_line": line,
        }
    return None


def format_playbook_line(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    """Format a bullet into playbook line format."""
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def get_next_global_id(playbook_text: str) -> int:
    """Extract the highest global ID in the playbook and return the next one."""
    max_id = 0
    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            id_match = re.search(r"-(\d+)$", parsed["id"])
            if id_match:
                max_id = max(max_id, int(id_match.group(1)))
    return max_id + 1


def playbook_bullet_ids(playbook_text: str) -> List[str]:
    """Every bullet id currently in the playbook, in document order."""
    ids = []
    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            ids.append(parsed["id"])
    return ids


def iter_bullets(playbook_text: str):
    """Yield ``(line_index, section_key, parsed_bullet)`` for each bullet."""
    section = "general"
    for idx, line in enumerate(playbook_text.strip().split("\n")):
        stripped = line.strip()
        if stripped.startswith("##"):
            section = normalize_section(stripped[2:])
            continue
        parsed = parse_playbook_line(line)
        if parsed:
            yield idx, section, parsed


# ---------------------------------------------------------------------------
# Counter layer  (ported verbatim)
# ---------------------------------------------------------------------------


def update_bullet_counts(playbook_text: str, bullet_tags: Any) -> str:
    """Update helpful/harmful counts based on reflector tags (Counter layer).

    Deterministic: an LLM never rewrites the numbers, it only emits tags.
    """
    lines = playbook_text.strip().split("\n")
    updated_lines: List[str] = []

    tag_map: Dict[str, str] = {}
    if isinstance(bullet_tags, list) and len(bullet_tags) > 0:
        for tag in bullet_tags:
            if isinstance(tag, dict):
                # Accept both 'id' and 'bullet' keys, as the reference does.
                bullet_id = tag.get("id") or tag.get("bullet", "")
                tag_value = tag.get("tag", "neutral")
                if bullet_id:
                    tag_map[bullet_id] = tag_value

    if not tag_map:
        logger.debug("No valid bullet tags found to update counts")
        return playbook_text

    for line in lines:
        if line.strip().startswith("#") or not line.strip():
            updated_lines.append(line)
            continue

        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in tag_map:
            tag = tag_map[parsed["id"]]
            if tag == "helpful":
                parsed["helpful"] += 1
            elif tag == "harmful":
                parsed["harmful"] += 1
            # neutral: no change
            updated_lines.append(
                format_playbook_line(
                    parsed["id"], parsed["helpful"], parsed["harmful"], parsed["content"]
                )
            )
        else:
            updated_lines.append(line)

    return "\n".join(updated_lines)


# ---------------------------------------------------------------------------
# Delta merge  (ported verbatim: ADD-only, deterministic, non-LLM)
# ---------------------------------------------------------------------------


def apply_curator_operations(
    playbook_text: str, operations: List[Dict[str, Any]], next_id: int
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """Apply curator delta operations to the playbook.

    Only ``ADD`` is supported, matching the reference implementation (its
    UPDATE / MERGE / DELETE / CREATE_META branches are unimplemented stubs).
    The merge itself is plain string manipulation -- this is the "incremental
    delta update" that the paper's ablation shows to be load-bearing.

    Returns ``(new_playbook, next_id, added)`` where ``added`` is a list of
    ``{"id", "section", "content"}`` for each bullet actually inserted.
    """
    lines = playbook_text.strip().split("\n")

    # Build the set of sections that actually exist in the playbook.
    sections = set()
    for line in lines:
        if line.strip().startswith("##"):
            sections.add(normalize_section(line.strip()[2:]))

    bullets_to_add: List[Tuple[str, str]] = []
    added: List[Dict[str, Any]] = []

    for op in operations:
        if not isinstance(op, dict):
            continue
        if op.get("type") != "ADD":
            continue

        section_raw = op.get("section", "others")
        section = normalize_section(section_raw)
        if section not in sections:
            logger.warning("Section '%s' not found, adding to OTHERS", section_raw)
            section = "others"

        content = str(op.get("content", "")).strip().replace("\n", " ")
        if not content:
            continue

        new_id = f"{get_section_slug(section)}-{next_id:05d}"
        next_id += 1

        bullets_to_add.append((section, format_playbook_line(new_id, 0, 0, content)))
        added.append({"id": new_id, "section": section, "content": content})

    if not bullets_to_add:
        return playbook_text, next_id, added

    # Rebuild, flushing each section's pending bullets just before the next
    # header (i.e. appending them at the end of their own section).
    final_lines: List[str] = []
    current_section: Optional[str] = None
    pending = list(bullets_to_add)

    for line in lines:
        if line.strip().startswith("##"):
            if current_section:
                final_lines.extend(b for s, b in pending if s == current_section)
                pending = [(s, b) for s, b in pending if s != current_section]
            current_section = normalize_section(line.strip()[2:])
        final_lines.append(line)

    if current_section:
        final_lines.extend(b for s, b in pending if s == current_section)
        pending = [(s, b) for s, b in pending if s != current_section]

    if pending:
        logger.warning("%d bullets have no matching section, adding to OTHERS", len(pending))
        others_idx = next(
            (i for i, line in enumerate(final_lines) if normalize_section(line.strip().lstrip("#")) == "others"),
            -1,
        )
        orphans = [b for _, b in pending]
        if others_idx >= 0:
            for offset, bullet in enumerate(orphans):
                final_lines.insert(others_idx + 1 + offset, bullet)
        else:
            final_lines.extend(orphans)

    return "\n".join(final_lines), next_id, added


# ---------------------------------------------------------------------------
# Stats  (ported verbatim)
# ---------------------------------------------------------------------------


def get_playbook_stats(playbook_text: str) -> Dict[str, Any]:
    """Generate statistics about the playbook (shown to the curator)."""
    stats: Dict[str, Any] = {
        "total_bullets": 0,
        "high_performing": 0,  # helpful > 5, harmful < 2
        "problematic": 0,  # harmful >= helpful, harmful > 0
        "unused": 0,  # helpful + harmful == 0
        "by_section": {},
    }

    current_section = "general"
    for line in playbook_text.strip().split("\n"):
        if line.strip().startswith("##"):
            current_section = line.strip()[2:].strip()
            continue

        parsed = parse_playbook_line(line)
        if not parsed:
            continue

        stats["total_bullets"] += 1
        if parsed["helpful"] > 5 and parsed["harmful"] < 2:
            stats["high_performing"] += 1
        elif parsed["harmful"] >= parsed["helpful"] and parsed["harmful"] > 0:
            stats["problematic"] += 1
        elif parsed["helpful"] + parsed["harmful"] == 0:
            stats["unused"] += 1

        bucket = stats["by_section"].setdefault(
            current_section, {"count": 0, "helpful": 0, "harmful": 0}
        )
        bucket["count"] += 1
        bucket["helpful"] += parsed["helpful"]
        bucket["harmful"] += parsed["harmful"]

    return stats


# ---------------------------------------------------------------------------
# JSON extraction  (ported verbatim)
# ---------------------------------------------------------------------------


def _find_json_objects(text: str) -> List[str]:
    """Find JSON objects using balanced brace counting (string-escape aware)."""
    objects: List[str] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            brace_count = 1
            start = i
            i += 1
            while i < len(text) and brace_count > 0:
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                elif text[i] == '"':
                    i += 1
                    while i < len(text) and text[i] != '"':
                        if text[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            if brace_count == 0:
                objects.append(text[start:i])
        else:
            i += 1
    return objects


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from model output, handling various formats."""
    if not text:
        return None
    try:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        for match in re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        for candidate in _find_json_objects(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    except Exception as exc:  # pragma: no cover - defensive, matches reference
        logger.warning("Failed to extract JSON: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Bullet extraction for the reflector  (ported verbatim)
# ---------------------------------------------------------------------------


def extract_playbook_bullets(playbook_text: str, bullet_ids) -> str:
    """Render the bullets the generator cited, for the reflector's prompt."""
    if not bullet_ids:
        return "(No bullets used by generator)"

    wanted = set(bullet_ids)
    found = [
        format_playbook_line(p["id"], p["helpful"], p["harmful"], p["content"])
        for _, _, p in iter_bullets(playbook_text)
        if p["id"] in wanted
    ]

    if not found:
        return "(Generator referenced bullet IDs but none were found in playbook)"

    return "\n".join(found)


# ---------------------------------------------------------------------------
# Budget enforcement (new -- see module docstring)
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Rough token count.  ``tiktoken`` is not a dependency of this repo."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _bullet_numeric_id(bullet_id: str) -> int:
    match = re.search(r"-(\d+)$", bullet_id)
    return int(match.group(1)) if match else 0


def prune_to_budget(
    playbook_text: str,
    budget_tokens: int,
    min_bullets: int = 10,
    age_by_id: Optional[Dict[str, int]] = None,
    current_window: int = 0,
    unused_grace_windows: int = 5,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Deterministically prune the playbook back under ``budget_tokens``.

    Drop order (one bullet per iteration):

    1. ``harmful >= helpful and harmful > 0`` -- the reference's own
       "problematic" class, and the metadata filter the paper describes as
       grow-and-refine's first line of defence against context noise.
    2. ``helpful + harmful == 0`` and older than ``unused_grace_windows`` --
       the reference's "unused" class, with a grace period so a freshly added
       bullet is never pruned before it has had a chance to be cited.
    3. Lowest net utility (``helpful - harmful``).

    Ties break toward the lowest (oldest) numeric id.  Section headers are never
    removed and ids are never reused or renumbered.

    Returns ``(new_playbook, pruned)`` where ``pruned`` records what was dropped.
    """
    age_by_id = age_by_id or {}
    pruned: List[Dict[str, Any]] = []
    text = playbook_text

    while count_tokens(text) > budget_tokens:
        bullets = list(iter_bullets(text))
        if len(bullets) <= min_bullets:
            break

        def tier(entry):
            _, _, p = entry
            if p["harmful"] >= p["helpful"] and p["harmful"] > 0:
                return 0
            if p["helpful"] + p["harmful"] == 0:
                added = age_by_id.get(p["id"])
                if added is None or current_window - added >= unused_grace_windows:
                    return 1
                return 3  # still in its grace period
            return 2

        candidates = [e for e in bullets if tier(e) != 3]
        if not candidates:
            # Everything left is inside its grace period; stop rather than
            # evict a bullet that has not yet had a chance to prove itself.
            break

        victim = min(
            candidates,
            key=lambda e: (tier(e), e[2]["helpful"] - e[2]["harmful"], _bullet_numeric_id(e[2]["id"])),
        )
        _, section, parsed = victim

        lines = text.strip().split("\n")
        kept = [
            line
            for line in lines
            if not (
                (p := parse_playbook_line(line)) is not None and p["id"] == parsed["id"]
            )
        ]
        text = "\n".join(kept)

        pruned.append(
            {
                "id": parsed["id"],
                "section": section,
                "content": parsed["content"],
                "helpful": parsed["helpful"],
                "harmful": parsed["harmful"],
                "tier": tier(victim),
            }
        )

    return text, pruned


def extract_cited_bullet_ids(text: str) -> List[str]:
    """Extract the playbook bullet ids the generator claims to have used.

    The generator here is a function-calling agent, not a JSON-emitting one, so
    the citation arrives as a ``PLAYBOOK_USED:`` line inside its reasoning.  The
    bracket form is the reference implementation's own non-JSON fallback.
    Returns ``[]`` when the model does not cite, which is a graceful no-op:
    ``update_bullet_counts`` leaves the playbook untouched and the curator still
    runs.
    """
    if not text:
        return []

    ordered: List[str] = []
    seen = set()

    def _collect(candidates):
        for cid in candidates:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)

    for line in re.findall(r"^[^\S\n]*PLAYBOOK_USED[^\S\n]*:[^\S\n]*(.*)$", text, re.MULTILINE | re.IGNORECASE):
        _collect(_BULLET_ID_PATTERN.findall(line))

    if not ordered:
        _collect(re.findall(r"\[([a-z]{3,}-\d{5})\]", text))

    return ordered
