"""Tests for the ACE playbook data model (agents/ace/playbook.py).

Pure string/JSON manipulation -- no network, no emulator, no VLM.
"""

import re

import pytest

from agents.ace.playbook import (
    ACE_SECTION_KEYS,
    ACE_SECTIONS,
    ACE_SLUG_MAP,
    apply_curator_operations,
    count_tokens,
    empty_playbook,
    extract_cited_bullet_ids,
    extract_json_from_text,
    extract_playbook_bullets,
    format_playbook_line,
    get_next_global_id,
    get_section_slug,
    parse_playbook_line,
    playbook_bullet_ids,
    prune_to_budget,
    update_bullet_counts,
)

# The generator's own bullet-id regex, from ace/ace/core/generator.py:115.
GENERATOR_ID_REGEX = re.compile(r"\[([a-z]{3,}-\d{5})\]")


# ---------------------------------------------------------------------------
# Sections and slugs
# ---------------------------------------------------------------------------


def test_all_section_slugs_are_lowercase_three_plus_and_unique():
    slugs = list(ACE_SLUG_MAP.values())
    for slug in slugs:
        assert re.fullmatch(r"[a-z]{3,}", slug), f"bad slug: {slug!r}"
    assert len(set(slugs)) == len(slugs)


@pytest.mark.parametrize("section_key", ACE_SECTION_KEYS)
def test_generator_regex_matches_every_section_slug(section_key):
    """Regression test for the reference implementation's silent slug bug.

    ace/utils.py's get_section_slug falls back to first-letters-of-words, which
    yields two-character slugs for some sections (e.g. "problem-solving
    heuristics" -> "ph"). The generator's citation regex requires three or more
    lowercase letters, so those bullets can never be cited and their
    helpful/harmful counters can never move. Every section we define must be
    citable.
    """
    bullet_id = f"{get_section_slug(section_key)}-00007"
    assert GENERATOR_ID_REGEX.findall(f"[{bullet_id}]") == [bullet_id]


def test_unknown_section_falls_back_to_misc():
    assert get_section_slug("something_we_never_defined") == "misc"


def test_empty_playbook_contains_every_section_header():
    playbook = empty_playbook()
    for header, _, _ in ACE_SECTIONS:
        assert f"## {header}" in playbook
    assert playbook_bullet_ids(playbook) == []


# ---------------------------------------------------------------------------
# Bullet parsing / formatting
# ---------------------------------------------------------------------------


def test_parse_format_roundtrip():
    line = format_playbook_line("nav-00003", 4, 1, "Walk into doors to use them")
    assert line == "[nav-00003] helpful=4 harmful=1 :: Walk into doors to use them"

    parsed = parse_playbook_line(line)
    assert parsed["id"] == "nav-00003"
    assert parsed["helpful"] == 4
    assert parsed["harmful"] == 1
    assert parsed["content"] == "Walk into doors to use them"


def test_parse_ignores_headers_and_prose():
    assert parse_playbook_line("## NAVIGATION AND MOVEMENT") is None
    assert parse_playbook_line("") is None
    assert parse_playbook_line("just some text") is None


def test_get_next_global_id_is_monotone_across_sections():
    playbook = "\n".join(
        [
            "## NAVIGATION AND MOVEMENT",
            "[nav-00001] helpful=0 harmful=0 :: a",
            "## BATTLE TACTICS",
            "[btl-00009] helpful=0 harmful=0 :: b",
        ]
    )
    assert get_next_global_id(playbook) == 10
    assert get_next_global_id(empty_playbook()) == 1


# ---------------------------------------------------------------------------
# Counter layer
# ---------------------------------------------------------------------------


def _two_bullet_playbook():
    return "\n".join(
        [
            "## NAVIGATION AND MOVEMENT",
            "[nav-00001] helpful=2 harmful=0 :: a",
            "",
            "## BATTLE TACTICS",
            "[btl-00002] helpful=0 harmful=1 :: b",
        ]
    )


def test_update_bullet_counts_helpful_harmful_neutral():
    updated = update_bullet_counts(
        _two_bullet_playbook(),
        [
            {"id": "nav-00001", "tag": "helpful"},
            {"id": "btl-00002", "tag": "harmful"},
        ],
    )
    assert "[nav-00001] helpful=3 harmful=0 :: a" in updated
    assert "[btl-00002] helpful=0 harmful=2 :: b" in updated

    neutral = update_bullet_counts(
        _two_bullet_playbook(), [{"id": "nav-00001", "tag": "neutral"}]
    )
    assert "[nav-00001] helpful=2 harmful=0 :: a" in neutral


def test_update_bullet_counts_empty_tags_is_noop():
    playbook = _two_bullet_playbook()
    assert update_bullet_counts(playbook, []) == playbook
    assert update_bullet_counts(playbook, None) == playbook


def test_update_bullet_counts_preserves_headers_and_blank_lines():
    updated = update_bullet_counts(
        _two_bullet_playbook(), [{"id": "nav-00001", "tag": "helpful"}]
    )
    assert "## NAVIGATION AND MOVEMENT" in updated
    assert "## BATTLE TACTICS" in updated
    assert updated.count("\n") == _two_bullet_playbook().count("\n")


def test_update_bullet_counts_accepts_legacy_bullet_key():
    updated = update_bullet_counts(
        _two_bullet_playbook(), [{"bullet": "nav-00001", "tag": "helpful"}]
    )
    assert "[nav-00001] helpful=3 harmful=0 :: a" in updated


# ---------------------------------------------------------------------------
# Delta merge (ADD-only)
# ---------------------------------------------------------------------------


def test_add_places_bullet_in_named_section():
    playbook, next_id, added = apply_curator_operations(
        empty_playbook(),
        [{"type": "ADD", "section": "battle_tactics", "content": "Use type advantage"}],
        1,
    )
    assert next_id == 2
    assert added == [
        {"id": "btl-00001", "section": "battle_tactics", "content": "Use type advantage"}
    ]

    lines = playbook.split("\n")
    header_idx = lines.index("## BATTLE TACTICS")
    bullet_idx = next(i for i, line in enumerate(lines) if "btl-00001" in line)
    assert bullet_idx > header_idx
    # And before the *next* header.
    next_header = next(
        (i for i, line in enumerate(lines) if i > header_idx and line.startswith("##")),
        len(lines),
    )
    assert bullet_idx < next_header


def test_add_unknown_section_falls_to_others():
    playbook, _, added = apply_curator_operations(
        empty_playbook(),
        [{"type": "ADD", "section": "quantum_chromodynamics", "content": "nope"}],
        1,
    )
    assert added[0]["section"] == "others"
    assert added[0]["id"] == "misc-00001"
    lines = playbook.split("\n")
    assert lines.index("## OTHERS") < next(
        i for i, line in enumerate(lines) if "misc-00001" in line
    )


def test_add_preserves_existing_bullets_and_ids():
    base, _, _ = apply_curator_operations(
        empty_playbook(),
        [{"type": "ADD", "section": "navigation_and_movement", "content": "first"}],
        1,
    )
    grown, next_id, added = apply_curator_operations(
        base,
        [{"type": "ADD", "section": "navigation_and_movement", "content": "second"}],
        2,
    )
    assert "[nav-00001] helpful=0 harmful=0 :: first" in grown
    assert "[nav-00002] helpful=0 harmful=0 :: second" in grown
    assert next_id == 3
    assert len(added) == 1


def test_non_add_operations_are_ignored():
    playbook, next_id, added = apply_curator_operations(
        empty_playbook(),
        [
            {"type": "DELETE", "bullet_id": "nav-00001"},
            {"type": "UPDATE", "bullet_id": "nav-00001", "content": "x"},
        ],
        1,
    )
    assert playbook == empty_playbook()
    assert next_id == 1
    assert added == []


def test_add_with_empty_content_is_skipped():
    playbook, next_id, added = apply_curator_operations(
        empty_playbook(),
        [{"type": "ADD", "section": "battle_tactics", "content": "   "}],
        1,
    )
    assert added == []
    assert next_id == 1
    assert playbook == empty_playbook()


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_raw_fenced_and_embedded():
    assert extract_json_from_text('{"a": 1}') == {"a": 1}
    assert extract_json_from_text('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json_from_text('blah blah {"a": 3} trailing') == {"a": 3}
    assert extract_json_from_text("not json at all") is None
    assert extract_json_from_text("") is None


def test_extract_json_handles_braces_inside_strings():
    payload = '{"reasoning": "use { and } carefully", "operations": []}'
    assert extract_json_from_text(payload)["operations"] == []


# ---------------------------------------------------------------------------
# Bullet extraction for the reflector
# ---------------------------------------------------------------------------


def test_extract_playbook_bullets_filters_unknown_ids():
    playbook = _two_bullet_playbook()
    rendered = extract_playbook_bullets(playbook, ["nav-00001", "zzz-99999"])
    assert "[nav-00001] helpful=2 harmful=0 :: a" in rendered
    assert "zzz-99999" not in rendered


def test_extract_playbook_bullets_sentinels():
    assert extract_playbook_bullets(_two_bullet_playbook(), []) == "(No bullets used by generator)"
    assert "none were found" in extract_playbook_bullets(_two_bullet_playbook(), ["zzz-99999"])


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


def test_citation_from_playbook_used_line():
    text = "I will walk north.\nPLAYBOOK_USED: [nav-00001, menu-00002]"
    assert extract_cited_bullet_ids(text) == ["nav-00001", "menu-00002"]


def test_citation_from_bracket_fallback():
    text = "Following [nav-00001] and also [btl-00004] here."
    assert extract_cited_bullet_ids(text) == ["nav-00001", "btl-00004"]


def test_citation_none_and_absent_are_empty():
    assert extract_cited_bullet_ids("PLAYBOOK_USED: none") == []
    assert extract_cited_bullet_ids("no citation at all") == []
    assert extract_cited_bullet_ids("") == []


def test_citation_deduplicates_preserving_order():
    text = "PLAYBOOK_USED: [nav-00001, nav-00001, btl-00002]"
    assert extract_cited_bullet_ids(text) == ["nav-00001", "btl-00002"]


def test_citation_line_wins_over_bracket_mentions():
    text = "I considered [btl-00009] but rejected it.\nPLAYBOOK_USED: [nav-00001]"
    assert extract_cited_bullet_ids(text) == ["nav-00001"]


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def _playbook_with(bullets):
    """bullets: list of (id, helpful, harmful, content)."""
    lines = ["## NAVIGATION AND MOVEMENT"]
    lines += [format_playbook_line(*b) for b in bullets]
    lines += ["", "## OTHERS"]
    return "\n".join(lines)


def test_prune_drops_harmful_before_unused_before_low_utility():
    playbook = _playbook_with(
        [
            ("nav-00001", 5, 1, "x" * 60),  # useful
            ("nav-00002", 0, 3, "y" * 60),  # harmful  -> tier 0
            ("nav-00003", 0, 0, "z" * 60),  # unused   -> tier 1
            ("nav-00004", 1, 0, "w" * 60),  # low net  -> tier 2
        ]
    )
    pruned_book, dropped = prune_to_budget(
        playbook, budget_tokens=count_tokens(playbook) - 20, min_bullets=1,
        age_by_id={}, current_window=99, unused_grace_windows=5,
    )
    assert dropped[0]["id"] == "nav-00002"
    assert "nav-00002" not in pruned_book
    if len(dropped) > 1:
        assert dropped[1]["id"] == "nav-00003"


def test_prune_respects_min_bullets_floor():
    playbook = _playbook_with([(f"nav-0000{i}", 0, 0, "q" * 80) for i in range(1, 5)])
    pruned_book, _ = prune_to_budget(
        playbook, budget_tokens=1, min_bullets=3, age_by_id={}, current_window=99
    )
    assert len(playbook_bullet_ids(pruned_book)) == 3


def test_prune_never_touches_section_headers():
    playbook = _playbook_with([(f"nav-0000{i}", 0, 0, "q" * 80) for i in range(1, 5)])
    pruned_book, _ = prune_to_budget(
        playbook, budget_tokens=1, min_bullets=1, age_by_id={}, current_window=99
    )
    assert "## NAVIGATION AND MOVEMENT" in pruned_book
    assert "## OTHERS" in pruned_book


def test_prune_protects_bullets_inside_grace_period():
    playbook = _playbook_with([(f"nav-0000{i}", 0, 0, "q" * 80) for i in range(1, 5)])
    age = {f"nav-0000{i}": 10 for i in range(1, 5)}
    pruned_book, dropped = prune_to_budget(
        playbook, budget_tokens=1, min_bullets=1,
        age_by_id=age, current_window=11, unused_grace_windows=5,
    )
    assert dropped == []
    assert pruned_book == playbook


def test_prune_never_reuses_ids():
    playbook = _playbook_with([(f"nav-0000{i}", 0, 5, "q" * 80) for i in range(1, 5)])
    pruned_book, _ = prune_to_budget(
        playbook, budget_tokens=1, min_bullets=1, age_by_id={}, current_window=99
    )
    # The id counter is derived from the max id ever seen, and pruning must not
    # lower it in a way that would let a future ADD collide.
    assert get_next_global_id(playbook) >= get_next_global_id(pruned_book)
    surviving = playbook_bullet_ids(pruned_book)
    assert len(set(surviving)) == len(surviving)


def test_prune_is_a_noop_when_under_budget():
    playbook = _two_bullet_playbook()
    pruned_book, dropped = prune_to_budget(playbook, budget_tokens=100_000, min_bullets=1)
    assert pruned_book == playbook
    assert dropped == []


def test_count_tokens_is_positive_for_nonempty():
    assert count_tokens("") == 0
    assert count_tokens("abcd") >= 1
