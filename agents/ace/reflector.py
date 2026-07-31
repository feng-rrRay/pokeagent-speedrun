"""ACE Reflector.

Mirrors ``ace/ace/core/reflector.py`` from the vendored reference clone, with
the reference's OpenAI client swapped for this repo's :class:`VLM` so that token
accounting, pricing and logging stay unified.

The Reflector diagnoses one segment and tags the bullets the generator cited.
It **never writes the playbook** -- that separation is core to ACE.  Every
failure path is non-fatal: a bad response yields no tags, which
``update_bullet_counts`` treats as a no-op, and the run continues.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.ace.playbook import extract_json_from_text
from agents.ace.prompts import ACE_REFLECTOR_PROMPT_NO_GT

logger = logging.getLogger(__name__)

REFLECTOR_MODULE_NAME = "ace_reflector"


class AceReflector:
    """Diagnoses a play segment and tags playbook bullets helpful/harmful/neutral."""

    def __init__(self, vlm, game_name: str = "Pokemon Red"):
        self.vlm = vlm
        self.game_name = game_name

    def reflect(
        self,
        task: str,
        trace: str,
        actions: str,
        environment_feedback: str,
        bullets_used: str,
        window_start: int,
        window_end: int,
        previous_reflection: str = "(none)",
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], List[Dict[str, str]]]:
        """Run one reflection round.

        Returns ``(raw_response, parsed_json, bullet_tags)``.  Any of these may
        be ``None``/empty on failure; callers must treat that as "no update".
        """
        prompt = ACE_REFLECTOR_PROMPT_NO_GT.format(
            game_name=self.game_name,
            task=task,
            trace=trace,
            actions=actions,
            previous_reflection=previous_reflection or "(none)",
            environment_feedback=environment_feedback,
            bullets_used=bullets_used,
            window_start=window_start,
            window_end=window_end,
        )

        try:
            response = self.vlm.get_text_query(prompt, REFLECTOR_MODULE_NAME)
        except Exception as exc:
            logger.error("ACE reflector call failed: %s", exc)
            return None, None, []

        if not isinstance(response, str) or not response.strip():
            logger.warning("ACE reflector returned an empty response")
            return None, None, []

        parsed = extract_json_from_text(response)
        tags = self._extract_bullet_tags(response, parsed)
        return response, parsed, tags

    @staticmethod
    def _extract_bullet_tags(response: str, parsed: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Pull ``bullet_tags`` out of the response.

        Prefers the fully parsed object; falls back to the reference's
        bracket-depth scan anchored on the literal ``"bullet_tags"`` key, which
        survives responses whose surrounding JSON is malformed.
        """
        if isinstance(parsed, dict):
            tags = parsed.get("bullet_tags")
            if isinstance(tags, list):
                return [t for t in tags if isinstance(t, dict)]

        try:
            start_idx = response.find('"bullet_tags"')
            if start_idx == -1:
                return []
            bracket_idx = response.find("[", start_idx)
            if bracket_idx == -1:
                return []

            depth = 0
            end_idx = bracket_idx
            for i in range(bracket_idx, len(response)):
                if response[i] == "[":
                    depth += 1
                elif response[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

            tags = json.loads(response[bracket_idx:end_idx])
            if isinstance(tags, list):
                return [t for t in tags if isinstance(t, dict)]
        except Exception as exc:
            logger.warning("Failed to extract bullet tags: %s", exc)

        return []


def filter_tags_to_playbook(bullet_tags: List[Dict[str, str]], known_ids) -> List[Dict[str, str]]:
    """Drop tags for ids that are not actually in the playbook.

    Guards against a hallucinated id silently creating no-op counter updates,
    and keeps the reflector honest about the "do not invent bullet ids" rule.
    """
    known = set(known_ids)
    kept = []
    for tag in bullet_tags:
        bullet_id = tag.get("id") or tag.get("bullet")
        if bullet_id in known:
            kept.append({"id": bullet_id, "tag": tag.get("tag", "neutral")})
    return kept
