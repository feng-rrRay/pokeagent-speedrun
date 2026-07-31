"""ACE Curator.

Mirrors ``ace/ace/core/curator.py`` from the vendored reference clone, with the
reference's OpenAI client swapped for this repo's :class:`VLM`.

The Curator emits **append-only delta operations** and never rewrites the
playbook itself; the merge is done by deterministic non-LLM code in
``agents.ace.playbook.apply_curator_operations``.  The paper's ablation shows
this incremental-delta design is the load-bearing component (removing it costs
-11.7 TGC / -27.8 SGC on AppWorld), and that full-context rewriting is what
causes context collapse.

Every failure path is non-fatal and returns the playbook unchanged.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from agents.ace.playbook import (
    ACE_SECTION_KEYS,
    apply_curator_operations,
    extract_json_from_text,
)
from agents.ace.prompts import ACE_CURATOR_PROMPT_NO_GT

logger = logging.getLogger(__name__)

CURATOR_MODULE_NAME = "ace_curator"


class AceCurator:
    """Turns a reflection into append-only playbook deltas."""

    def __init__(self, vlm, game_name: str = "Pokemon Red"):
        self.vlm = vlm
        self.game_name = game_name

    def curate(
        self,
        current_playbook: str,
        recent_reflection: str,
        question_context: str,
        current_step: int,
        total_samples: Any,
        token_budget: int,
        playbook_stats: Dict[str, Any],
        next_global_id: int = 1,
    ) -> Tuple[str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run one curation round.

        Returns ``(playbook, next_global_id, operations, added)``.  On any
        failure the playbook and id counter come back untouched.
        """
        prompt = ACE_CURATOR_PROMPT_NO_GT.format(
            game_name=self.game_name,
            current_step=current_step,
            total_samples=total_samples,
            token_budget=token_budget,
            playbook_stats=json.dumps(playbook_stats, indent=2),
            recent_reflection=recent_reflection,
            current_playbook=current_playbook,
            question_context=question_context,
            section_names="\n".join(f"- {key}" for key in ACE_SECTION_KEYS),
        )

        try:
            response = self.vlm.get_text_query(prompt, CURATOR_MODULE_NAME)
        except Exception as exc:
            logger.error("ACE curator call failed: %s", exc)
            return current_playbook, next_global_id, [], []

        if not isinstance(response, str) or not response.strip():
            logger.warning("ACE curator returned an empty response; skipping curation")
            return current_playbook, next_global_id, [], []

        try:
            operations = self._extract_and_validate_operations(response)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "ACE curator JSON parsing failed (%s); skipping curation. Preview: %s",
                exc,
                response[:300],
            )
            return current_playbook, next_global_id, [], []

        try:
            updated, next_global_id, added = apply_curator_operations(
                current_playbook, operations, next_global_id
            )
        except Exception as exc:
            logger.error("ACE curator failed to apply operations: %s", exc)
            return current_playbook, next_global_id, [], []

        return updated, next_global_id, operations, added

    @staticmethod
    def _extract_and_validate_operations(response: str) -> List[Dict[str, Any]]:
        """Validate the curator's JSON contract (same checks as the reference)."""
        info = extract_json_from_text(response)

        if not info:
            raise ValueError("Failed to extract valid JSON from curator response")
        if "reasoning" not in info:
            raise ValueError("JSON missing required 'reasoning' field")
        if "operations" not in info:
            raise ValueError("JSON missing required 'operations' field")
        if not isinstance(info["reasoning"], str):
            raise ValueError("'reasoning' field must be a string")
        if not isinstance(info["operations"], list):
            raise ValueError("'operations' field must be a list")

        for i, op in enumerate(info["operations"]):
            if not isinstance(op, dict):
                raise ValueError(f"Operation {i} must be a dictionary")
            if "type" not in op:
                raise ValueError(f"Operation {i} missing required 'type' field")
            if op["type"] == "ADD":
                missing = {"type", "section", "content"} - set(op.keys())
                if missing:
                    raise ValueError(f"ADD operation {i} missing fields: {sorted(missing)}")
            else:
                # Only ADD is implemented, here and in the reference.
                logger.warning("Curator emitted unsupported operation type '%s'; ignoring", op["type"])

        return info["operations"]
