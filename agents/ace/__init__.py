"""ACE (Agentic Context Engineering) scaffold support.

A port of https://github.com/ace-agent/ace (arXiv 2510.04618) to this repo's
reset-free Pokemon setting.  The reference implementation is vendored read-only
at the repository root under ``ace/``; nothing here imports from it.

Public surface is deliberately small: ``PokeAgent`` only touches
:func:`create_ace_controller` and the controller's three public methods.
"""

from agents.ace.controller import AceController, create_ace_controller
from agents.ace.playbook import (
    ACE_SECTIONS,
    ACE_SECTION_KEYS,
    ACE_SLUG_MAP,
    empty_playbook,
)

__all__ = [
    "AceController",
    "create_ace_controller",
    "ACE_SECTIONS",
    "ACE_SECTION_KEYS",
    "ACE_SLUG_MAP",
    "empty_playbook",
]
