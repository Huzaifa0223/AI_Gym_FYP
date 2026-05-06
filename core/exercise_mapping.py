"""
core.exercise_mapping — single source of truth for body-part → exercise routing.

The system has three layers that talk about exercises in different vocabularies:

* :mod:`core.exercise_classifier` — emits **body-part categories**
  (``arms``, ``back``, ``chest``, ``legs``, ``shoulders``, ``unknown``).
  These describe what the user is *moving*, not which exercise they are doing.
* :data:`BODY_PART_TO_EXERCISE` — this file. Maps body-part categories to
  the **specific exercise key** the rule engines and ML aggregator are
  actually keyed on.
* Rule engines / ML / configs — keyed on the **specific exercise key**
  (``bicep_curl``, ``bent_over_row``, ``push_up``).

Body parts that have no mapped exercise (``legs``, ``shoulders``,
``unknown``) are intentionally absent from the mapping. Callers that
``.get()`` them receive ``None`` and should fall through to a
:class:`~core.rule_engine.NullRuleEngine` / user-provided default.
This preserves the project's graceful-degradation behaviour for body
parts the system does not yet score.
"""
from __future__ import annotations

from typing import Final


BODY_PART_TO_EXERCISE: Final[dict[str, str]] = {
    "arms": "bicep_curl",
    "back": "bent_over_row",
    "chest": "push_up",
    # legs, shoulders, unknown intentionally absent — flow to
    # NullRuleEngine / caller default.
}


__all__ = ["BODY_PART_TO_EXERCISE"]
