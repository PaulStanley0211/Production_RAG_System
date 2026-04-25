"""Filter builder — converts plain dicts to Qdrant Filter objects.

Accepts simple dicts and produces Qdrant Filter trees. Supports:
- Exact match:  {"source": "manual.pdf"}
- One-of match: {"source": ["manual.pdf", "guide.pdf"]}
- Range match:  {"page_count": {"gte": 5, "lte": 100}}
- Combined:     all conditions are AND-ed together

Phase 3 enhancement could add OR / NOT support; for now AND covers 90% of cases.
"""

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)


class FilterBuilder:
    """Translate user-friendly filter dicts to Qdrant Filter objects."""

    def from_dict(self, filters: dict | None) -> Filter | None:
        """Build a Qdrant Filter from a flat dict. Returns None if empty/None."""
        if not filters:
            return None

        conditions: list[FieldCondition] = []
        for field, value in filters.items():
            condition = self._build_condition(field, value)
            if condition is not None:
                conditions.append(condition)

        if not conditions:
            return None

        # All conditions are AND-ed (must match)
        return Filter(must=conditions)

    def _build_condition(self, field: str, value) -> FieldCondition | None:
        """Build one condition from a (field, value) pair."""
        # List → match any of these values
        if isinstance(value, list):
            if not value:
                return None
            return FieldCondition(key=field, match=MatchAny(any=value))

        # Dict with range operators → numeric range
        if isinstance(value, dict):
            range_kwargs = {}
            if "gte" in value:
                range_kwargs["gte"] = value["gte"]
            if "lte" in value:
                range_kwargs["lte"] = value["lte"]
            if "gt" in value:
                range_kwargs["gt"] = value["gt"]
            if "lt" in value:
                range_kwargs["lt"] = value["lt"]
            if not range_kwargs:
                return None
            return FieldCondition(key=field, range=Range(**range_kwargs))

        # Scalar → exact match
        return FieldCondition(key=field, match=MatchValue(value=value))