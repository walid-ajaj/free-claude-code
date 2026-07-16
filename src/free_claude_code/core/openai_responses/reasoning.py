"""Reasoning and thinking conversion helpers for OpenAI Responses."""

from collections.abc import Mapping
from typing import Any

from .tools import optional_str


def reasoning_text_from_item(item: Mapping[str, Any]) -> str | None:
    content_parts = _text_parts_from_items(
        item.get("content"), item_type="reasoning_text"
    )
    if content_parts:
        return "\n".join(content_parts)
    summary_parts = _text_parts_from_items(
        item.get("summary"), item_type="summary_text"
    )
    if summary_parts:
        return "\n".join(summary_parts)
    return None


def combine_reasoning(existing: str | None, addition: str | None) -> str | None:
    if addition is None:
        return existing
    if existing is None:
        return addition
    if existing == "":
        return addition
    if addition == "":
        return existing
    return f"{existing}\n{addition}"


def responses_reasoning_to_anthropic_fields(value: Any) -> dict[str, Any]:
    """Preserve Responses reasoning enablement and effort for application routing."""
    if not isinstance(value, Mapping):
        return {}
    effort = value.get("effort")
    if effort == "none":
        return {"thinking": {"type": "disabled", "enabled": False}}
    fields: dict[str, Any] = {}
    if any(item is not None for item in value.values()):
        fields["thinking"] = {"type": "adaptive", "enabled": True}
    if isinstance(effort, str) and effort.strip():
        fields["output_config"] = {"effort": effort}
    return fields


def _text_parts_from_items(value: Any, *, item_type: str) -> list[str]:
    if not isinstance(value, list):
        return []
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("type") == item_type:
            text = optional_str(item.get("text"))
            if text is not None:
                parts.append(text)
    return parts
