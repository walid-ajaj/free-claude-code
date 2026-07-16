"""Validation helpers for OpenAI-chat ``extra_body`` passthrough."""

from typing import Any

CANONICAL_OPENAI_CHAT_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "metadata",
        "stop",
        "stop_sequences",
        "stream_options",
        "reasoning",
        "reasoning_effort",
        "reasoning_tokens",
        "thinking",
        "thinking_budget_tokens",
        "chat_template_kwargs",
    }
)


def validate_extra_body_does_not_override_canonical_fields(
    extra: dict[str, Any],
) -> None:
    """Reject extras that would replace FCC-owned chat-completion fields."""
    bad = CANONICAL_OPENAI_CHAT_BODY_KEYS & extra.keys()
    if bad:
        raise ValueError(
            f"extra_body must not override canonical request fields: {sorted(bad)}"
        )
