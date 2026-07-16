"""Canonical reasoning intent resolved at the application boundary."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest


class ReasoningEffort(StrEnum):
    """Provider-neutral reasoning effort ordered from least to most compute."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class ReasoningPolicy:
    """Resolved request intent before provider-specific wire translation."""

    enabled: bool
    effort: ReasoningEffort | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.enabled and (
            self.effort is not None or self.budget_tokens is not None
        ):
            raise ValueError("Disabled reasoning cannot carry effort or budget.")
        if self.budget_tokens is not None and (
            isinstance(self.budget_tokens, bool) or self.budget_tokens <= 0
        ):
            raise ValueError("Reasoning budget must be positive.")

    @classmethod
    def off(cls) -> ReasoningPolicy:
        """Return a disabled policy."""
        return cls(enabled=False)

    @classmethod
    def on(
        cls,
        *,
        effort: ReasoningEffort | None = None,
        budget_tokens: int | None = None,
    ) -> ReasoningPolicy:
        """Return an enabled policy with optional compute intent."""
        return cls(
            enabled=True,
            effort=effort,
            budget_tokens=budget_tokens,
        )


def resolve_reasoning_policy(
    request: MessagesRequest,
    *,
    route_enabled: bool,
) -> ReasoningPolicy:
    """Combine route configuration and protocol request intent exactly once."""
    if not route_enabled or _request_disables_reasoning(request):
        return ReasoningPolicy.off()

    return ReasoningPolicy.on(
        effort=_request_effort(request.output_config),
        budget_tokens=(
            request.thinking.budget_tokens if request.thinking is not None else None
        ),
    )


def _request_disables_reasoning(request: MessagesRequest) -> bool:
    thinking = request.thinking
    if thinking is not None:
        if thinking.type == "disabled":
            return True
        if "enabled" in thinking.model_fields_set and thinking.enabled is False:
            return True

    output_config = request.output_config
    if not isinstance(output_config, dict):
        return False
    effort = output_config.get("effort")
    return isinstance(effort, str) and effort.strip().lower() == "none"


def _request_effort(output_config: Any) -> ReasoningEffort | None:
    if not isinstance(output_config, dict):
        return None
    raw_effort = output_config.get("effort")
    if not isinstance(raw_effort, str):
        return None
    try:
        return ReasoningEffort(raw_effort.strip().lower())
    except ValueError:
        return None
