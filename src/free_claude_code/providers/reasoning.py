"""Shared conversion primitives for provider-owned reasoning encoders."""

from collections.abc import Sequence

from free_claude_code.application.reasoning import ReasoningEffort, ReasoningPolicy

_EFFORT_ORDER = tuple(ReasoningEffort)
_EFFORT_BUDGET_TOKENS = {
    ReasoningEffort.MINIMAL: 256,
    ReasoningEffort.LOW: 1_024,
    ReasoningEffort.MEDIUM: 2_048,
    ReasoningEffort.HIGH: 4_096,
    ReasoningEffort.XHIGH: 8_192,
    ReasoningEffort.MAX: 16_384,
}


def reasoning_budget_tokens(policy: ReasoningPolicy) -> int | None:
    """Return an explicit budget or FCC's documented effort-to-budget mapping."""
    if not policy.enabled:
        return None
    if policy.budget_tokens is not None:
        return policy.budget_tokens
    if policy.effort is None:
        return None
    return _EFFORT_BUDGET_TOKENS[policy.effort]


def reasoning_effort(
    policy: ReasoningPolicy,
    supported: Sequence[ReasoningEffort],
    *,
    default: ReasoningEffort | None = None,
) -> ReasoningEffort | None:
    """Return the closest supported effort, preferring the higher tier on ties."""
    if not policy.enabled:
        return None
    requested: ReasoningEffort | None = None
    if policy.budget_tokens is not None:
        requested = _effort_for_budget(policy.budget_tokens)
    elif policy.effort is not None:
        requested = policy.effort
    if requested is None:
        return default
    return _closest_supported_effort(requested, supported)


def _effort_for_budget(budget_tokens: int) -> ReasoningEffort:
    for effort in _EFFORT_ORDER:
        if budget_tokens <= _EFFORT_BUDGET_TOKENS[effort]:
            return effort
    return ReasoningEffort.MAX


def _closest_supported_effort(
    requested: ReasoningEffort,
    supported: Sequence[ReasoningEffort],
) -> ReasoningEffort:
    if not supported:
        raise ValueError("At least one supported reasoning effort is required.")
    target = _EFFORT_ORDER.index(requested)
    return min(
        supported,
        key=lambda effort: (
            abs(_EFFORT_ORDER.index(effort) - target),
            -_EFFORT_ORDER.index(effort),
        ),
    )
