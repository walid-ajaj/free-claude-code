import pytest

from free_claude_code.application.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.reasoning import (
    reasoning_budget_tokens,
    reasoning_effort,
)


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (ReasoningEffort.MINIMAL, 256),
        (ReasoningEffort.LOW, 1024),
        (ReasoningEffort.MEDIUM, 2048),
        (ReasoningEffort.HIGH, 4096),
        (ReasoningEffort.XHIGH, 8192),
        (ReasoningEffort.MAX, 16384),
    ],
)
def test_fcc_effort_budget_ladder(
    effort: ReasoningEffort,
    budget: int,
) -> None:
    assert reasoning_budget_tokens(ReasoningPolicy.on(effort=effort)) == budget


def test_explicit_budget_wins_over_effort_for_budget_provider() -> None:
    policy = ReasoningPolicy.on(
        effort=ReasoningEffort.MAX,
        budget_tokens=777,
    )

    assert reasoning_budget_tokens(policy) == 777


def test_explicit_budget_wins_over_effort_for_effort_provider() -> None:
    policy = ReasoningPolicy.on(
        effort=ReasoningEffort.MINIMAL,
        budget_tokens=5000,
    )

    assert (
        reasoning_effort(
            policy,
            (
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            ),
        )
        == ReasoningEffort.HIGH
    )


def test_effort_mapping_prefers_higher_level_on_tie() -> None:
    assert (
        reasoning_effort(
            ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM),
            (ReasoningEffort.LOW, ReasoningEffort.HIGH),
        )
        == ReasoningEffort.HIGH
    )


def test_effort_mapping_uses_provider_default_only_without_compute_hint() -> None:
    assert (
        reasoning_effort(
            ReasoningPolicy.on(),
            (ReasoningEffort.LOW, ReasoningEffort.HIGH),
            default=ReasoningEffort.LOW,
        )
        == ReasoningEffort.LOW
    )


def test_disabled_policy_has_no_budget_or_effort() -> None:
    policy = ReasoningPolicy.off()

    assert reasoning_budget_tokens(policy) is None
    assert reasoning_effort(policy, (ReasoningEffort.HIGH,)) is None


def test_effort_mapping_requires_supported_levels() -> None:
    with pytest.raises(ValueError, match="At least one"):
        reasoning_effort(
            ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
            (),
        )
