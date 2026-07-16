import pytest

from free_claude_code.application.reasoning import (
    ReasoningEffort,
    ReasoningPolicy,
    resolve_reasoning_policy,
)
from free_claude_code.core.anthropic.models import Message, MessagesRequest


def _request(**overrides: object) -> MessagesRequest:
    data: dict[str, object] = {
        "model": "provider/model",
        "messages": [Message(role="user", content="hello")],
    }
    data.update(overrides)
    return MessagesRequest.model_validate(data)


@pytest.mark.parametrize(
    "message_request",
    [
        _request(thinking={"type": "disabled"}),
        _request(thinking={"type": "adaptive", "enabled": False}),
        _request(output_config={"effort": "none"}),
    ],
)
def test_request_can_disable_reasoning(message_request: MessagesRequest) -> None:
    assert (
        resolve_reasoning_policy(message_request, route_enabled=True)
        == ReasoningPolicy.off()
    )


def test_route_gate_overrides_request_compute_intent() -> None:
    request = _request(
        thinking={"type": "enabled", "budget_tokens": 8192},
        output_config={"effort": "max"},
    )

    assert (
        resolve_reasoning_policy(request, route_enabled=False) == ReasoningPolicy.off()
    )


def test_resolver_preserves_explicit_effort_and_budget() -> None:
    request = _request(
        thinking={"type": "enabled", "budget_tokens": 4096},
        output_config={"effort": "xhigh"},
    )

    assert resolve_reasoning_policy(request, route_enabled=True) == ReasoningPolicy.on(
        effort=ReasoningEffort.XHIGH,
        budget_tokens=4096,
    )


@pytest.mark.parametrize(
    ("output_config", "expected"),
    [
        ({"effort": " HIGH "}, ReasoningEffort.HIGH),
        ({"effort": "ultracode"}, None),
        ({"effort": 3}, None),
        (None, None),
    ],
)
def test_resolver_accepts_only_canonical_efforts(
    output_config: object,
    expected: ReasoningEffort | None,
) -> None:
    policy = resolve_reasoning_policy(
        _request(output_config=output_config),
        route_enabled=True,
    )

    assert policy == ReasoningPolicy.on(effort=expected)


def test_resolver_allows_enabled_reasoning_without_a_budget() -> None:
    policy = resolve_reasoning_policy(
        _request(thinking={"type": "enabled"}),
        route_enabled=True,
    )

    assert policy == ReasoningPolicy.on()


def test_disabled_policy_rejects_compute_controls() -> None:
    with pytest.raises(ValueError, match="Disabled reasoning"):
        ReasoningPolicy(enabled=False, effort=ReasoningEffort.LOW)


@pytest.mark.parametrize("budget", [0, -1, True])
def test_enabled_policy_rejects_invalid_budgets(budget: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ReasoningPolicy.on(budget_tokens=budget)
