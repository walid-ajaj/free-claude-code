import pytest

from free_claude_code.application.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.openai_chat.reasoning import (
    encode_cerebras_reasoning,
    encode_fireworks_reasoning,
    encode_minimax_reasoning,
    encode_ollama_reasoning,
    encode_sambanova_reasoning,
)


def _request(model: str) -> MessagesRequest:
    return MessagesRequest(
        model=model,
        messages=[Message(role="user", content="hello")],
    )


@pytest.mark.parametrize(
    ("model", "policy", "expected"),
    [
        (
            "MiniMax-M3",
            ReasoningPolicy.on(),
            {
                "reasoning_split": True,
                "thinking": {"type": "adaptive"},
            },
        ),
        (
            "MiniMax-M3",
            ReasoningPolicy.off(),
            {
                "reasoning_split": True,
                "thinking": {"type": "disabled"},
            },
        ),
        (
            "MiniMax-M2.7",
            ReasoningPolicy.off(),
            {"reasoning_split": True},
        ),
    ],
)
def test_minimax_reasoning_contract(
    model: str,
    policy: ReasoningPolicy,
    expected: dict[str, object],
) -> None:
    body: dict[str, object] = {}

    encode_minimax_reasoning(body, _request(model), policy)

    assert body == {"extra_body": expected}


@pytest.mark.parametrize(
    ("model", "policy", "expected"),
    [
        (
            "gpt-oss-120b",
            ReasoningPolicy.on(),
            {"reasoning_effort": "medium"},
        ),
        ("gpt-oss-120b", ReasoningPolicy.off(), {}),
        ("zai-glm-4.7", ReasoningPolicy.on(), {}),
        (
            "zai-glm-4.7",
            ReasoningPolicy.off(),
            {"reasoning_effort": "none"},
        ),
        (
            "gemma-4-31b",
            ReasoningPolicy.on(),
            {"reasoning_effort": "medium"},
        ),
        (
            "gemma-4-31b",
            ReasoningPolicy.off(),
            {"reasoning_effort": "none"},
        ),
        ("llama-3.3-70b", ReasoningPolicy.on(), {}),
    ],
)
def test_cerebras_reasoning_contract(
    model: str,
    policy: ReasoningPolicy,
    expected: dict[str, object],
) -> None:
    body: dict[str, object] = {}

    encode_cerebras_reasoning(body, _request(model), policy)

    assert body == expected


@pytest.mark.parametrize(
    ("model", "policy", "expected"),
    [
        (
            "DeepSeek-V3.2",
            ReasoningPolicy.on(),
            {"chat_template_kwargs": {"enable_thinking": True}},
        ),
        (
            "deepseek-v3p2",
            ReasoningPolicy.off(),
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            "gemma-4-31B-it",
            ReasoningPolicy.on(),
            {"chat_template_kwargs": {"enable_thinking": True}},
        ),
        ("gpt-oss-120b", ReasoningPolicy.on(), None),
    ],
)
def test_sambanova_reasoning_contract(
    model: str,
    policy: ReasoningPolicy,
    expected: dict[str, object] | None,
) -> None:
    body: dict[str, object] = {}

    encode_sambanova_reasoning(body, _request(model), policy)

    assert body == ({"extra_body": expected} if expected is not None else {})


@pytest.mark.parametrize(
    ("model", "policy", "expected"),
    [
        (
            "accounts/fireworks/models/gpt-oss-120b",
            ReasoningPolicy.on(),
            {"reasoning_effort": "medium"},
        ),
        (
            "accounts/fireworks/models/gpt-oss-120b",
            ReasoningPolicy.off(),
            {},
        ),
        (
            "accounts/fireworks/models/deepseek-v3p1",
            ReasoningPolicy.on(),
            {"reasoning_effort": True},
        ),
        (
            "accounts/fireworks/models/deepseek-v3p2",
            ReasoningPolicy.off(),
            {"reasoning_effort": False},
        ),
        (
            "accounts/fireworks/models/deepseek-v4",
            ReasoningPolicy.on(effort=ReasoningEffort.XHIGH),
            {"reasoning_effort": "xhigh"},
        ),
        (
            "accounts/fireworks/models/deepseek-v4",
            ReasoningPolicy.off(),
            {"reasoning_effort": "none"},
        ),
        (
            "accounts/fireworks/models/qwen3-235b",
            ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
            {"reasoning_effort": "high"},
        ),
        (
            "accounts/fireworks/models/qwen3-235b",
            ReasoningPolicy.on(budget_tokens=3072),
            {"extra_body": {"reasoning_effort": 3072}},
        ),
        (
            "accounts/fireworks/models/qwen3-235b",
            ReasoningPolicy.off(),
            {"reasoning_effort": "none"},
        ),
        (
            "accounts/fireworks/models/glm-5p1",
            ReasoningPolicy.on(),
            {"reasoning_effort": True},
        ),
        (
            "accounts/fireworks/models/glm-5p2",
            ReasoningPolicy.on(),
            {"reasoning_effort": "max"},
        ),
        (
            "accounts/fireworks/models/glm-5p2",
            ReasoningPolicy.off(),
            {"reasoning_effort": "none"},
        ),
        ("accounts/fireworks/models/llama-v4", ReasoningPolicy.on(), {}),
    ],
)
def test_fireworks_reasoning_contract(
    model: str,
    policy: ReasoningPolicy,
    expected: dict[str, object],
) -> None:
    body: dict[str, object] = {}

    encode_fireworks_reasoning(body, _request(model), policy)

    assert body == expected


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ReasoningPolicy.on(), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "max"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "max"),
        (ReasoningPolicy.off(), "none"),
    ],
)
def test_ollama_reasoning_contract(
    policy: ReasoningPolicy,
    expected: str,
) -> None:
    body: dict[str, object] = {}

    encode_ollama_reasoning(body, _request("gpt-oss:20b"), policy)

    assert body == {"reasoning_effort": expected}
