"""Provider-specific reasoning encoders for OpenAI-compatible chat APIs."""

from typing import Any

from free_claude_code.application.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.reasoning import (
    reasoning_budget_tokens,
    reasoning_effort,
)

_LOW_MEDIUM_HIGH = (
    ReasoningEffort.LOW,
    ReasoningEffort.MEDIUM,
    ReasoningEffort.HIGH,
)
_MINIMAL_TO_XHIGH = (
    ReasoningEffort.MINIMAL,
    ReasoningEffort.LOW,
    ReasoningEffort.MEDIUM,
    ReasoningEffort.HIGH,
    ReasoningEffort.XHIGH,
)
_LOW_TO_MAX = (
    ReasoningEffort.LOW,
    ReasoningEffort.MEDIUM,
    ReasoningEffort.HIGH,
    ReasoningEffort.MAX,
)
_FIREWORKS_V4_EFFORTS = (
    ReasoningEffort.LOW,
    ReasoningEffort.MEDIUM,
    ReasoningEffort.HIGH,
    ReasoningEffort.XHIGH,
    ReasoningEffort.MAX,
)


def encode_vercel_reasoning(
    body: dict[str, Any],
    _request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Use Vercel AI Gateway's OpenAI-compatible effort control."""
    _encode_optional_effort(body, policy, _MINIMAL_TO_XHIGH, disabled="none")


def encode_huggingface_reasoning(
    body: dict[str, Any],
    _request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Use Hugging Face Inference Providers' normalized effort control."""
    _encode_optional_effort(body, policy, _MINIMAL_TO_XHIGH, disabled="none")


def encode_cohere_reasoning(
    body: dict[str, Any],
    _request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Map Cohere's binary none/high compatibility contract."""
    body["reasoning_effort"] = "high" if policy.enabled else "none"


def encode_wafer_reasoning(
    body: dict[str, Any],
    _request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Map Wafer's documented thinking toggle."""
    _extra_body(body)["thinking"] = {
        "type": "enabled" if policy.enabled else "disabled"
    }


def encode_kimi_reasoning(
    body: dict[str, Any],
    request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Disable configurable Kimi thinking while leaving mandatory models alone."""
    model = request.model.lower()
    if "kimi-k2.7" in model and "code" in model:
        return
    if not policy.enabled:
        _extra_body(body)["thinking"] = {"type": "disabled"}


def encode_minimax_reasoning(
    body: dict[str, Any],
    request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Use MiniMax M3 adaptive/disabled control; M2 reasoning is mandatory."""
    extra = _extra_body(body)
    extra["reasoning_split"] = True
    if "minimax-m3" not in request.model.lower():
        return
    extra["thinking"] = {"type": "adaptive" if policy.enabled else "disabled"}


def encode_cerebras_reasoning(
    body: dict[str, Any],
    request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Apply only the model-family controls documented by Cerebras."""
    model = request.model.lower()
    if "gpt-oss" in model:
        if not policy.enabled:
            return
        effort = reasoning_effort(
            policy,
            _LOW_MEDIUM_HIGH,
            default=ReasoningEffort.MEDIUM,
        )
        assert effort is not None
        body["reasoning_effort"] = effort.value
        return
    if "zai-glm" in model and not policy.enabled:
        body["reasoning_effort"] = "none"
        return
    if "gemma-4" in model:
        if not policy.enabled:
            body["reasoning_effort"] = "none"
            return
        effort = reasoning_effort(
            policy,
            _LOW_MEDIUM_HIGH,
            default=ReasoningEffort.MEDIUM,
        )
        assert effort is not None
        body["reasoning_effort"] = effort.value


def encode_groq_reasoning(
    body: dict[str, Any],
    request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Apply Groq's distinct Qwen and GPT-OSS reasoning contracts."""
    model = request.model.lower()
    if "gpt-oss" in model:
        if not policy.enabled:
            return
        effort = reasoning_effort(
            policy,
            _LOW_MEDIUM_HIGH,
            default=ReasoningEffort.MEDIUM,
        )
        assert effort is not None
        body["reasoning_effort"] = effort.value
        return
    if "qwen" in model and "3" in model:
        body["reasoning_effort"] = "default" if policy.enabled else "none"


def encode_sambanova_reasoning(
    body: dict[str, Any],
    request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Use SambaNova's documented model-family thinking toggle."""
    model = request.model.lower()
    if not (
        ("deepseek" in model and _has_version(model, "v3.2", "v3p2"))
        or "gemma-4" in model
    ):
        return
    chat_template = _nested_dict(_extra_body(body), "chat_template_kwargs")
    chat_template["enable_thinking"] = policy.enabled


def encode_fireworks_reasoning(
    body: dict[str, Any],
    request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Apply Fireworks' documented model-specific reasoning validation."""
    model = request.model.lower()
    if "gpt-oss" in model or "harmony" in model or "minimax-m2" in model:
        if not policy.enabled:
            return
        effort = reasoning_effort(
            policy,
            _LOW_MEDIUM_HIGH,
            default=ReasoningEffort.MEDIUM,
        )
        assert effort is not None
        body["reasoning_effort"] = effort.value
        return
    if "deepseek" in model and "v4" in model:
        if not policy.enabled:
            body["reasoning_effort"] = "none"
            return
        effort = reasoning_effort(
            policy,
            _FIREWORKS_V4_EFFORTS,
            default=ReasoningEffort.HIGH,
        )
        assert effort is not None
        body["reasoning_effort"] = effort.value
        return
    if "qwen3" in model:
        if not policy.enabled:
            body["reasoning_effort"] = "none"
            return
        if policy.budget_tokens is not None:
            _extra_body(body)["reasoning_effort"] = policy.budget_tokens
            return
        effort = reasoning_effort(policy, _LOW_MEDIUM_HIGH)
        if effort is not None:
            body["reasoning_effort"] = effort.value
        return
    if "glm-5.2" in model or "glm-5p2" in model:
        if not policy.enabled:
            body["reasoning_effort"] = "none"
            return
        effort = reasoning_effort(
            policy,
            (ReasoningEffort.HIGH, ReasoningEffort.MAX),
            default=ReasoningEffort.MAX,
        )
        assert effort is not None
        body["reasoning_effort"] = effort.value
        return
    if (
        "deepseek" in model and _has_version(model, "v3.1", "v3p1", "v3.2", "v3p2")
    ) or ("glm-" in model or "glm_" in model):
        body["reasoning_effort"] = policy.enabled


def encode_zai_reasoning(
    body: dict[str, Any],
    _request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Use Z.ai's thinking object and preserve agent reasoning history."""
    _extra_body(body)["thinking"] = (
        {"type": "enabled", "clear_thinking": False}
        if policy.enabled
        else {"type": "disabled"}
    )


def encode_ollama_reasoning(
    body: dict[str, Any],
    _request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Map Ollama's none/low/medium/high effort contract."""
    if not policy.enabled:
        body["reasoning_effort"] = "none"
        return
    effort = reasoning_effort(
        policy,
        _LOW_TO_MAX,
        default=ReasoningEffort.HIGH,
    )
    assert effort is not None
    body["reasoning_effort"] = effort.value


def encode_llamacpp_reasoning(
    body: dict[str, Any],
    _request: MessagesRequest,
    policy: ReasoningPolicy,
) -> None:
    """Use llama.cpp's per-request numeric thinking budget."""
    if not policy.enabled:
        _extra_body(body)["thinking_budget_tokens"] = 0
        return
    budget = reasoning_budget_tokens(policy)
    if budget is not None:
        _extra_body(body)["thinking_budget_tokens"] = budget


def _encode_optional_effort(
    body: dict[str, Any],
    policy: ReasoningPolicy,
    supported: tuple[ReasoningEffort, ...],
    *,
    disabled: str,
) -> None:
    if not policy.enabled:
        body["reasoning_effort"] = disabled
        return
    effort = reasoning_effort(policy, supported)
    if effort is not None:
        body["reasoning_effort"] = effort.value


def _extra_body(body: dict[str, Any]) -> dict[str, Any]:
    extra = body.setdefault("extra_body", {})
    if not isinstance(extra, dict):
        raise TypeError("OpenAI extra_body must be an object.")
    return extra


def _nested_dict(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object.")
    return value


def _has_version(model: str, *versions: str) -> bool:
    return any(version in model for version in versions)
