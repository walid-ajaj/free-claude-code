"""Declarative profiles for providers with no adapter-specific runtime behavior."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.reasoning import ReasoningPolicy
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest

from .base_url import openai_v1_base_url
from .extra_body import validate_extra_body_does_not_override_canonical_fields
from .reasoning import (
    encode_cerebras_reasoning,
    encode_cohere_reasoning,
    encode_fireworks_reasoning,
    encode_groq_reasoning,
    encode_huggingface_reasoning,
    encode_kimi_reasoning,
    encode_llamacpp_reasoning,
    encode_minimax_reasoning,
    encode_ollama_reasoning,
    encode_sambanova_reasoning,
    encode_vercel_reasoning,
    encode_wafer_reasoning,
    encode_zai_reasoning,
)
from .request_policy import OpenAIChatPostprocessor, OpenAIChatRequestPolicy


@dataclass(frozen=True, slots=True)
class OpenAIChatProfile:
    """Immutable behavior differences for one ordinary OpenAI-chat provider."""

    request_policy: OpenAIChatRequestPolicy
    postprocessors: tuple[OpenAIChatPostprocessor, ...] = ()
    reasoning_encoder: OpenAIChatPostprocessor | None = None
    normalize_base_url: bool = False
    reasoning_delta_field: Literal["reasoning_content", "reasoning"] = (
        "reasoning_content"
    )

    @property
    def provider_name(self) -> str:
        return self.request_policy.provider_name

    def base_url(self, configured: str) -> str:
        return openai_v1_base_url(configured) if self.normalize_base_url else configured

    def reasoning_delta(self, delta: Any) -> str | None:
        value = getattr(delta, self.reasoning_delta_field, None)
        return value if isinstance(value, str) else None

    @property
    def request_postprocessors(self) -> tuple[OpenAIChatPostprocessor, ...]:
        """Return generic transforms followed by the provider reasoning encoder."""
        if self.reasoning_encoder is None:
            return self.postprocessors
        return (*self.postprocessors, self.reasoning_encoder)


def _apply_cohere_request_quirks(
    body: dict[str, Any],
    request: MessagesRequest,
    _policy: ReasoningPolicy,
) -> None:
    _merge_allowed_cohere_extra_body(body, request.extra_body)


_COHERE_EXTRA_BODY_KEYS = frozenset(
    {
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "seed",
    }
)


def _merge_allowed_cohere_extra_body(body: dict[str, Any], extra_body: Any) -> None:
    if extra_body in (None, {}):
        return
    if not isinstance(extra_body, Mapping):
        raise InvalidRequestError("Cohere extra_body must be an object when provided.")

    unsupported = sorted(
        str(key) for key in extra_body if key not in _COHERE_EXTRA_BODY_KEYS
    )
    if unsupported:
        raise InvalidRequestError(
            "Cohere extra_body supports only these keys: "
            f"{sorted(_COHERE_EXTRA_BODY_KEYS)}. Unsupported: {unsupported}"
        )
    body.update({str(key): deepcopy(value) for key, value in extra_body.items()})


OPENAI_CHAT_PROFILES: dict[str, OpenAIChatProfile] = {
    "mistral_codestral": OpenAIChatProfile(
        OpenAIChatRequestPolicy(provider_name="CODESTRAL")
    ),
    "opencode": OpenAIChatProfile(OpenAIChatRequestPolicy(provider_name="OPENCODE")),
    "opencode_go": OpenAIChatProfile(
        OpenAIChatRequestPolicy(provider_name="OPENCODE_GO")
    ),
    "vercel": OpenAIChatProfile(
        OpenAIChatRequestPolicy(provider_name="VERCEL", include_extra_body=True),
        reasoning_encoder=encode_vercel_reasoning,
    ),
    "huggingface": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="HUGGINGFACE",
            include_extra_body=True,
            reasoning_replay=ReasoningReplayMode.DISABLED,
        ),
        reasoning_encoder=encode_huggingface_reasoning,
    ),
    "cohere": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="COHERE",
            strip_message_names=True,
            unsupported_body_keys=frozenset(
                {
                    "audio",
                    "logit_bias",
                    "metadata",
                    "modalities",
                    "n",
                    "parallel_tool_calls",
                    "prediction",
                    "service_tier",
                    "store",
                    "top_logprobs",
                }
            ),
        ),
        postprocessors=(_apply_cohere_request_quirks,),
        reasoning_encoder=encode_cohere_reasoning,
    ),
    "wafer": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="WAFER",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        reasoning_encoder=encode_wafer_reasoning,
    ),
    "kimi": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="KIMI",
            reject_extra_body_message=(
                "Kimi Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        reasoning_encoder=encode_kimi_reasoning,
    ),
    "minimax": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="MINIMAX",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            max_tokens_field="max_completion_tokens",
        ),
        reasoning_encoder=encode_minimax_reasoning,
    ),
    "cerebras": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="CEREBRAS",
            include_extra_body=True,
            max_tokens_field="max_completion_tokens",
        ),
        reasoning_encoder=encode_cerebras_reasoning,
    ),
    "groq": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="GROQ",
            include_extra_body=True,
            max_tokens_field="max_completion_tokens",
            strip_message_names=True,
            unsupported_body_keys=frozenset({"logprobs", "logit_bias", "top_logprobs"}),
            normalize_n_to_one=True,
        ),
        reasoning_encoder=encode_groq_reasoning,
    ),
    "sambanova": OpenAIChatProfile(
        OpenAIChatRequestPolicy(provider_name="SAMBANOVA", include_extra_body=True),
        reasoning_encoder=encode_sambanova_reasoning,
    ),
    "fireworks": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="FIREWORKS",
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        reasoning_encoder=encode_fireworks_reasoning,
    ),
    "zai": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="ZAI",
            reject_extra_body_message=(
                "Z.ai Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        reasoning_encoder=encode_zai_reasoning,
    ),
    "ollama_cloud": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="OLLAMA_CLOUD",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            reasoning_replay=ReasoningReplayMode.REASONING,
        ),
        reasoning_encoder=encode_ollama_reasoning,
        reasoning_delta_field="reasoning",
    ),
    "llamacpp": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="LLAMACPP",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        reasoning_encoder=encode_llamacpp_reasoning,
        normalize_base_url=True,
    ),
    "ollama": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="OLLAMA",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            reasoning_replay=ReasoningReplayMode.REASONING,
        ),
        reasoning_encoder=encode_ollama_reasoning,
        normalize_base_url=True,
        reasoning_delta_field="reasoning",
    ),
}
