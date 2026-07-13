"""Single source of truth for model configuration, shared by every evaluator.

The two ``model_utils.py`` modules (repo root, and ``gsme_q_mt/``) intentionally
remain separate -- their client plumbing differs -- but they must agree on
*which* models exist and what their parameters are. Historically they drifted,
so the same ``MODEL=`` string could work for Logic-Q-MT and fail for GSME-Q-MT.

This module holds that shared state as **plain dicts**, not dataclass instances,
because each ``model_utils`` defines its own ``LocalModelConfig`` with different
fields and defaults. Each one calls :func:`local_config_kwargs` to build its own
dataclass, so neither file depends on the other's types.

Transport policy (``backend``):
  * ``completions``  -- raw ``/v1/completions`` with an ``apply_chat_template``
    prompt and token-level logprobs. Used by the Qwen family, which is how the
    published Qwen numbers were produced. Do not change this for those models.
  * ``openai_chat``  -- ``/v1/chat/completions``. Used by gpt-oss (the server's
    harmony parser returns clean ``content`` plus separate ``reasoning_content``)
    and by any custom / unregistered local model.
"""

import dataclasses
import json
import os
from typing import Any, Dict, List, Optional

# Transport identifiers.
BACKEND_COMPLETIONS = "completions"
BACKEND_OPENAI_CHAT = "openai_chat"

# Backend used for models that are not in the registry (custom / user-supplied).
DEFAULT_CUSTOM_BACKEND = BACKEND_OPENAI_CHAT

# ---------------------------------------------------------------------------
# Local (vLLM-served) models.
#
# `base_url` is deliberately omitted: each model_utils supplies its own default
# and the port is overridden via VLLM_PORT / --port. `reasoning_parser` is the
# vLLM `--reasoning-parser` value used by scripts/serve_vllm.sh (None = omit).
# ---------------------------------------------------------------------------
_QWEN_THINKING = dict(
    enable_reasoning=True,
    # NOTE: Qwen chat templates already emit the opening "<think>" tag.
    thinking_start_token="",
    thinking_end_token="</think>",
    backend=BACKEND_COMPLETIONS,
    reasoning_parser="qwen3",
)

_QWEN_INSTRUCT = dict(
    enable_reasoning=False,
    thinking_start_token="",
    thinking_end_token="",
    backend=BACKEND_COMPLETIONS,
    reasoning_parser=None,
)

_GPT_OSS = dict(
    enable_reasoning=True,
    thinking_start_token="<|channel|>analysis<|message|>",
    thinking_end_token="<|end|>",
    backend=BACKEND_OPENAI_CHAT,
    # vLLM's harmony reasoning parser. The exact name is version-dependent;
    # older builds may expect "openai" instead.
    reasoning_parser="openai_gptoss",
)


def _local(tokenizer_name: str, **overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"tokenizer_name": tokenizer_name}
    cfg.update(overrides)
    return cfg


LOCAL_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # -- Qwen3 thinking ------------------------------------------------------
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8": _local("Qwen/Qwen3-30B-A3B-Thinking-2507-FP8", **_QWEN_THINKING),
    "Qwen/Qwen3-30B-A3B-Thinking-2507": _local("Qwen/Qwen3-30B-A3B-Thinking-2507", **_QWEN_THINKING),
    "Qwen/Qwen3-4B-Thinking-2507": _local("Qwen/Qwen3-4B-Thinking-2507", **_QWEN_THINKING),
    "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8": _local("Qwen/Qwen3-Next-80B-A3B-Thinking-FP8", **_QWEN_THINKING),
    # -- Qwen3.5 -------------------------------------------------------------
    "Qwen/Qwen3.5-4B": _local("Qwen/Qwen3.5-4B", **_QWEN_THINKING),
    "Qwen/Qwen3.5-9B": _local("Qwen/Qwen3.5-9B", **_QWEN_THINKING),
    "Qwen/Qwen3.5-27B": _local("Qwen/Qwen3.5-27B", **_QWEN_THINKING),
    "Qwen/Qwen3.5-27B-FP8": _local("Qwen/Qwen3.5-27B-FP8", **_QWEN_THINKING),
    "Qwen/Qwen3.5-35B-A3B": _local("Qwen/Qwen3.5-35B-A3B", **_QWEN_THINKING),
    "Qwen/Qwen3.5-35B-A3B-FP8": _local("Qwen/Qwen3.5-35B-A3B-FP8", **_QWEN_THINKING),
    "Qwen/Qwen3.5-122B-A10B-FP8": _local("Qwen/Qwen3.5-122B-A10B-FP8", **_QWEN_THINKING),
    # -- Qwen3 instruct (non-reasoning; previously GSME-only) -----------------
    "Qwen/Qwen3-30B-A3B-Instruct-2507": _local("Qwen/Qwen3-30B-A3B-Instruct-2507", **_QWEN_INSTRUCT),
    "qwen3-30b-instruct-2507": _local("Qwen/Qwen3-30B-A3B-Instruct-2507", **_QWEN_INSTRUCT),
    # -- gpt-oss (chat/completions; harmony parsed server-side) ---------------
    "openai/gpt-oss-20B": _local("openai/gpt-oss-20B", **_GPT_OSS),
    "openai/gpt-oss-120B": _local("openai/gpt-oss-120B", **_GPT_OSS),
}

# ---------------------------------------------------------------------------
# Hosted model pricing (USD per token for GPT; USD per 1M tokens for Gemini).
# ---------------------------------------------------------------------------
GPT_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-5": {"prompt_tokens": 1.25 / 1_000_000, "completion_tokens": 10 / 1_000_000},
    "gpt-5-mini": {"prompt_tokens": 0.25 / 1_000_000, "completion_tokens": 2 / 1_000_000},
    "gpt-5.4": {"prompt_tokens": 2.5 / 1_000_000, "completion_tokens": 15 / 1_000_000},
}

GEMINI_COSTS: Dict[str, Dict[str, float]] = {
    "gemini-3.1-pro-preview": {"in": 2.00, "out": 12.00},
    "gemini-3-pro-preview": {"in": 2.00, "out": 12.00},
    # NOTE: the two model_utils disagreed on this rate (root said 3.00, gsme
    # said 2.00). Root's value is used as canonical; this affects only the
    # reported `cost_usd` field, never a metric.
    "gemini-3-flash-preview": {"in": 0.50, "out": 3.00},
    "gemini-3.1-flash-lite-preview": {"in": 0.25, "out": 1.50},
}

CLAUDE_MODELS: List[str] = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------
# Bumped whenever LOCAL_MODEL_CONFIGS changes, so the model_utils modules (which
# cache dataclass instances) can tell when their view is stale.
_REVISION = 0


def revision() -> int:
    return _REVISION


def local_config_kwargs(model_name: str, config_cls: type) -> Dict[str, Any]:
    """Return kwargs for ``config_cls`` (a LocalModelConfig dataclass).

    Fields the dataclass does not declare (e.g. ``backend``) are dropped, and
    ``None`` values are omitted so the dataclass default applies.
    """
    entry = LOCAL_MODEL_CONFIGS[model_name]
    allowed = {f.name for f in dataclasses.fields(config_cls)}
    return {k: v for k, v in entry.items() if k in allowed and v is not None}


def build_local_configs(config_cls: type) -> Dict[str, Any]:
    """Build ``{model_name: config_cls(...)}`` for the whole registry."""
    return {
        name: config_cls(**local_config_kwargs(name, config_cls))
        for name in LOCAL_MODEL_CONFIGS
    }


def get_backend(model_name: str) -> str:
    """Transport to use for a local model (see module docstring)."""
    entry = LOCAL_MODEL_CONFIGS.get(model_name)
    if entry is None:
        return DEFAULT_CUSTOM_BACKEND
    return entry.get("backend", DEFAULT_CUSTOM_BACKEND)


def get_reasoning_parser(model_name: str) -> Optional[str]:
    """vLLM ``--reasoning-parser`` value for a registered local model."""
    entry = LOCAL_MODEL_CONFIGS.get(model_name)
    return entry.get("reasoning_parser") if entry else None


def is_registered_local(model_name: str) -> bool:
    return model_name in LOCAL_MODEL_CONFIGS


def known_models() -> List[str]:
    """Every model name the suite recognises, for error messages."""
    return sorted(
        list(LOCAL_MODEL_CONFIGS) + list(GPT_COSTS) + list(GEMINI_COSTS) + CLAUDE_MODELS
    )


def suggest(model_name: str, n: int = 5) -> List[str]:
    """Closest known model names, for a helpful error on typos."""
    import difflib

    return difflib.get_close_matches(model_name, known_models(), n=n, cutoff=0.4)


def register_local_model(
    model_key: str,
    tokenizer_name: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    enable_reasoning: bool = True,
    thinking_start_token: str = "",
    thinking_end_token: str = "",
    backend: str = DEFAULT_CUSTOM_BACKEND,
    reasoning_parser: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> Dict[str, Any]:
    """Register (or override) a custom local / OpenAI-compatible model."""
    global _REVISION
    entry = _local(
        tokenizer_name or model_key,
        base_url=base_url,
        enable_reasoning=enable_reasoning,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
        backend=backend,
        reasoning_parser=reasoning_parser,
        api_key_env=api_key_env,
    )
    LOCAL_MODEL_CONFIGS[model_key] = entry
    _REVISION += 1
    return entry


def load_model_config_file(path: str) -> str:
    """Load a custom model definition (YAML or JSON) and register it.

    Expected keys: ``model_name`` (required), plus any of ``tokenizer_name``,
    ``base_url``, ``api_key_env``, ``backend``, ``enable_reasoning``,
    ``thinking_start_token``, ``thinking_end_token``, ``reasoning_parser``.

    Returns the registered model name.
    """
    with open(path, "r") as f:
        text = f.read()
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # optional dependency; only needed for YAML configs

        spec = yaml.safe_load(text)

    if not isinstance(spec, dict) or "model_name" not in spec:
        raise ValueError(f"{path}: model config must be a mapping with 'model_name'.")

    name = spec.pop("model_name")
    register_local_model(name, **spec)
    return name


def is_hosted_model(model_name: str) -> bool:
    """True for API-served models (GPT / Gemini / Claude families)."""
    name = (model_name or "").lower()
    return (
        model_name in GPT_COSTS
        or model_name in GEMINI_COSTS
        or model_name in CLAUDE_MODELS
        or name.startswith("gpt-")
        or "gemini" in name
        or name.startswith("claude")
    )


def validate_model_name(model_name: str, extra_allowed: Optional[List[str]] = None) -> None:
    """Raise a helpful error if `model_name` cannot be routed to any backend.

    Replaces the old hardcoded argparse ``choices=[...]`` allowlists, which made
    it impossible to evaluate a model that was not baked into the source.

    A name with a hosted-model prefix (``gpt-...``, ``gemini-...``) that is absent
    from the cost tables is allowed and emits a warning. This supports new API
    models while still flagging possible typos.
    """
    if model_name in (extra_allowed or []):
        return
    if is_registered_local(model_name):
        return
    if model_name in GPT_COSTS or model_name in GEMINI_COSTS or model_name in CLAUDE_MODELS:
        return

    if is_hosted_model(model_name):
        hint = suggest(model_name)
        note = f" Did you mean: {', '.join(hint)}?" if hint else ""
        print(
            f"WARNING: {model_name!r} is not a known model. It will be routed to the "
            f"hosted backend by name prefix, and cost tracking is disabled for it."
            f"{note}"
        )
        return

    # Unknown name + a local endpoint -> assume it's served there (inference).
    if ensure_local_registered(model_name):
        return

    hint = suggest(model_name)
    msg = [f"Unknown model: {model_name!r}."]
    if hint:
        msg.append(f"Did you mean: {', '.join(hint)}?")
    msg.append(
        "If it is a local model, start its vLLM server and set VLLM_PORT or "
        "VLLM_BASE_URL (then it is inferred automatically). Otherwise pass "
        "--model-config <file>, or pick a known model."
    )
    raise ValueError(" ".join(msg))


def api_key_env_for(model_name: str) -> Optional[str]:
    """Env var holding the API key for a custom OpenAI-compatible model."""
    entry = LOCAL_MODEL_CONFIGS.get(model_name) or {}
    return entry.get("api_key_env")


def resolve_base_url(
    model_name: str,
    default: str = "http://127.0.0.1:8011/v1",
    port: Optional[str] = None,
) -> str:
    """Resolve a served model endpoint with one suite-wide precedence.

    A URL explicitly pinned by ``--model-config`` wins. Otherwise a complete
    ``VLLM_BASE_URL`` wins over an explicit port, then ``VLLM_PORT``, then the
    caller's default.
    """
    entry = LOCAL_MODEL_CONFIGS.get(model_name) or {}
    if entry.get("base_url"):
        return entry["base_url"]
    if os.environ.get("VLLM_BASE_URL"):
        return os.environ["VLLM_BASE_URL"]
    resolved_port = port or os.environ.get("VLLM_PORT")
    if resolved_port:
        return f"http://127.0.0.1:{resolved_port}/v1"
    return default


# ---------------------------------------------------------------------------
# Inference fallback for served models.
#
# A model that is not a known API model, when a local vLLM endpoint is available
# (VLLM_PORT), is assumed to be served there over /v1/chat/completions. This is
# how a person reasons -- "I started a server and passed a model name, so use the
# server" -- and it means a freshly-served HuggingFace model works by name alone:
# no config file, no registry entry, and every subprocess infers the same thing
# independently from `--model <name>` + VLLM_PORT. `--model-config` is then only
# for overrides (a remote gateway's API key, forcing the raw /completions path,
# a non-default tokenizer).
# ---------------------------------------------------------------------------
def local_endpoint_available() -> bool:
    return bool(os.environ.get("VLLM_PORT") or os.environ.get("VLLM_BASE_URL"))


def ensure_local_registered(model_name: str) -> bool:
    """If `model_name` is an unknown model but a local endpoint is configured,
    register it as an OpenAI-compatible chat model. Returns True iff the model is
    (now) a registered local model.
    """
    if not model_name:
        return False
    if is_registered_local(model_name):
        return True
    if is_hosted_model(model_name):
        return False
    if not local_endpoint_available():
        return False
    register_local_model(
        model_name,
        tokenizer_name=model_name,
        backend=BACKEND_OPENAI_CHAT,
        enable_reasoning=True,
    )
    print(f"INFO: '{model_name}' is unknown; assuming it is served locally over "
          f"/v1/chat/completions (VLLM_PORT/VLLM_BASE_URL). Use --model-config "
          f"to override.")
    return True
