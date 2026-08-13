import os
import time
import copy
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List


# ============================================================================
# Environment helpers
# ============================================================================

def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_keys_enabled() -> bool:
    return _env_bool("TWENTYQ_DEBUG_KEYS", False)


# The whole suite uses ONE variable for the local vLLM server (VLLM_PORT) and one
# default (8011). `VLLM_BASE_URL` overrides it outright for a non-local endpoint.
DEFAULT_VLLM_PORT = "8011"


def _local_vllm_port() -> str:
    return _env_str("VLLM_PORT", DEFAULT_VLLM_PORT)


def _suite_model_registry():
    """Return the top-level registry when 20Q is launched by the suite."""
    try:
        import model_registry
        return model_registry
    except ImportError:
        return None


def _openai_compatible_api_key(model: str) -> str:
    """Resolve an optional protected-endpoint key; local vLLM uses a dummy."""
    registry = _suite_model_registry()
    if registry is not None:
        env_name = registry.api_key_env_for(model)
        if env_name:
            value = _env_str(env_name)
            if not value:
                raise RuntimeError(
                    f"Model {model!r} requires API key environment variable {env_name}."
                )
            return value
    # The OpenAI SDK requires a non-empty string, even though ordinary local
    # vLLM servers do not authenticate it.
    return "EMPTY"


def _local_vllm_base_url() -> str:
    override = _env_str("VLLM_BASE_URL")
    if override:
        return override.rstrip("/")
    return f"http://127.0.0.1:{_local_vllm_port()}/v1"


def _openai_compatible_base_url(model: str) -> str:
    registry = _suite_model_registry()
    if registry is not None and registry.is_registered_local(model):
        return registry.resolve_base_url(model, default=_local_vllm_base_url())
    return _local_vllm_base_url()


def _is_suite_registered_local(model: str) -> bool:
    registry = _suite_model_registry()
    return bool(registry is not None and registry.is_registered_local(model))


# ============================================================================
# Unified generation defaults
# ============================================================================

MAX_TOKENS = 32768

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": MAX_TOKENS,
    "reasoning_effort": "high",
}

time_gap = {
    "gpt-4": 3,
    "gpt-3.5-turbo": 0.5,
    "gpt-5": 1,
    "gpt-5-mini": 0.5,
    "gemini-3.1-pro-preview": 2,
    "gemini-3-flash-preview": 1,
    "gemini-3.1-flash-lite-preview": 1,
    "mistral-small-latest": 1,
    "mistral-medium-latest": 1,
    "mistral-large-latest": 1,
    "qwen_instruct_4b": 1,
    "qwen_thinking_4b": 1,
    "qwen_thinking_30b": 1,
    "qwen_4b": 1,
    "qwen_30b": 1,
    "qwen_instruct_30b": 1,
    "gpt_oss_20b": 1,
    "qwen3-4b-instruct-local": 0,
    "qwen3-4b-local": 0,
    "qwen3-30b-local": 0,
    "qwen3-30b-instruct-local": 0,
    "llama3.1-8b-local": 0,
}

openai_client = None

co = None
genai = None
glm = None
claude_client = None
llama_client = None
mistral_client = None
ChatMessage = None
genai_types = None


def _make_local_openai_client(base_url: str, api_key: str = "EMPTY", timeout: float = 1200.0):
    from openai import OpenAI

    kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "max_retries": 0,
        "timeout": timeout,
    }
    try:
        import httpx
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
    except Exception:
        pass
    return OpenAI(**kwargs)


def _print_openai_exception_debug(exc: Exception) -> None:
    if os.getenv("TWENTYQ_DEBUG_OPENAI", "").strip().lower() not in {"1", "true", "yes"}:
        return

    print(f"[OPENAI_DEBUG] exception_type={type(exc).__name__}")
    for attr in ("status_code", "code", "type", "message"):
        value = getattr(exc, attr, None)
        if value is not None:
            print(f"[OPENAI_DEBUG] {attr}={value!r}")

    response = getattr(exc, "response", None)
    if response is not None:
        print(f"[OPENAI_DEBUG] response_status={getattr(response, 'status_code', None)!r}")
        try:
            print(f"[OPENAI_DEBUG] response_text={response.text[:2000]!r}")
        except Exception:
            pass

        request = getattr(response, "request", None)
        if request is not None:
            print(f"[OPENAI_DEBUG] request_method={getattr(request, 'method', None)!r}")
            print(f"[OPENAI_DEBUG] request_url={str(getattr(request, 'url', ''))!r}")


def _debug_secret(name: str, value: str) -> None:
    if _debug_keys_enabled() and value:
        print(f"{name}: ****{value[-4:]}")


def _debug_value(name: str, value: str) -> None:
    if _debug_keys_enabled() and value:
        print(f"{name}: {value}")

def _get_openai_client():
    global openai_client
    if openai_client is not None:
        return openai_client

    azure_key = _env_str("AZURE_OPENAI_API_KEY")
    endpoint = _env_str("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if azure_key:
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set, but GPT model was requested.")

        from openai import AzureOpenAI

        openai_client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=endpoint,
            api_version=_env_str("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            timeout=600,
        )
        _debug_secret("AZURE_OPENAI_API_KEY", azure_key)
        _debug_value("AZURE_OPENAI_ENDPOINT", endpoint)
        return openai_client

    api_key = _env_str("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI

        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": 600}
        base_url = _env_str("OPENAI_BASE_URL").rstrip("/")
        if base_url:
            kwargs["base_url"] = base_url
        openai_client = OpenAI(**kwargs)
        _debug_secret("OPENAI_API_KEY", api_key)
        _debug_value("OPENAI_BASE_URL", base_url)
        return openai_client

    raise RuntimeError(
        "No GPT credentials found. Set AZURE_OPENAI_API_KEY (+ "
        "AZURE_OPENAI_ENDPOINT), or OPENAI_API_KEY (+ optional OPENAI_BASE_URL)."
    )


def _get_cohere_client():
    global co
    if co is not None:
        return co
    api_key = _env_str("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not set, but Cohere model was requested.")
    import cohere
    co = cohere.Client(api_key)
    _debug_secret("COHERE_API_KEY", api_key)
    return co


def _get_google_genai_modules():
    global genai, genai_types
    if genai is not None:
        return genai, genai_types
    try:
        from google import genai as google_genai
        from google.genai import types as google_genai_types
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed, but Gemini model was requested.") from exc
    genai = google_genai
    genai_types = google_genai_types
    return genai, genai_types


def _get_anthropic_client():
    global claude_client
    if claude_client is not None:
        return claude_client
    api_key = _env_str("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set, but Claude model was requested.")
    from anthropic import Anthropic
    claude_client = Anthropic(api_key=api_key)
    _debug_secret("ANTHROPIC_API_KEY", api_key)
    return claude_client


def _get_together_client():
    global llama_client
    if llama_client is not None:
        return llama_client
    api_key = _env_str("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set, but remote Llama model was requested.")
    from openai import OpenAI
    llama_client = OpenAI(api_key=api_key, base_url="https://api.together.xyz")
    _debug_secret("TOGETHER_API_KEY", api_key)
    return llama_client


def _get_mistral_client():
    global mistral_client, ChatMessage
    if mistral_client is not None and ChatMessage is not None:
        return mistral_client, ChatMessage
    api_key = _env_str("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set, but Mistral model was requested.")
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage as MistralChatMessage
    mistral_client = MistralClient(api_key=api_key)
    ChatMessage = MistralChatMessage
    _debug_secret("MISTRAL_API_KEY", api_key)
    return mistral_client, ChatMessage


# ============================================================================
# Model config
# ============================================================================

@dataclass
class LocalModelConfig:
    name: str
    family: str
    tokenizer_name_or_path: str
    served_model_name: str
    base_url: str
    enable_reasoning: bool = True
    thinking_start_token: str = ""
    thinking_end_token: str = "</think>"


LOCAL_MODEL_PATHS = {
    "Qwen/Qwen3-4B-Instruct-2507": {
        "family": "qwen",
        "path": "Qwen/Qwen3-4B-Instruct-2507",
    },
    "Qwen/Qwen3-4B-Thinking-2507": {
        "family": "qwen",
        "path": "Qwen/Qwen3-4B-Thinking-2507",
    },
    "Qwen/Qwen3-30B-A3B-Thinking-2507": {
        "family": "qwen",
        "path": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "family": "llama",
        "path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "family": "qwen",
        "path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    },
    "openai/gpt-oss-20b": {
        "family": "gpt_oss",
        "path": "openai/gpt-oss-20b",
    },
}

LOCAL_MODEL_ALIASES = {
    "qwen3-4b-instruct-local": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-local": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-30b-local": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "llama3.1-8b-local": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen3-30b-instruct-local": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen_instruct_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

REMOTE_QWEN_SPECS = {
    "qwen_instruct_4b": {
        "tokenizer_path": "Qwen/Qwen3-4B-Instruct-2507",
        "server_model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "enable_reasoning": False,
        "thinking_start_token": "",
        "thinking_end_token": "",
    },
    "qwen_thinking_4b": {
        "tokenizer_path": "Qwen/Qwen3-4B-Thinking-2507",
        "server_model_name": "Qwen/Qwen3-4B-Thinking-2507",
        "enable_reasoning": True,
        "thinking_start_token": "",
        "thinking_end_token": "</think>",
    },
    "qwen_thinking_30b": {
        "tokenizer_path": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "server_model_name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "enable_reasoning": True,
        "thinking_start_token": "",
        "thinking_end_token": "</think>",
    },
    "qwen_instruct_30b": {
        "tokenizer_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "server_model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "enable_reasoning": False,
        "thinking_start_token": "",
        "thinking_end_token": "",
    },
    "qwen_4b": {
        "tokenizer_path": "Qwen/Qwen3-4B-Thinking-2507",
        "server_model_name": "Qwen/Qwen3-4B-Thinking-2507",
        "enable_reasoning": True,
        "thinking_start_token": "",
        "thinking_end_token": "</think>",
    },
    "qwen_30b": {
        "tokenizer_path": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "server_model_name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "enable_reasoning": True,
        "thinking_start_token": "",
        "thinking_end_token": "</think>",
    },
}

LOCAL_MODEL_CONFIGS: Dict[str, LocalModelConfig] = {
    "qwen3-4b-instruct-local": LocalModelConfig(
        name="qwen3-4b-instruct-local",
        family="qwen",
        tokenizer_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
        served_model_name="Qwen/Qwen3-4B-Instruct-2507",
        base_url=_local_vllm_base_url(),
        enable_reasoning=False,
        thinking_start_token="",
        thinking_end_token="",
    ),
    "qwen3-4b-local": LocalModelConfig(
        name="qwen3-4b-local",
        family="qwen",
        tokenizer_name_or_path="Qwen/Qwen3-4B-Thinking-2507",
        served_model_name="Qwen/Qwen3-4B-Thinking-2507",
        base_url=_local_vllm_base_url(),
        enable_reasoning=True,
        thinking_start_token="",
        thinking_end_token="</think>",
    ),
    "qwen3-30b-local": LocalModelConfig(
        name="qwen3-30b-local",
        family="qwen",
        tokenizer_name_or_path="Qwen/Qwen3-30B-A3B-Thinking-2507",
        served_model_name="Qwen/Qwen3-30B-A3B-Thinking-2507",
        base_url=_local_vllm_base_url(),
        enable_reasoning=True,
        thinking_start_token="",
        thinking_end_token="</think>",
    ),
    "qwen3-30b-instruct-local": LocalModelConfig(
        name="qwen3-30b-instruct-local",
        family="qwen",
        tokenizer_name_or_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
        served_model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_url=_local_vllm_base_url(),
        enable_reasoning=False,
        thinking_start_token="",
        thinking_end_token="",
    ),
    "llama3.1-8b-local": LocalModelConfig(
        name="llama3.1-8b-local",
        family="llama",
        tokenizer_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        served_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        base_url=_local_vllm_base_url(),
        enable_reasoning=False,
        thinking_start_token="",
        thinking_end_token="",
    ),
}

LOCAL_MODEL_CONFIGS["Qwen/Qwen3-4B-Instruct-2507"] = LOCAL_MODEL_CONFIGS["qwen3-4b-instruct-local"]
LOCAL_MODEL_CONFIGS["Qwen/Qwen3-4B-Thinking-2507"] = LOCAL_MODEL_CONFIGS["qwen3-4b-local"]
LOCAL_MODEL_CONFIGS["Qwen/Qwen3-30B-A3B-Thinking-2507"] = LOCAL_MODEL_CONFIGS["qwen3-30b-local"]
LOCAL_MODEL_CONFIGS["Qwen/Qwen3-30B-A3B-Instruct-2507"] = LOCAL_MODEL_CONFIGS["qwen3-30b-instruct-local"]
LOCAL_MODEL_CONFIGS["meta-llama/Meta-Llama-3.1-8B-Instruct"] = LOCAL_MODEL_CONFIGS["llama3.1-8b-local"]

_LOCAL_CLIENT_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_TOKENIZER_DEBUG_PRINTED: set[str] = set()


# ============================================================================
# Helpers
# ============================================================================

def _sleep_for_model(model: str):
    time.sleep(time_gap.get(model, 1))


def _resolve_local_model_name(model: str) -> str:
    if model in LOCAL_MODEL_PATHS:
        return model
    if model in LOCAL_MODEL_ALIASES:
        return LOCAL_MODEL_ALIASES[model]
    return model


def get_local_model_config(model: str) -> Optional[LocalModelConfig]:
    resolved = _resolve_local_model_name(model)
    return LOCAL_MODEL_CONFIGS.get(model) or LOCAL_MODEL_CONFIGS.get(resolved)


def _get_or_create_openai_client(base_url: str, api_key: str = "EMPTY"):
    cache_key = f"{base_url}|{api_key}"
    if cache_key not in _LOCAL_CLIENT_CACHE:
        _LOCAL_CLIENT_CACHE[cache_key] = _make_local_openai_client(base_url, api_key=api_key)
    return _LOCAL_CLIENT_CACHE[cache_key]


def _looks_like_local_path(path: str) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    path = path.strip()
    return (
        path.startswith("/")
        or path.startswith("./")
        or path.startswith("../")
        or path.startswith("~/")
    )


def _is_hf_repo_id(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    s = s.strip()
    if _looks_like_local_path(s):
        return False
    return "/" in s or "-" in s or "_" in s


def _build_fallback_chat_prompt(message: List[dict]) -> str:
    text = ""
    for m in message:
        role = m.get("role", "user")
        content = m.get("content", "")
        text += f"{role}: {content}\n"
    text += "assistant: "
    return text


# When tokenizer/model paths use local filesystem locations,
# local dev machines may not have them. Map those paths to Hub ids.
_TOKENIZER_HUB_FALLBACK: Dict[str, str] = {
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def _resolve_tokenizer_pretrained_spec(tokenizer_path: str) -> Tuple[str, bool]:
    """
    Returns (pretrained_model_name_or_path, local_files_only).

    Order:
    1. real local dir
    2. TWENTYQ_LOCAL_TOKENIZER_ROOT + suffix after .../models/
    3. explicit Hub fallback
    4. heuristic Hub fallback by basename
    """
    if not tokenizer_path:
        raise ValueError("tokenizer_path is empty")

    expanded = os.path.expanduser(tokenizer_path)
    if os.path.isdir(expanded):
        return expanded, True

    root = os.getenv("TWENTYQ_LOCAL_TOKENIZER_ROOT", "").strip()
    if root:
        root_exp = os.path.expanduser(root)
        rel = None
        for marker in ("/models/", "\\models\\"):
            if marker in tokenizer_path:
                rel = tokenizer_path.split(marker, 1)[1].replace("\\", "/")
                break
        if rel:
            candidate = os.path.join(root_exp, *rel.split("/"))
            if os.path.isdir(candidate):
                return candidate, True

    if tokenizer_path in _TOKENIZER_HUB_FALLBACK:
        return _TOKENIZER_HUB_FALLBACK[tokenizer_path], False

    base = os.path.basename(os.path.normpath(tokenizer_path))
    lower = base.lower()
    if "gpt-oss" in lower:
        return "openai/gpt-oss-20b", False
    if lower.startswith("qwen") or "qwen" in lower:
        return f"Qwen/{base}", False
    if "llama" in lower:
        return f"meta-llama/{base}", False

    raise OSError(
        f"Tokenizer path does not exist locally: {tokenizer_path!r}. "
        "Clone the tokenizer locally, set TWENTYQ_LOCAL_TOKENIZER_ROOT to the parent of the "
        "`models` folder containing the tokenizer, or extend _TOKENIZER_HUB_FALLBACK."
    )


def _load_tokenizer_cached(tokenizer_path: str, allow_remote_fallback: bool = False):
    from transformers import AutoTokenizer

    cache_key = f"{tokenizer_path}|remote={allow_remote_fallback}"
    if cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    try:
        load_target, local_only = _resolve_tokenizer_pretrained_spec(tokenizer_path)
        tok = AutoTokenizer.from_pretrained(
            load_target,
                trust_remote_code=True,
                local_files_only=local_only,
                cache_dir=_env_str("HUGGINGFACE_HUB_CACHE") or None,
        )
        if os.getenv("TWENTYQ_DEBUG_TOKENIZER", "").strip().lower() in {"1", "true", "yes"}:
            print(
                "[TOKENIZER_DEBUG] "
                f"tokenizer_path={tokenizer_path!r} "
                f"load_target={load_target!r} "
                f"local_files_only={local_only} "
                f"class={tok.__class__.__name__} "
                f"has_chat_template={bool(getattr(tok, 'chat_template', None))}"
            )
        _TOKENIZER_CACHE[cache_key] = tok
        return tok
    except Exception as e:
        print(f"Warning: failed to resolve tokenizer via local/root/hub fallback for {tokenizer_path}: {e}")

    expanded_path = os.path.expanduser(tokenizer_path)
    if os.path.isdir(expanded_path):
        try:
            tok = AutoTokenizer.from_pretrained(
                expanded_path,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=_env_str("HUGGINGFACE_HUB_CACHE") or None,
            )
            _TOKENIZER_CACHE[cache_key] = tok
            return tok
        except Exception as e:
            print(f"Warning: failed to load local tokenizer from {expanded_path}: {e}")

    if allow_remote_fallback and _is_hf_repo_id(tokenizer_path):
        try:
            tok = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=False,
                cache_dir=_env_str("HUGGINGFACE_HUB_CACHE") or None,
            )
            _TOKENIZER_CACHE[cache_key] = tok
            return tok
        except Exception as e:
            print(f"Warning: failed to load remote tokenizer from repo {tokenizer_path}: {e}")

    _TOKENIZER_CACHE[cache_key] = None
    return None


def _count_tokens_with_tokenizer(text: str, tokenizer=None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(str(text).split())


def _get_remote_qwen_client(model: str):
    if model not in REMOTE_QWEN_SPECS:
        raise ValueError(f"Unsupported remote qwen model: {model}")
    return _get_or_create_openai_client(
        _openai_compatible_base_url(model),
        api_key=_openai_compatible_api_key(model),
    )


def _apply_chat_template_safely(tokenizer, message, family: str, enable_reasoning: bool = True):
    if tokenizer is None:
        return _build_fallback_chat_prompt(message)

    if family == "qwen":
        if enable_reasoning:
            try:
                prompt = tokenizer.apply_chat_template(
                    conversation=message,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_reasoning=True,
                    add_special_tokens=True,
                )
                _debug_print_prompt_once(tokenizer, prompt)
                return prompt
            except TypeError:
                pass

        try:
            prompt = tokenizer.apply_chat_template(
                conversation=message,
                add_generation_prompt=True,
                tokenize=False,
                add_special_tokens=True,
            )
            _debug_print_prompt_once(tokenizer, prompt)
            return prompt
        except TypeError:
            pass

    try:
        prompt = tokenizer.apply_chat_template(
            conversation=message,
            add_generation_prompt=True,
            tokenize=False,
        )
        _debug_print_prompt_once(tokenizer, prompt)
        return prompt
    except TypeError:
        pass

    try:
        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        _debug_print_prompt_once(tokenizer, prompt)
        return prompt
    except TypeError:
        pass

    return _build_fallback_chat_prompt(message)


def _debug_print_prompt_once(tokenizer, prompt: str) -> None:
    if os.getenv("TWENTYQ_DEBUG_TOKENIZER", "").strip().lower() not in {"1", "true", "yes"}:
        return

    key = f"{id(tokenizer)}:{hash(prompt[:512])}"
    if key in _TOKENIZER_DEBUG_PRINTED:
        return
    _TOKENIZER_DEBUG_PRINTED.add(key)

    print(
        "[TOKENIZER_DEBUG] "
        f"prompt_len_chars={len(prompt)} "
        f"prompt_head={prompt[:300]!r} "
        f"prompt_tail={prompt[-300:]!r}"
    )


def _strip_start_token(text: str, start_token: str) -> str:
    if not text or not start_token:
        return text.strip()
    t = text.strip()
    if t.startswith(start_token):
        return t[len(start_token):].strip()
    idx = t.find(start_token)
    if idx >= 0:
        return t[idx + len(start_token):].strip()
    return t


def _split_reasoning_text(
    response_text: str,
    tokenizer=None,
    thinking_start_token: str = "",
    thinking_end_token: str = "</think>",
):
    response_text = (response_text or "").strip()
    if not response_text:
        return {
            "text": "",
            "cot": "",
            "num_thinking_tokens": 0,
        }

    cot = ""
    final_output = response_text

    if thinking_end_token and thinking_end_token in response_text:
        cot, final_output = response_text.split(thinking_end_token, 1)
        cot = _strip_start_token(cot, thinking_start_token)
        final_output = final_output.strip()
    else:
        final_output = response_text.strip()

    num_thinking_tokens = _count_tokens_with_tokenizer(cot, tokenizer) if cot else 0

    return {
        "text": final_output,
        "cot": cot,
        "num_thinking_tokens": num_thinking_tokens,
    }


_GPT_OSS_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")
_GPT_OSS_CHANNEL_BLOCK_RE = re.compile(
    r"<\|channel\|>(?P<channel>[^<]+?)(?:<\|constrain\|>[^<]+)?<\|message\|>(?P<body>.*?)(?=(?:<\|start\|>|<\|end\|>|$))",
    re.DOTALL,
)


def _clean_gpt_oss_payload(text: str) -> str:
    if not text:
        return ""
    s = _GPT_OSS_CONTROL_TOKEN_RE.sub(" ", text)
    s = s.replace("\\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _coerce_chat_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and item["text"] is not None:
                    parts.append(str(item["text"]))
                elif "content" in item and item["content"] is not None:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if "text" in content and content["text"] is not None:
            return str(content["text"])
        if "content" in content and content["content"] is not None:
            return str(content["content"])
    return str(content)


def _extract_gpt_oss_text_from_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    candidates = []
    for call in tool_calls:
        function = getattr(call, "function", None)
        if function is None and isinstance(call, dict):
            function = call.get("function")
        if function is None:
            continue

        arguments = getattr(function, "arguments", None)
        if arguments is None and isinstance(function, dict):
            arguments = function.get("arguments")
        if arguments is None:
            continue

        arg_text = str(arguments)
        candidates.append(arg_text)
    if not candidates:
        return ""
    return _clean_gpt_oss_payload("\n".join(candidates))


def extract_gpt_oss_content_and_cot(raw_content: Any, tool_calls: Any = None) -> Tuple[str, str]:
    text = _coerce_chat_content_to_text(raw_content)
    if not text:
        tool_text = _extract_gpt_oss_text_from_tool_calls(tool_calls)
        return tool_text, ""

    channel_blocks = _GPT_OSS_CHANNEL_BLOCK_RE.findall(text)
    if channel_blocks:
        final_parts = []
        commentary_parts = []
        analysis_parts = []
        tool_parts = []
        for channel_raw, body_raw in channel_blocks:
            channel = channel_raw.strip().lower()
            body = _clean_gpt_oss_payload(body_raw)
            if not body:
                continue
            if " to=" in channel:
                tool_parts.append(body)
            if channel.startswith("final"):
                final_parts.append(body)
            elif channel.startswith("commentary"):
                commentary_parts.append(body)
            elif channel.startswith("analysis"):
                analysis_parts.append(body)

        cot = "\n\n".join(analysis_parts).strip()
        if final_parts:
            return final_parts[-1], cot
        if commentary_parts:
            return commentary_parts[-1], cot
        if tool_parts:
            return tool_parts[-1], cot

        tool_text = _extract_gpt_oss_text_from_tool_calls(tool_calls)
        if tool_text:
            return tool_text, cot
        return "", cot

    return _clean_gpt_oss_payload(text), ""


def _usage_get_int(obj: Any, *keys: str) -> int:
    for key in keys:
        value = None
        if isinstance(obj, dict):
            value = obj.get(key)
        else:
            value = getattr(obj, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                continue
    return 0


def _extract_reasoning_tokens_from_usage(usage: Any) -> int:
    if usage is None:
        return 0

    direct = _usage_get_int(usage, "reasoning_tokens", "reasoningTokenCount")
    if direct > 0:
        return direct

    details_candidates = []
    if isinstance(usage, dict):
        details_candidates.extend([
            usage.get("completion_tokens_details"),
            usage.get("output_tokens_details"),
        ])
    else:
        details_candidates.extend([
            getattr(usage, "completion_tokens_details", None),
            getattr(usage, "output_tokens_details", None),
        ])

    for details in details_candidates:
        if details is None:
            continue
        value = _usage_get_int(details, "reasoning_tokens", "reasoningTokenCount")
        if value > 0:
            return value

    return 0


def _request_completion_with_fallback(
    client,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str = "medium",
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    extra_kwargs = extra_kwargs or {}

    request_kwargs = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    request_kwargs.update(extra_kwargs)

    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort

    if os.getenv("TWENTYQ_DEBUG_OPENAI", "").strip().lower() in {"1", "true", "yes"}:
        print(
            "[OPENAI_DEBUG] completions.create "
            f"model={model!r} "
            f"temperature={temperature!r} "
            f"top_p={top_p!r} "
            f"max_tokens={max_tokens!r} "
            f"reasoning_effort={request_kwargs.get('reasoning_effort')!r} "
            f"logprobs={request_kwargs.get('logprobs')!r} "
            f"prompt_head={prompt[:300]!r} "
            f"prompt_tail={prompt[-300:]!r}"
        )

    try:
        return client.completions.create(**request_kwargs)
    except TypeError:
        request_kwargs.pop("reasoning_effort", None)
        request_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        return client.completions.create(**request_kwargs)
    except Exception as e:
        _print_openai_exception_debug(e)
        raise


def _generate_vllm_completion(
    message: list,
    config: LocalModelConfig,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    allow_remote_tokenizer_fallback = (
        os.getenv("ALLOW_REMOTE_TOKENIZER_FALLBACK", "0").strip().lower()
        in {"1", "true", "yes"}
    )

    tokenizer = _load_tokenizer_cached(
        config.tokenizer_name_or_path,
        allow_remote_fallback=allow_remote_tokenizer_fallback,
    )

    raw_prompt_text = _apply_chat_template_safely(
        tokenizer,
        message,
        family=config.family,
        enable_reasoning=config.enable_reasoning,
    )

    client = _get_or_create_openai_client(
        config.base_url, api_key=_openai_compatible_api_key(config.name)
    )
    res = _request_completion_with_fallback(
        client=client,
        model=config.served_model_name,
        prompt=raw_prompt_text,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort if config.enable_reasoning else None,
        extra_kwargs={"logprobs": 1},
    )

    response_text = res.choices[0].text if res.choices else ""
    parsed = _split_reasoning_text(
        response_text=response_text,
        tokenizer=tokenizer,
        thinking_start_token=config.thinking_start_token,
        thinking_end_token=config.thinking_end_token,
    )

    return res, parsed


def unpack_model_response(ret):
    """
    Normalize model return format to:
    (raw_response, answer, cot, think_tokens)
    """
    if isinstance(ret, tuple):
        if len(ret) == 4:
            return ret
        if len(ret) == 3:
            raw, answer, cot = ret
            return raw, answer, cot, 0
        if len(ret) == 2:
            raw, answer = ret
            return raw, answer, "", 0
        if len(ret) == 1:
            return None, str(ret[0]), "", 0

    if isinstance(ret, str):
        return None, ret, "", 0

    return None, str(ret), "", 0


def _normalize_gemini_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "\n".join([p for p in parts if p])
    return str(content)


def _to_gemini_contents_and_system_instruction(messages):
    contents = []
    system_parts = []

    for msg in messages:
        role = (msg.get("role") or "user").lower()
        text = _normalize_gemini_text_content(msg.get("content", ""))
        if not text:
            continue

        if role == "system":
            system_parts.append(text)
            continue

        if role == "user":
            gemini_role = "user"
        elif role in ("assistant", "model"):
            gemini_role = "model"
        elif role == "tool":
            gemini_role = "tool"
        else:
            raise ValueError(f"Unsupported Gemini message role: {role}")

        contents.append({
            "role": gemini_role,
            "parts": [{"text": text}],
        })

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return contents, system_instruction


def _extract_gemini_usage(resp: Any):
    usage = getattr(resp, "usage_metadata", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage_metadata") or resp.get("usageMetadata")

    pt = ct = tt = tot = 0
    if usage is not None:
        pt = getattr(usage, "prompt_token_count", 0) or getattr(usage, "promptTokenCount", 0) or 0
        ct = (
            getattr(usage, "response_token_count", 0)
            or getattr(usage, "responseTokenCount", 0)
            or getattr(usage, "candidates_token_count", 0)
            or getattr(usage, "candidatesTokenCount", 0)
            or 0
        )
        tt = getattr(usage, "thoughts_token_count", 0) or getattr(usage, "thoughtsTokenCount", 0) or 0
        tot = getattr(usage, "total_token_count", 0) or getattr(usage, "totalTokenCount", 0) or 0

        if isinstance(usage, dict):
            pt = usage.get("prompt_token_count", usage.get("promptTokenCount", pt)) or 0
            ct = usage.get(
                "response_token_count",
                usage.get(
                    "responseTokenCount",
                    usage.get("candidates_token_count", usage.get("candidatesTokenCount", ct)),
                ),
            ) or 0
            tt = usage.get("thoughts_token_count", usage.get("thoughtsTokenCount", tt)) or 0
            tot = usage.get("total_token_count", usage.get("totalTokenCount", tot)) or 0

    return int(pt), int(ct), int(tt), int(tot)


# ============================================================================
# Remote API model responses
# ============================================================================

def _is_gpt5_family_model(model_name: str) -> bool:
    m = str(model_name).lower()
    return ("gpt-5" in m or "gpt5" in m)


def is_hosted_gpt_model(model_name: str) -> bool:
    m = str(model_name).lower()
    if _is_gpt5_family_model(m):
        return True
    return m in {"gpt-4", "gpt-3.5-turbo"}


def _use_max_completion_tokens(model_name: str) -> bool:
    flag = os.getenv("OPENAI_USE_MAX_COMPLETION_TOKENS", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if _is_gpt5_family_model(model_name):
        return True
    m = str(model_name).lower()
    for prefix in ("o1", "o3"):
        if m == prefix or m.startswith(prefix + "-") or m.startswith(prefix + "_"):
            return True
    return False


def _error_suggests_use_max_completion_tokens(err: Exception) -> bool:
    s = str(err).lower()
    return (
        "max_completion_tokens" in s
        and "max_tokens" in s
        and ("unsupported" in s or "not supported" in s)
    )


def _is_client_bad_request(err: Exception) -> bool:
    if type(err).__name__ == "BadRequestError":
        return True
    sc = getattr(err, "status_code", None)
    if sc == 400:
        return True
    resp = getattr(err, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 400:
        return True
    return False


def hosted_gpt_chat_completion(
    message: list,
    model="gpt-5-mini",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
    client=None,
    sleep_for_model: bool = True,
):
    active_client = _get_openai_client() if client is None else client

    if sleep_for_model:
        _sleep_for_model(model)

    def _build_gpt_chat_kwargs(use_max_completion: bool) -> dict:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": message,
        }

        if use_max_completion:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        if not _is_gpt5_family_model(model):
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p

        if _is_gpt5_family_model(model):
            kwargs["reasoning_effort"] = reasoning_effort

        return kwargs

    def _create_chat_completion(kwargs: dict):
        try:
            return active_client.chat.completions.create(**kwargs)
        except TypeError:
            if "reasoning_effort" in kwargs:
                reff = kwargs.pop("reasoning_effort")
                kwargs["extra_body"] = {"reasoning_effort": reff}
            return active_client.chat.completions.create(**kwargs)

    use_mct = _use_max_completion_tokens(model)
    kwargs = _build_gpt_chat_kwargs(use_mct)

    try:
        res = _create_chat_completion(dict(kwargs))
    except Exception as e:
        if not use_mct and _error_suggests_use_max_completion_tokens(e):
            retry_kwargs = _build_gpt_chat_kwargs(True)
            res = _create_chat_completion(dict(retry_kwargs))
        elif _is_client_bad_request(e):
            raise
        else:
            print(e)
            time.sleep(time_gap.get(model, 3) * 2)
            return hosted_gpt_chat_completion(
                message,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                client=active_client,
                sleep_for_model=sleep_for_model,
            )

    try:
        text = res.choices[0].message.content

        usage = getattr(res, "usage", None)
        num_thinking_tokens = 0
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                num_thinking_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
            else:
                num_thinking_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0)

        return res, text, "", num_thinking_tokens
    except Exception as e:
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return hosted_gpt_chat_completion(
            message,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            client=active_client,
            sleep_for_model=sleep_for_model,
        )


def gpt_response(
    message: list,
    model="gpt-5-mini",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return hosted_gpt_chat_completion(
        message,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )


def cohere_response(
    message: list,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    client = _get_cohere_client()

    msg = copy.deepcopy(message[:-1])
    new_msg = message[-1]["content"]
    for m in msg:
        m.update({"role": "CHATBOT" if m["role"] in ["system", "assistant"] else "USER"})
        m.update({"message": m.pop("content")})

    try:
        text = client.chat(chat_history=msg, message=new_msg).text
        return None, text, "", 0
    except Exception as e:
        print(e)
        time.sleep(1)
        return cohere_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def palm_response(
    message: list,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    raise NotImplementedError("PaLM is not enabled in this file. Use Gemini instead.")


def gemini_response(
    message: list,
    model="gemini-3-flash-preview",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    google_genai, google_genai_types = _get_google_genai_modules()

    try:
        if _env_bool("GOOGLE_GENAI_USE_VERTEXAI", False):
            project = _env_str("GOOGLE_CLOUD_PROJECT")
            location = _env_str("GOOGLE_CLOUD_LOCATION")
            if not project or not location:
                raise RuntimeError(
                    "GOOGLE_GENAI_USE_VERTEXAI is enabled, but GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION is not set."
                )
            _debug_value("GOOGLE_GENAI_USE_VERTEXAI", "True")
            _debug_value("GOOGLE_CLOUD_PROJECT", project)
            _debug_value("GOOGLE_CLOUD_LOCATION", location)
            client = google_genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            api_key = _env_str("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set, but Gemini model was requested. "
                    "Set GOOGLE_API_KEY or enable Vertex AI with GOOGLE_GENAI_USE_VERTEXAI=True."
                )
            _debug_secret("GOOGLE_API_KEY", api_key)
            client = google_genai.Client(api_key=api_key)

        contents, system_instruction = _to_gemini_contents_and_system_instruction(message)

        # Gemini requires at least one non-system message, so seed a minimal user turn
        # when callers provide only a system prompt.
        if not contents and system_instruction:
            contents = [{"role": "user", "parts": [{"text": "Please start."}]}]
        elif not contents:
            raise ValueError("No messages found for Gemini request.")

        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if google_genai_types is not None:
            gen_config = google_genai_types.GenerateContentConfig(**config_kwargs)
        else:
            gen_config = config_kwargs

        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config,
        )

        text = getattr(resp, "text", None)
        if text is None:
            text = str(resp)

        pt, ct, tt, tot = _extract_gemini_usage(resp)
        return resp, text, "", tt

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        print(e)
        time.sleep(3)
        return gemini_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def claude_response(
    message,
    model="claude-3-sonnet-20240229",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    client = _get_anthropic_client()

    msg = []
    for m in message:
        role = m["role"] if m["role"] == "user" else "assistant"
        if msg and msg[-1]["role"] == role:
            msg[-1]["content"] += m["content"]
        else:
            msg.append({"role": role, "content": m["content"]})
    try:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": msg,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        res = client.messages.create(**kwargs)
        return res, res.content[0].text, "", 0
    except Exception as e:
        print(e)
        time.sleep(3)
        return claude_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def llama_response(
    message,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    client = _get_together_client()

    try:
        chat_completion = client.chat.completions.create(
            messages=message,
            model="meta-llama/Llama-2-70b-chat-hf",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return chat_completion, chat_completion.choices[0].message.content, "", 0
    except Exception as e:
        print(e)
        time.sleep(1)
        return llama_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def mistral_response(
    message: list,
    model="mistral-large-latest",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    client, message_cls = _get_mistral_client()

    msg = [message_cls(role=m["role"], content=m["content"]) for m in message]
    try:
        res = client.chat(model=model, messages=msg)
        return res, res.choices[0].message.content, "", 0
    except Exception as e:
        print(e)
        time.sleep(1)
        return mistral_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def qwen_response(
    message: list,
    model="qwen_thinking_4b",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
    _attempt: int = 1,
):
    if model not in REMOTE_QWEN_SPECS:
        raise ValueError(f"Unsupported qwen remote model: {model}")

    spec = REMOTE_QWEN_SPECS[model]
    tokenizer_path = spec["tokenizer_path"]
    server_model_name = spec["server_model_name"]
    enable_reasoning = spec["enable_reasoning"]
    thinking_start_token = spec.get("thinking_start_token", "")
    thinking_end_token = spec.get("thinking_end_token", "</think>")
    client = _get_remote_qwen_client(model)

    try:
        allow_remote_tokenizer_fallback = (
            os.getenv("ALLOW_REMOTE_TOKENIZER_FALLBACK", "0").strip().lower()
            in {"1", "true", "yes"}
        )

        qwen_tokenizer = _load_tokenizer_cached(
            tokenizer_path,
            allow_remote_fallback=allow_remote_tokenizer_fallback,
        )

        if qwen_tokenizer is not None:
            raw_prompt_text = _apply_chat_template_safely(
                qwen_tokenizer,
                message,
                family="qwen",
                enable_reasoning=enable_reasoning,
            )
        else:
            print(
                f"Warning: tokenizer unavailable for {tokenizer_path}. "
                f"Falling back to plain chat prompt for remote API call."
            )
            raw_prompt_text = _build_fallback_chat_prompt(message)

        chat_completion = _request_completion_with_fallback(
            client=client,
            model=server_model_name,
            prompt=raw_prompt_text,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort if enable_reasoning else None,
            extra_kwargs={"logprobs": 1},
        )

        response_text = chat_completion.choices[0].text if chat_completion.choices else ""
        parsed = _split_reasoning_text(
            response_text=response_text,
            tokenizer=qwen_tokenizer,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        )
        return chat_completion, parsed["text"], parsed["cot"], parsed["num_thinking_tokens"]

    except Exception as e:
        _print_openai_exception_debug(e)
        print(e)
        max_attempts = int(os.getenv("TWENTYQ_MAX_QWEN_RETRIES", "3"))
        if _attempt >= max_attempts:
            raise RuntimeError(
                f"qwen_response failed after {_attempt} attempts for model={model}"
            ) from e
        time.sleep(1)
        return qwen_response(
            message,
            model,
            temperature,
            top_p,
            max_tokens,
            reasoning_effort,
            _attempt=_attempt + 1,
        )


def gpt_oss_20b_response(
    message: list,
    model="gpt_oss_20b",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    client = _get_or_create_openai_client(
        _openai_compatible_base_url(model),
        api_key=_openai_compatible_api_key(model),
    )
    server_model_name = "openai/gpt-oss-20b"

    try:
        request_kwargs = {
            "model": server_model_name,
            "messages": message,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        try:
            res = client.chat.completions.create(**request_kwargs)
        except TypeError:
            request_kwargs.pop("reasoning_effort", None)
            request_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
            res = client.chat.completions.create(**request_kwargs)

        raw_content = getattr(res.choices[0].message, "content", "")
        raw_tool_calls = getattr(res.choices[0].message, "tool_calls", None)
        final_output, cot = extract_gpt_oss_content_and_cot(raw_content, tool_calls=raw_tool_calls)

        num_thinking_tokens = 0
        usage = getattr(res, "usage", None)
        if usage is not None:
            num_thinking_tokens = _extract_reasoning_tokens_from_usage(usage)

        if num_thinking_tokens == 0 and cot:
            tok = None
            try:
                tok = _load_tokenizer_cached("openai/gpt-oss-20b")
            except Exception:
                tok = None
            num_thinking_tokens = _count_tokens_with_tokenizer(cot, tok)

        return res, final_output, cot, num_thinking_tokens
    except Exception as e:
        print(e)
        time.sleep(1)
        return gpt_oss_20b_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


# ============================================================================
# Local model responses via vLLM
# ============================================================================

def _local_vllm_response(
    message: list,
    model: str,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    config = get_local_model_config(model)
    if config is None:
        raise ValueError(f"Unknown local model: {model}")

    try:
        res, parsed = _generate_vllm_completion(
            message=message,
            config=config,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return res, parsed["text"], parsed["cot"], parsed["num_thinking_tokens"]
    except Exception as e:
        print(e)
        time.sleep(1)
        return _local_vllm_response(
            message=message,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )


def local_qwen_30b_instruct_response(
    message: list,
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def local_qwen_4b_instruct_response(
    message: list,
    model="Qwen/Qwen3-4B-Instruct-2507",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def local_qwen_4b_response(
    message: list,
    model="Qwen/Qwen3-4B-Thinking-2507",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def local_qwen_30b_response(
    message: list,
    model="Qwen/Qwen3-30B-A3B-Thinking-2507",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def local_llama31_8b_response(
    message: list,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


# ============================================================================
# Router
# ============================================================================

def _unsupported_model_response(
    message,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    raise NotImplementedError(f"Unsupported model: {model}")


def _local_endpoint_configured() -> bool:
    """True if a local vLLM endpoint was explicitly configured."""
    return bool(os.environ.get("VLLM_PORT") or os.environ.get("VLLM_BASE_URL"))


def local_chat_response(
    message,
    model,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=None,
    **_,
):
    """Generic OpenAI-compatible chat path for an unrecognized local model.

    Talks to the vLLM server on VLLM_PORT (or VLLM_BASE_URL) over
    /v1/chat/completions -- the portable path every server supports. Reasoning
    text, when the server exposes it, arrives as `reasoning_content`
    (vLLM/SGLang) or `reasoning` (OpenRouter). Returns the 4-tuple contract
    (raw_response, visible_text, thinking_text, think_tokens).
    """
    client = _get_or_create_openai_client(
        _openai_compatible_base_url(model),
        api_key=_openai_compatible_api_key(model),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=message,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    text = _coerce_chat_content_to_text(content).strip()
    cot = _coerce_chat_content_to_text(reasoning).strip()
    think = _extract_reasoning_tokens_from_usage(getattr(resp, "usage", None))
    return resp, text, cot, think


def get_response_method(model):
    exact_methods = {
        "Qwen/Qwen3-4B-Instruct-2507": local_qwen_4b_instruct_response,
        "Qwen/Qwen3-4B-Thinking-2507": local_qwen_4b_response,
        "Qwen/Qwen3-30B-A3B-Thinking-2507": local_qwen_30b_response,
        "meta-llama/Meta-Llama-3.1-8B-Instruct": local_llama31_8b_response,
        "Qwen/Qwen3-30B-A3B-Instruct-2507": local_qwen_30b_instruct_response,

        "qwen3-4b-instruct-local": local_qwen_4b_instruct_response,
        "qwen3-4b-local": local_qwen_4b_response,
        "qwen3-30b-local": local_qwen_30b_response,
        "llama3.1-8b-local": local_llama31_8b_response,
        "qwen3-30b-instruct-local": local_qwen_30b_instruct_response,

        "qwen_instruct_4b": qwen_response,
        "qwen_thinking_4b": qwen_response,
        "qwen_thinking_30b": qwen_response,
        "qwen_instruct_30b": qwen_response,

        "qwen_4b": qwen_response,
        "qwen_30b": qwen_response,

        "gpt-5": gpt_response,
        "gpt-5-mini": gpt_response,
        "gpt-5.4": gpt_response,
        "gpt-5.2": gpt_response,
    }

    if model in exact_methods:
        return exact_methods[model]

    # A --model-config entry takes precedence over name-based hosted routing.
    # 20Q uses the portable Chat Completions transport for custom models; the
    # config supplies its base_url and optional api_key_env.
    if _is_suite_registered_local(model):
        return local_chat_response

    model_lower = str(model).lower()

    if model_lower.startswith("gpt"):
        if model_lower == "gpt_oss_20b":
            return gpt_oss_20b_response
        return gpt_response
    if model_lower.startswith("cohere"):
        return cohere_response
    if model_lower.startswith("palm"):
        return palm_response
    if model_lower.startswith("claude"):
        return claude_response
    if model_lower.startswith("llama"):
        return llama_response
    if model_lower.startswith("mistral"):
        return mistral_response
    if model_lower.startswith("gemini"):
        return gemini_response
    if model_lower.startswith("qwen_"):
        return qwen_response

    # Unknown model + a configured local endpoint -> assume it is served there
    # over /v1/chat/completions (mirrors model_utils.py's inference fallback, so
    # a freshly-served HuggingFace model works by name alone).
    if _local_endpoint_configured():
        return local_chat_response

    def _unsupported(*args, **kwargs):
        raise NotImplementedError(
            f"Unsupported model: {model}. If it is a local model, start its vLLM "
            f"server and set VLLM_PORT."
        )

    return _unsupported
