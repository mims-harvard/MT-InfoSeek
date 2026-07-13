"""Utility functions for calling models."""

import hashlib
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Set
import asyncio
import dataclasses
from dataclasses import dataclass
import re

from openai import AsyncOpenAI, AsyncAzureOpenAI
import httpx
import tenacity
from tenacity import retry
import transformers

import chat_backends
import model_registry
from model_registry import (
    BACKEND_OPENAI_CHAT,
    get_backend,
    register_local_model as _registry_register_local_model,
)

wait_random_exponential = tenacity.wait_random_exponential
stop_after_attempt = tenacity.stop_after_attempt

MAX_TOKS = 65536  # Default upper bound for remote models.
LOCAL_REQUEST_TIMEOUT_S = 1800.0
_GPT_OSS_DEBUG_FINGERPRINTS_SEEN: Set[str] = set()

@dataclass
class GenerationResult:
  """Result from a single generation call."""
  text: str
  num_thinking_tokens: int = 0
  cot: str = ""
  cost_usd: float = 0.0


@dataclass
class LocalModelConfig:
  """Configuration for a local model."""
  tokenizer_name: str
  base_url: str = "http://127.0.0.1:8011/v1"
  enable_reasoning: bool = True
  thinking_start_token: str = ""
  thinking_end_token: str = ""


# Local models hosted via vLLM. Definitions live in model_registry.py so that
# this module and gsme_q_mt/model_utils.py cannot drift apart.
LOCAL_MODEL_CONFIGS: Dict[str, LocalModelConfig] = model_registry.build_local_configs(
    LocalModelConfig
)
_LOCAL_CONFIGS_REVISION = model_registry.revision()


def _sync_local_configs() -> None:
  """Rebuild the dataclass view when the shared registry has changed.

  Models can be registered *or overridden* at runtime (e.g. `--model-config`)
  after this module built its dict at import, so a stale view would silently
  send requests to the wrong endpoint.
  """
  global _LOCAL_CONFIGS_REVISION
  if _LOCAL_CONFIGS_REVISION == model_registry.revision():
    return
  LOCAL_MODEL_CONFIGS.clear()
  LOCAL_MODEL_CONFIGS.update(model_registry.build_local_configs(LocalModelConfig))
  _LOCAL_CONFIGS_REVISION = model_registry.revision()

# Import the Gemini SDK if available; fail only when a Gemini model is requested.
# Non-Gemini runs do not require google-genai.
try:
  from google import genai
  from google.genai import types as genai_types
except ImportError:
  genai = None
  genai_types = None

# Hosted model pricing. Defined in model_registry.py (shared with GSME-Q-MT).
GPT_COSTS = model_registry.GPT_COSTS
GEMINI_COSTS = model_registry.GEMINI_COSTS

def _gemini_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
  # find best prefix match
  key = None
  for k in GEMINI_COSTS.keys():
    if model == k or model.startswith(k):
      key = k
      break
  if key is None:
    return 0.0

  spec = GEMINI_COSTS[key]
  pt = float(prompt_tokens or 0)
  ct = float(completion_tokens or 0)
  in_rate = float(spec["in"])
  out_rate = float(spec["out"])

  return pt * in_rate / 1_000_000.0 + ct * out_rate / 1_000_000.0

# Claude models (shared; see model_registry.py)
CLAUDE_MODELS = model_registry.CLAUDE_MODELS

_azure_openai_client: Optional[Any] = None  # AsyncAzureOpenAI or AsyncOpenAI
_anthropic_client: Optional[httpx.AsyncClient] = None
_local_clients: Dict[Tuple[str, str], AsyncOpenAI] = {}
_gemini_client: Optional[Any] = None

_tokenizer: Any = None


def get_tokenizer(tokenizer_name: str):
  """Get or create a tokenizer."""
  global _tokenizer
  if _tokenizer is None:
    _tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
  return _tokenizer


def get_local_model_config(model_name: str, port: str = None) -> Optional[LocalModelConfig]:
  """Get config for a local model, returns None if not found.

  Returns a copy when a port override applies, so a port passed on one call
  cannot leak into later calls via the shared registry object. A `base_url`
  pinned in the registry (e.g. a hosted OpenAI-compatible endpoint) wins over
  `port`.
  """
  _sync_local_configs()
  config = LOCAL_MODEL_CONFIGS.get(model_name)
  if config is None:
    return None
  base_url = model_registry.resolve_base_url(
      model_name, default=config.base_url, port=port
  )
  return dataclasses.replace(config, base_url=base_url)


def _build_local_apply_chat_template_kwargs(
    model: str,
    messages: List[Dict[str, str]],
    config: LocalModelConfig,
) -> Dict[str, Any]:
  apply_kwargs: Dict[str, Any] = {
      "conversation": messages,
      "add_generation_prompt": True,
      "tokenize": False,
  }
  if config.enable_reasoning:
    if "qwen" in model.lower():
      apply_kwargs["enable_reasoning"] = True
    apply_kwargs["add_special_tokens"] = True
  return apply_kwargs


def build_local_raw_prompt_text(
    model: str,
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    port: Optional[str] = None,
) -> str:
  """Render the local-model raw prompt via apply_chat_template."""
  if config is None:
    config = get_local_model_config(model, port)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  tokenizer = get_tokenizer(config.tokenizer_name)
  apply_kwargs = _build_local_apply_chat_template_kwargs(model, messages, config)
  try:
    return tokenizer.apply_chat_template(**apply_kwargs)
  except Exception as e:
    print(f"\nERROR: Failed to apply chat template with error {e}.\nThe messages were:\n{messages}\n")
    raise


def _append_qwen_cot_prefill(raw_prompt_text: str, cot_prefill: str) -> str:
  suffix = "<|im_start|>assistant\n<think>\n"
  if not raw_prompt_text.endswith(suffix):
    tail = raw_prompt_text[-200:]
    raise ValueError(
        "Qwen reasoning prompt did not end with the expected assistant think prefix. "
        f"Prompt tail: {tail!r}"
    )
  prefill = cot_prefill.rstrip()
  if not prefill:
    return raw_prompt_text
  return raw_prompt_text + prefill + "\n\n"


def build_local_raw_prompt_with_cot_prefill(
    model: str,
    messages: List[Dict[str, str]],
    cot_prefill: str,
    config: Optional[LocalModelConfig] = None,
    port: Optional[str] = None,
) -> str:
  """Render prompt text and inject a reasoning prefill inside the generation CoT."""
  if config is None:
    config = get_local_model_config(model, port)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  raw_prompt_text = build_local_raw_prompt_text(model, messages, config=config)
  if not cot_prefill.strip():
    return raw_prompt_text
  if "qwen" in model.lower():
    return _append_qwen_cot_prefill(raw_prompt_text, cot_prefill)
  raise NotImplementedError(
      "CoT-prefill prompt injection is currently implemented only for Qwen local models."
  )


def register_local_model(
    model_key: str,
    tokenizer_name: str,
    base_url: Optional[str] = None,
    enable_reasoning: bool = True,
    thinking_start_token: str = "",
    thinking_end_token: str = "",
    backend: str = BACKEND_OPENAI_CHAT,
    reasoning_parser: Optional[str] = None,
    api_key_env: Optional[str] = None,
):
  """Register a new local model, writing through to the shared registry.

  Custom models default to the `/v1/chat/completions` transport, which is what
  every OpenAI-compatible server (vLLM, SGLang, Ollama, OpenRouter) supports.
  """
  _registry_register_local_model(
      model_key,
      tokenizer_name,
      base_url=base_url,
      enable_reasoning=enable_reasoning,
      thinking_start_token=thinking_start_token,
      thinking_end_token=thinking_end_token,
      backend=backend,
      reasoning_parser=reasoning_parser,
      api_key_env=api_key_env,
  )
  LOCAL_MODEL_CONFIGS[model_key] = LocalModelConfig(
      **model_registry.local_config_kwargs(model_key, LocalModelConfig)
  )


def get_azure_openai_client():
  """Get or create the client used for GPT models.

  Prefers Azure OpenAI (how the published numbers were produced). Falls back to
  the standard OpenAI API when no Azure key is configured, which -- together
  with `OPENAI_BASE_URL` -- also covers OpenAI-compatible gateways such as
  OpenRouter, Ollama and SGLang.
  """
  global _azure_openai_client
  if _azure_openai_client is None:
    if os.environ.get("AZURE_OPENAI_API_KEY"):
      _azure_openai_client = AsyncAzureOpenAI(
          api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
          azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
          api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
          timeout=600,
      )
    elif os.environ.get("OPENAI_API_KEY"):
      _azure_openai_client = AsyncOpenAI(
          api_key=os.environ["OPENAI_API_KEY"],
          base_url=os.environ.get("OPENAI_BASE_URL") or None,
          timeout=600,
      )
    else:
      raise RuntimeError(
          "No GPT credentials found. Set AZURE_OPENAI_API_KEY (+ "
          "AZURE_OPENAI_ENDPOINT), or OPENAI_API_KEY (+ optional "
          "OPENAI_BASE_URL for an OpenAI-compatible gateway)."
      )
  return _azure_openai_client


def get_anthropic_client() -> httpx.AsyncClient:
  """Get or create the Anthropic async client."""
  global _anthropic_client
  if _anthropic_client is None:
    _anthropic_client = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        headers={
            "Content-Type": "application/json",
            "Anthropic-Version": "2023-06-01",
            "X-Api-Key": os.environ.get("ANTHROPIC_API_KEY", ""),
        },
        timeout=600,
    )
  return _anthropic_client


def get_local_client(
    base_url: str = "http://127.0.0.1:8011/v1",
    api_key: str = "EMPTY",
) -> AsyncOpenAI:
  """Get or create a local/OpenAI-compatible async client.

  Cached per (base_url, api_key): a single global client would otherwise pin the
  first base_url ever requested and silently send later requests to the wrong
  endpoint.
  """
  key = (base_url, api_key)
  if key not in _local_clients:
    _local_clients[key] = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=LOCAL_REQUEST_TIMEOUT_S,
        max_retries=0,
    )
  return _local_clients[key]


def get_gemini_client() -> Any:
  """Get or create the Google GenAI client."""
  global _gemini_client
  if genai is None:
    raise ImportError(
        "google.genai is not installed. Please install the google-genai package."
    )
  if _gemini_client is None:
    use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    if not use_vertexai:
      gemini_api_key = os.environ.get("GEMINI_API_KEY")
      _gemini_client = genai.Client(api_key=gemini_api_key).aio
    else:
      # Use Vertex AI: Automatically loads GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
      try:
        # New google-genai versions use `vertexai`.
        _gemini_client = genai.Client(vertexai=True).aio
      except TypeError:
        # Backward compatibility with older google-genai versions.
        _gemini_client = genai.Client(vertex_ai=True).aio
  return _gemini_client


def _gemini_exception_status_code(exc: Exception) -> Optional[int]:
  """Best-effort extraction of HTTP status code from SDK exceptions."""
  # Common direct attributes on wrapped exceptions.
  for attr in ("status_code", "status", "code", "http_status"):
    try:
      value = getattr(exc, attr, None)
      if isinstance(value, int):
        return value
    except Exception:
      pass

  # httpx-style errors can carry response objects.
  try:
    resp = getattr(exc, "response", None)
    if resp is not None:
      status = getattr(resp, "status_code", None)
      if isinstance(status, int):
        return status
  except Exception:
    pass

  return None


def _is_retryable_gemini_exception(exc: Exception) -> bool:
  """Retry policy for Gemini transient/capacity errors (notably 429)."""
  status = _gemini_exception_status_code(exc)
  if status in (408, 409, 429, 500, 502, 503, 504):
    return True

  msg = str(exc).lower()
  retry_signals = (
      "429",
      "resource exhausted",
      "too many requests",
      "rate limit",
      "quota",
      "temporarily unavailable",
      "service unavailable",
      "deadline exceeded",
      "internal",
      "unavailable",
  )
  return any(sig in msg for sig in retry_signals)


def load_cache_file(cache_file: str) -> Dict[str, Any]:
  """Load cache from a JSONL file."""
  cache: Dict[str, Any] = {}
  if not os.path.exists(cache_file):
    return cache

  with open(cache_file, "r") as f:
    for line in f:
      entry = json.loads(line)
      prompt = entry["prompt"]

      text = entry.get("completion", entry.get("text", ""))
      cache[prompt] = {
          "text": text,
          "num_thinking_tokens": entry.get("num_thinking_tokens", 0),
          "cot": entry.get("cot", ""),
          "cost_usd": entry.get("cost_usd", 0.0),
      }
  return cache


def jsonify_prompt(prompt: List[Dict[str, str]]) -> str:
  """Convert prompt to JSON string for caching."""
  return json.dumps(prompt)
#########
# GPT-OSS / chat-completions helpers -- single implementation in chat_backends.py
# so this module and gsme_q_mt/model_utils.py cannot drift.
#########
_GPT_OSS_CONTROL_TOKEN_RE = chat_backends._GPT_OSS_CONTROL_TOKEN_RE
_coerce_chat_content_to_text = chat_backends.coerce_chat_content_to_text
_message_like_get = chat_backends.message_like_get
_message_like_dump = chat_backends.message_like_dump
_extract_gpt_oss_from_message_like = chat_backends.extract_gpt_oss_from_message_like
_usage_get_int = chat_backends.usage_get_int
_extract_reasoning_tokens_from_usage = chat_backends.extract_reasoning_tokens_from_usage
_build_local_gpt_oss_request_kwargs = chat_backends.build_chat_request_kwargs
_create_local_gpt_oss_chat_completion = chat_backends.create_chat_completion

def _sanitize_debug_fragment(value: Any) -> str:
  raw = str(value) if value is not None else "na"
  cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
  return cleaned or "na"


def _truncate_debug_text(text: str, max_chars: int = 500) -> str:
  if len(text) <= max_chars:
    return text
  return text[:max_chars] + "...<truncated>"


def _safe_token_count(tokenizer: Any, text: str) -> Optional[int]:
  if not text:
    return 0
  try:
    return len(tokenizer.encode(text))
  except Exception:
    return None


def _render_gpt_oss_harmony_prompt_tokens(
    messages: List[Dict[str, str]],
    reasoning_effort: str,
) -> Tuple[Optional[int], Optional[str]]:
  try:
    from vllm.entrypoints.openai.parser.harmony_utils import (
        get_system_message,
        parse_chat_inputs_to_harmony_messages,
        render_for_completion,
    )

    harmony_messages = [get_system_message(reasoning_effort=reasoning_effort)]
    harmony_messages.extend(parse_chat_inputs_to_harmony_messages(messages))
    token_ids = render_for_completion(harmony_messages)
    return len(token_ids), None
  except Exception as exc:
    return None, f"{type(exc).__name__}: {exc}"


def _dump_gpt_oss_request_debug(
    *,
    model: str,
    messages: List[Dict[str, str]],
    request_kwargs: Dict[str, Any],
    reasoning_effort: str,
    tokenizer: Any,
    error: Exception,
    debug_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
  debug_context = debug_context or {}
  request_summary = {
      key: value
      for key, value in request_kwargs.items()
      if key != "messages"
  }
  fingerprint_material = {
      "model": model,
      "request_summary": request_summary,
      "messages": messages,
      "debug_context": debug_context,
      "error": f"{type(error).__name__}: {error}",
  }
  fingerprint = hashlib.sha256(
      json.dumps(fingerprint_material, sort_keys=True, default=str).encode("utf-8")
  ).hexdigest()
  if fingerprint in _GPT_OSS_DEBUG_FINGERPRINTS_SEEN:
    return None
  _GPT_OSS_DEBUG_FINGERPRINTS_SEEN.add(fingerprint)

  message_summaries: List[Dict[str, Any]] = []
  for idx, msg in enumerate(messages):
    content_text = _coerce_chat_content_to_text(msg.get("content", ""))
    message_summaries.append({
        "index": idx,
        "role": msg.get("role"),
        "content_chars": len(content_text),
        "content_tokens_hf": _safe_token_count(tokenizer, content_text),
        "content_preview": _truncate_debug_text(content_text),
    })

  harmony_prompt_tokens, harmony_error = _render_gpt_oss_harmony_prompt_tokens(
      messages=messages,
      reasoning_effort=reasoning_effort,
  )

  dump_payload = {
      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
      "cwd": os.getcwd(),
      "pid": os.getpid(),
      "slurm_job_id": os.getenv("SLURM_JOB_ID"),
      "model": model,
      "reasoning_effort": reasoning_effort,
      "error_type": type(error).__name__,
      "error_message": str(error),
      "debug_context": debug_context,
      "request_kwargs": request_summary,
      "message_summaries": message_summaries,
      "messages": messages,
      "harmony_rendered_prompt_tokens": harmony_prompt_tokens,
      "harmony_render_error": harmony_error,
      "joined_message_chars": sum(
          summary["content_chars"] for summary in message_summaries
      ),
      "joined_message_tokens_hf": sum(
          summary["content_tokens_hf"] or 0 for summary in message_summaries
      ),
  }

  debug_dir = os.getenv(
      "QUESTBENCH_GPT_OSS_DEBUG_DIR",
      os.path.join(os.getcwd(), "logs_evaluation"),
  )
  os.makedirs(debug_dir, exist_ok=True)

  sample_frag = _sanitize_debug_fragment(debug_context.get("sample_id"))
  turn_frag = _sanitize_debug_fragment(debug_context.get("turn"))
  call_frag = _sanitize_debug_fragment(debug_context.get("call_index"))
  dump_name = (
      f"gpt_oss_request_error_sample{sample_frag}_turn{turn_frag}_"
      f"call{call_frag}_{os.getpid()}_{int(time.time())}.json"
  )
  dump_path = os.path.join(debug_dir, dump_name)
  with open(dump_path, "w", encoding="utf-8") as f:
    json.dump(dump_payload, f, indent=2, ensure_ascii=False)
  return dump_path

def _gpt_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
  if model not in GPT_COSTS:
    return 0.0
  p = GPT_COSTS[model]["prompt_tokens"]
  c = GPT_COSTS[model]["completion_tokens"]
  return float(prompt_tokens) * p + float(completion_tokens) * c


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=5),
)
async def openai_chat_request(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> Dict[str, Any]:
  client = get_azure_openai_client()
  response = await client.chat.completions.create(
      model=model,
      messages=messages,
      **generation_config
  )

  usage = response.usage
  # Chat Completions: usage.completion_tokens_details.reasoning_tokens
  try:
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
      reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    else:
      reasoning_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0)
  except Exception:
    reasoning_tokens = 0

  return {
      "choices": [{"message": {"content": response.choices[0].message.content}}],
      "usage": {
          "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
          "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
          "reasoning_tokens": reasoning_tokens,
      }
  }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def claude_request(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> Dict[str, Any]:
  """Async request to Anthropic Claude API."""
  client = get_anthropic_client()

  system_content = None
  filtered_messages = []
  for msg in messages:
    if msg["role"] == "system":
      system_content = msg["content"]
    else:
      filtered_messages.append(msg)

  data = {
      "model": model,
      "messages": filtered_messages,
      "max_tokens": generation_config.get("max_tokens", MAX_TOKS),
  }
  if system_content:
    data["system"] = system_content

  for key in ["temperature", "top_p"]:
    if key in generation_config:
      data[key] = generation_config[key]

  response = await client.post("/v1/messages", json=data)
  response.raise_for_status()
  result = response.json()

  if "content" not in result:
    raise ValueError(f"Unexpected response format: {result}")

  return result


def _extract_gemini_usage(resp: Any) -> Tuple[int, int, int, int]:
  """
  Returns: (prompt_tokens, response_tokens, thoughts_tokens, total_tokens)
  """
  usage = getattr(resp, "usage_metadata", None)
  if usage is None and isinstance(resp, dict):
    usage = resp.get("usage_metadata") or resp.get("usageMetadata")

  pt = ct = tt = tot = 0

  if usage is not None:
    # object-like
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

    # dict-like
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

  pt, ct, tt, tot = int(pt), int(ct), int(tt), int(tot)

  return pt, ct, tt, tot


def _gemini_output_tokens_for_cost(
    prompt_tokens: int,
    response_tokens: int,
    total_tokens: int,
    thoughts_tokens: int = 0,
) -> int:
  """Compute billable output tokens for Gemini cost estimation.

  Prefer explicit response/candidate tokens from usage metadata.
  Fall back to total - prompt when response token count is unavailable.
  Includes thinking tokens (`thoughts_tokens`) in output billing.
  """
  tt = max(0, int(thoughts_tokens or 0))
  if response_tokens > 0:
    return int(response_tokens) + tt
  if total_tokens > 0:
    return max(0, int(total_tokens) - int(prompt_tokens))
  return tt


def _normalize_gemini_text_content(content: Any) -> str:
  """Normalize message content into plain text for Gemini contents."""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: List[str] = []
    for item in content:
      if isinstance(item, str):
        parts.append(item)
      elif isinstance(item, dict):
        # OpenAI-style multimodal block: {"type": "text", "text": "..."}
        if item.get("type") == "text" and "text" in item:
          parts.append(str(item.get("text", "")))
        elif "text" in item:
          parts.append(str(item.get("text", "")))
    return "\n".join([p for p in parts if p])
  return str(content)


def _to_gemini_contents_and_system_instruction(
    messages: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
  """Convert OpenAI-style chat messages to Gemini contents + system instruction.

  Preserves turn boundaries: one input message -> one Gemini content item.
  """
  contents: List[Dict[str, Any]] = []
  system_parts: List[str] = []

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


@retry(
  stop=stop_after_attempt(10),
  wait=wait_random_exponential(multiplier=1, max=60),
  retry=tenacity.retry_if_exception(_is_retryable_gemini_exception),
  reraise=True,
)
async def gemini_request(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> GenerationResult:
  """Async request to Google Gemini API using google.genai client SDK."""
  client = get_gemini_client()
  contents, system_instruction = _to_gemini_contents_and_system_instruction(messages)

  if not contents:
    raise ValueError("No non-system messages found for Gemini request.")

  config_kwargs = {
      "temperature": generation_config["temperature"],
      "max_output_tokens": generation_config["max_output_tokens"],
  }
  if system_instruction:
    config_kwargs["system_instruction"] = system_instruction

  if genai_types is not None:
    gen_config = genai_types.GenerateContentConfig(**config_kwargs)
  else:
    gen_config = config_kwargs

  resp = await client.models.generate_content(
      model=model,
      contents=contents,
      config=gen_config,
  )

  text = getattr(resp, "text", None)
  if text is None:
    text = str(resp)

  pt, ct, tt, tot = _extract_gemini_usage(resp)
  output_tokens_for_cost = _gemini_output_tokens_for_cost(pt, ct, tot, tt)
  cost = _gemini_cost_usd(model, pt, output_tokens_for_cost)

  return GenerationResult(
      text=text,
      num_thinking_tokens=tt,
      cot="",
      cost_usd=cost,
  )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=5),
    reraise=True,
)
async def local_gpt_oss_chat_request(
    model: str,
    port: str,
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    debug_context: Optional[Dict[str, Any]] = None,
    **generation_config
) -> GenerationResult:
  """Async request to local GPT-OSS via Chat Completions API."""
  if config is None:
    config = get_local_model_config(model, port)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  client = get_local_client(config.base_url)
  tokenizer = get_tokenizer(config.tokenizer_name)
  request_kwargs, reasoning_effort = _build_local_gpt_oss_request_kwargs(
      model,
      messages,
      generation_config=generation_config,
  )
  try:
    response = await _create_local_gpt_oss_chat_completion(
        client,
        request_kwargs,
        reasoning_effort,
    )
  except Exception as exc:
    try:
      dump_path = _dump_gpt_oss_request_debug(
          model=model,
          messages=messages,
          request_kwargs=request_kwargs,
          reasoning_effort=reasoning_effort,
          tokenizer=tokenizer,
          error=exc,
          debug_context=debug_context,
      )
      if dump_path:
        print(f"[GPT_OSS_DEBUG_DUMP] wrote request debug to {dump_path}")
    except Exception as dump_exc:
      print(
          "[GPT_OSS_DEBUG_DUMP] failed to write debug dump: "
          f"{type(dump_exc).__name__}: {dump_exc}"
      )
    raise

  choice = response.choices[0]
  message = choice.message
  final_output, cot = _extract_gpt_oss_from_message_like(message)

  num_thinking_tokens = 0
  usage = getattr(response, "usage", None)
  if usage is not None:
    num_thinking_tokens = _extract_reasoning_tokens_from_usage(usage)

  if num_thinking_tokens == 0 and cot:
    num_thinking_tokens = len(tokenizer.encode(cot))
  if num_thinking_tokens == 0 and usage is not None:
    completion_tokens = _usage_get_int(
        usage,
        "completion_tokens",
        "output_tokens",
        "response_token_count",
        "responseTokenCount",
    )
    if completion_tokens > 0:
      visible_output_tokens = len(tokenizer.encode(final_output)) if final_output else 0
      num_thinking_tokens = max(0, completion_tokens - visible_output_tokens)

  return GenerationResult(
      text=final_output.strip(),
      num_thinking_tokens=num_thinking_tokens,
      cot=cot.strip(),
      cost_usd=0.0,
  )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=5),
    reraise=True,
)
async def local_openai_chat_request(
    model: str,
    port: str,
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    **generation_config
) -> GenerationResult:
  """Chat-completions request to any OpenAI-compatible server.

  Used for custom / unregistered local models. Works against vLLM, SGLang,
  Ollama and OpenRouter, which all expose `/v1/chat/completions` (and, unlike
  `/v1/completions`, do so universally). Reasoning text, when the server exposes
  it, arrives as `reasoning_content` (vLLM/SGLang) or `reasoning` (OpenRouter).
  """
  if config is None:
    config = get_local_model_config(model, port)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  api_key_env = model_registry.api_key_env_for(model)
  api_key = os.environ.get(api_key_env, "EMPTY") if api_key_env else "EMPTY"
  client = get_local_client(config.base_url, api_key=api_key)

  request_kwargs, reasoning_effort = chat_backends.build_chat_request_kwargs(
      model, messages, generation_config
  )
  response = await chat_backends.create_chat_completion(
      client, request_kwargs, reasoning_effort
  )

  message = response.choices[0].message
  text, cot = chat_backends.parse_generic_chat_message(message)

  num_thinking_tokens = _extract_reasoning_tokens_from_usage(
      getattr(response, "usage", None)
  )
  if num_thinking_tokens == 0 and cot:
    try:
      num_thinking_tokens = len(get_tokenizer(config.tokenizer_name).encode(cot))
    except Exception:
      num_thinking_tokens = 0

  return GenerationResult(
      text=text,
      num_thinking_tokens=num_thinking_tokens,
      cot=cot,
      cost_usd=0.0,
  )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=5),
    reraise=True,
)
async def local_model_request(
    model: str,
    port: str,
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    debug_context: Optional[Dict[str, Any]] = None,
    **generation_config
) -> GenerationResult:
  """Async request to local model via vLLM.

  Transport is chosen by the shared registry: Qwen stays on raw
  `/v1/completions` (as the published numbers were produced); gpt-oss and any
  custom model use `/v1/chat/completions`.
  """
  if config is None:
    config = get_local_model_config(model, port)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  if get_backend(model) == BACKEND_OPENAI_CHAT:
    if "gpt-oss" in model.lower():
      return await local_gpt_oss_chat_request(
          model=model,
          port=port,
          messages=messages,
          config=config,
          debug_context=debug_context,
          **generation_config,
      )
    return await local_openai_chat_request(
        model=model,
        port=port,
        messages=messages,
        config=config,
        **generation_config,
    )

  client = get_local_client(config.base_url)
  tokenizer = get_tokenizer(config.tokenizer_name)
  raw_prompt_text = build_local_raw_prompt_text(model, messages, config=config)
  
  response = await client.completions.create(
      model=model,
      prompt=raw_prompt_text,
      logprobs=1,
      echo=False,
      temperature=generation_config.get("temperature", 0.6),
      top_p=generation_config.get("top_p", 0.95),
      max_tokens=generation_config.get("max_tokens", MAX_TOKS),
  )

  choice = response.choices[0]
  response_text = choice.text

  if config.enable_reasoning and config.thinking_end_token:
    tokens = choice.logprobs.tokens if choice.logprobs else []
    # split by cot end token
    if config.thinking_end_token in tokens:
      num_thinking_tokens = tokens.index(config.thinking_end_token)
      cot, final_output = response_text.split(config.thinking_end_token, 1)
    elif config.thinking_end_token in response_text:
      cot, final_output = response_text.split(config.thinking_end_token, 1)
      num_thinking_tokens = len(tokenizer.encode(cot))
    else:
      num_thinking_tokens = len(tokens) if tokens else 0
      print(f"\nWARNING: No {config.thinking_end_token} token found in response:\n{response_text}\n")
      cot = response_text
      final_output = ""
    cot, final_output = cot.strip(), final_output.strip()

    # clean up cot start token
    if config.thinking_start_token and cot.startswith(config.thinking_start_token):
      cot = cot[len(config.thinking_start_token):].strip()
    elif config.thinking_start_token:
      cot = cot.split(config.thinking_start_token, 1)[-1].strip()
  
  else:
    num_thinking_tokens = 0
    cot = ""
    final_output = response_text.strip()

  return GenerationResult(
      text=final_output,
      num_thinking_tokens=num_thinking_tokens,
      cot=cot,
      cost_usd=0.0,
  )


async def local_model_request_with_cot_prefill(
    model: str,
    port: str,
    messages: List[Dict[str, str]],
    cot_prefill: str,
    config: Optional[LocalModelConfig] = None,
    debug_context: Optional[Dict[str, Any]] = None,
    **generation_config
) -> GenerationResult:
  """Async request to a local model with injected text inside the generated CoT."""
  if config is None:
    config = get_local_model_config(model, port)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  if "gpt-oss" in model.lower():
    raise NotImplementedError(
        "local_model_request_with_cot_prefill is currently implemented only for Qwen local models."
    )

  client = get_local_client(config.base_url)
  tokenizer = get_tokenizer(config.tokenizer_name)
  raw_prompt_text = build_local_raw_prompt_with_cot_prefill(
      model=model,
      messages=messages,
      cot_prefill=cot_prefill,
      config=config,
  )
  if debug_context is not None:
    debug_context = dict(debug_context)
    debug_context["cot_prefill_chars"] = len(cot_prefill)

  response = await client.completions.create(
      model=model,
      prompt=raw_prompt_text,
      logprobs=1,
      echo=False,
      temperature=generation_config.get("temperature", 0.6),
      top_p=generation_config.get("top_p", 0.95),
      max_tokens=generation_config.get("max_tokens", MAX_TOKS),
  )

  choice = response.choices[0]
  response_text = choice.text

  if config.enable_reasoning and config.thinking_end_token:
    tokens = choice.logprobs.tokens if choice.logprobs else []
    if config.thinking_end_token in tokens:
      num_thinking_tokens = tokens.index(config.thinking_end_token)
      cot, final_output = response_text.split(config.thinking_end_token, 1)
    elif config.thinking_end_token in response_text:
      cot, final_output = response_text.split(config.thinking_end_token, 1)
      num_thinking_tokens = len(tokenizer.encode(cot))
    else:
      num_thinking_tokens = len(tokens) if tokens else 0
      print(f"\nWARNING: No {config.thinking_end_token} token found in response:\n{response_text}\n")
      cot = response_text
      final_output = ""
    cot, final_output = cot.strip(), final_output.strip()
    if config.thinking_start_token and cot.startswith(config.thinking_start_token):
      cot = cot[len(config.thinking_start_token):].strip()
    elif config.thinking_start_token:
      cot = cot.split(config.thinking_start_token, 1)[-1].strip()
  else:
    num_thinking_tokens = 0
    cot = ""
    final_output = response_text.strip()

  effective_cot = cot
  prefill = cot_prefill.strip()
  if prefill:
    effective_cot = (prefill + "\n\n" + cot).strip() if cot else prefill

  return GenerationResult(
      text=final_output,
      num_thinking_tokens=num_thinking_tokens,
      cot=effective_cot,
      cost_usd=0.0,
  )


def _root_url(base_url_v1: str) -> str:
    return base_url_v1[:-3] if base_url_v1.endswith("/v1") else base_url_v1

async def vllm_count_tokens(
    base_url_v1: str,
    model: str,
    *,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    add_special_tokens: bool = False,
) -> int:
    """
    Calls vLLM Tokenizer API (/tokenize). Returns TokenizeResponse.count.
    - `messages`: apply chat preprocessing.
    - `prompt`: encodes the raw prompt.
    """
    url = f"{_root_url(base_url_v1)}/tokenize"
    payload = {"model": model, "add_special_tokens": add_special_tokens}
    if messages is not None:
        payload["messages"] = messages
    else:
        payload["prompt"] = prompt or ""

    async with httpx.AsyncClient(timeout=30.0) as h:
        r = await h.post(url, json=payload)
        r.raise_for_status()
        return int(r.json()["count"])
      
  
@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def local_mistral_model_request(
    model: str,
    port: str,
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    **generation_config
) -> GenerationResult:
  """Async request to local Mistral model via vLLM using streaming chat completions API."""
  if config is None:
    config = get_local_model_config(model, port)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  client = get_local_client(config.base_url)

  # Use streaming to capture reasoning_content separately
  stream = await client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=generation_config.get("temperature", 0.6),
      top_p=generation_config.get("top_p", 0.95),
      max_tokens=generation_config.get("max_tokens", MAX_TOKS),
      stream=True,
      stream_options={"include_usage": True},
  )

  reasoning_chunks = []
  content_chunks = []
  round_usage = 0
  num_completion_tokens = 0
  num_thinking_tokens = 0

  async for chunk in stream:
    # Extract usage from final chunk (which has empty choices)
    if hasattr(chunk, "usage") and chunk.usage is not None:
      usage = chunk.usage
      round_usage += 1
      num_completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    
    # Skip chunks with no choices (like the final usage-only chunk)
    if not chunk.choices or len(chunk.choices) == 0:
      continue
    
    delta = chunk.choices[0].delta
    
    # Capture reasoning_content (thinking tokens)
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
      reasoning_chunks.append(delta.reasoning_content)
    
    # Capture regular content
    if hasattr(delta, "content") and delta.content is not None:
      content_chunks.append(delta.content)

  assert round_usage == 1, f"Expected 1 usage chunk, got {round_usage}"

  cot = "".join(reasoning_chunks).strip()
  final_output = "".join(content_chunks).strip()

  # Estimate thinking tokens from content lengths
  if cot:
    num_thinking_tokens = await vllm_count_tokens(config.base_url, model, prompt=cot)

  return GenerationResult(
      text=final_output,
      num_thinking_tokens=num_thinking_tokens,
      cot=cot,
      cost_usd=0.0,
  )


def is_local_model(model_name: str) -> bool:
  return get_local_model_config(model_name) is not None


def is_gpt_model(model_name: str) -> bool:
  return model_name in GPT_COSTS or model_name.startswith("gpt-")


async def async_generate_single(
    model_name: str,
    port: str,
    messages: List[Dict[str, str]],
    generation_config: Dict[str, Any],
    local_model_config: Optional[LocalModelConfig] = None,
    debug_context: Optional[Dict[str, Any]] = None,
) -> GenerationResult:
  """Single async generation for one message/conversation.
  
  This is the core dispatch function that routes to model-specific APIs.
  No internal semaphore - caller is responsible for concurrency control.
  
  Args:
    model_name: Name of the model to use
    messages: Single conversation (list of message dicts)
    generation_config: Generation parameters (temperature, max_tokens, etc.)
    local_model_config: Config for local models (optional, auto-detected if None)
  
  Returns:
    GenerationResult with text, cot, num_thinking_tokens, cost_usd
  """
  # An unknown model, when a vLLM endpoint is available, is assumed to be served
  # there over /v1/chat/completions (registers it as a local chat model).
  model_registry.ensure_local_registered(model_name)

  if is_local_model(model_name):
    if "mistral" in model_name.lower():
      # Mistral models use streaming chat completions API
      return await local_mistral_model_request(
          model_name, port, messages, config=local_model_config, **generation_config
      )
    else:
      return await local_model_request(
          model_name,
          port,
          messages,
          config=local_model_config,
          debug_context=debug_context,
          **generation_config,
      )
  elif is_gpt_model(model_name):
    resp = await openai_chat_request(model_name, messages, **generation_config)
    raw_text = resp["choices"][0]["message"]["content"]
    text = raw_text if isinstance(raw_text, str) else ("" if raw_text is None else str(raw_text))
    usage = resp["usage"]
    pt = usage["prompt_tokens"]
    ct = usage["completion_tokens"]
    rt = usage.get("reasoning_tokens", 0)
    cost = _gpt_cost_usd(model_name, pt, ct)
    return GenerationResult(
        text=text,
        num_thinking_tokens=rt,
        cost_usd=cost,
    )
  elif model_name in CLAUDE_MODELS or model_name.startswith("claude"):
    resp = await claude_request(model_name, messages, **generation_config)
    text = resp["content"][0]["text"]
    return GenerationResult(text=text, cost_usd=0.0)
  elif model_name in GEMINI_COSTS.keys() or "gemini" in model_name.lower():
    return await gemini_request(model_name, messages, **generation_config)
  else:
    raise ValueError(f"Unknown model: {model_name}")


async def async_generate_single_with_cot_prefill(
    model_name: str,
    port: str,
    messages: List[Dict[str, str]],
    cot_prefill: str,
    generation_config: Dict[str, Any],
    local_model_config: Optional[LocalModelConfig] = None,
    debug_context: Optional[Dict[str, Any]] = None,
) -> GenerationResult:
  """Single async generation with an injected reasoning prefix inside CoT."""
  if not is_local_model(model_name):
    raise ValueError(
        f"CoT-prefill generation is only supported for local models, got {model_name}."
    )
  return await local_model_request_with_cot_prefill(
      model=model_name,
      port=port,
      messages=messages,
      cot_prefill=cot_prefill,
      config=local_model_config,
      debug_context=debug_context,
      **generation_config,
  )


async def async_batch_generate(
    model_name: str,
    port: str,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    max_concurrent: int = 64,
    local_model_config: Optional[LocalModelConfig] = None,
) -> List[GenerationResult]:
  """Unified async batch generation for all model types.
  
  Uses internal semaphore for concurrency control.
  For single-message generation without semaphore, use async_generate_single.
  """
  if not batch_messages:
    return []

  semaphore = asyncio.Semaphore(max_concurrent)

  async def generate_one(idx: int, messages: List[Dict[str, str]]) -> Tuple[int, GenerationResult]:
    async with semaphore:
      try:
        result = await async_generate_single(
            model_name, port, messages, generation_config, local_model_config
        )
        return idx, result
      except Exception as e:
        print(f"Error generating response for index {idx}: {e}")
        raise

  tasks = [generate_one(i, msg) for i, msg in enumerate(batch_messages)]
  results_with_idx = await asyncio.gather(*tasks, return_exceptions=True)

  processed_results = []
  for r in results_with_idx:
    if isinstance(r, Exception):
      raise r
    processed_results.append(r)

  processed_results.sort(key=lambda x: x[0])
  return [result for _, result in processed_results]


def model_call_wrapper(
    model_name: str,
    port: str,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    local_model_config: Optional[LocalModelConfig] = None,
    max_concurrent: int = 64,
) -> List[GenerationResult]:
  """Wrapper for calling various types of models."""
  return asyncio.run(async_batch_generate(
      model_name,
      port=port,
      batch_messages=batch_messages,
      generation_config=generation_config,
      max_concurrent=max_concurrent,
      local_model_config=local_model_config,
  ))


def cached_generate(
    batch_prompts: List[List[Dict[str, str]]],
    model_name: str,
    port: Optional[str] = None,
    cache: Optional[Dict[str, Any]] = None,
    cache_file: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    parallel_model_calls: bool = True,
    local_model_config: Optional[LocalModelConfig] = None,
) -> Tuple[List[str], List[int], List[str], List[float]]:
  """
  Backwards-compatible cached generate.

  gsm.py expects:
    batch_responses, think_token_num, all_cots, cost_usd = cached_generate(
        batch_prompts, model_name, model_url, cache=..., cache_file=...,
        generation_config=..., parallel_model_calls=...
    )
  """
  if generation_config is None:
    generation_config = {}

  max_concurrent = 128 if parallel_model_calls else 1
  if not model_name in LOCAL_MODEL_CONFIGS:
    max_concurrent = 64 if parallel_model_calls else 1
  if "gemini" in model_name.lower():
    max_concurrent = 8 if parallel_model_calls else 1

  def _result_to_cache_dict(r: GenerationResult) -> Dict[str, Any]:
    return {
        "text": r.text,
        "num_thinking_tokens": r.num_thinking_tokens,
        "cot": r.cot,
        "cost_usd": r.cost_usd,
    }

  def _cache_dict_to_outputs(v: Any) -> Tuple[str, int, str, float]:
    if isinstance(v, dict):
      return (
          str(v.get("text", "")),
          int(v.get("num_thinking_tokens", 0) or 0),
          str(v.get("cot", "")),
          float(v.get("cost_usd", 0.0) or 0.0),
      )
    if isinstance(v, tuple):
      text = str(v[0]) if len(v) > 0 else ""
      nt = int(v[1]) if len(v) > 1 else 0
      cot = str(v[2]) if len(v) > 2 else ""
      cost = float(v[3]) if len(v) > 3 else 0.0
      return text, nt, cot, cost
    return str(v), 0, "", 0.0

  if cache is None:
    results = model_call_wrapper(
        model_name,
        port=port,
        batch_messages=batch_prompts,
        generation_config=generation_config,
        local_model_config=local_model_config,
        max_concurrent=max_concurrent,
    )
    batch_responses = [r.text for r in results]
    all_num_thinking_tokens = [r.num_thinking_tokens for r in results]
    all_cots = [r.cot for r in results]
    cost_usd = [r.cost_usd for r in results]
    return batch_responses, all_num_thinking_tokens, all_cots, cost_usd

  new_batch_prompts = []
  new_prompt_indices = []
  for i, prompt in enumerate(batch_prompts):
    jp = jsonify_prompt(prompt)
    if jp not in cache:
      new_batch_prompts.append(prompt)
      new_prompt_indices.append(i)

  if new_batch_prompts:
    batch_results = model_call_wrapper(
        model_name,
        port=port,
        batch_messages=new_batch_prompts,
        generation_config=generation_config,
        local_model_config=local_model_config,
        max_concurrent=max_concurrent,
    )

    for prompt, result in zip(new_batch_prompts, batch_results):
      jp = jsonify_prompt(prompt)
      cache[jp] = _result_to_cache_dict(result)

      if cache_file:
        cache_entry = {
            "prompt": jp,
            "completion": result.text,
            "num_thinking_tokens": result.num_thinking_tokens,
            "cot": result.cot,
            "cost_usd": result.cost_usd,
        }
        with open(cache_file, "a") as f:
          f.write(json.dumps(cache_entry) + "\n")

  batch_responses: List[str] = []
  all_num_thinking_tokens: List[int] = []
  all_cots: List[str] = []
  cost_usd: List[float] = []

  for prompt in batch_prompts:
    jp = jsonify_prompt(prompt)
    text_output, num_tokens, cot, cost = _cache_dict_to_outputs(cache[jp])
    batch_responses.append(text_output)
    all_num_thinking_tokens.append(num_tokens)
    all_cots.append(cot)
    cost_usd.append(cost)

  return batch_responses, all_num_thinking_tokens, all_cots, cost_usd


async def cleanup_clients():
  """Close all async clients."""
  global _azure_openai_client, _anthropic_client, _gemini_client

  if _anthropic_client:
    await _anthropic_client.aclose()
    _anthropic_client = None

  if _azure_openai_client:
    await _azure_openai_client.close()
    _azure_openai_client = None

  for client in list(_local_clients.values()):
    await client.close()
  _local_clients.clear()

  if _gemini_client:
    await _gemini_client.aclose()
    _gemini_client = None
