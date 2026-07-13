"""Utility functions for calling models."""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import dataclasses
from dataclasses import dataclass
import time

from openai import AsyncOpenAI, AsyncAzureOpenAI
import httpx

import chat_backends
import model_registry
from model_registry import BACKEND_OPENAI_CHAT, get_backend
import requests
import tenacity
from tenacity import retry
try:
  import transformers
except ImportError:
  transformers = None
import threading

try:
  from google import genai as google_genai
  from google.genai import types as google_genai_types
except ImportError:
  google_genai = None
  google_genai_types = None

try:
  import google.auth as google_auth
  from google.auth.transport.requests import AuthorizedSession, Request
except ImportError:
  google_auth = None
  AuthorizedSession = None
  Request = None

wait_random_exponential = tenacity.wait_random_exponential
stop_after_attempt = tenacity.stop_after_attempt

MAX_TOKS = 65536

_LOOP_THREAD: Optional[threading.Thread] = None
_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LOOP_READY: Optional[threading.Event] = None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
  global _LOOP_THREAD, _LOOP, _LOOP_READY

  if _LOOP is not None and not _LOOP.is_closed():
    return _LOOP

  _LOOP_READY = threading.Event()

  def _runner():
    global _LOOP
    _LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_LOOP)
    _LOOP_READY.set()
    _LOOP.run_forever()

  _LOOP_THREAD = threading.Thread(target=_runner, daemon=True)
  _LOOP_THREAD.start()
  _LOOP_READY.wait()
  return _LOOP

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
  base_url: Optional[str] = None
  enable_reasoning: bool = True
  thinking_start_token: str = ""
  thinking_end_token: str = ""


# Local models hosted via vLLM. Definitions live in the repo-root
# model_registry.py, shared with the root model_utils.py.
LOCAL_MODEL_CONFIGS: Dict[str, LocalModelConfig] = model_registry.build_local_configs(
    LocalModelConfig
)
_LOCAL_CONFIGS_REVISION = model_registry.revision()


def _sync_local_configs() -> None:
  """Rebuild the dataclass view when the shared registry has changed."""
  global _LOCAL_CONFIGS_REVISION
  if _LOCAL_CONFIGS_REVISION == model_registry.revision():
    return
  LOCAL_MODEL_CONFIGS.clear()
  LOCAL_MODEL_CONFIGS.update(model_registry.build_local_configs(LocalModelConfig))
  _LOCAL_CONFIGS_REVISION = model_registry.revision()

legacy_genai = None
# Fail only when a Gemini model requires the legacy SDK. Non-Gemini runs do not
# require google-generativeai, even if GOOGLE_API_KEY is exported as an empty
# value by .env.example.
if os.environ.get("GOOGLE_API_KEY"):
  try:
    import google.generativeai as legacy_genai
    legacy_genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
  except ImportError:
    legacy_genai = None


def _env_flag(name: str, default: bool = False) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _use_vertex_ai_for_gemini() -> bool:
  return _env_flag("GOOGLE_GENAI_USE_VERTEXAI", False)


def _env_int(name: str, default: int) -> int:
  value = os.environ.get(name)
  if value is None:
    return default
  try:
    return int(value)
  except ValueError:
    return default


def _is_gemini_model(model_name: str) -> bool:
  return model_name in GEMINI_COSTS or "gemini" in (model_name or "").lower()


def _env_str(name: str, default: str = "") -> str:
  return os.environ.get(name, default).strip()


def _gemini_max_output_tokens(generation_config: Dict[str, Any]) -> int:
  env_value = _env_int("QUESTBENCH_GEMINI_MAX_OUTPUT_TOKENS", 0)
  if env_value > 0:
    return env_value
  return generation_config.get("max_tokens", MAX_TOKS)


def _apply_gemini_thinking_config(model: str, generation_config: Dict[str, Any]) -> None:
  thinking_level = _env_str("QUESTBENCH_GEMINI_THINKING_LEVEL")
  thinking_budget = _env_int("QUESTBENCH_GEMINI_THINKING_BUDGET", -999999)

  if thinking_level:
    generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
  elif thinking_budget != -999999:
    generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

# Remote GPT models via Azure OpenAI
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

# Claude models
CLAUDE_MODELS = model_registry.CLAUDE_MODELS

_azure_openai_client: Optional[AsyncAzureOpenAI] = None
_anthropic_client: Optional[httpx.AsyncClient] = None
# _local_client: Optional[AsyncOpenAI] = None
_local_clients: Dict[Tuple[str, str], AsyncOpenAI] = {}

# _tokenizer: Any = None
_tokenizer_cache: Dict[str, Any] = {}


def get_tokenizer(tokenizer_name: str):
  global _tokenizer_cache
  if transformers is None:
    raise ImportError("transformers is required for local tokenizer loading. Install release requirements first.")
  if tokenizer_name in _tokenizer_cache:
    return _tokenizer_cache[tokenizer_name]

  token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
  kwargs = {"trust_remote_code": True}

  try:
    if token:
      try:
        tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name, token=token, **kwargs)
      except TypeError:
        tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=token, **kwargs)
    else:
      tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
  except Exception:
    tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True, **kwargs)

  _tokenizer_cache[tokenizer_name] = tok
  return tok


def resolve_local_model_key(model_name: str) -> Optional[str]:
  """Registry key for `model_name`: exact match, else longest substring match.

  The substring fallback is historical (callers sometimes pass a suffixed name).
  Matching longest-first is required now that the registry contains both
  `...-Thinking-2507` and `...-Thinking-2507-FP8`, where the former is a
  substring of the latter and dict order would otherwise decide the winner.
  """
  if not model_name:
    return None
  _sync_local_configs()
  if model_name in LOCAL_MODEL_CONFIGS:
    return model_name
  mn = model_name.lower()
  candidates = [k for k in LOCAL_MODEL_CONFIGS if k.lower() in mn]
  if not candidates:
    return None
  return max(candidates, key=len)


def get_local_model_config(model_name: str, port: str = None) -> Optional[LocalModelConfig]:
  key = resolve_local_model_key(model_name)
  if key is None:
    return None
  config = LOCAL_MODEL_CONFIGS[key]
  pinned = (model_registry.LOCAL_MODEL_CONFIGS.get(key) or {}).get("base_url")
  if port and not pinned:
    return dataclasses.replace(config, base_url=f"http://127.0.0.1:{port}/v1")
  return config


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
  """Register a new local model, writing through to the shared registry."""
  model_registry.register_local_model(
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
  the standard OpenAI API when no Azure key is set; with `OPENAI_BASE_URL` this
  also covers OpenAI-compatible gateways (OpenRouter, Ollama, SGLang).
  """
  global _azure_openai_client
  if _azure_openai_client is None:
    if os.environ.get("AZURE_OPENAI_API_KEY"):
      _azure_openai_client = AsyncAzureOpenAI(
          api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
          azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
          api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
      )
    elif os.environ.get("OPENAI_API_KEY"):
      _azure_openai_client = AsyncOpenAI(
          api_key=os.environ["OPENAI_API_KEY"],
          base_url=os.environ.get("OPENAI_BASE_URL") or None,
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
        timeout=120.0,
    )
  return _anthropic_client

def get_local_client(base_url: str = "http://127.0.0.1:8011/v1", api_key: str = "EMPTY") -> AsyncOpenAI:
  global _local_clients
  key = (base_url, api_key)
  if key not in _local_clients:
    _local_clients[key] = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )
  return _local_clients[key]


def _resolve_local_base_url(config: LocalModelConfig, generation_config: Dict[str, Any]) -> str:
  # One variable, one default, suite-wide: VLLM_PORT (8011); VLLM_BASE_URL overrides.
  return (
      config.base_url
      or generation_config.get("base_url")
      or os.environ.get("VLLM_BASE_URL")
      or f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '8011')}/v1"
  )


async def local_openai_chat_request(
    model: str,
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    **generation_config
) -> GenerationResult:
  """Chat-completions request for gpt-oss and custom OpenAI-compatible models.

  Mirrors the repo-root `model_utils.local_gpt_oss_chat_request` exactly (same
  shared parsing + reasoning-token accounting from chat_backends), so GSME-Q-MT
  and the other datasets report identical thinking-token counts.
  """
  if config is None:
    config = get_local_model_config(model)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  key = resolve_local_model_key(model) or model
  api_key_env = model_registry.api_key_env_for(key)
  api_key = os.environ.get(api_key_env, "EMPTY") if api_key_env else "EMPTY"
  client = get_local_client(_resolve_local_base_url(config, generation_config), api_key=api_key)

  gen_cfg = {k: v for k, v in generation_config.items() if k != "base_url"}
  request_kwargs, reasoning_effort = chat_backends.build_chat_request_kwargs(
      model, messages, gen_cfg
  )
  response = await chat_backends.create_chat_completion(client, request_kwargs, reasoning_effort)

  message = response.choices[0].message
  if "gpt-oss" in model.lower():
    text, cot = chat_backends.extract_gpt_oss_from_message_like(message)
  else:
    text, cot = chat_backends.parse_generic_chat_message(message)

  num_thinking_tokens = chat_backends.extract_reasoning_tokens_from_usage(
      getattr(response, "usage", None)
  )
  if num_thinking_tokens == 0 and cot:
    try:
      num_thinking_tokens = len(get_tokenizer(config.tokenizer_name).encode(cot))
    except Exception:
      num_thinking_tokens = 0

  return GenerationResult(
      text=text.strip(),
      num_thinking_tokens=num_thinking_tokens,
      cot=cot.strip(),
      cost_usd=0.0,
  )


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


def _gpt_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
  if model not in GPT_COSTS:
    return 0.0
  p = GPT_COSTS[model]["prompt_tokens"]
  c = GPT_COSTS[model]["completion_tokens"]
  return float(prompt_tokens) * p + float(completion_tokens) * c

import re
_FINAL_MARK_RE = re.compile(r"<\|channel\|>final<\|message\|>")

def split_gpt_oss(response_text: str) -> tuple[str, str]:
  if not response_text:
    return "", ""
  m = _FINAL_MARK_RE.search(response_text)
  if not m:
    return response_text.strip(), ""  # if no final, treat all as cot
  cot = response_text[:m.start()].strip()
  final_output = response_text[m.end():].strip()
  if "<|return|>" in final_output:
    final_output = final_output.split("<|return|>", 1)[0].strip()
  return cot, final_output


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
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
  Returns: (prompt_tokens, candidates_tokens, thoughts_tokens, total_tokens)
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

  if tt == 0 and tot > 0:
    est = tot - pt - ct
    if est > 0:
      tt = est

  return pt, ct, tt, tot


def _to_gemini_messages(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
  gemini_messages = []
  system_instruction = None

  for msg in messages:
    if msg["role"] == "system":
      system_instruction = msg["content"]
    else:
      role = "user" if msg["role"] == "user" else "model"
      gemini_messages.append({
          "role": role,
          "parts": [{"text": msg["content"]}]
      })

  combined_messages = []
  for msg in gemini_messages:
    if combined_messages and combined_messages[-1]["role"] == msg["role"]:
      combined_messages[-1]["parts"].extend(msg["parts"])
    else:
      combined_messages.append(msg)

  if not combined_messages and system_instruction:
    combined_messages = [{"role": "user", "parts": [{"text": "Please start."}]}]
  elif not combined_messages:
    raise ValueError("No messages found for Gemini request.")

  return combined_messages, system_instruction


def _extract_gemini_text_from_dict(resp: Dict[str, Any]) -> str:
  candidates = resp.get("candidates") or []
  text_parts = []
  for candidate in candidates:
    content = candidate.get("content") or {}
    for part in content.get("parts") or []:
      if "text" in part:
        text_parts.append(str(part["text"]))
  if text_parts:
    return "\n".join(text_parts)
  return json.dumps(resp)


def _vertex_rest_endpoint(project: str, location: str, model: str) -> str:
  host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
  model_path = f"projects/{project}/locations/{location}/publishers/google/models/{model}"
  return f"https://{host}/v1/{model_path}:generateContent"


def _vertex_gemini_request_rest(
    model: str,
    combined_messages: List[Dict[str, Any]],
    system_instruction: Optional[str],
    **generation_config
) -> GenerationResult:
  if google_auth is None or AuthorizedSession is None:
    raise RuntimeError(
        "google-auth is not installed, but Vertex AI REST transport was requested."
    )

  project = _env_str("GOOGLE_CLOUD_PROJECT")
  location = _env_str("GOOGLE_CLOUD_LOCATION")
  if not project or not location:
    raise RuntimeError(
        "GOOGLE_GENAI_USE_VERTEXAI=True requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION."
    )

  trust_env = _env_flag("QUESTBENCH_GOOGLE_TRUST_ENV", False)
  credentials, _ = google_auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
  auth_session = requests.Session()
  auth_session.trust_env = trust_env
  session = AuthorizedSession(credentials, auth_request=Request(auth_session))
  session.trust_env = trust_env

  endpoint = _vertex_rest_endpoint(project, location, model)
  gen_config = {
      "temperature": generation_config.get("temperature", 0.0),
      "maxOutputTokens": _gemini_max_output_tokens(generation_config),
  }
  if "top_p" in generation_config:
    gen_config["topP"] = generation_config["top_p"]
  _apply_gemini_thinking_config(model, gen_config)

  body = {
      "contents": combined_messages,
      "generationConfig": gen_config,
  }
  if system_instruction:
    body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

  timeout = _env_int("QUESTBENCH_GEMINI_TIMEOUT", 120)
  resp = session.post(endpoint, json=body, timeout=timeout)
  try:
    resp.raise_for_status()
  except Exception as e:
    raise RuntimeError(f"Vertex REST request failed: HTTP {resp.status_code}: {resp.text}") from e

  data = resp.json()
  text = _extract_gemini_text_from_dict(data)
  pt, ct, tt, tot = _extract_gemini_usage(data)
  cost = _gemini_cost_usd(model, pt, ct + tt)

  return GenerationResult(
      text=text,
      num_thinking_tokens=tt,
      cot="",
      cost_usd=cost,
  )


def _gemini_request_once_sync(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> GenerationResult:
  combined_messages, system_instruction = _to_gemini_messages(messages)
  temperature = generation_config.get("temperature", 0.0)
  max_output_tokens = _gemini_max_output_tokens(generation_config)

  if _use_vertex_ai_for_gemini():
    transport = _env_str("QUESTBENCH_GEMINI_TRANSPORT", "rest").lower()
    if transport == "rest":
      return _vertex_gemini_request_rest(
          model,
          combined_messages,
          system_instruction,
          **generation_config,
      )

    if google_genai is None:
      raise RuntimeError(
          "google-genai is not installed, but GOOGLE_GENAI_USE_VERTEXAI=True was requested."
      )

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not project or not location:
      raise RuntimeError(
          "GOOGLE_GENAI_USE_VERTEXAI=True requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION."
      )

    client = google_genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if "top_p" in generation_config:
      config_kwargs["top_p"] = generation_config["top_p"]
    thinking_level = _env_str("QUESTBENCH_GEMINI_THINKING_LEVEL")
    thinking_budget = _env_int("QUESTBENCH_GEMINI_THINKING_BUDGET", -999999)
    if google_genai_types is not None and thinking_level:
      config_kwargs["thinking_config"] = google_genai_types.ThinkingConfig(
          thinking_level=thinking_level.upper()
      )
    elif google_genai_types is not None and thinking_budget != -999999:
      config_kwargs["thinking_config"] = google_genai_types.ThinkingConfig(
          thinking_budget=thinking_budget
      )
    if system_instruction:
      config_kwargs["system_instruction"] = system_instruction

    if google_genai_types is not None:
      gen_config = google_genai_types.GenerateContentConfig(**config_kwargs)
    else:
      gen_config = config_kwargs

    resp = client.models.generate_content(
        model=model,
        contents=combined_messages,
        config=gen_config,
    )
  else:
    if legacy_genai is None:
      raise RuntimeError(
          "GOOGLE_API_KEY is not set, but Gemini model was requested. "
          "Set GOOGLE_API_KEY or enable Vertex AI with GOOGLE_GENAI_USE_VERTEXAI=True."
      )

    model_kwargs = {}
    if system_instruction:
      model_kwargs["system_instruction"] = system_instruction

    gen_model = legacy_genai.GenerativeModel(model, **model_kwargs)
    gen_config = legacy_genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    if len(combined_messages) > 1:
      chat = gen_model.start_chat(history=combined_messages[:-1])
      resp = chat.send_message(
          combined_messages[-1]["parts"][0]["text"],
          generation_config=gen_config,
      )
    else:
      resp = gen_model.generate_content(
          combined_messages[0]["parts"][0]["text"],
          generation_config=gen_config,
      )

  text = getattr(resp, "text", None)
  if text is None:
    text = str(resp)

  pt, ct, tt, tot = _extract_gemini_usage(resp)
  cost = _gemini_cost_usd(model, pt, ct + tt)

  return GenerationResult(
      text=text,
      num_thinking_tokens=tt,
      cot="",
      cost_usd=cost,
  )


def gemini_request_sync(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> GenerationResult:
  attempts = _env_int("QUESTBENCH_GEMINI_RETRIES", 5)
  attempts = max(1, attempts)
  last_err = None

  for attempt in range(1, attempts + 1):
    try:
      return _gemini_request_once_sync(model, messages, **generation_config)
    except Exception as e:
      if isinstance(e, (ValueError, RuntimeError)):
        raise
      last_err = e
      if attempt == attempts:
        break
      wait_s = min(30, 2 ** (attempt - 1))
      print(f"Gemini request failed on attempt {attempt}/{attempts}: {e}; retrying in {wait_s}s")
      time.sleep(wait_s)

  raise last_err


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    reraise=True,
)
async def gemini_request(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> GenerationResult:
  """Async request to Gemini via Google AI Studio or Vertex AI."""
  loop = asyncio.get_running_loop()
  return await loop.run_in_executor(
      None,
      lambda: gemini_request_sync(model, messages, **generation_config),
  )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def local_model_request(
    model: str,
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    **generation_config
) -> GenerationResult:
  """Async request to local model via vLLM.

  Transport comes from the shared registry: Qwen stays on raw `/v1/completions`
  (as the published Qwen numbers were produced); gpt-oss and custom models use
  `/v1/chat/completions`.
  """
  if config is None:
    config = get_local_model_config(model)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  registry_key = resolve_local_model_key(model) or model
  if get_backend(registry_key) == BACKEND_OPENAI_CHAT:
    return await local_openai_chat_request(
        model, messages, config=config, **generation_config
    )

  base_url = _resolve_local_base_url(config, generation_config)
  client = get_local_client(base_url)
  tokenizer = get_tokenizer(config.tokenizer_name)

  apply_kwargs = {
      "conversation": messages,
      "add_generation_prompt": True,
      "tokenize": False,
  }

  if config.enable_reasoning:
    if "gpt-oss" in model.lower():
      apply_kwargs["reasoning_effort"] = "medium"
    elif "qwen" in model.lower():
      apply_kwargs["enable_reasoning"] = True
    apply_kwargs["add_special_tokens"] = True

  raw_prompt_text = tokenizer.apply_chat_template(**apply_kwargs)

  request_kwargs = {
      "model": model,
      "prompt": raw_prompt_text,
      "logprobs": 2,
      "echo": False,
      "temperature": generation_config.get("temperature", 0.6),
      "top_p": generation_config.get("top_p", 0.95),
      "max_tokens": generation_config.get("max_tokens", MAX_TOKS),
  }
  response = await client.completions.create(
      **request_kwargs
  )

  choice = response.choices[0]
  response_text = choice.text

  if config.enable_reasoning and config.thinking_end_token:
    tokens = choice.logprobs.tokens if choice.logprobs else []
    if "gpt-oss" not in model.lower():
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
        cot = ""
        final_output = response_text
      cot, final_output = cot.strip(), final_output.strip()
      # clean up cot start token
      if config.thinking_start_token and cot.startswith(config.thinking_start_token):
        cot = cot[len(config.thinking_start_token):].strip()
      elif config.thinking_start_token:
        cot = cot.split(config.thinking_start_token, 1)[-1].strip()
      if config.thinking_start_token and not cot:
        if final_output.startswith(config.thinking_start_token):
          final_output = final_output[len(config.thinking_start_token):].strip()
        elif config.thinking_start_token in final_output:
          final_output = final_output.split(config.thinking_start_token, 1)[-1].strip()
      if config.thinking_end_token and final_output.startswith(config.thinking_end_token):
        final_output = final_output[len(config.thinking_end_token):].strip()
        
    else:
      # for gpt-oss models, split by final assistant message
      final_start_tokens = "<|start|>assistant<|channel|>final<|message|>"
      final_end_token = "<|return|>"
      response_text = "".join(tokens) if tokens else response_text
      
      cot, final_output = split_gpt_oss(response_text)
      
      if config.thinking_start_token:
        if cot.startswith(config.thinking_start_token):
          cot = cot[len(config.thinking_start_token):].strip()
        elif config.thinking_start_token in cot:
          cot = cot.split(config.thinking_start_token, 1)[-1].strip()
      
      if config.thinking_end_token and cot.endswith(config.thinking_end_token):
        cot = cot[:-len(config.thinking_end_token)].strip()
      
      if final_output.endswith(final_end_token):
        final_output = final_output[:-len(final_end_token)].strip()
      
      num_thinking_tokens = len(tokenizer.encode(cot)) if cot else 0
  
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
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    **generation_config
) -> GenerationResult:
  """Async request to local Mistral model via vLLM using streaming chat completions API."""
  if config is None:
    config = get_local_model_config(model)
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
  round_usage = 0  # DEBUG
  num_completion_tokens = 0
  num_thinking_tokens = 0

  async for chunk in stream:
    print(f"Chunk: choices={len(chunk.choices) if chunk.choices else 0}, usage={chunk.usage is not None}")
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


async def async_batch_generate(
    model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    max_concurrent: int = 64,
    local_model_config: Optional[LocalModelConfig] = None,
) -> List[GenerationResult]:
  """Unified async batch generation for all model types."""
  if not batch_messages:
    return []

  semaphore = asyncio.Semaphore(max_concurrent)

  # An unknown model, when a vLLM endpoint is available, is assumed to be served
  # there over /v1/chat/completions (registers it as a local chat model).
  model_registry.ensure_local_registered(model_name)

  async def generate_one(idx: int, messages: List[Dict[str, str]]) -> Tuple[int, GenerationResult]:
    async with semaphore:
      try:
        if is_local_model(model_name):
          if "mistral" in model_name.lower():  # NOTE: Mistral models have issues with tokenizers, switch to /chat/completions API instead
            result = await local_mistral_model_request(
                model_name, messages, config=local_model_config, **generation_config
            )
          else:
            result = await local_model_request(
                model_name, messages, config=local_model_config, **generation_config
            )
          return idx, result
        elif is_gpt_model(model_name):
          resp = await openai_chat_request(model_name, messages, **generation_config)
          text = resp["choices"][0]["message"]["content"]
          usage = resp["usage"]
          pt = usage["prompt_tokens"]
          ct = usage["completion_tokens"]
          rt = usage["reasoning_tokens"]
          cost = _gpt_cost_usd(model_name, pt, ct)
          return idx, GenerationResult(
              text=text,
              num_thinking_tokens=rt,
              cost_usd=cost,
          )
        elif model_name in CLAUDE_MODELS or model_name.startswith("claude"):
          resp = await claude_request(model_name, messages, **generation_config)
          text = resp["content"][0]["text"]
          # FIXME
          return idx, GenerationResult(text=text, cost_usd=0.0)
        elif _is_gemini_model(model_name):
          result = await gemini_request(model_name, messages, **generation_config)
          return idx, result
        else:
          raise ValueError(f"Unknown model: {model_name}")
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
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    local_model_config: Optional[LocalModelConfig] = None,
    max_concurrent: int = 64,
) -> List[GenerationResult]:
  if _is_gemini_model(model_name) and max_concurrent <= 1:
    return [
        gemini_request_sync(model_name, messages, **generation_config)
        for messages in batch_messages
    ]

  loop = _ensure_background_loop()
  fut = asyncio.run_coroutine_threadsafe(
      async_batch_generate(
          model_name,
          batch_messages=batch_messages,
          generation_config=generation_config,
          max_concurrent=max_concurrent,
          local_model_config=local_model_config,
      ),
      loop,
  )
  return fut.result()


def cached_generate(
    batch_prompts: List[List[Dict[str, str]]],
    model_name: str,
    model_url: Optional[str] = None,
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
  generation_config = dict(generation_config)
  if is_local_model(model_name) and model_url and isinstance(model_url, str):
    normalized = model_url.strip()
    if normalized.startswith("http://") or normalized.startswith("https://"):
      if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
      if normalized.endswith("/completions"):
        normalized = normalized[: -len("/completions")]
      if not normalized.endswith("/v1"):
        normalized = normalized.rstrip("/") + "/v1"
      generation_config.setdefault("base_url", normalized)

  max_concurrent = 64 if parallel_model_calls else 1
  # if not model_name in LOCAL_MODEL_CONFIGS:
  #   max_concurrent = 8 if parallel_model_calls else 1
  
  if not is_local_model(model_name):
    max_concurrent = 8 if parallel_model_calls else 1
  if _is_gemini_model(model_name):
    # Vertex AI's google-genai sync client is prone to RemoteProtocolError under
    # this evaluator's threaded batch fanout. Keep Gemini serial by default and
    # allow opting back into limited concurrency once the environment is stable.
    max_concurrent = _env_int("QUESTBENCH_GEMINI_MAX_CONCURRENT", 1) if parallel_model_calls else 1
    max_concurrent = max(1, max_concurrent)

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
  global _azure_openai_client, _anthropic_client, _local_client
  
  global _LOOP, _LOOP_THREAD
  if _LOOP is not None and not _LOOP.is_closed():
    try:
      _LOOP.call_soon_threadsafe(_LOOP.stop)
    except Exception:
      pass
  _LOOP = None
  _LOOP_THREAD = None

  if _anthropic_client:
    await _anthropic_client.aclose()
    _anthropic_client = None

  if _azure_openai_client:
    await _azure_openai_client.close()
    _azure_openai_client = None

  if _local_clients:
    for c in list(_local_clients.values()):
      try:
        await c.close()
      except Exception:
        pass
    _local_clients = {}
