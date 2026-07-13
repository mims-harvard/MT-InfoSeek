"""Shared `/v1/chat/completions` request + parsing helpers.

Both model_utils modules import these so that the gpt-oss transport, its harmony
parsing, and reasoning-token accounting are *identical* across datasets. They
previously diverged: the repo-root client used chat/completions and the
server-reported `usage.reasoning_tokens`, while gsme_q_mt used raw
`/v1/completions` plus a hand-rolled harmony split and a re-encoded CoT token
count (which drifts, since decode->encode is not the identity).

Qwen models deliberately stay on the raw `/v1/completions` path in each
model_utils; that is how the published Qwen numbers were produced.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

MAX_TOKS = 65536

_GPT_OSS_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")

def _coerce_chat_content_to_text(content: Any) -> str:
  """Best-effort conversion from chat content payload to plain text."""
  if content is None:
    return ""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: List[str] = []
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
  return str(content)


def _message_like_get(obj: Any, key: str, default: Any = None) -> Any:
  """Get a field from either dict-like or object-like SDK payloads."""
  if obj is None:
    return default
  if isinstance(obj, dict):
    return obj.get(key, default)
  return getattr(obj, key, default)


def _message_like_dump(obj: Any) -> Any:
  """Best-effort conversion of SDK objects into dict/list payloads."""
  if obj is None or isinstance(obj, (str, int, float, bool, list, dict)):
    return obj

  for method_name in ("model_dump", "to_dict", "dict"):
    method = getattr(obj, method_name, None)
    if callable(method):
      try:
        return method(mode="python")
      except TypeError:
        try:
          return method()
        except Exception:
          pass
      except Exception:
        pass

  raw_dict = getattr(obj, "__dict__", None)
  if isinstance(raw_dict, dict):
    return {
        key: value
        for key, value in raw_dict.items()
        if not key.startswith("_")
    }

  return None


def _clean_gpt_oss_payload(text: str) -> str:
  if not text:
    return ""
  s = _GPT_OSS_CONTROL_TOKEN_RE.sub(" ", text)
  s = s.replace("\\n", "\n")
  s = re.sub(r"[ \t]+", " ", s)
  s = re.sub(r"\n{3,}", "\n\n", s)
  return s.strip()


def _extract_gpt_oss_text_from_tool_calls(tool_calls: Any) -> str:
  """Extract text payload from GPT-OSS tool calls, if any."""
  if not tool_calls:
    return ""
  candidates: List[str] = []
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
    try:
      parsed = json.loads(arg_text)
      if isinstance(parsed, dict):
        for key in ("response", "content", "message", "text"):
          value = parsed.get(key)
          if isinstance(value, str) and value.strip():
            candidates.append(value)
    except Exception:
      pass

  if not candidates:
    return ""
  return _clean_gpt_oss_payload("\n".join(candidates))


def _combine_gpt_oss_cot(*parts: str) -> str:
  deduped: List[str] = []
  for part in parts:
    clean = _clean_gpt_oss_payload(part)
    if not clean:
      continue
    if any(clean == existing or clean in existing for existing in deduped):
      continue
    deduped = [existing for existing in deduped if existing not in clean]
    deduped.append(clean)
  return "\n\n".join(deduped).strip()


_GPT_OSS_VISIBLE_ACTION_RE = re.compile(
    r"(?im)^\s*(Question:\s*.+?|Answer:\s*.+?)\s*$"
)


def _split_gpt_oss_visible_action(text: str) -> Tuple[str, str]:
  """Split text into visible Question/Answer output plus residual CoT."""
  clean_text = _clean_gpt_oss_payload(text)
  if not clean_text:
    return "", ""

  matches = list(_GPT_OSS_VISIBLE_ACTION_RE.finditer(clean_text))
  if matches:
    match = matches[-1]
    visible = match.group(1).strip()
    prefix = clean_text[:match.start()].strip()
    suffix = clean_text[match.end():].strip()
    cot_parts = [part for part in (prefix, suffix) if part]
    cot = "\n\n".join(cot_parts).strip()
    return visible, cot

  return "", clean_text


def _extract_gpt_oss_from_message_like(message: Any) -> Tuple[str, str]:
  """Extract visible output and CoT from GPT-OSS Chat Completions payloads."""
  message_dump = _message_like_dump(message)

  raw_content = _message_like_get(message, "content")
  raw_tool_calls = _message_like_get(message, "tool_calls")
  raw_reasoning = _message_like_get(message, "reasoning")
  if raw_reasoning in (None, "", []):
    raw_reasoning = _message_like_get(message, "reasoning_content")

  if isinstance(message_dump, dict):
    if raw_content in (None, "", []):
      raw_content = message_dump.get("content", raw_content)
    if raw_tool_calls in (None, "", []):
      raw_tool_calls = message_dump.get("tool_calls", raw_tool_calls)
    if raw_reasoning in (None, "", []):
      raw_reasoning = message_dump.get("reasoning", raw_reasoning)
      if raw_reasoning in (None, "", []):
        raw_reasoning = message_dump.get("reasoning_content", raw_reasoning)

  reasoning_text = _coerce_chat_content_to_text(raw_reasoning)
  content_text = _coerce_chat_content_to_text(raw_content)
  final_output, cot_from_content = _split_gpt_oss_visible_action(content_text)

  if not final_output and raw_tool_calls:
    tool_text = _extract_gpt_oss_text_from_tool_calls(raw_tool_calls)
    tool_output, _ = _split_gpt_oss_visible_action(tool_text)
    final_output = tool_output

  cot = _combine_gpt_oss_cot(reasoning_text, cot_from_content)
  return final_output.strip(), cot.strip()



def _usage_get_int(obj: Any, *keys: str) -> int:
  """Get an integer field from either dict-like or object-like usage payload."""
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
  """Extract reasoning token count from provider-specific usage fields."""
  if usage is None:
    return 0

  # Direct field on usage payload.
  direct = _usage_get_int(usage, "reasoning_tokens", "reasoningTokenCount")
  if direct > 0:
    return direct

  # Nested details fields used by different backends/APIs.
  details_candidates: List[Any] = []
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


def _build_local_gpt_oss_request_kwargs(
    model: str,
    messages: List[Dict[str, str]],
    generation_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
  request_kwargs: Dict[str, Any] = {
      "model": model,
      "messages": messages,
      "temperature": generation_config.get("temperature", 0.6),
      "top_p": generation_config.get("top_p", 0.95),
      "max_tokens": generation_config.get("max_tokens", MAX_TOKS),
  }
  # Optional: only gpt-oss style servers accept it. Custom models omit it.
  reasoning_effort = generation_config.get("reasoning_effort")
  if reasoning_effort:
    request_kwargs["reasoning_effort"] = reasoning_effort
  return request_kwargs, reasoning_effort


async def _create_local_gpt_oss_chat_completion(
    client: AsyncOpenAI,
    request_kwargs: Dict[str, Any],
    reasoning_effort: str,
):
  """Issue a Chat Completions request, retrying with extra_body when needed."""
  try:
    return await client.chat.completions.create(**request_kwargs)
  except TypeError:
    retry_kwargs = dict(request_kwargs)
    retry_kwargs.pop("reasoning_effort", None)
    extra_body = dict(retry_kwargs.get("extra_body", {}))
    extra_body["reasoning_effort"] = reasoning_effort
    retry_kwargs["extra_body"] = extra_body
    return await client.chat.completions.create(**retry_kwargs)



# ---------------------------------------------------------------------------
# Generic OpenAI-compatible chat parsing (custom models: vLLM, SGLang, Ollama,
# OpenRouter, ...). Unlike gpt-oss there is no harmony envelope: the server
# returns clean `content` and, when it exposes reasoning, `reasoning_content`
# (vLLM/SGLang) or `reasoning` (OpenRouter).
# ---------------------------------------------------------------------------
def parse_generic_chat_message(message: Any) -> Tuple[str, str]:
  """Return ``(text, cot)`` from a chat-completions message."""
  raw_content = _message_like_get(message, "content")
  raw_reasoning = _message_like_get(message, "reasoning_content")
  if raw_reasoning in (None, "", []):
    raw_reasoning = _message_like_get(message, "reasoning")

  dump = _message_like_dump(message)
  if isinstance(dump, dict):
    if raw_content in (None, "", []):
      raw_content = dump.get("content", raw_content)
    if raw_reasoning in (None, "", []):
      raw_reasoning = dump.get("reasoning_content", dump.get("reasoning"))

  text = _coerce_chat_content_to_text(raw_content).strip()
  cot = _coerce_chat_content_to_text(raw_reasoning).strip()
  return text, cot


# Public aliases (the leading underscore is historical).
coerce_chat_content_to_text = _coerce_chat_content_to_text
message_like_get = _message_like_get
message_like_dump = _message_like_dump
extract_gpt_oss_from_message_like = _extract_gpt_oss_from_message_like
extract_reasoning_tokens_from_usage = _extract_reasoning_tokens_from_usage
usage_get_int = _usage_get_int
build_chat_request_kwargs = _build_local_gpt_oss_request_kwargs
create_chat_completion = _create_local_gpt_oss_chat_completion
