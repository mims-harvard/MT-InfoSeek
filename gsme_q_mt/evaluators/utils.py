from typing import Any, Dict, List, Optional, Set
import re
import ast
import json
import pandas as pd


# -----------------------------
# Small utilities (safe parsing)
# -----------------------------

def _safe_dict_field(x: Any) -> Dict[str, Any]:
  if x is None:
    return {}
  if isinstance(x, dict):
    return x
  if isinstance(x, str):
    s = x.strip()
    if not s:
      return {}
    try:
      obj = json.loads(s)
      return obj if isinstance(obj, dict) else {}
    except Exception:
      try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
      except Exception:
        return {}
  return {}


def _safe_list_field(x: Any) -> List[Any]:
  if x is None:
    return []
  if isinstance(x, list):
    return x
  if isinstance(x, (set, tuple)):
    return list(x)
  if isinstance(x, (dict,)):
    return list(x.keys())
  if not isinstance(x, str):
    return []
  s = x.strip()
  if not s:
    return []
  try:
    return json.loads(s)
  except Exception:
    try:
      return ast.literal_eval(s)
    except Exception:
      return []

def _as_cost_usd_list(costs_usd: Any, n: int) -> List[float]:
  if n <= 0:
    return []
  if costs_usd is None:
    return [0.0] * n
  if isinstance(costs_usd, list):
    out: List[float] = []
    for i in range(n):
      try:
        v = costs_usd[i]
      except Exception:
        v = None
      try:
        out.append(float(v) if v is not None else 0.0)
      except Exception:
        out.append(0.0)
    return out
  try:
    v = float(costs_usd)
    return [v] * n
  except Exception:
    return [0.0] * n


def _as_thinking_list(thinking_tokens: Any, n: int) -> List[int]:
  if n <= 0:
    return []
  if thinking_tokens is None:
    return [0] * n
  if isinstance(thinking_tokens, list):
    out: List[int] = []
    for i in range(n):
      try:
        v = thinking_tokens[i]
      except Exception:
        v = None
      try:
        out.append(int(v) if v is not None else 0)
      except Exception:
        out.append(0)
    return out
  try:
    v = int(thinking_tokens)
    return [v] * n
  except Exception:
    return [0] * n


def _try_float(x: Any) -> Optional[float]:
  if x is None:
    return None
  if isinstance(x, (int, float)):
    return float(x)
  if not isinstance(x, str):
    return None
  s = x.strip()
  if not s:
    return None
  # common patterns: "Answer: 50", "50", "50.0"
  s2 = s
  if "answer:" in s2.lower():
    s2 = re.split(r"answer\s*:", s2, flags=re.IGNORECASE)[-1].strip()
  m = re.search(r"[-+]?\d+(?:\.\d+)?", s2)
  if not m:
    return None
  try:
    return float(m.group(0))
  except Exception:
    return None


def _float_equal(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
  if a is None or b is None:
    return False
  try:
    return abs(float(a) - float(b)) <= tol
  except Exception:
    return False


# -----------------------------
# Multi-turn parsing helpers
# -----------------------------
_CTRL_TAG_RE = re.compile(r"<\|[^>]*\|>")  # <|...|>

def _strip_thinking(response: Optional[str], model_name: str = "") -> str:
  if not response:
    return ""

  t = response
  mn = (model_name or "").lower()

  if "gpt-oss" in mn:
    final_marker = "<|channel|>final<|message|>"
    if final_marker in t:
      t = t.split(final_marker, 1)[-1]
      t = t.split("<|return|>", 1)[0]
    else:
      # best-effort: drop analysis and any closing markers
      if "<|channel|>analysis<|message|>" in t:
        t = t.split("<|channel|>analysis<|message|>", 1)[-1]
      if "<|end|>" in t:
        t = t.rsplit("<|end|>", 1)[-1]
  else:
    # Common reasoning end tokens (e.g. Qwen's </think>)
    for end_tok in ("</think>", "[/THINK]"):
      if end_tok in t or end_tok.lower() in t.lower():
        # case-insensitive split for [/THINK] variants
        if end_tok.startswith("["):
          t = re.split(re.escape(end_tok), t, flags=re.IGNORECASE)[-1]
        else:
          t = re.split(re.escape(end_tok), t, flags=re.IGNORECASE)[-1]
        break

  # remove any leftover control tags to avoid apply_chat_template TemplateError
  t = _CTRL_TAG_RE.sub("", t)
  return t.strip()


def _extract_questions_vars(text: str) -> List[str]:
  """
  Extract variable names the model asked for.

  Accepts formats like:
    - "Choice: 1,2" (not used in multi-turn)
    - "Question: What is V?"
    - "Questions: V, F, S"
    - "Ask: V; S"
    - "What is the value of V?"
  Returns raw var tokens (strings).
  """
  if text is None:
    return []
  t = _strip_thinking(text)

  # strongest: explicit "Question:" lines
  qs = re.findall(r"(?:Question|Questions)\s*:\s*(.+)", t, flags=re.IGNORECASE)
  cand_chunks = qs[:] if qs else []

  # fallback: "What is X?" occurrences
  whatis = re.findall(r"\bWhat\s+is\s+([A-Za-z][A-Za-z0-9_]*)\b", t, flags=re.IGNORECASE)
  if whatis:
    return [w.strip() for w in whatis]

  # fallback chunk: "Ask:" or "I need:" or "Please provide:"
  for key in ["Ask:", "Need:", "Please provide:", "I need:", "Request:"]:
    if key.lower() in t.lower():
      cand_chunks.append(t)

  # if no chunks, try a simple token scan for single var ask
  if not cand_chunks:
    m = re.search(r"\b([A-Za-z][A-Za-z0-9_]*)\b", t)
    if m and len(t) <= 80:
      return [m.group(1)]

  out: List[str] = []
  for ch in cand_chunks:
    # take after the first colon if present
    if ":" in ch:
      ch = ch.split(":", 1)[-1]
    # split by comma/semicolon/newline
    parts = re.split(r"[,\n;]+", ch)
    for p in parts:
      v = p.strip()
      if not v:
        continue
      m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\b", v)
      if m:
        out.append(m.group(1))
  return out


def _is_answer_action(text: str) -> bool:
  if text is None:
    return False
  t = _strip_thinking(text).lower()
  return ("answer:" in t) or bool(re.match(r"^\s*Answer\s*:\s*[-+]?\d+(\.\d+)?\s*$", t))


# -----------------------------
# Leaf and forbidden computation
# -----------------------------

def _parse_pred_values_block(pred_values_str: Any) -> Dict[str, float]:
  """
  Parse "Pred Values" block like:
    F = 0.5
    V = 100
  into dict.
  """
  if pred_values_str is None:
    return {}
  if isinstance(pred_values_str, dict):
    out = {}
    for k, v in pred_values_str.items():
      fv = _try_float(v)
      if fv is not None:
        out[str(k)] = float(fv)
    return out
  if not isinstance(pred_values_str, str):
    return {}
  out: Dict[str, float] = {}
  for line in pred_values_str.splitlines():
    if "=" not in line:
      continue
    k, v = line.split("=", 1)
    k = k.strip()
    v = v.strip()
    fv = _try_float(v)
    if k and fv is not None:
      out[k] = float(fv)
  return out


def _infer_leaf_nodes(datum: pd.Series) -> Set[str]:
  """
  Best-effort leaf node inference:
    1) prefer dataset fields: leaf_nodes_all, relevant_leaf
    2) else leaf = keys(Pred Values) minus LHS variables in Equations
  """
  for key in ["leaf_nodes_all", "leaf_nodes", "relevant_leaf"]:
    if key in datum and pd.notna(datum[key]):
      lst = _safe_list_field(datum.get(key))
      if lst:
        return set(str(x) for x in lst if str(x))
  eqs = _safe_dict_field(datum.get("Equations", "{}"))
  lhs_vars: Set[str] = set()
  if isinstance(eqs, dict):
    for k in eqs.keys():
      if isinstance(k, str) and "=" in k:
        lhs_vars.add(k.split("=", 1)[0].strip())
  pred_map = _parse_pred_values_block(datum.get("Pred Values", None))
  value_vars = set(pred_map.keys())
  leaf = value_vars - lhs_vars
  return set(str(x) for x in leaf)


def _all_vars_in_problem(datum: pd.Series) -> Set[str]:
  eqs = _safe_dict_field(datum.get("Equations", "{}"))
  vars_desc = _safe_dict_field(datum.get("Variables", "{}"))
  pred_map = _parse_pred_values_block(datum.get("Pred Values", None))

  out: Set[str] = set()
  out |= set(str(k) for k in vars_desc.keys())
  out |= set(str(k) for k in pred_map.keys())

  if isinstance(eqs, dict):
    for eq in eqs.keys():
      if not isinstance(eq, str):
        continue
      for tok in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", eq):
        out.add(tok)
  return out


def _prf1(pred: Set[str], gt: Set[str]) -> Dict[str, float]:
  inter = len(pred & gt)
  p = inter / max(1, len(pred))
  r = inter / max(1, len(gt))
  f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
  return {"precision": p, "recall": r, "f1": f1}

def _find_first_answer_log(turn_logs):
    for rec in turn_logs:
        if rec.get("action") in ("ANSWER", "EARLY_ANSWER"):
            return rec.get("turn"), bool(rec.get("force_answer", False))
    return None, None

def _qset_metrics(pred: Set[str], gt: Set[str]) -> Dict[str, Any]:
  # similarity: jaccard
  inter = len(pred & gt)
  union = len(pred | gt)
  jaccard = (inter / union) if union > 0 else 1.0

  # over-ask / under-ask
  extra = pred - gt
  missing = gt - pred
  num_extra = len(extra)
  num_missing = len(missing)

  # extend of over-ask
  gt_size = len(gt)
  pred_size = len(pred)
  over_rate = (num_extra / gt_size) if gt_size > 0 else (1.0 if num_extra > 0 else 0.0)
  under_rate = (num_missing / gt_size) if gt_size > 0 else 0.0

  signed_diff = num_extra - num_missing

  # exactly accuracy
  exact = (pred == gt)

  return {
    "qset_exact": bool(exact),
    "qset_jaccard": float(jaccard),

    "qset_pred_size": int(pred_size),
    "qset_gt_size": int(gt_size),
    "qset_intersection": int(inter),
    "qset_union": int(union),

    "qset_num_extra": int(num_extra),
    "qset_num_missing": int(num_missing),
    "qset_over_rate": float(over_rate),
    "qset_under_rate": float(under_rate),
    "qset_signed_extra_minus_missing": int(signed_diff),

    "qset_extra": sorted(list(extra)),
    "qset_missing": sorted(list(missing)),
  }