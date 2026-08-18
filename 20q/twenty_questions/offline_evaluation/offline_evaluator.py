#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import httpx
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from . import candidate_list as candidate_bank
except ImportError:
    try:
        import candidate_list as candidate_bank
    except Exception as e:
        raise ImportError(
            "Could not import candidate_list.py. Put offline_evaluator.py in the same directory as candidate_list.py, "
            "or add that directory to PYTHONPATH."
        ) from e


# =========================
# Candidate pools
# =========================
POOL_MAP = {
    "Animals_EVAL_POOL": candidate_bank.Animals_EVAL_POOL,
    "Places_EVAL_POOL": candidate_bank.Places_EVAL_POOL,
    "Food_EVAL_POOL": candidate_bank.Food_EVAL_POOL,
    "Objects_EVAL_POOL": candidate_bank.Objects_EVAL_POOL,
    "BIG_BENCH_CONCEPT_EVAL_POOL": candidate_bank.BIG_BENCH_CONCEPT_EVAL_POOL,
    "THING200_EVAL_POOL": candidate_bank.THING200_EVAL_POOL,
    "COMMON_EVAL_POOL": candidate_bank.COMMON_EVAL_POOL,
    "common": candidate_bank.COMMON,
    "bigbench": candidate_bank.BIG_BENCH_CONCEPT,
    "thing": candidate_bank.THING200,
}


# =========================
# Helpers
# =========================
QUESTION_RE = re.compile(
    r"\b(?:Is|Are|Does|Do|Can|Could|Would|Should|Will|Have|Has)\s+X\b[^?]*\?",
    flags=re.IGNORECASE,
)
QUESTION_RE_GENERIC = re.compile(
    r"\b(?:Is|Are|Does|Do|Can|Could|Would|Should|Will|Have|Has)\b[^?]*\?",
    flags=re.IGNORECASE,
)

YES_RE = re.compile(r"\bYes\b", flags=re.IGNORECASE)
NO_RE = re.compile(r"\bNo\b", flags=re.IGNORECASE)
PASS_RE = re.compile(
    r"\b(?:Pass|Partially|Partial|Maybe|Unknown|Uncertain|Cannot determine|Can't determine|Not sure|Unsure)\b",
    flags=re.IGNORECASE,
)


def normalize_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_entity(s: str) -> str:
    s = normalize_text(s).lower()
    s = re.sub(r'^[\s"\'`\(\)\[\]\{\}\.,!?;:]+', '', s)
    s = re.sub(r'[\s"\'`\(\)\[\]\{\}\.,!?;:]+$', '', s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_question(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_answer_label(x: str) -> Optional[str]:
    x = normalize_text(x).lower()

    if x in {"yes", "y"}:
        return "yes"
    if x in {"no", "n"}:
        return "no"
    if x in {"pass", "partial", "partially", "maybe", "unknown", "uncertain", "unsure", "not sure"}:
        return "pass"

    if "partially" in x or "partial" in x:
        return "pass"
    if "pass" in x:
        return "pass"
    if "unknown" in x or "uncertain" in x or "unsure" in x or "not sure" in x:
        return "pass"
    if "maybe" in x:
        return "pass"

    return None


def parse_yes_no_pass(text: str) -> Optional[str]:
    text = normalize_text(text)

    has_yes = bool(YES_RE.search(text))
    has_no = bool(NO_RE.search(text))
    has_pass = bool(PASS_RE.search(text))

    m = re.match(
        r"\s*(Yes|No|Pass|Partially|Partial|Maybe|Unknown|Uncertain|Unsure|Not sure)\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_answer_label(m.group(1))

    if has_yes and not has_no and not has_pass:
        return "yes"
    if has_no and not has_yes and not has_pass:
        return "no"
    if has_pass and not has_yes and not has_no:
        return "pass"

    if has_yes and has_no:
        return None
    if has_yes and has_pass and not has_no:
        return "pass"
    if has_no and has_pass and not has_yes:
        return "pass"

    return None


def looks_like_question(text: str) -> bool:
    text = normalize_text(text)
    if QUESTION_RE.search(text):
        return True
    if text.endswith("?") and QUESTION_RE_GENERIC.search(text):
        return True
    return False


def extract_question_candidates(text: str) -> List[str]:
    text = normalize_text(text)
    qs = QUESTION_RE.findall(text)
    if not qs:
        qs = QUESTION_RE_GENERIC.findall(text)
    return [normalize_question(q) for q in qs]


def extract_best_question(text: str) -> Optional[str]:
    text = normalize_text(text)

    if looks_like_question(text) and len(text) < 260:
        return normalize_question(text)

    qs = extract_question_candidates(text)
    if not qs:
        return None

    return qs[-1]


def entropy_from_probs(probs: Dict[str, float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in probs.values() if p > 0)


def normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    z = sum(probs.values())
    if z <= 0:
        n = len(probs)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in probs}
    return {k: v / z for k, v in probs.items()}


def sha1_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def chunked(seq: Sequence[str], size: int) -> List[List[str]]:
    return [list(seq[i:i + size]) for i in range(0, len(seq), size)]


def maybe_tqdm(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


# =========================
# Parsing logs
# =========================
@dataclass
class Turn:
    idx: int
    question: str
    answer: str
    duplicate: bool = False


def extract_turns_from_history(history: Sequence[Dict[str, Any]]) -> List[Turn]:
    turns: List[Turn] = []
    pending_q: Optional[str] = None
    seen_questions: set[str] = set()

    for msg in history:
        role = msg.get("role", "")
        content = normalize_text(msg.get("content", ""))
        if not content:
            continue

        if role == "assistant":
            q = extract_best_question(content)
            pending_q = q
        elif role == "user":
            ans = parse_yes_no_pass(content)
            if pending_q is not None and ans is not None:
                duplicate = pending_q.lower() in seen_questions
                turns.append(
                    Turn(
                        idx=len(turns) + 1,
                        question=pending_q,
                        answer=ans,
                        duplicate=duplicate,
                    )
                )
                seen_questions.add(pending_q.lower())
                pending_q = None
            else:
                pending_q = None

    return turns


# =========================
# Judge backend
# =========================
class BaseJudge:
    def judge(self, candidate: str, question: str) -> str:
        raise NotImplementedError

    def judge_batch(self, candidates: Sequence[str], question: str) -> Dict[str, str]:
        return {c: self.judge(c, question) for c in candidates}


class JsonCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: Dict[str, str] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class LocalVLLMJudge(BaseJudge):
    """
    Batch judge via a local vLLM server exposing an OpenAI-compatible chat API.

    Important robustness behavior:
    1. Uses a smaller default batch size externally (recommended: 4).
    2. Uses a large default max_tokens to avoid truncation.
    3. If batch parsing is incomplete, automatically falls back to per-candidate judging.
    4. Never silently assigns unparsed candidates to "pass".
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8011/v1",
        api_key: str = "EMPTY",
        cache_path: str | Path = "judge_cache.json",
        temperature: float = 0.0,
        max_tokens: int = 32768,
        sleep_s: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 5,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for LocalVLLMJudge")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )
        self.model = model
        self.base_url = base_url
        self.cache = JsonCache(cache_path)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sleep_s = sleep_s
        self.max_retries = max_retries

    def _messages_single(self, candidate: str, question: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a strict oracle used inside a 20-questions benchmark.\n"
            "You will be given a candidate entity and one yes/no style question about X.\n"
            "Your task is to answer what the truthful response would be IF X were exactly that candidate.\n\n"
            "Allowed outputs:\n"
            "- Yes\n"
            "- No\n"
            "- Pass\n\n"
            "Use Pass when the question is ambiguous, malformed, underspecified, only partially true, context-dependent, "
            "or cannot be answered cleanly with a definite Yes or No.\n"
            "Return EXACTLY one token: Yes or No or Pass."
        )
        user_prompt = (
            f"Candidate: {candidate}\n"
            f"Question: {question}\n\n"
            "Return exactly one token: Yes / No / Pass"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _messages_batch(self, candidates: Sequence[str], question: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a strict oracle used inside a 20-questions benchmark.\n"
            "You will be given one yes/no style question about X and a numbered list of candidate entities.\n"
            "For each candidate, answer what the truthful response would be IF X were exactly that candidate.\n\n"
            "Allowed labels:\n"
            "- Yes\n"
            "- No\n"
            "- Pass\n\n"
            "Use Pass when the question is ambiguous, malformed, underspecified, only partially true, context-dependent, "
            "or cannot be answered cleanly with a definite Yes or No.\n\n"
            "Output format requirements:\n"
            "Return exactly one line per candidate.\n"
            "Each line must be exactly:\n"
            "<index>. <Yes|No|Pass>\n"
            "Examples:\n"
            "1. Yes\n"
            "2. No\n"
            "3. Pass\n"
            "Do not include candidate names.\n"
            "Do not include explanations.\n"
            "Do not omit any candidate.\n"
            "Do not add any extra text before or after the answer list."
        )

        lines = [f"{i + 1}. {c}" for i, c in enumerate(candidates)]
        user_prompt = (
            f"Question: {question}\n\n"
            f"Candidates:\n" + "\n".join(lines) + "\n\n"
            "Return exactly one line per candidate in the form:\n"
            "1. Yes\n"
            "2. No\n"
            "3. Pass"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_model_answer(self, text: str) -> str:
        text = normalize_text(text)

        m = re.match(
            r"^(Yes|No|Pass|Partially|Partial|Maybe|Unknown|Uncertain|Unsure)\b",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            ans = normalize_answer_label(m.group(1))
            return ans if ans is not None else "pass"

        if re.search(r"\bYes\b", text, flags=re.IGNORECASE) and not re.search(r"\bNo\b", text, flags=re.IGNORECASE):
            return "yes"
        if re.search(r"\bNo\b", text, flags=re.IGNORECASE) and not re.search(r"\bYes\b", text, flags=re.IGNORECASE):
            return "no"
        if re.search(r"\bPass\b", text, flags=re.IGNORECASE):
            return "pass"
        if re.search(r"\bPartially\b|\bPartial\b|\bMaybe\b|\bUnknown\b|\bUncertain\b|\bUnsure\b", text, flags=re.IGNORECASE):
            return "pass"

        return "pass"

    def _parse_batch_output(self, text: str, candidates: Sequence[str]) -> Tuple[Dict[str, str], bool]:
        """
        Returns:
            (parsed_output, is_complete)

        is_complete=True only if every candidate got a parsed answer.
        Never assigns default 'pass' to unparsed items.
        """
        out: Dict[str, str] = {}
        lines = [normalize_text(x) for x in str(text).splitlines() if normalize_text(x)]

        idx_to_ans: Dict[int, str] = {}
        seen_idx = set()

        for line in lines:
            m = re.match(
                r"^\s*(\d+)\s*[\.\:\-\)]\s*(Yes|No|Pass|Partially|Partial|Maybe|Unknown|Uncertain|Unsure)\b",
                line,
                flags=re.IGNORECASE,
            )
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(candidates) and idx not in seen_idx:
                    ans = normalize_answer_label(m.group(2)) or "pass"
                    idx_to_ans[idx] = ans
                    seen_idx.add(idx)
                continue

        for i, cand in enumerate(candidates):
            if i in idx_to_ans:
                out[cand] = idx_to_ans[i]

        is_complete = len(out) == len(candidates)
        return out, is_complete

    def _chat_completion_kwargs(self, messages: List[Dict[str, str]], use_max_completion_tokens: bool = False) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
        }
        if use_max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens

        if not str(self.model).startswith("gpt-5"):
            kwargs["temperature"] = self.temperature

        return kwargs

    def _request(self, messages: List[Dict[str, str]]) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                try:
                    resp = self.client.chat.completions.create(
                        **self._chat_completion_kwargs(messages, use_max_completion_tokens=False)
                    )
                except Exception as e:
                    if "max_completion_tokens" not in str(e) and "max_tokens" not in str(e):
                        raise
                    resp = self.client.chat.completions.create(
                        **self._chat_completion_kwargs(messages, use_max_completion_tokens=True)
                    )
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))
                else:
                    break

        if last_err is not None:
            raise last_err
        raise RuntimeError("Unknown vLLM request failure")

    def judge(self, candidate: str, question: str) -> str:
        key = sha1_key(
            self.model,
            self.base_url,
            normalize_entity(candidate),
            normalize_question(question),
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            text = self._request(self._messages_single(candidate, question))
            ans = self._parse_model_answer(text)
        except Exception as e:
            print(f"[WARN] judge failed for candidate={candidate!r}, question={question!r}: {e}")
            ans = "pass"

        self.cache.set(key, ans)
        return ans

    def judge_batch(self, candidates: Sequence[str], question: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        uncached: List[str] = []

        q_norm = normalize_question(question)

        # 1. read cache first
        for cand in candidates:
            key = sha1_key(self.model, self.base_url, normalize_entity(cand), q_norm)
            cached = self.cache.get(key)
            if cached is not None:
                result[cand] = cached
            else:
                uncached.append(cand)

        if not uncached:
            return result

        # 2. try one batch request for remaining candidates
        batch_out: Dict[str, str] = {}
        batch_complete = False
        raw_text = ""

        try:
            raw_text = self._request(self._messages_batch(uncached, question))
            batch_out, batch_complete = self._parse_batch_output(raw_text, uncached)
        except Exception as e:
            print(f"[WARN] batch judge failed for question={question!r}, batch_size={len(uncached)}: {e}")
            batch_out = {}
            batch_complete = False

        # 3. if parse incomplete, fallback to single judge for missing candidates
        if not batch_complete:
            parsed_n = len(batch_out)
            expected_n = len(uncached)
            print(
                f"[WARN] incomplete batch parse for question={question!r}: "
                f"parsed {parsed_n}/{expected_n}. Falling back to single judge for missing candidates."
            )

            if raw_text:
                print("=== RAW JUDGE BATCH OUTPUT BEGIN ===")
                print(raw_text)
                print("=== RAW JUDGE BATCH OUTPUT END ===")

            missing = [cand for cand in uncached if cand not in batch_out]
            for cand in missing:
                batch_out[cand] = self.judge(cand, question)

        # 4. cache fill
        for cand, ans in batch_out.items():
            key = sha1_key(self.model, self.base_url, normalize_entity(cand), q_norm)
            self.cache.set(key, ans)
            result[cand] = ans

        return result

    def flush(self) -> None:
        self.cache.flush()


class HostedGPTJudge(BaseJudge):
    """
    Judge via the same hosted GPT path used by guesser/examiner models.
    """

    def __init__(
        self,
        model: str,
        cache_path: str | Path = "judge_cache.json",
        temperature: float = 0.0,
        max_tokens: int = 32768,
        sleep_s: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 5,
    ):
        self.model = model
        self.cache = JsonCache(cache_path)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sleep_s = sleep_s
        self.timeout = timeout
        self.max_retries = max_retries

        try:
            from twenty_questions.models import hosted_gpt_chat_completion
        except Exception as e:
            raise ImportError(
                "HostedGPTJudge requires twenty_questions.models to be importable."
            ) from e

        self._hosted_gpt_chat_completion = hosted_gpt_chat_completion
        self.backend_id = f"hosted_gpt|{os.environ.get('AZURE_OPENAI_ENDPOINT', '').rstrip('/') or 'azure_env'}"

    def _messages_single(self, candidate: str, question: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a strict oracle used inside a 20-questions benchmark.\n"
            "You will be given a candidate entity and one yes/no style question about X.\n"
            "Your task is to answer what the truthful response would be IF X were exactly that candidate.\n\n"
            "Allowed outputs:\n"
            "- Yes\n"
            "- No\n"
            "- Pass\n\n"
            "Use Pass when the question is ambiguous, malformed, underspecified, only partially true, context-dependent, "
            "or cannot be answered cleanly with a definite Yes or No.\n"
            "Return EXACTLY one token: Yes or No or Pass."
        )
        user_prompt = (
            f"Candidate: {candidate}\n"
            f"Question: {question}\n\n"
            "Return exactly one token: Yes / No / Pass"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _messages_batch(self, candidates: Sequence[str], question: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a strict oracle used inside a 20-questions benchmark.\n"
            "You will be given one yes/no style question about X and a numbered list of candidate entities.\n"
            "For each candidate, answer what the truthful response would be IF X were exactly that candidate.\n\n"
            "Allowed labels:\n"
            "- Yes\n"
            "- No\n"
            "- Pass\n\n"
            "Use Pass when the question is ambiguous, malformed, underspecified, only partially true, context-dependent, "
            "or cannot be answered cleanly with a definite Yes or No.\n\n"
            "Output format requirements:\n"
            "Return exactly one line per candidate.\n"
            "Each line must be exactly:\n"
            "<index>. <Yes|No|Pass>\n"
            "Examples:\n"
            "1. Yes\n"
            "2. No\n"
            "3. Pass\n"
            "Do not include candidate names.\n"
            "Do not include explanations.\n"
            "Do not omit any candidate.\n"
            "Do not add any extra text before or after the answer list."
        )

        lines = [f"{i + 1}. {c}" for i, c in enumerate(candidates)]
        user_prompt = (
            f"Question: {question}\n\n"
            f"Candidates:\n" + "\n".join(lines) + "\n\n"
            "Return exactly one line per candidate in the form:\n"
            "1. Yes\n"
            "2. No\n"
            "3. Pass"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_model_answer(self, text: str) -> str:
        text = normalize_text(text)

        m = re.match(
            r"^(Yes|No|Pass|Partially|Partial|Maybe|Unknown|Uncertain|Unsure)\b",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            ans = normalize_answer_label(m.group(1))
            return ans if ans is not None else "pass"

        if re.search(r"\bYes\b", text, flags=re.IGNORECASE) and not re.search(r"\bNo\b", text, flags=re.IGNORECASE):
            return "yes"
        if re.search(r"\bNo\b", text, flags=re.IGNORECASE) and not re.search(r"\bYes\b", text, flags=re.IGNORECASE):
            return "no"
        if re.search(r"\bPass\b", text, flags=re.IGNORECASE):
            return "pass"
        if re.search(r"\bPartially\b|\bPartial\b|\bMaybe\b|\bUnknown\b|\bUncertain\b|\bUnsure\b", text, flags=re.IGNORECASE):
            return "pass"

        return "pass"

    def _parse_batch_output(self, text: str, candidates: Sequence[str]) -> Tuple[Dict[str, str], bool]:
        out: Dict[str, str] = {}
        lines = [normalize_text(x) for x in str(text).splitlines() if normalize_text(x)]

        idx_to_ans: Dict[int, str] = {}
        seen_idx = set()

        for line in lines:
            m = re.match(
                r"^\s*(\d+)\s*[\.\:\-\)]\s*(Yes|No|Pass|Partially|Partial|Maybe|Unknown|Uncertain|Unsure)\b",
                line,
                flags=re.IGNORECASE,
            )
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(candidates) and idx not in seen_idx:
                    ans = normalize_answer_label(m.group(2)) or "pass"
                    idx_to_ans[idx] = ans
                    seen_idx.add(idx)
                continue

        for i, cand in enumerate(candidates):
            if i in idx_to_ans:
                out[cand] = idx_to_ans[i]

        is_complete = len(out) == len(candidates)
        return out, is_complete

    def _request(self, messages: List[Dict[str, str]]) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                _, text, _, _ = self._hosted_gpt_chat_completion(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    sleep_for_model=self.sleep_s > 0,
                )
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)
                return text or ""
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))
                else:
                    break

        if last_err is not None:
            raise last_err
        raise RuntimeError("Unknown hosted GPT request failure")

    def judge(self, candidate: str, question: str) -> str:
        key = sha1_key(
            self.model,
            self.backend_id,
            normalize_entity(candidate),
            normalize_question(question),
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            text = self._request(self._messages_single(candidate, question))
            ans = self._parse_model_answer(text)
        except Exception as e:
            print(f"[WARN] hosted judge failed for candidate={candidate!r}, question={question!r}: {e}")
            ans = "pass"

        self.cache.set(key, ans)
        return ans

    def judge_batch(self, candidates: Sequence[str], question: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        uncached: List[str] = []

        q_norm = normalize_question(question)

        for cand in candidates:
            key = sha1_key(self.model, self.backend_id, normalize_entity(cand), q_norm)
            cached = self.cache.get(key)
            if cached is not None:
                result[cand] = cached
            else:
                uncached.append(cand)

        if not uncached:
            return result

        batch_out: Dict[str, str] = {}
        batch_complete = False
        raw_text = ""

        try:
            raw_text = self._request(self._messages_batch(uncached, question))
            batch_out, batch_complete = self._parse_batch_output(raw_text, uncached)
        except Exception as e:
            print(f"[WARN] hosted batch judge failed for question={question!r}, batch_size={len(uncached)}: {e}")
            batch_out = {}
            batch_complete = False

        if not batch_complete:
            parsed_n = len(batch_out)
            expected_n = len(uncached)
            print(
                f"[WARN] incomplete hosted batch parse for question={question!r}: "
                f"parsed {parsed_n}/{expected_n}. Falling back to single judge for missing candidates."
            )

            if raw_text:
                print("=== RAW HOSTED JUDGE BATCH OUTPUT BEGIN ===")
                print(raw_text)
                print("=== RAW HOSTED JUDGE BATCH OUTPUT END ===")

            missing = [cand for cand in uncached if cand not in batch_out]
            for cand in missing:
                batch_out[cand] = self.judge(cand, question)

        for cand, ans in batch_out.items():
            key = sha1_key(self.model, self.backend_id, normalize_entity(cand), q_norm)
            self.cache.set(key, ans)
            result[cand] = ans

        return result

    def flush(self) -> None:
        self.cache.flush()


# =========================
# Evaluator
# =========================
@dataclass
class EvaluatorConfig:
    match_prob: float = 0.98
    mismatch_prob: float = 0.02
    pass_prob: float = 0.50
    topk_prune: int = 0
    judge_batch_size: int = 4
    show_progress: bool = True


class OfflineEvaluator:
    def __init__(self, candidates: Sequence[str], judge: BaseJudge, cfg: EvaluatorConfig):
        self.candidates = list(dict.fromkeys(candidates))
        self.judge = judge
        self.cfg = cfg

    def initial_belief(self) -> Dict[str, float]:
        n = len(self.candidates)
        return {c: 1.0 / n for c in self.candidates}

    def answer_map(
        self,
        question: str,
        active_candidates: Iterable[str],
        episode_desc: Optional[str] = None,
        turn_idx: Optional[int] = None,
    ) -> Dict[str, str]:
        active_candidates = list(active_candidates)
        if not active_candidates:
            return {}

        batch_size = max(1, self.cfg.judge_batch_size)
        candidate_batches = chunked(active_candidates, batch_size)

        amap: Dict[str, str] = {}
        desc = "judge batches"
        if episode_desc is not None and turn_idx is not None:
            desc = f"{episode_desc} | turn {turn_idx} | judge batches"

        iterator = maybe_tqdm(
            candidate_batches,
            disable=not self.cfg.show_progress,
            desc=desc,
            leave=False,
        )

        for batch in iterator:
            amap.update(self.judge.judge_batch(batch, question))

        return amap

    def posterior_given_observed_answer(
        self,
        prior: Dict[str, float],
        answer_map: Dict[str, str],
        observed_answer: str,
    ) -> Dict[str, float]:
        probs: Dict[str, float] = {}

        for c, p in prior.items():
            pred = answer_map[c]

            if pred == observed_answer:
                likelihood = self.cfg.match_prob
            elif pred == "pass":
                likelihood = self.cfg.pass_prob
            else:
                likelihood = self.cfg.mismatch_prob

            probs[c] = p * likelihood

        probs = normalize_probs(probs)

        if self.cfg.topk_prune and self.cfg.topk_prune > 0 and len(probs) > self.cfg.topk_prune:
            top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[: self.cfg.topk_prune]
            probs = normalize_probs(dict(top))

        return probs

    def posterior_for_hyp_answer(
        self,
        prior: Dict[str, float],
        answer_map: Dict[str, str],
        hypothetical_answer: str,
    ) -> Dict[str, float]:
        probs: Dict[str, float] = {}
        for c, p in prior.items():
            probs[c] = p if answer_map[c] == hypothetical_answer else 0.0
        return normalize_probs(probs)

    def question_eig(
        self,
        prior: Dict[str, float],
        answer_map: Dict[str, str],
    ) -> Tuple[float, float, float, Dict[str, float]]:
        h_prior = entropy_from_probs(prior)

        answer_dist: Dict[str, float] = defaultdict(float)
        for c, p in prior.items():
            answer_dist[answer_map[c]] += p

        exp_post_h = 0.0
        for ans, p_ans in answer_dist.items():
            post = self.posterior_for_hyp_answer(prior, answer_map, ans)
            exp_post_h += p_ans * entropy_from_probs(post)

        eig = h_prior - exp_post_h
        norm_eig = eig / (h_prior + 1e-12)
        return eig, norm_eig, h_prior, dict(answer_dist)

    def evaluate_turns(
        self,
        turns: Sequence[Turn],
        episode_desc: Optional[str] = None,
    ) -> Dict[str, Any]:
        prior = self.initial_belief()
        results: List[Dict[str, Any]] = []

        remaining_entropy_pre = [entropy_from_probs(prior)]
        remaining_entropy_post: List[float] = []

        turn_iter = maybe_tqdm(
            turns,
            disable=not self.cfg.show_progress,
            desc=f"{episode_desc} | turns" if episode_desc else "turns",
            leave=False,
        )

        for turn in turn_iter:
            active_candidates = list(prior.keys())
            amap = self.answer_map(
                turn.question,
                active_candidates,
                episode_desc=episode_desc,
                turn_idx=turn.idx,
            )
            eig, norm_eig, h_prior, answer_dist = self.question_eig(prior, amap)

            posterior = self.posterior_given_observed_answer(prior, amap, turn.answer)
            h_post = entropy_from_probs(posterior)

            match_mass = sum(prior[c] for c, a in amap.items() if a == turn.answer)
            pass_mass = sum(prior[c] for c, a in amap.items() if a == "pass")

            results.append({
                "turn": turn.idx,
                "question": turn.question,
                "observed_answer": turn.answer,
                "duplicate": turn.duplicate,
                "n_active_candidates": len(prior),
                "entropy_pre": h_prior,
                "raw_eig": eig,
                "normalized_eig": norm_eig,
                "answer_distribution": answer_dist,
                "support_mass_matching_observed": match_mass,
                "support_mass_pass": pass_mass,
                "entropy_post": h_post,
                "top_candidates_pre": [
                    {"candidate": c, "prob": p}
                    for c, p in sorted(prior.items(), key=lambda x: x[1], reverse=True)[:10]
                ],
                "top_candidates_post": [
                    {"candidate": c, "prob": p}
                    for c, p in sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:10]
                ],
            })

            prior = posterior
            remaining_entropy_pre.append(entropy_from_probs(prior))
            remaining_entropy_post.append(h_post)

        summary = {
            "num_turns": len(turns),
            "avg_normalized_eig": sum(r["normalized_eig"] for r in results) / max(len(results), 1),
            "avg_raw_eig": sum(r["raw_eig"] for r in results) / max(len(results), 1),
            "remaining_entropy_curve_post": remaining_entropy_post,
            "remaining_entropy_curve_pre": remaining_entropy_pre[:-1],
            "initial_entropy": remaining_entropy_pre[0] if remaining_entropy_pre else 0.0,
            "final_entropy": (
                remaining_entropy_post[-1]
                if remaining_entropy_post
                else (remaining_entropy_pre[0] if remaining_entropy_pre else 0.0)
            ),
            "cumulative_raw_eig": sum(r["raw_eig"] for r in results),
            "cumulative_normalized_eig": sum(r["normalized_eig"] for r in results),
        }

        return {"turn_metrics": results, "summary": summary}


# =========================
# IO
# =========================
def load_input(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def infer_pool_name(ep: Dict[str, Any], default_pool: str) -> str:
    item = normalize_entity(str(ep.get("item", "")))
    if item:
        for pool_name, pool in POOL_MAP.items():
            normalized_pool = {normalize_entity(x) for x in pool}
            if item in normalized_pool:
                if pool_name == "COMMON_EVAL_POOL":
                    continue
                return pool_name
    return default_pool


def make_judge(_pool_name: str, args: argparse.Namespace) -> BaseJudge:
    if args.judge_backend == "hosted_gpt":
        return HostedGPTJudge(
            model=args.model,
            cache_path=args.cache_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            sleep_s=args.sleep,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

    return LocalVLLMJudge(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        cache_path=args.cache_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sleep_s=args.sleep,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline evaluator for 20Q logs: normalized EIG + remaining entropy curve using a judge model."
    )
    ap.add_argument("--input", required=True, help="Path to JSON or JSONL file containing episodes.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--pool", default="COMMON_EVAL_POOL", choices=sorted(POOL_MAP), help="Candidate pool to use, unless --infer-pool is set.")
    ap.add_argument("--infer-pool", action="store_true", help="Infer a category pool from ep['item'] when possible.")

    ap.add_argument(
        "--judge-backend",
        choices=["openai_compatible", "hosted_gpt"],
        default="openai_compatible",
        help="Use openai_compatible for vLLM or provider endpoints; use hosted_gpt to reuse twenty_questions.models GPT/Azure backend.",
    )
    ap.add_argument("--model", required=True, help="Judge model name.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1", help="OpenAI-compatible base URL for the judge endpoint.")
    ap.add_argument("--api-key", default="EMPTY", help="API key for the OpenAI-compatible judge endpoint.")
    ap.add_argument("--cache-path", default="judge_cache.json", help="Cache file for judge outputs.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Judge generation budget, including reasoning tokens (default: 32768).",
    )
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between judge calls.")
    ap.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout for each judge call.")
    ap.add_argument("--max-retries", type=int, default=5, help="Max retries per judge request.")
    ap.add_argument(
        "--judge-batch-size",
        type=int,
        default=4,
        help="Number of candidates judged per request (default: 4).",
    )

    ap.add_argument("--match-prob", type=float, default=0.98)
    ap.add_argument("--mismatch-prob", type=float, default=0.02)
    ap.add_argument("--pass-prob", type=float, default=0.50, help="Likelihood used when judge returns pass.")
    ap.add_argument("--topk-prune", type=int, default=0, help="If > 0, keep only top-k posterior candidates after each turn.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")

    args = ap.parse_args()

    episodes = load_input(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    judge_cache: Dict[str, BaseJudge] = {}

    episode_iter = maybe_tqdm(
        list(enumerate(episodes)),
        disable=args.no_progress,
        desc="episodes",
        leave=True,
    )

    for ep_idx, ep in episode_iter:
        pool_name = infer_pool_name(ep, args.pool) if args.infer_pool else args.pool

        if pool_name not in judge_cache:
            judge_cache[pool_name] = make_judge(pool_name, args)

        judge = judge_cache[pool_name]
        candidates = POOL_MAP[pool_name]

        cfg = EvaluatorConfig(
            match_prob=args.match_prob,
            mismatch_prob=args.mismatch_prob,
            pass_prob=args.pass_prob,
            topk_prune=args.topk_prune,
            judge_batch_size=args.judge_batch_size,
            show_progress=not args.no_progress,
        )
        evaluator = OfflineEvaluator(candidates, judge, cfg)

        history = ep.get("history_g", [])
        turns = extract_turns_from_history(history)

        item = ep.get("item", None)
        episode_desc = f"ep{ep_idx}"
        if item:
            episode_desc += f" [{item}]"

        metrics = evaluator.evaluate_turns(turns, episode_desc=episode_desc)

        row = {
            "episode_index": ep_idx,
            "item": ep.get("item"),
            "pool_name": pool_name,
            "num_candidates": len(candidates),
            "parsed_num_turns": len(turns),
            "parsed_turns": [
                {
                    "turn": t.idx,
                    "question": t.question,
                    "answer": t.answer,
                    "duplicate": t.duplicate,
                }
                for t in turns
            ],
            **metrics,
        }
        all_rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for j in judge_cache.values():
        flush = getattr(j, "flush", None)
        if callable(flush):
            flush()

    print(f"Wrote {len(all_rows)} evaluated episodes to {out_path}")


if __name__ == "__main__":
    main()
