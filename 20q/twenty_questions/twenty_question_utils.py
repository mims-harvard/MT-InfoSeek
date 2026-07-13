import re


def _normalize_free_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def looks_like_guess(text: str) -> bool:
    if not isinstance(text, str):
        return False

    t = _normalize_free_text(text).lower()
    if not t or t.endswith("?"):
        return False

    guess_patterns = [
        r'^\s*x\s+is\s+["\']?.+["\']?\s*[\.\!\?]?\s*$',
        r'^\s*is\s+it\s+["\']?.+["\']?\??\s*$',
        r'^\s*my\s+guess\s+is\s+["\']?.+["\']?\s*[\.\!\?]?\s*$',
        r'^\s*the\s+answer\s+is\s+["\']?.+["\']?\s*[\.\!\?]?\s*$',
        r'^\s*i\s+think\s+it\s+is\s+["\']?.+["\']?\s*[\.\!\?]?\s*$',
        r'^\s*it\s+is\s+["\']?.+["\']?\s*[\.\!\?]?\s*$',
        r'^\s*["\']?.+["\']?\s*[\.\!\?]?\s*$',
    ]
    return any(re.match(p, t) for p in guess_patterns)


def extract_guess_entity(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = _normalize_free_text(text)

    patterns = [
        r'^\s*x\s+is\s+["\']?(.+?)["\']?\s*[\.\!\?]?\s*$',
        r'^\s*is\s+it\s+["\']?(.+?)["\']?\??\s*$',
        r'^\s*my\s+guess\s+is\s+["\']?(.+?)["\']?\s*[\.\!\?]?\s*$',
        r'^\s*the\s+answer\s+is\s+["\']?(.+?)["\']?\s*[\.\!\?]?\s*$',
        r'^\s*i\s+think\s+it\s+is\s+["\']?(.+?)["\']?\s*[\.\!\?]?\s*$',
        r'^\s*it\s+is\s+["\']?(.+?)["\']?\s*[\.\!\?]?\s*$',
        r'^\s*["\']?(.+?)["\']?\s*[\.\!\?]?\s*$',
    ]

    for p in patterns:
        m = re.match(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def normalize_guess_entity(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = _normalize_free_text(text).lower()
    t = re.sub(r'^[\s"\'`\(\)\[\]\{\}\.,!?;:]+', '', t)
    t = re.sub(r'[\s"\'`\(\)\[\]\{\}\.,!?;:]+$', '', t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()
