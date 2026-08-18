#!/usr/bin/env python3
"""
Clean 'Rewritten Problem' in GSM CSV:

1) Remove all natural-language text inside square brackets [...] everywhere.
2) Inside the "Variables:" block only, drop lines that are single variables with NO assignment.
   Example lines to drop (after step 1):
     B
     C
     foo
   Keep lines like:
     A = 30
     W=40
     G1 = S1/2

Usage:
  python clean_rewritten_problem.py \
    --input gsm_CSP_full_ksufficient_kmax4_N30_M1.csv \
    --output gsm_CSP_full_ksufficient_kmax4_N30_M1.cleaned.csv
"""

import argparse
import re
import pandas as pd


BRACKET_RE = re.compile(r"\s*\[[^\]]*\]")  # remove bracket + content, plus leading spaces
SPACE_RE = re.compile(r"[ \t]+")

ASSIGN_RE = re.compile(r"^\s*[A-Za-z]\w*\s*=")     # "A = ..." or "A=..."
BARE_VAR_RE = re.compile(r"^\s*[A-Za-z]\w*\s*$")   # "A" only


def strip_bracket_text(s: str) -> str:
    if s is None:
        return s
    if not isinstance(s, str):
        s = str(s)

    s2 = BRACKET_RE.sub("", s)

    # Clean up extra spaces per line, keep newlines
    s2 = "\n".join(SPACE_RE.sub(" ", line).rstrip() for line in s2.splitlines())
    return s2.strip()


def drop_unassigned_vars_in_variables_block(s: str) -> str:
    """
    Only edits the Variables: section (from 'Variables:' line up to the next 'Equations:' line).
    """
    lines = s.splitlines()

    # find "Variables:" line
    var_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "variables:":
            var_idx = i
            break
    if var_idx is None:
        return s

    # find end at "Equations:" (first occurrence after Variables:)
    end_idx = None
    for j in range(var_idx + 1, len(lines)):
        if lines[j].strip().lower().startswith("equations:"):
            end_idx = j
            break
    if end_idx is None:
        # no equations block, treat rest as variables block
        end_idx = len(lines)

    # filter variable lines
    kept = []
    for line in lines[var_idx + 1 : end_idx]:
        t = line.strip()
        if not t:
            kept.append(line)  # keep blank lines
            continue

        # drop "bare var" lines, keep assignments and everything else
        if BARE_VAR_RE.match(t) and not ASSIGN_RE.match(t):
            continue

        kept.append(line)

    # optional: avoid a long run of blank lines after deletions (keep at most 1 consecutive blank)
    cleaned_kept = []
    prev_blank = False
    for line in kept:
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue
        cleaned_kept.append(line)
        prev_blank = is_blank

    out_lines = lines[: var_idx + 1] + cleaned_kept + lines[end_idx:]
    return "\n".join(out_lines).strip()


def process_rewritten(text: str) -> str:
    s = strip_bracket_text(text)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--col", default="Rewritten Problem")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    if args.col not in df.columns:
        raise ValueError(f"Column not found: {args.col}. Available: {list(df.columns)}")

    df[args.col] = df[args.col].map(process_rewritten)

    df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()