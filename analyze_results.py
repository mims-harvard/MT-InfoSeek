"""Summarize a run into one comparable table + summary.json.

Each evaluator writes a different result schema; this reads them and reports the
metrics that matter across datasets:

  * accuracy            -- final-answer correctness (GSME and 20Q)
  * final sufficiency   -- did the model ever obtain enough info to determine the
                           answer (logic_q / genereg: last turn not ambiguous;
                           gsme: all held-out variables asked)
  * turn-to-sufficiency -- turns until that point, over the sufficient subset
                           (meaningful only at one question/turn)
  * avg turns           -- turns used per episode

Logic-Q-MT and GeneReg-MT use final sufficiency as their reported outcome; their
answer accuracy is intentionally omitted. 20-Questions additionally reports the
offline judge's question EIG and final entropy when that stage was run.

    python analyze_results.py --results-dir logs/run_gpt-5-mini_smoke --datasets all
"""
import argparse
import glob
import json
import os

import numpy as np

ALL = ["logic_q_mt", "gsme_q_mt", "gsme_q_mt_ext", "genereg_mt", "20q"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _nanmean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.nanmean(xs)) if xs and not all(np.isnan(xs)) else None


# ── logic_q / genereg share the turn-log sufficiency logic ───────────────────
def _episode_sufficiency(ep):
    """(is_final_sufficient, turns_to_sufficiency|nan) from a turn_logs episode."""
    tl = ep.get("turn_logs") or []
    is_known = (not tl[-1]["is_ambiguous"]) if tl else False
    if not is_known:
        return False, float("nan")
    turns_to = sum(1 for t in tl if t.get("is_ambiguous")) + 1
    return True, float(turns_to)


def summarize_turnlog_dataset(result_json, metrics_key, episodes_key):
    d = json.load(open(result_json))
    m = d[metrics_key]
    episodes = d.get(episodes_key, [])

    suff, tts = [], []
    for ep in episodes:
        ok, t = _episode_sufficiency(ep)
        suff.append(1.0 if ok else 0.0)
        if ok:
            tts.append(t)

    return {
        "n_episodes": m.get("total_episodes", len(episodes)),
        # For these tasks, the benchmark outcome is whether the acquired
        # information became sufficient. Do not headline answer accuracy: a
        # model can guess correctly while still lacking sufficient evidence.
        "accuracy": None,
        "final_sufficiency": _mean(suff),
        "turn_to_sufficiency": _mean(tts),
        "avg_turns": m.get("avg_num_turns_used", m.get("avg_questions_used")),
        "avg_thinking_tokens": m.get("avg_thinking_tokens"),
    }


def find_and_summarize_logic(ds_dir):
    hits = glob.glob(os.path.join(ds_dir, "**", "*_results.json"), recursive=True)
    if not hits:
        return None
    return summarize_turnlog_dataset(sorted(hits)[-1], "metrics_original", "episodes_original")


def find_and_summarize_genereg(ds_dir):
    hits = glob.glob(os.path.join(ds_dir, "**", "*_results.json"), recursive=True)
    if not hits:
        return None
    return summarize_turnlog_dataset(sorted(hits)[-1], "metrics", "episodes")


# ── gsme: set-containment sufficiency ────────────────────────────────────────
def _gsme_turn_to_sufficiency(ep):
    """Turn index by which every held-out variable was asked (nan if never)."""
    need = set(ep.get("var_gt") or [])
    if not need:
        return float("nan")
    got = set()
    for t in ep.get("turn_logs") or []:
        for v in (t.get("asked_used") or t.get("asked_vars_raw") or []):
            got.add(v)
        if need <= got:
            return float(t.get("turn", 0) + 1)
    return float("nan")


def find_and_summarize_gsme(ds_dir):
    eps = glob.glob(os.path.join(ds_dir, "*.episodes.jsonl"))
    summ = glob.glob(os.path.join(ds_dir, "*_summary.json"))
    if not eps:
        return None
    episodes = [json.loads(l) for l in open(sorted(eps)[-1])]

    suff, tts, turns, acc, think = [], [], [], [], []
    for e in episodes:
        is_suff = float(e.get("var_recall") or 0.0) >= 1.0
        suff.append(1.0 if is_suff else 0.0)
        if is_suff:
            tts.append(_gsme_turn_to_sufficiency(e))
        turns.append(e.get("turns_used"))
        acc.append(1.0 if e.get("ans_correct") else 0.0)

    tt = None
    if summ:
        s = json.load(open(sorted(summ)[-1]))
        tt = s.get("metrics", {}).get("thinking_tokens", {})
        tt = tt.get("mean") if isinstance(tt, dict) else None
    else:
        tt = None

    return {
        "n_episodes": len(episodes),
        "accuracy": _mean(acc),
        "final_sufficiency": _mean(suff),
        "turn_to_sufficiency": _nanmean(tts),
        "avg_turns": _mean(turns),
        "avg_thinking_tokens": tt,
    }


# ── 20q: game outcome + offline question quality ────────────────────────────
def find_and_summarize_20q(ds_dir):
    hits = glob.glob(os.path.join(ds_dir, "**", "*.json"), recursive=True)
    # run.py writes three files per run: the dialogue log, a *_cot.json trace,
    # and a *_metrics.json summary. We want the dialogue log (the JSON array).
    hits = [h for h in hits
            if not any(s in os.path.basename(h) for s in ("_cot", "_metrics"))
            and f"{os.sep}offline_evaluation{os.sep}" not in h]
    if not hits:
        return None

    # A full suite run writes one dialogue log for Common and one for THING200.
    # Keep the newest completed log for each pool, then aggregate both.
    latest_by_pool = {}
    for path in hits:
        candidate = json.load(open(path))
        if not isinstance(candidate, list) or not candidate:
            continue
        pool = candidate[0].get("run_config", {}).get("dataset")
        key = pool or path
        previous = latest_by_pool.get(key)
        if previous is None or os.path.getmtime(path) > os.path.getmtime(previous[0]):
            latest_by_pool[key] = (path, candidate)

    data = []
    for _, candidate in latest_by_pool.values():
        data.extend(candidate)
    if not data:
        return None
    n = len(data)
    succ = [d for d in data if d.get("state") == 1]

    offline_dir = os.path.join(ds_dir, "offline_evaluation")
    offline_hits = glob.glob(os.path.join(offline_dir, "question_quality_*.jsonl"))
    if not offline_hits:  # backward compatibility with pre-pool output
        offline_hits = glob.glob(os.path.join(offline_dir, "question_quality.jsonl"))
    offline_rows = []
    for path in sorted(offline_hits):
        with open(path) as f:
            offline_rows.extend(json.loads(line) for line in f if line.strip())

    eig = [row.get("summary", {}).get("avg_normalized_eig") for row in offline_rows]
    final_entropy = [row.get("summary", {}).get("final_entropy") for row in offline_rows]
    return {
        "n_episodes": n,
        "accuracy": len(succ) / n if n else None,
        "final_sufficiency": None,   # not defined for 20Q
        # turns over ALL episodes (a failed game runs to its turn cap), so the
        # number is meaningful even when nothing was solved.
        "turn_to_sufficiency": _mean([d.get("turn") for d in succ]) if succ else None,
        "avg_turns": _mean([d.get("turn") for d in data]),
        "avg_thinking_tokens": None,
        "avg_question_eig": _mean(eig),
        "avg_final_entropy": _mean(final_entropy),
        "offline_n_episodes": len(offline_rows) if offline_rows else None,
    }


SUMMARIZERS = {
    "logic_q_mt": find_and_summarize_logic,
    "genereg_mt": find_and_summarize_genereg,
    "gsme_q_mt": find_and_summarize_gsme,
    "gsme_q_mt_ext": find_and_summarize_gsme,
    "20q": find_and_summarize_20q,
}

COLUMNS = [
    ("n_episodes", "N", "{:d}"),
    ("accuracy", "accuracy", "{:.3f}"),
    ("final_sufficiency", "final-suff", "{:.3f}"),
    ("turn_to_sufficiency", "turns-to-suff", "{:.2f}"),
    ("avg_turns", "avg-turns", "{:.2f}"),
    ("avg_thinking_tokens", "think-tok", "{:.0f}"),
    ("avg_question_eig", "q-EIG", "{:.3f}"),
    ("avg_final_entropy", "final-H", "{:.3f}"),
]


def fmt(val, spec):
    if val is None:
        return "-"
    try:
        return spec.format(val)
    except (ValueError, TypeError):
        return str(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--datasets", default="all")
    args = ap.parse_args()

    datasets = ALL if args.datasets.strip() == "all" else [
        d.strip() for d in args.datasets.split(",") if d.strip()]

    rows = {}
    for ds in datasets:
        ds_dir = os.path.join(args.results_dir, ds)
        try:
            rows[ds] = SUMMARIZERS[ds](ds_dir) if os.path.isdir(ds_dir) else None
        except Exception as exc:  # never let one bad file sink the summary
            print(f"  [warn] {ds}: could not summarize ({type(exc).__name__}: {exc})")
            rows[ds] = None

    # Markdown table.
    header = "| dataset | " + " | ".join(h for _, h, _ in COLUMNS) + " |"
    sep = "|" + "|".join(["---"] * (len(COLUMNS) + 1)) + "|"
    lines = [header, sep]
    for ds in datasets:
        r = rows.get(ds)
        if r is None:
            lines.append(f"| {ds} | " + " | ".join(["_no results_"] + [""] * (len(COLUMNS) - 1)) + " |")
            continue
        cells = [fmt(r.get(key), spec) for key, _, spec in COLUMNS]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    print("\n" + table + "\n")
    print("final-suff = fraction of episodes where enough info was obtained; "
          "turns-to-suff over that subset. Logic/GeneReg report final-suff, not "
          "answer accuracy. q-EIG/final-H come from 20Q offline evaluation.")

    out = os.path.join(args.results_dir, "summary.json")
    with open(out, "w") as f:
        json.dump({"results_dir": args.results_dir, "datasets": rows}, f, indent=2)
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(f"\nWrote {out} and summary.md")


if __name__ == "__main__":
    main()
