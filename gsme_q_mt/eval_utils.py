import pandas as pd
import json
import re
import statistics


def safe_mean(xs):
    return float(sum(xs) / len(xs)) if xs else None

def safe_median(xs):
    try:
        return float(statistics.median(xs)) if xs else None
    except Exception:
        return None

def safe_min(xs):
    return float(min(xs)) if xs else None

def safe_max(xs):
    return float(max(xs)) if xs else None

def col_numeric(series):
    if series is None:
        return []
    vals = []
    for v in series.tolist():
        if v is None:
            continue
        try:
            if isinstance(v, bool):
                vals.append(float(v))
            else:
                fv = float(v)
                if fv == fv:
                    vals.append(fv)
        except Exception:
            pass
    return vals

def numeric_stats_from_list(vals):
    return {
        "mean": safe_mean(vals),
        "median": safe_median(vals),
        "min": safe_min(vals),
        "max": safe_max(vals),
        "count": int(len(vals)),
    }

def numeric_stats_from_col(df, col):
    if df is None or col not in df.columns:
        return None
    vals = col_numeric(df[col])
    return numeric_stats_from_list(vals)


def pred_set_size_stats(df, pred_col="pred_q"):
    pred_set_sizes = []
    if df is not None and pred_col in df.columns:
        for x in df[pred_col].tolist():
            if isinstance(x, (set, list, tuple)):
                pred_set_sizes.append(len(x))
                continue
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    continue
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        pred_set_sizes.append(len(arr))
                        continue
                except Exception:
                    pass
                # fallback:
                nums = re.findall(r"\b[0-9]+\b", s)
                if nums:
                    pred_set_sizes.append(len(set(nums)))
                    continue
                toks = re.findall(r"\b[A-Za-z]\w*\b", s)
                if toks:
                    pred_set_sizes.append(len(set(toks)))
    return {
        "avg_pred_set_size": safe_mean(pred_set_sizes) if pred_set_sizes else None,
        "median_pred_set_size": safe_median(pred_set_sizes) if pred_set_sizes else None,
        "min_pred_set_size": safe_min(pred_set_sizes) if pred_set_sizes else None,
        "max_pred_set_size": safe_max(pred_set_sizes) if pred_set_sizes else None,
        "num_pred_set_entries": int(len(pred_set_sizes)),
    }
    
    
def numeric_stats_from_series(series):
    vals = col_numeric(series)
    return numeric_stats_from_list(vals)

def get_cost_stats_from_results(df):
    # prefer per-sample columns if present
    out = {}
    if df is None:
        return {"thinking_tokens": None, "cost_usd": None}

    if "thinking_tokens" in df.columns:
        out["thinking_tokens"] = numeric_stats_from_series(df["thinking_tokens"])
        if out["thinking_tokens"] is not None:
            # also add totals for convenience
            vals = col_numeric(df["thinking_tokens"])
            out["thinking_tokens"]["total"] = float(sum(vals)) if vals else None
    else:
        out["thinking_tokens"] = None

    if "cost_usd" in df.columns:
        out["cost_usd"] = numeric_stats_from_series(df["cost_usd"])
        if out["cost_usd"] is not None:
            vals = col_numeric(df["cost_usd"])
            out["cost_usd"]["total"] = float(sum(vals)) if vals else None
    else:
        out["cost_usd"] = None

    return out

def _pick_correct_col(df, eval_mode):
    if eval_mode in ["mt_all", "mt_one"] and df is not None and "ans_correct" in df.columns:
        return "ans_correct"
    return "correct"
    
def compute_metrics(df, eval_mode):
    pred_col = "var_pred" if (eval_mode in ["mt_all","mt_one"] and df is not None and "var_pred" in df.columns) else "pred_q"
    pred_set_stats = pred_set_size_stats(df, pred_col=pred_col)
    correct_col = _pick_correct_col(df, eval_mode)

    num_rows = int(len(df)) if df is not None else 0
    num_scored = int(df[correct_col].notna().sum()) if (df is not None and correct_col in df.columns) else 0

    acc = None
    if df is not None and num_scored > 0:
        try:
            acc = float(df.loc[df[correct_col].notna(), correct_col].mean())
        except Exception:
            acc = None

    correct_count = None
    incorrect_count = None
    if df is not None and num_scored > 0:
        try:
            correct_count = int(df.loc[df[correct_col] == True].shape[0])
            incorrect_count = int(df.loc[df[correct_col] == False].shape[0])
        except Exception:
            pass

    # by depth
    acc_by_depth = {}
    depth_col = None
    for c in ["max_depth", "depth"]:
        if df is not None and c in df.columns:
            depth_col = c
            break
    if df is not None and depth_col is not None and num_scored > 0:
        try:
            tmp = df.loc[df["correct"].notna()].groupby(depth_col)["correct"].mean()
            acc_by_depth = {str(k): float(v) for k, v in tmp.to_dict().items()}
        except Exception:
            acc_by_depth = {}

    # cost + pred set
    cost_pack = get_cost_stats_from_results(df)
    # pred_set_stats = pred_set_size_stats(df)

    # oe metrics (optional)
    mt_metrics = {}
    if df is not None and eval_mode in ["mt_all", "mt_one"]:
        cols_to_summarize = [
        # answer track
        "ans_correct",
        "ans_turn",
        "ans_forced",
        "ans_early",

        # qset track (NEW)
        "qset_exact",
        "qset_jaccard",
        "qset_num_extra",
        "qset_num_missing",
        "qset_over_rate",
        "qset_under_rate",
        "qset_signed_extra_minus_missing",
        "qset_pred_size",
        "qset_gt_size",

        # cost track
        "thinking_tokens",
        "cost_usd",
        ]
        for col in cols_to_summarize:
            st = numeric_stats_from_col(df, col)
            if st is not None:
                mt_metrics[col] = st

    return {
        "num_rows_total": num_rows,
        "num_rows_scored": num_scored,
        "accuracy": acc,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy_by_depth": acc_by_depth,
        "thinking_tokens": cost_pack.get("thinking_tokens"),
        "cost_usd": cost_pack.get("cost_usd"),
        "pred_set_size": pred_set_stats,
        "mt_metrics": mt_metrics if mt_metrics else None,
    }
    
