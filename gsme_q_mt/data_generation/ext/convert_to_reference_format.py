from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REFERENCE_COLUMNS = [
    "sample_id",
    "problem_id",
    "Full Problem",
    "CSP",
    "Full Answer",
    "Variables",
    "Equations",
    "depth",
    "Pred Values",
    "Heldout Value",
    "Rewritten Problem",
    "Possible Questions",
    "Given_Conditions",
    "k",
    "diff_score",
    "goal_var",
    "leaf_nodes_all",
    "relevant_leaf",
    "ancestors_goal",
    "dist_max",
    "dist_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert generated gsme_q_mt_ext.csv into a reference-style 21-column CSV."
    )
    parser.add_argument("--input", required=True, help="Input generated CSV path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <input>.reference.csv",
    )
    parser.add_argument(
        "--drop_unassigned_in_rewritten",
        action="store_true",
        help="Drop bare variable lines from the Variables block of Rewritten Problem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".reference.csv")

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [convert_row(dict(row), args.drop_unassigned_in_rewritten) for row in reader]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote converted file: {output_path}")
    print(f"Converted rows: {len(rows)}")


def convert_row(row: Dict[str, Any], drop_unassigned_in_rewritten: bool) -> Dict[str, Any]:
    normalized = {key: normalize_field(value) for key, value in row.items()}

    full_problem = str(normalized.get("Full Problem", ""))
    pred_values_text = str(normalized.get("Pred Values", ""))
    pred_values = parse_pred_values(pred_values_text)
    goal_var = str(normalized.get("goal_var", infer_goal_var(full_problem)))

    leaf_nodes_all = ensure_list(normalized.get("leaf_nodes_all"))
    relevant_leaf = ensure_list(normalized.get("relevant_leaf"))
    heldout = ensure_list(normalized.get("Heldout Value"))
    variables_payload = ensure_dict(normalized.get("Variables"))
    equations_payload = ensure_dict(normalized.get("Equations"))

    reference_given_conditions = sorted(
        [var for var in relevant_leaf if var not in set(heldout)],
        key=sort_key,
    )
    reference_possible_questions = sorted(leaf_nodes_all, key=sort_key)

    variable_descs = build_variable_descriptions(
        all_variables=extract_variable_order(full_problem, pred_values, variables_payload),
        variable_roles=variables_payload,
        pred_values=pred_values,
        goal_var=goal_var,
    )
    equation_descs = build_equation_descriptions(equations_payload)

    csp_text = build_reference_csp(
        variable_descs=variable_descs,
        pred_values=pred_values,
        known_value_vars=leaf_nodes_all,
        equations_payload=equations_payload,
        equation_descs=equation_descs,
        goal_var=goal_var,
        full_problem=full_problem,
    )
    rewritten_problem = reconstruct_problem_text_from_csp(
        csp_text=csp_text,
        given_vars=reference_given_conditions,
        value_dict=pred_values,
        drop_unassigned=drop_unassigned_in_rewritten,
    )

    converted = {
        "sample_id": normalized.get("sample_id", ""),
        "problem_id": normalized.get("problem_id", ""),
        "Full Problem": full_problem,
        "CSP": csp_text,
        "Full Answer": stringify_scalar(normalized.get("Full Answer", normalized.get("answer", ""))),
        "Variables": json.dumps(variable_descs, ensure_ascii=False),
        "Equations": json.dumps(equation_descs, ensure_ascii=False),
        "depth": stringify_scalar(normalized.get("depth", "")),
        "Pred Values": pred_values_text,
        "Heldout Value": json.dumps(heldout, ensure_ascii=False),
        "Rewritten Problem": rewritten_problem,
        "Possible Questions": json.dumps(reference_possible_questions, ensure_ascii=False),
        "Given_Conditions": json.dumps(reference_given_conditions, ensure_ascii=False),
        "k": stringify_scalar(normalized.get("k", "")),
        "diff_score": stringify_scalar(normalized.get("diff_score", "")),
        "goal_var": goal_var,
        "leaf_nodes_all": json.dumps(leaf_nodes_all, ensure_ascii=False),
        "relevant_leaf": json.dumps(relevant_leaf, ensure_ascii=False),
        "ancestors_goal": json.dumps(ensure_list(normalized.get("ancestors_goal")), ensure_ascii=False),
        "dist_max": stringify_scalar(normalized.get("dist_max", "")),
        "dist_mean": stringify_scalar(normalized.get("dist_mean", "")),
    }
    return converted


def normalize_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "[{" and stripped[-1] in "]}":
        try:
            return json.loads(stripped)
        except Exception:
            try:
                return ast.literal_eval(stripped)
            except Exception:
                return value
    return value


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except Exception:
            try:
                loaded = ast.literal_eval(stripped)
                if isinstance(loaded, (list, tuple, set)):
                    return [str(item) for item in loaded]
            except Exception:
                return [stripped]
    return [str(value)]


def ensure_dict(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): str(val) for key, val in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, dict):
                return {str(key): str(val) for key, val in loaded.items()}
        except Exception:
            try:
                loaded = ast.literal_eval(stripped)
                if isinstance(loaded, dict):
                    return {str(key): str(val) for key, val in loaded.items()}
            except Exception:
                return {}
    return {}


def stringify_scalar(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_pred_values(pred_values_text: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for line in pred_values_text.splitlines():
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        key = lhs.strip()
        raw = rhs.strip()
        try:
            numeric = float(raw)
            values[key] = int(numeric) if numeric.is_integer() else numeric
        except Exception:
            values[key] = raw
    return values


def infer_goal_var(full_problem: str) -> str:
    match = re.search(r"What is ([A-Za-z0-9_]+)\?", full_problem)
    return match.group(1) if match else ""


def extract_variable_order(
    full_problem: str,
    pred_values: Mapping[str, Any],
    variables_payload: Mapping[str, str],
) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for line in full_problem.splitlines():
        if "=" not in line:
            continue
        lhs = line.split("=", 1)[0].strip()
        if lhs not in seen:
            ordered.append(lhs)
            seen.add(lhs)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", line.split("=", 1)[1]):
            if token not in seen and not token.isnumeric():
                ordered.append(token)
                seen.add(token)
    for source in [pred_values.keys(), variables_payload.keys()]:
        for key in source:
            if key not in seen:
                ordered.append(str(key))
                seen.add(key)
    return ordered


def build_variable_descriptions(
    all_variables: Sequence[str],
    variable_roles: Mapping[str, str],
    pred_values: Mapping[str, Any],
    goal_var: str,
) -> Dict[str, str]:
    descs: Dict[str, str] = {}
    for var in sorted(all_variables, key=sort_key):
        role = variable_roles.get(var, "")
        if var == goal_var or role == "goal":
            descs[var] = f"Target quantity for {var}"
        elif role == "known_leaf":
            descs[var] = f"Known input quantity {var}"
        elif role == "known_leaf_distractor":
            descs[var] = f"Known distractor quantity {var}"
        elif role == "derived":
            descs[var] = f"Intermediate quantity {var}"
        elif role == "derived_distractor":
            descs[var] = f"Distractor intermediate quantity {var}"
        else:
            descs[var] = f"Quantity {var}"
    return descs


def build_equation_descriptions(equations_payload: Mapping[str, str]) -> Dict[str, str]:
    descs: Dict[str, str] = {}
    for expression in equations_payload:
        descs[expression] = describe_equation(expression, equations_payload[expression])
    return descs


def describe_equation(expression: str, rule_type: str) -> str:
    compact = compact_expression(expression)
    if "=" not in compact:
        return f"{compact}."
    lhs, rhs = [part.strip() for part in compact.split("=", 1)]
    if rule_type == "add_xy":
        a, b = [item.strip() for item in rhs.split("+", 1)]
        return f"{lhs} is the sum of {a} and {b}."
    if rule_type == "sub_xy":
        a, b = [item.strip() for item in rhs.split("-", 1)]
        return f"{lhs} is {a} minus {b}."
    if rule_type == "mul_k":
        a, k = [item.strip() for item in rhs.split("*", 1)]
        return f"{lhs} is {a} multiplied by {k}."
    if rule_type == "div_k":
        a, k = [item.strip() for item in rhs.split("/", 1)]
        return f"{lhs} is {a} divided by {k}."
    if rule_type == "add_k":
        a, k = [item.strip() for item in rhs.split("+", 1)]
        return f"{lhs} is {k} more than {a}."
    if rule_type == "sub_k":
        a, k = [item.strip() for item in rhs.split("-", 1)]
        return f"{lhs} is {k} less than {a}."
    if rule_type == "sum_mul_k":
        match = re.match(r"\((.+)\)\s*\*\s*(.+)", rhs)
        if match:
            summed, k = match.groups()
            a, b = [item.strip() for item in summed.split("+", 1)]
            return f"{lhs} is {k.strip()} times the sum of {a} and {b}."
    return f"{lhs} is derived using rule type {rule_type}."


def build_reference_csp(
    variable_descs: Mapping[str, str],
    pred_values: Mapping[str, Any],
    known_value_vars: Sequence[str],
    equations_payload: Mapping[str, str],
    equation_descs: Mapping[str, str],
    goal_var: str,
    full_problem: str,
) -> str:
    lines: List[str] = ["Variables:"]
    known_set = set(known_value_vars)
    for var in sorted(variable_descs, key=sort_key):
        desc = variable_descs[var]
        if var in pred_values and var in known_set:
            lines.append(f"{var} = {format_value(pred_values[var])} [{desc}]")
        else:
            lines.append(f"{var} [{desc}]")
    lines.append("")
    lines.append("Equations:")
    for expression in order_equations(equations_payload.keys(), full_problem):
        lines.append(f"{compact_expression(expression)} [{equation_descs[expression]}]")
    lines.append("")
    lines.append("Goal:")
    goal_desc = variable_descs.get(goal_var, f"Target quantity for {goal_var}")
    lines.append(f"{goal_var} [{goal_desc}]")
    return "\n".join(lines)


def reconstruct_problem_text_from_csp(
    csp_text: str,
    given_vars: Sequence[str],
    value_dict: Mapping[str, Any],
    drop_unassigned: bool,
) -> str:
    given_set = set(given_vars)
    lines = csp_text.splitlines()
    out: List[str] = []
    section = None
    var_with_desc = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*(?:=\s*([^\[]+))?\s*\[(.+)\]\s*$")
    for raw in lines:
        stripped = raw.strip()
        if stripped == "Variables:":
            section = "vars"
            out.append("Variables:")
            continue
        if stripped == "Equations:":
            section = "eqs"
            out.append("Equations:")
            continue
        if stripped == "Goal:":
            section = "goal"
            out.append("Goal:")
            continue
        if section == "vars":
            if not stripped:
                out.append("")
                continue
            match = var_with_desc.match(stripped)
            if not match:
                continue
            var = match.group(1)
            if var in given_set and var in value_dict:
                out.append(f"{var} = {format_value(value_dict[var])}")
            elif not drop_unassigned:
                out.append(var)
            continue
        if section == "eqs":
            if not stripped:
                out.append("")
                continue
            if "[" in stripped and stripped.endswith("]"):
                out.append(stripped.rsplit("[", 1)[0].rstrip())
            else:
                out.append(stripped)
            continue
        if section == "goal":
            if not stripped:
                continue
            if "[" in stripped and stripped.endswith("]"):
                out.append(stripped.rsplit("[", 1)[0].rstrip())
            else:
                out.append(stripped)
            continue
    return "\n".join(compact_blank_lines(out)).strip()


def compact_blank_lines(lines: Sequence[str]) -> List[str]:
    out: List[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        out.append(line)
        prev_blank = is_blank
    return out


def compact_expression(expression: str) -> str:
    left, right = [part.strip() for part in expression.split("=", 1)]
    right = re.sub(r"\s+", " ", right)
    right = right.replace(" * ", "*").replace(" / ", "/")
    right = right.replace(" + ", " + ").replace(" - ", " - ")
    right = right.replace("( ", "(").replace(" )", ")")
    return f"{left} = {right}"


def order_equations(expressions: Iterable[str], full_problem: str) -> List[str]:
    ordered: List[str] = []
    expression_set = set(expressions)
    for line in full_problem.splitlines():
        stripped = line.strip()
        if "=" not in stripped or stripped.startswith("What is "):
            continue
        if stripped in expression_set:
            ordered.append(stripped)
    remaining = [expr for expr in expressions if expr not in set(ordered)]
    ordered.extend(sorted(remaining, key=lambda item: sort_key(item.split("=", 1)[0].strip())))
    return ordered


def format_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def sort_key(name: str) -> Tuple[str, int]:
    prefix = "".join(ch for ch in name if ch.isalpha())
    digits = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(digits or 0)


if __name__ == "__main__":
    main()
