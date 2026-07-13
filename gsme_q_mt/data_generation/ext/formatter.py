from __future__ import annotations

import json
import random
import string
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ksufficient import choose_k_sufficient_holdout


def format_sample_for_symbolic_output(
    sample: Mapping[str, Any],
    rng: random.Random,
    holdout_k: int = 1,
    holdout_candidates: int = 200,
    holdout_top_m: int = 5,
    holdout_difficulty_mode: str = "distance",
) -> Dict[str, Any]:
    alias_map = build_alias_map(sample["all_variables"])
    equations = [alias_rule(rule, alias_map) for rule in sample["equations"]]
    values = {alias_map[key]: value for key, value in sample["pred_values"].items()}
    given_conditions = [alias_map[var] for var in sample["known_variables"]]
    leaf_nodes_all = [alias_map[var] for var in sample["leaf_nodes_all"]]
    relevant_leaf = [alias_map[var] for var in sample["relevant_leaf"]]
    ancestors_goal = [alias_map[var] for var in sample["ancestors_goal"]]
    distractor_vars = [alias_map[var] for var in sample["distractor_vars"]]
    goal_var = alias_map[sample["goal_var"]]

    heldout, holdout_meta = choose_k_sufficient_holdout(
        equations=equations,
        edges=alias_graph(sample["graph"], alias_map)["edges"],
        values=values,
        relevant_leaf=relevant_leaf if relevant_leaf else given_conditions,
        goal_var=goal_var,
        k=holdout_k,
        rng=rng,
        n_candidates=holdout_candidates,
        difficulty_mode=holdout_difficulty_mode,
        top_m=holdout_top_m,
    )
    visible_given = [var for var in given_conditions if var not in heldout]

    variable_map = build_variable_map(
        all_variables=[alias_map[var] for var in sample["all_variables"]],
        given_conditions=given_conditions,
        goal_var=goal_var,
        distractor_vars=distractor_vars,
    )
    equation_map = {item["expression"]: item["rule_type"] for item in equations}

    full_problem = render_full_problem(
        known_variables=given_conditions,
        values=values,
        equations=equations,
        goal_var=goal_var,
    )
    csp = render_csp(
        all_variables=[alias_map[var] for var in sample["all_variables"]],
        known_variables=given_conditions,
        values=values,
        equations=equations,
        goal_var=goal_var,
        variable_map=variable_map,
    )
    rewritten_problem = render_rewritten_problem(
        all_variables=[alias_map[var] for var in sample["all_variables"]],
        visible_given=visible_given,
        values=values,
        equations=equations,
        goal_var=goal_var,
    )
    pred_values_text = "\n".join(
        f"{var} = {normalize_number(values[var])}"
        for var in ordered_variables_for_display(values.keys())
    )

    dist_max = holdout_meta["dist_max"]
    dist_mean = holdout_meta["dist_mean"]
    diff_score = holdout_meta["diff_score"]

    problem_id = hashlib.md5(full_problem.encode("utf-8")).hexdigest()[:12]
    output_sample_id = f"{problem_id}_k{len(heldout)}_{holdout_meta.get('holdout_rank', 0)}"

    return {
        "sample_id": output_sample_id,
        "problem_id": problem_id,
        "Full Problem": full_problem,
        "CSP": csp,
        "Full Answer": normalize_number(sample["answer"]),
        "Variables": variable_map,
        "Equations": equation_map,
        "depth": sample["depth"],
        "Pred Values": pred_values_text,
        "Heldout Value": heldout,
        "Rewritten Problem": rewritten_problem,
        "Possible Questions": leaf_nodes_all,
        "Given_Conditions": visible_given,
        "k": len(heldout),
        "diff_score": diff_score,
        "goal_var": goal_var,
        "leaf_nodes_all": leaf_nodes_all,
        "relevant_leaf": relevant_leaf,
        "ancestors_goal": ancestors_goal,
        "distractor_vars": distractor_vars,
        "dist_max": dist_max,
        "dist_mean": dist_mean,
        "num_vars": sample["num_vars"],
        "num_rules": sample["num_rules"],
        "answer": normalize_number(sample["answer"]),
        "all_variables": [alias_map[var] for var in sample["all_variables"]],
        "graph": alias_graph(sample["graph"], alias_map),
        "rule_records": equations,
        "pred_values_dict": values,
        "has_distractor": sample["has_distractor"],
        "has_merge": sample["has_merge"],
        "ksufficient_found": holdout_meta.get("ksufficient_found", False),
        "y_cand_count": holdout_meta.get("y_cand_count"),
    }


def build_alias_map(variables: Sequence[str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    counters: Dict[str, int] = {}
    for index, var in enumerate(sorted(variables, key=sort_key)):
        prefix = string.ascii_uppercase[index % len(string.ascii_uppercase)]
        counters[prefix] = counters.get(prefix, 0) + 1
        alias_map[var] = f"{prefix}{counters[prefix]}"
    return alias_map


def alias_graph(graph: Mapping[str, Any], alias_map: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                **node,
                "id": alias_map[node["id"]],
            }
            for node in graph["nodes"]
        ],
        "edges": [
            {
                "source": alias_map[edge["source"]],
                "target": alias_map[edge["target"]],
            }
            for edge in graph["edges"]
        ],
        "rules": [
            alias_rule(rule, alias_map)
            for rule in graph["rules"]
        ],
    }


def alias_rule(rule: Mapping[str, Any], alias_map: Mapping[str, str]) -> Dict[str, Any]:
    output = alias_map[rule["output"]]
    inputs = [alias_map[item] for item in rule["inputs"]]
    expression = to_alias_expression(rule, alias_map)
    return {
        "output": output,
        "inputs": inputs,
        "constant": rule["constant"],
        "rule_type": rule["rule_type"],
        "expression": expression,
    }


def to_alias_expression(rule: Mapping[str, Any], alias_map: Mapping[str, str]) -> str:
    output = alias_map[rule["output"]]
    inputs = [alias_map[item] for item in rule["inputs"]]
    constant = rule["constant"]
    rule_type = rule["rule_type"]
    if rule_type == "add_xy":
        return f"{output} = {inputs[0]} + {inputs[1]}"
    if rule_type == "sub_xy":
        return f"{output} = {inputs[0]} - {inputs[1]}"
    if rule_type == "mul_k":
        return f"{output} = {inputs[0]} * {constant}"
    if rule_type == "div_k":
        return f"{output} = {inputs[0]} / {constant}"
    if rule_type == "add_k":
        return f"{output} = {inputs[0]} + {constant}"
    if rule_type == "sub_k":
        return f"{output} = {inputs[0]} - {constant}"
    if rule_type == "sum_mul_k":
        return f"{output} = ({inputs[0]} + {inputs[1]}) * {constant}"
    raise ValueError(f"unknown rule type: {rule_type}")


def build_variable_map(
    all_variables: Sequence[str],
    given_conditions: Sequence[str],
    goal_var: str,
    distractor_vars: Sequence[str],
) -> Dict[str, str]:
    distractor_set = set(distractor_vars)
    given_set = set(given_conditions)
    variable_map: Dict[str, str] = {}
    for var in all_variables:
        if var == goal_var:
            role = "goal"
        elif var in given_set and var in distractor_set:
            role = "known_leaf_distractor"
        elif var in given_set:
            role = "known_leaf"
        elif var in distractor_set:
            role = "derived_distractor"
        else:
            role = "derived"
        variable_map[var] = role
    return variable_map


def render_full_problem(
    known_variables: Sequence[str],
    values: Mapping[str, Any],
    equations: Sequence[Mapping[str, Any]],
    goal_var: str,
) -> str:
    lines = [f"{var} = {normalize_number(values[var])}" for var in ordered_variables_for_display(known_variables)]
    lines.extend(item["expression"] for item in equations)
    lines.append(f"What is {goal_var}?")
    return "\n".join(lines)


def render_csp(
    all_variables: Sequence[str],
    known_variables: Sequence[str],
    values: Mapping[str, Any],
    equations: Sequence[Mapping[str, Any]],
    goal_var: str,
    variable_map: Mapping[str, str],
) -> str:
    lines = ["Variables:"]
    known_set = set(known_variables)
    for var in ordered_variables_for_display(all_variables):
        if var in known_set:
            lines.append(f"{var} = {normalize_number(values[var])} [{variable_map[var]}]")
        else:
            lines.append(f"{var} [{variable_map[var]}]")
    lines.append("")
    lines.append("Equations:")
    for item in equations:
        lines.append(f"{item['expression']} [{item['rule_type']}]")
    lines.append("")
    lines.append("Goal:")
    lines.append(f"{goal_var} [goal]")
    return "\n".join(lines)


def render_rewritten_problem(
    all_variables: Sequence[str],
    visible_given: Sequence[str],
    values: Mapping[str, Any],
    equations: Sequence[Mapping[str, Any]],
    goal_var: str,
) -> str:
    visible_set = set(visible_given)
    lines = ["Variables:"]
    for var in ordered_variables_for_display(all_variables):
        if var in visible_set:
            lines.append(f"{var} = {normalize_number(values[var])}")
        else:
            lines.append(var)
    lines.append("")
    lines.append("Equations:")
    for item in equations:
        lines.append(item["expression"])
    lines.append("")
    lines.append("Goal:")
    lines.append(goal_var)
    return "\n".join(lines)


def ordered_variables_for_display(items: Iterable[str]) -> List[str]:
    return sorted(items, key=sort_key)


def sort_key(name: str) -> Tuple[str, int]:
    prefix = "".join(ch for ch in name if ch.isalpha())
    digits = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(digits or 0)


def normalize_number(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

def json_ready_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(row)


def csv_ready_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat
