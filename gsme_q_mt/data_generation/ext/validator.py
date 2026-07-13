from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from graph_utils import nx

try:  # pragma: no cover
    import sympy as sp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    sp = None


def validate_sample(
    sample: Mapping[str, Any],
    requested_num_vars: int,
    requested_num_rules: int,
    requested_depth: int,
    tolerance_vars: int = 3,
    tolerance_rules: int = 2,
    tolerance_depth: int = 0,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    graph = nx.DiGraph()
    for node in sample["graph"]["nodes"]:
        graph.add_node(node["id"])
    for edge in sample["graph"]["edges"]:
        graph.add_edge(edge["source"], edge["target"])

    if not nx.is_directed_acyclic_graph(graph):
        errors.append("graph is not acyclic")

    goal_var = sample["goal_var"]
    if goal_var not in graph.nodes:
        errors.append("goal var missing from graph")
    else:
        ancestors = nx.ancestors(graph, goal_var)
        leaf_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]
        if not any(node in ancestors or node == goal_var for node in leaf_nodes):
            errors.append("goal not reachable from any leaf")

    if abs(sample["num_vars"] - requested_num_vars) > tolerance_vars:
        errors.append("num_vars too far from requested target")
    if abs(sample["num_rules"] - requested_num_rules) > tolerance_rules:
        errors.append("num_rules too far from requested target")
    if abs(sample["depth"] - requested_depth) > tolerance_depth:
        errors.append("depth mismatch")

    try:
        computed_values = _recompute_values(sample)
    except Exception as exc:
        errors.append(f"value recomputation failed: {exc}")
    else:
        for name, value in computed_values.items():
            if sample["pred_values"][name] != value:
                errors.append(f"value mismatch for {name}")
                break

    if set(sample["distractor_vars"]) & (set(nx.ancestors(graph, goal_var)) | {goal_var}):
        errors.append("distractor vars overlap with goal ancestry")

    try:
        uniquely_determined = _unique_target_via_sympy(sample)
    except Exception as exc:
        errors.append(f"symbolic validation failed: {exc}")
    else:
        if not uniquely_determined:
            errors.append("target is not uniquely determined by known values and equations")

    return not errors, errors


def _recompute_values(sample: Mapping[str, Any]) -> Dict[str, int]:
    values: Dict[str, int] = {
        name: sample["pred_values"][name] for name in sample["known_variables"]
    }
    remaining = {rule["output"]: rule for rule in sample["equations"]}
    stalled = False
    while remaining and not stalled:
        stalled = True
        for output, rule in list(remaining.items()):
            if all(parent in values for parent in rule["inputs"]):
                x = values[rule["inputs"][0]]
                y = values[rule["inputs"][1]] if len(rule["inputs"]) > 1 else None
                rule_type = rule["rule_type"]
                constant = rule["constant"]
                if rule_type == "add_xy":
                    values[output] = x + int(y)
                elif rule_type == "sub_xy":
                    values[output] = x - int(y)
                elif rule_type == "mul_k":
                    values[output] = x * int(constant)
                elif rule_type == "div_k":
                    values[output] = x // int(constant)
                elif rule_type == "add_k":
                    values[output] = x + int(constant)
                elif rule_type == "sub_k":
                    values[output] = x - int(constant)
                elif rule_type == "sum_mul_k":
                    values[output] = (x + int(y)) * int(constant)
                else:
                    raise ValueError(f"unknown rule type: {rule_type}")
                del remaining[output]
                stalled = False
    if remaining:
        missing = ", ".join(sorted(remaining))
        raise ValueError(f"could not recompute all variables: {missing}")
    return values


def _unique_target_via_sympy(sample: Mapping[str, Any]) -> bool:
    if sp is None:
        return True
    variables = {name: sp.Symbol(name, integer=True) for name in sample["all_variables"]}
    equations = []
    for known in sample["known_variables"]:
        equations.append(sp.Eq(variables[known], sample["pred_values"][known]))
    for rule in sample["equations"]:
        output = variables[rule["output"]]
        inputs = [variables[item] for item in rule["inputs"]]
        constant = rule["constant"]
        rule_type = rule["rule_type"]
        if rule_type == "add_xy":
            equations.append(sp.Eq(output, inputs[0] + inputs[1]))
        elif rule_type == "sub_xy":
            equations.append(sp.Eq(output, inputs[0] - inputs[1]))
        elif rule_type == "mul_k":
            equations.append(sp.Eq(output, inputs[0] * int(constant)))
        elif rule_type == "div_k":
            equations.append(sp.Eq(output * int(constant), inputs[0]))
        elif rule_type == "add_k":
            equations.append(sp.Eq(output, inputs[0] + int(constant)))
        elif rule_type == "sub_k":
            equations.append(sp.Eq(output, inputs[0] - int(constant)))
        elif rule_type == "sum_mul_k":
            equations.append(sp.Eq(output, (inputs[0] + inputs[1]) * int(constant)))

    solution = sp.solve(equations, list(variables.values()), dict=True)
    if len(solution) != 1:
        return False
    solved_value = solution[0].get(variables[sample["goal_var"]])
    return solved_value == sample["answer"]
