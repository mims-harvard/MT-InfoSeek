# horn_sat_utils.py
# Shared utilities for Horn CNF parsing, unit propagation, and semantic "Known" via refutation.
# Designed to be imported by make_data_new.py, holdout_utils_new.py, verify_results.py.

from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

Literal = Tuple[str, bool]           # (var, value)
Clause = Set[Literal]               # set of literals
Clauses = List[Clause]              # CNF as list of clauses

CONTRADICTION = "CONTRADICTION"
INFEASIBLE = "INFEASIBLE"


def negate_lit(lit: str) -> str:
    """Toggle 'x' <-> 'not x'."""
    return lit[4:] if lit.startswith("not ") else f"not {lit}"


def parse_clauses(rules_obj: Any) -> Clauses:
    """
    Accepts:
      - RuleTree-like object with .serialize() -> list[list[str]]
      - CNF list[list[str]]
      - stringified CNF list[list[str]]
    Returns: list[set[(var, bool)]]
    """
    if hasattr(rules_obj, "serialize") and callable(getattr(rules_obj, "serialize")):
        raw_rules = rules_obj.serialize()
    elif isinstance(rules_obj, str):
        raw_rules = ast.literal_eval(rules_obj)
    else:
        raw_rules = rules_obj

    clauses: Clauses = []
    for rule in raw_rules:
        clause: Clause = set()
        for lit in rule:
            if lit.startswith("not "):
                clause.add((lit[4:], False))
            else:
                clause.add((lit, True))
        clauses.append(clause)
    return clauses


def facts_to_assignment(full_facts: Iterable[str]) -> Union[Dict[str, bool], str]:
    """
    Convert {'a','not b'} to {'a':True,'b':False}.
    Returns CONTRADICTION if both a and not a appear.
    """
    assignment: Dict[str, bool] = {}
    for f in full_facts:
        if f.startswith("not "):
            var, val = f[4:], False
        else:
            var, val = f, True
        if var in assignment and assignment[var] != val:
            return CONTRADICTION
        assignment[var] = val
    return assignment


def solve_unit_prop(clauses: Clauses, context: Dict[str, bool]) -> Union[Dict[str, bool], str]:
    """
    Unit propagation with empty-clause contradiction detection.
    Returns closure assignment dict or CONTRADICTION.
    """
    assignment = dict(context)

    while True:
        changed = False
        for clause in clauses:
            satisfied = False
            unknown_lits: List[Literal] = []
            false_lits_count = 0

            for var, val in clause:
                if var in assignment:
                    if assignment[var] == val:
                        satisfied = True
                        break
                    else:
                        false_lits_count += 1
                else:
                    unknown_lits.append((var, val))

            if satisfied:
                continue

            # All literals false => empty clause => contradiction
            if false_lits_count == len(clause):
                return CONTRADICTION

            # Unit clause => infer the remaining literal
            if len(unknown_lits) == 1 and false_lits_count == len(clause) - 1:
                var, val = unknown_lits[0]
                if var not in assignment:
                    assignment[var] = val
                    changed = True
                elif assignment[var] != val:
                    return CONTRADICTION

        if not changed:
            break

    return assignment


def infer_closure(clauses: Clauses, full_facts: Iterable[str]) -> Tuple[bool, Dict[str, bool]]:
    """
    Returns (is_contradiction, closure_assignment_dict).
    """
    base = facts_to_assignment(full_facts)
    if base == CONTRADICTION:
        return True, {}
    res = solve_unit_prop(clauses, base)
    if res == CONTRADICTION:
        return True, {}
    return False, res


def forced_value_via_refutation(clauses: Clauses, base_ctx: Dict[str, bool], var: str, return_base_res: bool = False):
    """
    Semantic Known(var | base_ctx) for Horn CNF using two UP-based satisfiability checks:
      - var forced True  iff (base_ctx ∧ var=False) is UNSAT
      - var forced False iff (base_ctx ∧ var=True) is UNSAT
    Returns:
      True/False if forced, None if not forced, INFEASIBLE if base_ctx itself UNSAT.
    """
    base_res = solve_unit_prop(clauses, base_ctx)
    if base_res == CONTRADICTION:
        return (INFEASIBLE, {}) if return_base_res else INFEASIBLE

    sat_true = solve_unit_prop(clauses, {**base_res, var: True}) != CONTRADICTION
    sat_false = solve_unit_prop(clauses, {**base_res, var: False}) != CONTRADICTION

    if not sat_true and not sat_false:
        return (INFEASIBLE, base_res) if return_base_res else INFEASIBLE
    if sat_true and not sat_false:
        return (True, base_res) if return_base_res else True
    if sat_false and not sat_true:
        return (False, base_res) if return_base_res else False
    return (None, base_res) if return_base_res else None


def forced_value_from_facts(clauses: Clauses, full_facts: Iterable[str] | Dict[str, bool], var: str) -> Union[bool, None, str]:
    """
    Same as forced_value_via_refutation, but takes 'not x' fact format.
    """
    base = full_facts if isinstance(full_facts, dict) else facts_to_assignment(full_facts)
    if base == CONTRADICTION:
        return INFEASIBLE
    return forced_value_via_refutation(clauses, base, var)


def get_inferrable_var_values(clauses: Clauses, full_facts: Iterable[str], valid_vars: Iterable[str], keep_fact_vars: bool = False, use_int: bool = True) -> Dict[str, bool]:
    """Force a truth value for each variable in `valid_vars` given `full_facts`.

    Returns a dict over `valid_vars` (minus the fact variables unless
    `keep_fact_vars`). With `use_int` (default) the values are 1 / 0 for forced
    True / False and 2 for undetermined; otherwise True / False / None.
    """
    base = facts_to_assignment(full_facts)
    if base == CONTRADICTION:
        raise ValueError("full_facts + clauses lead to contradiction")

    all_vars: Set[str] = set()
    if valid_vars is not None:
        all_vars.update(set(valid_vars))
    else:
        for clause in clauses:
            for var, _ in clause:
                all_vars.add(var)
    if not keep_fact_vars:
        all_vars.difference_update(base.keys())

    inferrable: Dict[str, bool] = {}
    for var in all_vars:
        val = forced_value_via_refutation(clauses, base, var)
        if val == INFEASIBLE:
            raise ValueError(f"Variable {var} is infeasible under full_facts and clauses")
        if use_int:
            inferrable[var] = int(val) if val is not None else 2
        else:
            inferrable[var] = val
            
    return inferrable
