import ast
import collections
import itertools
import os
from typing import List
from SimpleLogic.horn_sat_utils import parse_clauses, forced_value_from_facts, solve_unit_prop

DATA_DIR = os.environ.get("DATA_DIR", "")


def verify_row(data, counter, verbose=True):
    clauses = parse_clauses(data['rules'])

    known_true = ast.literal_eval(data.get('known_facts', '[]'))
    known_false = ast.literal_eval(data.get('known_untrue_facts', '[]')) if 'known_untrue_facts' in data else []

    context = {k: True for k in known_true}
    for k in known_false:
        context[k] = False

    goal = data['goal']
    gt_qs_list = ast.literal_eval(data['gt_qs'])
    if isinstance(gt_qs_list[0], str):
        gt_qs_list = [[gt_q] for gt_q in gt_qs_list]  # Compatible with original questbench k=1 format

    # Check 1: Context underspecified (goal should not already be known)
    base_forced = forced_value_from_facts(clauses, context, goal)
    if base_forced in (True, False):
        if verbose:
            print("❌ FAIL: Goal is already known from context.")
        if data["k"] not in counter['failed (goal inferred from context)'].keys():
            counter['failed (goal inferred from context)'][data["k"]] = 0
        counter['failed (goal inferred from context)'][data["k"]] += 1
        return False

    def build_table(qs):
        """
        Return (table, any_consistent). table maps answer tuples -> goal boolean.
        """
        table = {}
        any_consistent = False
        for answers in itertools.product([True, False], repeat=len(qs)):
            hyp_context = context.copy()
            hyp_context.update(zip(qs, answers))
            if solve_unit_prop(clauses, hyp_context) == "CONTRADICTION":
                continue
            any_consistent = True  # at least one consistent branch

            forced = forced_value_from_facts(clauses, hyp_context, goal)
            if forced == "INFEASIBLE":
                raise ValueError("Weird inconsistency between checkers")
                # continue  # if both are goals leads to contradiction; should be rare if hyp_context was satisfiable, but safe
            if forced is None:
                return None, True  # insufficient: goal not semantically determined in this branch

            table[answers] = forced  # this branch leads to a deterministic goal value
        return table, any_consistent

    def subset_is_sufficient(subset):
        """
        A subset is sufficient only if:
        1. No consistent branch returned forced is None (i.e., goal is not determined)
        2. There exists at least one consistent branch (any_consistent=True)
        """
        table, any_consistent = build_table(subset)
        if table is None:
            return False  # insufficient (some consistent branch doesn't determine)
        return any_consistent  # sufficient iff there exists at least one consistent branch and all determine

    def essentiality_holds(table, qs):
        k = len(qs)
        for i in range(k):
            essential = False
            for others in itertools.product([True, False], repeat=k-1):
                t0 = []
                t1 = []
                j = 0
                for p in range(k):
                    if p == i:
                        t0.append(False)
                        t1.append(True)
                    else:
                        t0.append(others[j])
                        t1.append(others[j])
                        j += 1
                t0 = tuple(t0)
                t1 = tuple(t1)
                if t0 in table and t1 in table and table[t0] != table[t1]:
                    essential = True
                    break
            if not essential:
                return False
        return True

    for qs in gt_qs_list:
        # Check 2: Sufficiency (all consistent branches must determine goal)
        table, any_consistent = build_table(qs)
        if table is None:
            if verbose:
                print(f"❌ FAIL: Some consistent branch for qs={qs} does not determine the goal.")
            if data["k"] not in counter['failed (insufficient branch)'].keys():
                counter['failed (insufficient branch)'][data["k"]] = 0
            counter['failed (insufficient branch)'][data["k"]] += 1
            return False
        if not any_consistent:
            if verbose:
                print(f"❌ FAIL: All branches for qs={qs} are contradictions (degenerate problem).")
            if data["k"] not in counter['failed (insufficient branch)'].keys():
                counter['failed (insufficient branch)'][data["k"]] = 0
            counter['failed (insufficient branch)'][data["k"]] += 1
            return False

        # Check 3: Local essentiality (each variable must be essential under consistent assignments)
        if len(qs) > 1 and not essentiality_holds(table, qs):
            if verbose:
                print(f"❌ FAIL: qs={qs} is sufficient but not minimal (violates essentiality).")
                print(f"Ground truth qs: {qs}")
                print(f"Context: {context}")
                print(f"Rules: {clauses}")
                print(f"Goal: {goal}")
                print(f"Table: {table}")
            if data["k"] not in counter['failed (not minimal)'].keys():
                counter['failed (not minimal)'][data["k"]] = 0
            counter['failed (not minimal)'][data["k"]] += 1
            return False

        # Check 4: Local minimality by subsets of size k-1
        for i in range(len(qs)):
            subset = qs[:i] + qs[i+1:]
            if subset and subset_is_sufficient(subset):
                if verbose:
                    print(f"❌ FAIL: Subset {subset} is sufficient (Not Minimal).")
                if data["k"] not in counter['failed (not minimal)'].keys():
                    counter['failed (not minimal)'][data["k"]] = 0
                counter['failed (not minimal)'][data["k"]] += 1
                return False

        # Check 5: Global minimality - no k-1 subset from entire search space is sufficient
        k = int(data['k']) if 'k' in data else len(qs)
        if k >= 1:
            # Build search space from all_valid_qs, excluding goal and known variables
            all_valid_qs = set([var[4:] if var.startswith("not ") else var for var in sum(ast.literal_eval(data["rules"]), [])]) - set(ast.literal_eval(data["known_facts"]) + ast.literal_eval(data["known_untrue_facts"]) + [data["goal"]])
            known_vars = set(context.keys())
            search_space = [q for q in all_valid_qs if q != goal and q not in known_vars]
            
            # Check all subsets of size k-1
            for subset in itertools.combinations(search_space, k - 1):
                if subset_is_sufficient(list(subset)):
                    if verbose:
                        print(f"❌ FAIL: Global minimality violated - subset {list(subset)} of size k-1 is sufficient.")
                    if data["k"] not in counter['failed (not globally minimal)'].keys():
                        counter['failed (not globally minimal)'][data["k"]] = 0
                    counter['failed (not globally minimal)'][data["k"]] += 1
                    return False

    if verbose:
        print(f"✅ Row Verified (k={data['k'] if 'k' in data else 'N/A'})")
    if data["k"] not in counter['verified'].keys():
        counter['verified'][data["k"]] = 0
    counter['verified'][data["k"]] += 1
    return True


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from argparse import ArgumentParser
    
    args = ArgumentParser()
    args.add_argument("--input_csv", type=str, default=os.path.join(DATA_DIR, "logic_q_mt.csv"), help="Path to the CSV file with results to verify.")
    
    arguments = args.parse_args()

    counter = {'verified': {1:0}, 'failed (goal inferred from context)': {1:0}, 'failed (insufficient branch)': {1:0}, 'failed (not minimal)': {1:0}, 'failed (not globally minimal)': {1:0}}
    data = pd.read_csv(arguments.input_csv)
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Verifying results"):
        print(f"Verifying row {idx}:")
        verify_row(row, counter)

    print("\nVerification Summary:")
    for key, value in counter.items():
        print(f"{key}: {value}")