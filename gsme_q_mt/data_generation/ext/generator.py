from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graph_utils import nx


UNARY_RULES = ("mul_k", "div_k", "add_k", "sub_k")
BINARY_RULES = ("add_xy", "sub_xy", "sum_mul_k")


@dataclass
class RuleSpec:
    output: str
    rule_type: str
    inputs: List[str]
    constant: Optional[int]
    expr: str


class ComplexGSMEGenerator:
    """Graph-first GSME-style arithmetic generator."""

    def __init__(
        self,
        rng: random.Random,
        min_value: int = 2,
        max_value: int = 40,
        allow_distractors: bool = True,
    ) -> None:
        self.rng = rng
        self.min_value = min_value
        self.max_value = max_value
        self.allow_distractors = allow_distractors
        self._node_index = 0

    def generate_sample(
        self,
        sample_id: str,
        num_vars: int,
        num_rules: int,
        depth: int,
        domain: str,
        target_relevant_leaves: Optional[int] = None,
    ) -> Dict[str, Any]:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        max_relevant_rules = depth + max(0, depth - 1)
        target_rules = max(depth, min(num_rules, max_relevant_rules))
        if self.allow_distractors:
            target_rules = max(depth, num_rules)
        target_vars = max(target_rules + 1, num_vars)
        if target_relevant_leaves is None:
            sampled_target_relevant_leaves = min(depth + 1, max(2, min(depth + 1, depth // 2 + 2)))
        else:
            sampled_target_relevant_leaves = max(1, min(target_relevant_leaves, depth + 1))

        graph = nx.DiGraph()
        node_values: Dict[str, int] = {}
        node_depths: Dict[str, int] = {}
        rules: List[RuleSpec] = []

        main_leaf = self._new_node()
        graph.add_node(main_leaf)
        node_values[main_leaf] = self._rand_value()
        node_depths[main_leaf] = 0
        depth_buckets: Dict[int, List[str]] = {0: [main_leaf]}
        main_chain = [main_leaf]
        merge_count = 0

        rules_remaining = target_rules
        extra_budget = max(0, target_vars - (target_rules + 1))
        aux_quota = min(max(0, target_rules - depth), max(0, depth - 1))
        relevant_leaf_budget = max(0, sampled_target_relevant_leaves - 1)

        for step in range(1, depth + 1):
            remaining_steps_including_this = depth - step + 1
            remaining_aux_slots = max(0, depth - step)
            create_aux = step >= 2 and aux_quota > 0 and (
                aux_quota > remaining_aux_slots or self.rng.random() < 0.65
            )
            force_relevant_leaf = relevant_leaf_budget > 0 and (
                relevant_leaf_budget >= remaining_steps_including_this or self.rng.random() < 0.8
            )

            side_parent: Optional[str] = None
            if create_aux:
                aux_node, aux_rule, added_relevant_leafs = self._add_auxiliary_node(
                    graph=graph,
                    node_values=node_values,
                    node_depths=node_depths,
                    depth_buckets=depth_buckets,
                    desired_depth=step - 1,
                    force_new_leaf=force_relevant_leaf,
                )
                rules.append(aux_rule)
                rules_remaining -= 1
                aux_quota -= 1
                side_parent = aux_node
                relevant_leaf_budget = max(0, relevant_leaf_budget - added_relevant_leafs)
                extra_budget = max(0, extra_budget - 1)
            elif force_relevant_leaf or extra_budget > 0 or self.rng.random() < 0.5:
                side_parent = self._new_leaf(
                    graph=graph,
                    node_values=node_values,
                    node_depths=node_depths,
                    depth_buckets=depth_buckets,
                )
                if force_relevant_leaf:
                    relevant_leaf_budget = max(0, relevant_leaf_budget - 1)
                extra_budget = max(0, extra_budget - 1)

            main_prev = main_chain[-1]
            main_node = self._new_node()
            rule = self._build_rule_for_node(
                graph=graph,
                output=main_node,
                primary_parent=main_prev,
                side_parent=side_parent,
                node_values=node_values,
                node_depths=node_depths,
                desired_depth=step,
            )
            graph.add_node(main_node)
            for parent in rule.inputs:
                graph.add_edge(parent, main_node)
            node_depths[main_node] = step
            node_values[main_node] = self._evaluate_rule(rule, node_values)
            depth_buckets.setdefault(step, []).append(main_node)
            rules.append(rule)
            main_chain.append(main_node)
            rules_remaining -= 1
            if len(rule.inputs) > 1:
                merge_count += 1

        goal_var = main_chain[-1]

        while rules_remaining > 0:
            if self.allow_distractors and self.rng.random() < 0.7:
                _, distractor_rules = self._add_distractor_component(
                    graph=graph,
                    node_values=node_values,
                    node_depths=node_depths,
                    depth_buckets=depth_buckets,
                    max_rules=min(rules_remaining, 2 + self.rng.randint(0, 2)),
                )
                rules.extend(distractor_rules)
                rules_remaining -= len(distractor_rules)
            else:
                break

        if len(graph.nodes) < target_vars:
            shortfall = target_vars - len(graph.nodes)
            for _ in range(shortfall):
                self._new_leaf(
                    graph=graph,
                    node_values=node_values,
                    node_depths=node_depths,
                    depth_buckets=depth_buckets,
                )

        metadata = self._extract_metadata(
            sample_id=sample_id,
            graph=graph,
            rules=rules,
            node_values=node_values,
            node_depths=node_depths,
            goal_var=goal_var,
            merge_count=merge_count,
            requested_num_vars=num_vars,
            requested_num_rules=num_rules,
            requested_depth=depth,
            domain=domain,
            requested_target_relevant_leaves=sampled_target_relevant_leaves,
        )
        return metadata

    def _extract_metadata(
        self,
        sample_id: str,
        graph: nx.DiGraph,
        rules: Sequence[RuleSpec],
        node_values: Dict[str, int],
        node_depths: Dict[str, int],
        goal_var: str,
        merge_count: int,
        requested_num_vars: int,
        requested_num_rules: int,
        requested_depth: int,
        domain: str,
        requested_target_relevant_leaves: int,
    ) -> Dict[str, Any]:
        ancestors_goal = sorted(nx.ancestors(graph, goal_var))
        goal_scope = set(ancestors_goal) | {goal_var}
        leaf_nodes_all = sorted(node for node in graph.nodes if graph.in_degree(node) == 0)
        relevant_leaf = sorted(
            node for node in leaf_nodes_all if node in goal_scope
        )
        known_variables = leaf_nodes_all[:]
        distractor_vars = sorted(node for node in graph.nodes if node not in goal_scope)
        rule_by_output = {rule.output: rule for rule in rules}
        equations = [
            {
                "output": rule.output,
                "rule_type": rule.rule_type,
                "inputs": rule.inputs,
                "constant": rule.constant,
                "expression": rule.expr,
            }
            for rule in rules
        ]
        possible_questions = sorted(
            node for node in graph.nodes if node not in set(known_variables) and node != goal_var
        )
        graph_payload = {
            "nodes": [
                {
                    "id": node,
                    "depth": node_depths[node],
                    "value": node_values[node],
                    "is_leaf": graph.in_degree(node) == 0,
                    "is_goal": node == goal_var,
                }
                for node in graph.nodes
            ],
            "edges": [{"source": u, "target": v} for u, v in graph.edges],
            "rules": equations,
        }

        return {
            "sample_id": sample_id,
            "domain": domain,
            "goal_var": goal_var,
            "answer": node_values[goal_var],
            "num_vars": graph.number_of_nodes(),
            "num_rules": len(rules),
            "depth": max(node_depths[node] for node in goal_scope),
            "requested_num_vars": requested_num_vars,
            "requested_num_rules": requested_num_rules,
            "requested_depth": requested_depth,
            "requested_target_relevant_leaves": requested_target_relevant_leaves,
            "all_variables": sorted(graph.nodes),
            "equations": equations,
            "equation_map": {key: value.expr for key, value in rule_by_output.items()},
            "known_variables": known_variables,
            "leaf_nodes_all": leaf_nodes_all,
            "relevant_leaf": relevant_leaf,
            "distractor_vars": distractor_vars,
            "ancestors_goal": ancestors_goal,
            "has_distractor": bool(distractor_vars),
            "has_merge": merge_count > 0,
            "pred_values": {node: node_values[node] for node in sorted(graph.nodes)},
            "graph": graph_payload,
            "possible_questions": possible_questions,
        }

    def _add_auxiliary_node(
        self,
        graph: nx.DiGraph,
        node_values: Dict[str, int],
        node_depths: Dict[str, int],
        depth_buckets: Dict[int, List[str]],
        desired_depth: int,
        force_new_leaf: bool = False,
    ) -> Tuple[str, RuleSpec, int]:
        if desired_depth <= 0:
            leaf = self._new_leaf(
                graph=graph,
                node_values=node_values,
                node_depths=node_depths,
                depth_buckets=depth_buckets,
            )
            raise ValueError(f"cannot create auxiliary derived node at depth {desired_depth}: {leaf}")

        base_parent = self._pick_existing_node_at_depth(depth_buckets, desired_depth - 1)
        side_parent: Optional[str] = None
        added_relevant_leafs = 0
        if force_new_leaf or self.rng.random() < 0.7:
            side_parent = self._new_leaf(
                graph=graph,
                node_values=node_values,
                node_depths=node_depths,
                depth_buckets=depth_buckets,
            )
            added_relevant_leafs = 1
        output = self._new_node()
        rule = self._build_rule_for_node(
            graph=graph,
            output=output,
            primary_parent=base_parent,
            side_parent=side_parent,
            node_values=node_values,
            node_depths=node_depths,
            desired_depth=desired_depth,
        )
        graph.add_node(output)
        for parent in rule.inputs:
            graph.add_edge(parent, output)
        node_depths[output] = desired_depth
        node_values[output] = self._evaluate_rule(rule, node_values)
        depth_buckets.setdefault(desired_depth, []).append(output)
        return output, rule, added_relevant_leafs

    def _add_distractor_component(
        self,
        graph: nx.DiGraph,
        node_values: Dict[str, int],
        node_depths: Dict[str, int],
        depth_buckets: Dict[int, List[str]],
        max_rules: int,
    ) -> Tuple[List[str], List[RuleSpec]]:
        component_nodes: List[str] = []
        rules: List[RuleSpec] = []
        leaf = self._new_leaf(
            graph=graph,
            node_values=node_values,
            node_depths=node_depths,
            depth_buckets=depth_buckets,
        )
        component_nodes.append(leaf)
        previous = leaf
        for local_depth in range(1, max_rules + 1):
            side_parent: Optional[str] = None
            if self.rng.random() < 0.5:
                side_parent = self._new_leaf(
                    graph=graph,
                    node_values=node_values,
                    node_depths=node_depths,
                    depth_buckets=depth_buckets,
                )
                component_nodes.append(side_parent)
            output = self._new_node()
            rule = self._build_rule_for_node(
                graph=graph,
                output=output,
                primary_parent=previous,
                side_parent=side_parent,
                node_values=node_values,
                node_depths=node_depths,
                desired_depth=local_depth,
            )
            graph.add_node(output)
            for parent in rule.inputs:
                graph.add_edge(parent, output)
            node_depths[output] = local_depth
            node_values[output] = self._evaluate_rule(rule, node_values)
            depth_buckets.setdefault(local_depth, []).append(output)
            component_nodes.append(output)
            rules.append(rule)
            previous = output
        return component_nodes, rules

    def _build_rule_for_node(
        self,
        graph: nx.DiGraph,
        output: str,
        primary_parent: str,
        side_parent: Optional[str],
        node_values: Dict[str, int],
        node_depths: Dict[str, int],
        desired_depth: int,
    ) -> RuleSpec:
        if side_parent:
            binary_candidates = self._binary_candidates(
                output=output,
                primary_parent=primary_parent,
                side_parent=side_parent,
                node_values=node_values,
            )
            if binary_candidates:
                return self.rng.choice(binary_candidates)

        parent_value = node_values[primary_parent]
        unary_candidates = self._unary_candidates(output=output, parent=primary_parent, parent_value=parent_value)
        if not unary_candidates:
            raise ValueError(f"no valid unary rule candidates for {output} from {primary_parent}")
        return self.rng.choice(unary_candidates)

    def _evaluate_rule(self, rule: RuleSpec, node_values: Dict[str, int]) -> int:
        x = node_values[rule.inputs[0]]
        y = node_values[rule.inputs[1]] if len(rule.inputs) > 1 else None
        if rule.rule_type == "add_xy":
            value = x + int(y)
        elif rule.rule_type == "sub_xy":
            value = x - int(y)
        elif rule.rule_type == "mul_k":
            value = x * int(rule.constant)
        elif rule.rule_type == "div_k":
            value = x // int(rule.constant)
        elif rule.rule_type == "add_k":
            value = x + int(rule.constant)
        elif rule.rule_type == "sub_k":
            value = x - int(rule.constant)
        elif rule.rule_type == "sum_mul_k":
            value = (x + int(y)) * int(rule.constant)
        else:
            raise ValueError(f"unsupported rule type: {rule.rule_type}")
        return value

    def _binary_candidates(
        self,
        output: str,
        primary_parent: str,
        side_parent: str,
        node_values: Dict[str, int],
    ) -> List[RuleSpec]:
        x = node_values[primary_parent]
        y = node_values[side_parent]
        upper = self.max_value * 8
        candidates: List[RuleSpec] = []

        if self.min_value <= x + y <= upper:
            candidates.append(
                RuleSpec(output, "add_xy", [primary_parent, side_parent], None, f"{output} = {primary_parent} + {side_parent}")
            )
        if x >= y and self.min_value <= x - y <= upper:
            candidates.append(
                RuleSpec(output, "sub_xy", [primary_parent, side_parent], None, f"{output} = {primary_parent} - {side_parent}")
            )
        if y >= x and self.min_value <= y - x <= upper:
            candidates.append(
                RuleSpec(output, "sub_xy", [side_parent, primary_parent], None, f"{output} = {side_parent} - {primary_parent}")
            )
        for constant in range(2, 5):
            value = (x + y) * constant
            if self.min_value <= value <= upper:
                candidates.append(
                    RuleSpec(
                        output,
                        "sum_mul_k",
                        [primary_parent, side_parent],
                        constant,
                        f"{output} = ({primary_parent} + {side_parent}) * {constant}",
                    )
                )
        return candidates

    def _unary_candidates(self, output: str, parent: str, parent_value: int) -> List[RuleSpec]:
        upper = self.max_value * 8
        candidates: List[RuleSpec] = []
        for constant in range(2, 5):
            value = parent_value * constant
            if self.min_value <= value <= upper:
                candidates.append(
                    RuleSpec(output, "mul_k", [parent], constant, f"{output} = {parent} * {constant}")
                )
        for constant in range(2, 6):
            if parent_value % constant == 0:
                value = parent_value // constant
                if self.min_value <= value <= upper:
                    candidates.append(
                        RuleSpec(output, "div_k", [parent], constant, f"{output} = {parent} / {constant}")
                    )
        for constant in range(2, 13):
            value = parent_value + constant
            if self.min_value <= value <= upper:
                candidates.append(
                    RuleSpec(output, "add_k", [parent], constant, f"{output} = {parent} + {constant}")
                )
        for constant in range(1, min(12, parent_value - 1) + 1):
            value = parent_value - constant
            if self.min_value <= value <= upper:
                candidates.append(
                    RuleSpec(output, "sub_k", [parent], constant, f"{output} = {parent} - {constant}")
                )
        return candidates

    def _pick_existing_node_at_depth(
        self, depth_buckets: Dict[int, List[str]], desired_depth: int
    ) -> str:
        if desired_depth in depth_buckets and depth_buckets[desired_depth]:
            return self.rng.choice(depth_buckets[desired_depth])
        available_depths = sorted(depth_buckets)
        fallback_depth = max(depth for depth in available_depths if depth <= desired_depth)
        return self.rng.choice(depth_buckets[fallback_depth])

    def _new_leaf(
        self,
        graph: nx.DiGraph,
        node_values: Dict[str, int],
        node_depths: Dict[str, int],
        depth_buckets: Dict[int, List[str]],
    ) -> str:
        node = self._new_node()
        graph.add_node(node)
        node_values[node] = self._rand_value()
        node_depths[node] = 0
        depth_buckets.setdefault(0, []).append(node)
        return node

    def _new_node(self) -> str:
        node = f"x{self._node_index}"
        self._node_index += 1
        return node

    def _rand_value(self) -> int:
        return self.rng.randint(self.min_value, self.max_value)
