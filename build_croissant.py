"""Generate the Croissant metadata file for the multi-turn information-seeking benchmark.

Run from the repository root:

    pip install mlcroissant
    python build_croissant.py

Writes ``croissant.json`` next to the ``data/`` folder.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib

import mlcroissant as mlc

REPO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
OUT_PATH = REPO_ROOT / "croissant.json"

REPO_URL = "https://github.com/mims-harvard/multiturn-info-seek"
DATA_URL_BASE = "https://raw.githubusercontent.com/mims-harvard/multiturn-info-seek/main/data"

VERSION = "1.0.0"
RELEASE_DATE = _dt.datetime(2026, 5, 7)

CC_BY_4_0 = "https://creativecommons.org/licenses/by/4.0/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_object(
    *,
    file_id: str,
    name: str,
    description: str,
    filename: str,
    encoding: str,
) -> mlc.FileObject:
    path = DATA_DIR / filename
    return mlc.FileObject(
        id=file_id,
        name=name,
        description=description,
        content_url=f"{DATA_URL_BASE}/{filename}",
        encoding_formats=[encoding],
        sha256=sha256(path),
        content_size=f"{path.stat().st_size} B",
    )


def text_field(field_id: str, name: str, description: str, file_id: str, column: str) -> mlc.Field:
    return mlc.Field(
        id=field_id,
        name=name,
        description=description,
        data_types=[mlc.DataType.TEXT],
        source=mlc.Source(file_object=file_id, extract=mlc.Extract(column=column)),
    )


def int_field(field_id: str, name: str, description: str, file_id: str, column: str) -> mlc.Field:
    return mlc.Field(
        id=field_id,
        name=name,
        description=description,
        data_types=[mlc.DataType.INTEGER],
        source=mlc.Source(file_object=file_id, extract=mlc.Extract(column=column)),
    )


# ---------------------------------------------------------------------------
# FileObjects
# ---------------------------------------------------------------------------


fo_logicq = file_object(
    file_id="logic_q_mt_csv",
    name="Logic-Q-MT (CSV)",
    description=(
        "Logic-Q-MT instances: under-specified propositional logic problems with rule sets, "
        "known facts, forbidden queryable variables, ground-truth queries, and minimal "
        "sufficient sets. Derived from QuestBench Logic-Q (Li et al., 2025)."
    ),
    filename="logic_q_mt.csv",
    encoding="text/csv",
)

fo_grnmt = file_object(
    file_id="genereg_mt_jsonl",
    name="GeneReg-MT (JSONL)",
    description=(
        "GeneReg-MT records: each line is a task-family/problem record built on top of a "
        "published Boolean gene-regulatory network (Kadelka et al., 2024). Each record "
        "specifies the observed gene-expression assignment, target type, minimal sufficient "
        "query sets, feasible target values, and branches corresponding to compatible "
        "target-specific task instances. Targets are either attractor/steady-state "
        "identifiers (`target_type = attractor_id`) or marker-gene values "
        "(`target_type = marker_gene`)."
    ),
    filename="genereg_mt.jsonl",
    encoding="application/jsonlines",
)

fo_gsme = file_object(
    file_id="gsme_q_mt_csv",
    name="GSME-Q-MT (CSV)",
    description=(
        "GSME-Q-MT instances: under-specified GSM-style math word problems represented as "
        "CSPs with k unknown variables (k <= 4). Derived from QuestBench GSM-Q "
        "(Li et al., 2025)."
    ),
    filename="gsme_q_mt.csv",
    encoding="text/csv",
)

fo_gsme_ext = file_object(
    file_id="gsme_q_mt_ext_csv",
    name="GSME-Q-MT-Ext (CSV)",
    description=(
        "GSME-Q-MT-Ext: extended GSME-Q-MT instances with additional rule graphs, distractor "
        "variables, alternative-path metadata, and forbid-alternatives query sets."
    ),
    filename="gsme_q_mt_ext.csv",
    encoding="text/csv",
)

fo_20q = file_object(
    file_id="twentyq_concepts_py",
    name="20-Questions concept lists (Python)",
    description=(
        "Python source file defining the candidate-entity lists used for the 20-Questions "
        "task (BIG_BENCH_CONCEPT, Animals, Places, Food, Objects, COMMON, THING200). "
        "Adapted from the 20-Questions dataset accompanying Hu et al. (UoT, 2024)."
    ),
    filename="data_20q.py",
    encoding="text/x-python",
)


# ---------------------------------------------------------------------------
# RecordSets (one per tabular file; the .py concept file gets a minimal RecordSet)
# ---------------------------------------------------------------------------


logicq_rs = mlc.RecordSet(
    id="logic_q_mt_records",
    name="logic_q_mt_records",
    description="One row per Logic-Q-MT instance.",
    key=["logic_q_mt_records/sample_id"],
    fields=[
        int_field("logic_q_mt_records/sample_id", "sample_id", "Unique instance identifier.", fo_logicq.id, "sample_id"),
        int_field("logic_q_mt_records/k", "k", "Number of unknown variables (degree of underspecification).", fo_logicq.id, "k"),
        text_field("logic_q_mt_records/known_facts", "known_facts", "JSON-encoded list of attributes asserted true at start.", fo_logicq.id, "known_facts"),
        text_field("logic_q_mt_records/known_untrue_facts", "known_untrue_facts", "JSON-encoded list of attributes asserted false at start.", fo_logicq.id, "known_untrue_facts"),
        text_field("logic_q_mt_records/cannot_ask_facts", "cannot_ask_facts", "JSON-encoded list of attributes that cannot be queried.", fo_logicq.id, "cannot_ask_facts"),
        text_field("logic_q_mt_records/cannot_ask_facts_sets", "cannot_ask_facts_sets", "JSON-encoded list of forbid-alternatives sets.", fo_logicq.id, "cannot_ask_facts_sets"),
        text_field("logic_q_mt_records/goal", "goal", "Target attribute the model must determine.", fo_logicq.id, "goal"),
        text_field("logic_q_mt_records/rules", "rules", "JSON-encoded list of conjunctive rules.", fo_logicq.id, "rules"),
        int_field("logic_q_mt_records/max_depth", "max_depth", "Maximum derivation depth in the rule graph.", fo_logicq.id, "max_depth"),
        int_field("logic_q_mt_records/min_num_rules_needed", "min_num_rules_needed", "Minimum number of rules required to prove goal.", fo_logicq.id, "min_num_rules_needed"),
        int_field("logic_q_mt_records/num_constraints", "num_constraints", "Number of constraints in the instance.", fo_logicq.id, "num_constraints"),
        int_field("logic_q_mt_records/num_vars", "num_vars", "Number of variables in the instance.", fo_logicq.id, "num_vars"),
        text_field("logic_q_mt_records/all_qs", "all_qs", "JSON-encoded list of all candidate query attributes.", fo_logicq.id, "all_qs"),
        text_field("logic_q_mt_records/all_valid_qs", "all_valid_qs", "JSON-encoded list of valid (non-forbidden) queries.", fo_logicq.id, "all_valid_qs"),
        text_field("logic_q_mt_records/gt_qs", "gt_qs", "JSON-encoded list of canonical ground-truth queries.", fo_logicq.id, "gt_qs"),
        text_field("logic_q_mt_records/gt_q_to_derivations_min_rules", "gt_q_to_derivations_min_rules", "JSON-encoded mapping from gt query to minimum-rule derivations.", fo_logicq.id, "gt_q_to_derivations_min_rules"),
        text_field("logic_q_mt_records/gt_q_to_derivations_min_depth", "gt_q_to_derivations_min_depth", "JSON-encoded mapping from gt query to minimum-depth derivations.", fo_logicq.id, "gt_q_to_derivations_min_depth"),
        text_field("logic_q_mt_records/all_alternative_gt_qs", "all_alternative_gt_qs", "JSON-encoded list of alternative ground-truth queries.", fo_logicq.id, "all_alternative_gt_qs"),
        text_field("logic_q_mt_records/all_valid_qs_forbid_alternatives", "all_valid_qs_forbid_alternatives", "Valid queries under the forbid-alternatives mode.", fo_logicq.id, "all_valid_qs_forbid_alternatives"),
        int_field("logic_q_mt_records/num_all_alternative_gt_qs", "num_all_alternative_gt_qs", "Cardinality of all_alternative_gt_qs.", fo_logicq.id, "num_all_alternative_gt_qs"),
        int_field("logic_q_mt_records/num_all_valid_qs", "num_all_valid_qs", "Cardinality of all_valid_qs.", fo_logicq.id, "num_all_valid_qs"),
        int_field("logic_q_mt_records/num_all_valid_qs_forbid_alternatives", "num_all_valid_qs_forbid_alternatives", "Cardinality of all_valid_qs under forbid-alternatives.", fo_logicq.id, "num_all_valid_qs_forbid_alternatives"),
        text_field("logic_q_mt_records/inferred_variable_values", "inferred_variable_values", "JSON-encoded mapping of variables inferred from known facts.", fo_logicq.id, "inferred_variable_values"),
    ],
)

def _grn_field(name: str, description: str, dtype=mlc.DataType.TEXT) -> mlc.Field:
    return mlc.Field(
        id=f"genereg_mt_records/{name}",
        name=name,
        description=description,
        data_types=[dtype],
        source=mlc.Source(
            file_object=fo_grnmt.id,
            extract=mlc.Extract(json_path=f"$.{name}"),
        ),
    )


grn_rs = mlc.RecordSet(
    id="genereg_mt_records",
    name="genereg_mt_records",
    description=(
        "One JSON object per line. Each row is a GeneReg-MT problem/task-family record "
        "built on a published Boolean GRN model. The branches and feasible_target_values "
        "fields enumerate compatible target-specific task instances induced by the same "
        "observed assignment."
    ),
    key=["genereg_mt_records/group_id"],
    fields=[
        _grn_field("schema_version", "Schema version for the GeneReg-MT JSONL record."),
        _grn_field("group_id", "Unique task group identifier (encodes family, model, k, m, c, seed)."),
        _grn_field("family", "Task family within GeneReg-MT (e.g., GeneReg-SS, GeneReg-Marker)."),
        _grn_field("model", "Identifier of the underlying Boolean GRN model from Kadelka et al. (2024)."),
        _grn_field("n_nodes", "Number of nodes (genes / inputs) in the underlying GRN.", mlc.DataType.INTEGER),
        _grn_field("var_names", "JSON array of gene/input variable names in the GRN."),
        _grn_field("target_type", "What the model must determine (e.g., attractor_id, dyn_attr, dyn_marker)."),
        _grn_field("marker_gene", "Marker gene used as the target for marker-gene tasks; null for attractor-identification tasks."),
        _grn_field("marker_gene_idx", "Index of the marker gene in var_names; null for attractor-identification tasks.", mlc.DataType.INTEGER),
        _grn_field("observed", "JSON array of observed gene-expression assignments, represented as [gene_name, value] pairs."),
        _grn_field("k_min", "Minimum number of additional queries required to determine the target.", mlc.DataType.INTEGER),
        _grn_field("minimal_sufficient_sets", "JSON array of minimal sufficient query sets, represented by gene names."),
        _grn_field("minimal_sufficient_sets_idx", "JSON array of minimal sufficient query sets, represented by indices into var_names."),
        _grn_field("prompt_catalog_size", "Number of prompt/query candidates generated for the record.", mlc.DataType.INTEGER),
        _grn_field("prompt_catalog_is_capped", "Whether the prompt/query catalog was capped during construction; null when not applicable.", mlc.DataType.BOOL),
        _grn_field("feasible_target_values", "JSON array of target values compatible with the observed assignment."),
        _grn_field("n_feasible_target_values", "Number of feasible target values compatible with the observed assignment.", mlc.DataType.INTEGER),
        _grn_field("branches", "JSON array of branches corresponding to compatible target-specific task instances; each branch includes a target value and associated state metadata."),
        _grn_field("metadata", "JSON object containing construction metadata, including fixed-point computation, tested subsets, and forbid-profile/queryability information."),
    ],
)


def _slug(s: str) -> str:
    return s.replace(" ", "_")


def csv_field(file_id: str, rs_prefix: str, column: str, description: str, dtype=mlc.DataType.TEXT) -> mlc.Field:
    return mlc.Field(
        id=f"{rs_prefix}/{_slug(column)}",
        name=_slug(column),
        description=description,
        data_types=[dtype],
        source=mlc.Source(file_object=file_id, extract=mlc.Extract(column=column)),
    )


gsme_rs = mlc.RecordSet(
    id="gsme_q_mt_records",
    name="gsme_q_mt_records",
    description="One row per GSME-Q-MT instance.",
    key=["gsme_q_mt_records/sample_id"],
    fields=[
        csv_field(fo_gsme.id, "gsme_q_mt_records", "sample_id", "Unique instance identifier.", mlc.DataType.INTEGER),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "problem_id", "Source GSM-Q problem identifier.", mlc.DataType.INTEGER),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Full Problem", "Original GSM-style word problem text."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "CSP", "Symbolic CSP encoding of the problem."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Full Answer", "Ground-truth final answer."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Variables", "JSON-encoded list of CSP variables."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Equations", "JSON-encoded list of CSP equations."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "depth", "Reasoning depth of the problem.", mlc.DataType.INTEGER),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Pred Values", "JSON-encoded predicted variable values."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Heldout Value", "Held-out variable value used to construct underspecification."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Rewritten Problem", "Underspecified rewriting of the original problem."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Possible Questions", "JSON-encoded list of admissible follow-up questions."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "Given_Conditions", "JSON-encoded list of conditions stated to the model."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "k", "Number of unknown variables.", mlc.DataType.INTEGER),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "diff_score", "Difficulty score.", mlc.DataType.FLOAT),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "goal_var", "Target variable to be determined."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "leaf_nodes_all", "JSON-encoded list of all leaf nodes in the rule graph."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "relevant_leaf", "JSON-encoded list of leaves on the goal's ancestor path."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "ancestors_goal", "JSON-encoded list of ancestor nodes of the goal."),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "dist_max", "Maximum graph distance to the goal.", mlc.DataType.INTEGER),
        csv_field(fo_gsme.id, "gsme_q_mt_records", "dist_mean", "Mean graph distance to the goal.", mlc.DataType.FLOAT),
    ],
)

gsme_ext_rs = mlc.RecordSet(
    id="gsme_q_mt_ext_records",
    name="gsme_q_mt_ext_records",
    description="One row per GSME-Q-MT-Ext instance (superset schema of GSME-Q-MT).",
    key=["gsme_q_mt_ext_records/sample_id"],
    fields=[
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "sample_id", "Unique instance identifier.", mlc.DataType.INTEGER),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "problem_id", "Source problem identifier.", mlc.DataType.INTEGER),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "Full Problem", "Original problem text."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "CSP", "Symbolic CSP encoding."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "Full Answer", "Ground-truth final answer."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "Rewritten Problem", "Underspecified rewriting of the original problem."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "Possible Questions", "JSON-encoded list of admissible follow-up questions."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "Possible Questions Forbid Alternatives", "Admissible questions under forbid-alternatives mode."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "k", "Number of unknown variables.", mlc.DataType.INTEGER),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "diff_score", "Difficulty score.", mlc.DataType.FLOAT),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "goal_var", "Target variable."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "distractor_vars", "JSON-encoded distractor-variable list."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "num_vars", "Number of variables.", mlc.DataType.INTEGER),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "num_rules", "Number of rules.", mlc.DataType.INTEGER),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "depth", "Reasoning depth.", mlc.DataType.INTEGER),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "graph", "JSON-encoded rule graph."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "rule_records", "JSON-encoded per-rule provenance records."),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "has_distractor", "Whether the instance contains distractor variables.", mlc.DataType.BOOL),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "has_merge", "Whether the instance contains merge nodes.", mlc.DataType.BOOL),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "has_alternative_path", "Whether multiple derivation paths exist.", mlc.DataType.BOOL),
        csv_field(fo_gsme_ext.id, "gsme_q_mt_ext_records", "ksufficient_found", "Whether a k-sufficient set was found.", mlc.DataType.BOOL),
    ],
)

twentyq_rs = mlc.RecordSet(
    id="twentyq_concept_lists",
    name="twentyq_concept_lists",
    description=(
        "Logical record set describing the named concept lists declared in data_20q.py "
        "(BIG_BENCH_CONCEPT, Animals, Places, Food, Objects, COMMON, THING200). The file "
        "is Python source rather than a parseable table, so the sole field captures the "
        "raw file contents; downstream consumers should `import` the module and access "
        "the lists by name."
    ),
    fields=[
        mlc.Field(
            id="twentyq_concept_lists/source",
            name="source",
            description="Raw Python source defining the concept lists.",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(file_object=fo_20q.id, extract=mlc.Extract(file_property="content")),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Top-level Metadata (with built-in RAI fields)
# ---------------------------------------------------------------------------


metadata = mlc.Metadata(
    name="multi-turn-info-seek",
    description=(
        "A benchmark release for multi-turn information seeking, framed as solving "
        "an under-specified constraint satisfaction problem with k unknown variables. "
        "This Croissant file documents the public non-clinical subset of MT-INFOSEEK "
        f"v{VERSION}, including Logic-Q-MT, GSME-Q-MT, GSME-Q-MT-Ext, GeneReg-MT, "
        "and 20-Questions concept lists. ClinGuide-MT, used in the accompanying paper "
        "for clinical guideline-based evaluation, is not included in this release "
        "and should be documented separately if released. Models are evaluated along "
        "three axes: what they ask, when they ask it, and how acquired information "
        "affects the final answer."
    ),
    cite_as=(
        "@misc{huang2026mtinfoseek, "
        "title={Do LLMs Know What to Ask and When? Evaluating Multi-Turn Information Seeking}, "
        "author={Yepeng Huang and Jiawen Zhang and Michelle Dai and Xiaorui Su and "
        "Shanghua Gao and Zi Wang and Marinka Zitnik}, "
        "year={2026}}"
    ),
    url=REPO_URL,
    version=VERSION,
    date_created=RELEASE_DATE,
    date_published=RELEASE_DATE,
    license=[CC_BY_4_0],
    in_language=["en"],
    keywords=[
        "multi-turn",
        "information-seeking",
        "clarification questions",
        "constraint satisfaction",
        "underspecification",
        "LLM evaluation",
        "logic",
        "math word problems",
        "gene regulatory networks",
        "20 questions",
    ],
    creators=[
        mlc.Person(name="Yepeng Huang"),
        mlc.Person(name="Jiawen Zhang"),
        mlc.Person(name="Michelle Dai"),
        mlc.Person(name="Xiaorui Su"),
        mlc.Person(name="Shanghua Gao"),
        mlc.Person(name="Zi Wang"),
        mlc.Person(name="Marinka Zitnik"),
    ],
    distribution=[fo_logicq, fo_grnmt, fo_gsme, fo_gsme_ext, fo_20q],
    record_sets=[logicq_rs, grn_rs, gsme_rs, gsme_ext_rs, twentyq_rs],
    # ---------- RAI ----------
    data_use_cases=[
        (
            "Evaluating and stress-testing multi-turn information-seeking and "
            "clarification-question behavior in LLMs under under-specification of "
            "varying degree (k unknown variables)."
        ),
        (
            "Studying when models recognize that information is missing, how accurately "
            "they estimate how much is missing, whether they identify minimal sufficient "
            "queries, and whether they stop only when the answer is determined."
        ),
        (
            "Decoupling information-seeking from answer generation so that final-answer "
            "accuracy does not obscure differences in how models ask and decide when to "
            "answer."
        ),
    ],
    data_limitations=[
        "All task instances are in English only.",
        (
            "Logic-Q-MT and GSME-Q-MT/Ext draw from a fixed adjective and name vocabulary "
            "inherited from QuestBench, so the distribution over surface forms is narrow "
            "and may not transfer to richer natural-language settings."
        ),
        (
            "GeneReg-MT uses Boolean GRN models from Kadelka et al. (2024) that are biased "
            "toward immune-cell and developmental biology, so biological coverage is "
            "non-uniform."
        ),
        (
            "20-Questions candidate sets are skewed toward Western and English-language "
            "concepts (BIG-bench, UoT), and do not capture cultural or geographic diversity."
        ),
        (
            "k is bounded above (k <= 4 for GSME-Q-MT) and the benchmark does not directly "
            "evaluate behavior in the presence of noisy or contradictory oracle responses."
        ),
        (
            "ClinGuide-MT is evaluated in the accompanying paper but is not included in "
            "this public Croissant-described release. This release should therefore be "
            "interpreted as the public non-clinical subset of MT-INFOSEEK."
        ),
    ],
    data_biases=[
        (
            "Logic-Q-MT inherits any selection or templating biases of the QuestBench "
            "Logic-Q generator, including a fixed adjective/name vocabulary and rule-graph "
            "topology distribution."
        ),
        (
            "GSME-Q-MT/Ext inherit the topical and stylistic biases of GSM8K-style word "
            "problems (currency, household quantities, school-grade arithmetic)."
        ),
        (
            "GeneReg-MT inherits the publication bias of Kadelka et al. (2024) toward "
            "well-studied immune-cell and developmental signaling networks; rare or "
            "less-studied biological systems are under-represented."
        ),
        (
            "20-Questions concept lists are skewed toward Western, English-language, and "
            "popular-culture entities."
        ),
    ],
    personal_sensitive_information=[
        (
            "None. All task instances are programmatically constructed from public "
            "scientific or curated common-knowledge sources and contain no personal, "
            "demographic, medical, or otherwise sensitive information about real "
            "individuals."
        ),
    ],
    data_social_impact=(
        "Positive: better evaluation of clarification-question behavior can drive LLMs "
        "that are more conservative under under-specification and that issue targeted "
        "clarification questions instead of guessing, which improves reliability in "
        "high-stakes settings (medicine, science, education). "
        "Negative: as with any capability benchmark, scores can be misread as license to "
        "deploy in safety-critical settings; multi-turn information-seeking ability does "
        "not by itself imply factual or safety reliability, and biases inherited from "
        "the upstream sources may be reproduced if the benchmark is used as supervision."
    ),
)


# ---------------------------------------------------------------------------
# Render JSON-LD and inject extra fields not exposed by the mlcroissant API
# ---------------------------------------------------------------------------


issues = metadata.ctx.issues
jsonld = metadata.to_json()


# Provenance + synthetic-data flags per FileObject (RAI / PROV).
provenance_by_file_id = {
    fo_logicq.id: {
        "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": {
            "@id": "https://github.com/google-deepmind/questbench",
            "@type": "sc:CreativeWork",
            "name": "QuestBench Logic-Q",
            "license": CC_BY_4_0,
        },
        "prov:wasGeneratedBy": (
            "Logic-Q-MT instances are generated by sampling rule graphs from the "
            "QuestBench Logic-Q template, designating a goal attribute, removing one or "
            "more antecedent attributes to create k unknown variables (k = 1,2,3 in this "
            "release), enumerating all queries that could close the proof, and computing "
            "minimum-rule and minimum-depth derivations and forbid-alternatives sets. "
            "Instances are filtered for non-trivial multi-step reasoning."
        ),
    },
    fo_grnmt.id: {
        "rai:hasSyntheticData": False,
        "prov:wasDerivedFrom": {
            "@id": "https://doi.org/10.1038/s41597-024-03900-1",
            "@type": "sc:CreativeWork",
            "name": "Kadelka et al. (2024) Boolean GRN model collection",
            "license": "Open access via the original publication; no specific license declared.",
        },
        "prov:wasGeneratedBy": (
            "For each published Boolean GRN, fixed-point attractors and basins are "
            "computed by brute force; the script enumerates target attractors or marker "
            "genes, samples observed-node configurations, and searches for minimal "
            "sufficient query sets that disambiguate the target. Prompt catalogs, "
            "queryable-gene index sets, branches, and forbid-alternatives metadata are "
            "computed deterministically. Underlying biology is real; task-instance "
            "construction is programmatic."
        ),
    },
    fo_gsme.id: {
        "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": {
            "@id": "https://github.com/google-deepmind/questbench",
            "@type": "sc:CreativeWork",
            "name": "QuestBench GSM-Q",
            "license": CC_BY_4_0,
        },
        "prov:wasGeneratedBy": (
            "GSM-style word problems are converted into CSPs; one or more variables are "
            "held out to create k-unknown instances (k <= 4); the rewriter produces an "
            "underspecified natural-language problem; admissible follow-up questions, "
            "ancestor sets, leaf-node sets, and difficulty scores are computed."
        ),
    },
    fo_gsme_ext.id: {
        "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": {
            "@id": "https://github.com/google-deepmind/questbench",
            "@type": "sc:CreativeWork",
            "name": "QuestBench GSM-Q",
            "license": CC_BY_4_0,
        },
        "prov:wasGeneratedBy": (
            "Same generator as GSME-Q-MT, extended with rule-graph metadata, distractor "
            "variables, alternative-path detection, and forbid-alternatives query sets."
        ),
    },
    fo_20q.id: {
        "rai:hasSyntheticData": False,
        "prov:wasDerivedFrom": {
            "@id": "https://github.com/lzy-xmu/UoT",
            "@type": "sc:CreativeWork",
            "name": "UoT 20-Questions candidate-entity lists",
            "license": "No explicit license identified in the upstream release; users should consult the upstream source terms.",
        },
        "prov:wasGeneratedBy": (
            "The candidate-entity lists are reproduced verbatim from the UoT release and "
            "the BIG-bench 20-Questions concept set, with no modification."
        ),
    },
}


for distribution_entry in jsonld.get("distribution", []):
    fid = distribution_entry.get("@id")
    extras = provenance_by_file_id.get(fid)
    if extras:
        distribution_entry.update(extras)


# Ensure the @context advertises the rai: and prov: prefixes used above. mlcroissant's
# default @context already pulls in cr: / sc: / wd: / dct:; rai and prov are the two we
# need to add. Keep "@language": "en" (the official Croissant 1.0 examples do).
ctx = jsonld.get("@context", {})
if isinstance(ctx, dict):
    ctx.setdefault("rai", "http://mlcommons.org/croissant/RAI/")
    ctx.setdefault("prov", "http://www.w3.org/ns/prov#")
    jsonld["@context"] = ctx


with OUT_PATH.open("w") as f:
    json.dump(jsonld, f, indent=2, ensure_ascii=False)

print(f"Wrote {OUT_PATH}")
print(f"FileObjects: {len(jsonld.get('distribution', []))}")
print(f"RecordSets:  {len(jsonld.get('recordSet', []))}")
if issues.errors:
    print("Validation errors:")
    for e in issues.errors:
        print(" -", e)
if issues.warnings:
    print("Validation warnings:")
    for w in issues.warnings:
        print(" -", w)
