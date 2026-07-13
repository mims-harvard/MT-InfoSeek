"""
Boolean Network Utilities
=========================
Clean, efficient functions for Boolean network analysis.
"""

import os
import pickle
import time
import sys
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

# Pickle compatibility: some cached GRN artifacts were serialized with NumPy>=2
# and reference `numpy._core.*`. Map those module paths when running on NumPy<2.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if hasattr(np.core, "multiarray") and "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

# ============================================================================
# DATA LOADING
# ============================================================================

def load_models_from_pickle(folder: str) -> List[Dict]:
    """Load all Boolean network models from pickle files."""
    models = []
    for pf in sorted(os.listdir(folder)):
        if not pf.endswith('.pickle'):
            continue
        with open(os.path.join(folder, pf), 'rb') as f:
            data = pickle.load(f)
        
        F, I, var = data[0], data[1], data[2]
        constants = data[3] if len(data) > 3 else []
        
        models.append({
            'model': pf.replace('.pickle', ''),
            'F': F, 'I': I, 'var': var, 'constants': constants
        })
    return models


def load_models_with_text(folder: str) -> List[Dict]:
    """Load models from pickle files along with raw .txt Boolean expressions."""
    models = []
    for pf in sorted(os.listdir(folder)):
        if not pf.endswith('.pickle'):
            continue
        
        # Load pickle
        with open(os.path.join(folder, pf), 'rb') as f:
            data = pickle.load(f)
        
        F, I, var = data[0], data[1], data[2]
        constants = data[3] if len(data) > 3 else []
        
        # Load corresponding .txt file
        txt_file = os.path.join(folder, pf.replace('.pickle', '.txt'))
        raw_rules = None
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    raw_rules = f.read()
            except UnicodeDecodeError:
                # Some files may be binary or use different encoding
                try:
                    with open(txt_file, 'r', encoding='latin-1') as f:
                        raw_rules = f.read()
                except:
                    raw_rules = None  # Skip if unreadable
        
        models.append({
            'model': pf.replace('.pickle', ''),
            'F': F, 'I': I, 'var': var, 'constants': constants,
            'raw_rules': raw_rules
        })
    return models


def filter_models(models: List[Dict], min_nodes: int = 1, max_nodes: int = 100) -> List[Dict]:
    """Filter models by node count."""
    return [m for m in models if min_nodes <= len(m['var']) <= max_nodes]


# ============================================================================
# CORE DYNAMICS
# ============================================================================

def sync_update(F: List, I: List, state: int, n: int) -> int:
    """Apply synchronous update. State is an integer (bit vector)."""
    next_state = 0
    for node in range(n):
        regs = I[node]
        k = len(regs)
        tt_idx = sum(((state >> r) & 1) << (k - 1 - i) for i, r in enumerate(regs))
        if F[node][tt_idx]:
            next_state |= (1 << node)
    return next_state


def state_to_tuple(state: int, n: int) -> Tuple[int, ...]:
    """Convert integer state to tuple."""
    return tuple((state >> i) & 1 for i in range(n))


def tuple_to_state(t: Tuple[int, ...]) -> int:
    """Convert tuple to integer state."""
    return sum(v << i for i, v in enumerate(t))


# ============================================================================
# ATTRACTOR CLASSES
# ============================================================================

@dataclass
class Attractor:
    """Single attractor (fixed point or cycle)."""
    states: List[int]       # States as integers
    var_names: List[str]
    
    @property
    def period(self) -> int:
        return len(self.states)
    
    @property
    def is_fixed_point(self) -> bool:
        return self.period == 1
    
    def active_genes(self, state: int) -> List[str]:
        """Get active genes in a state."""
        return [name for i, name in enumerate(self.var_names) if (state >> i) & 1]
    
    def __repr__(self):
        if self.is_fixed_point:
            return f"FixedPoint({self.active_genes(self.states[0])})"
        return f"Cycle(period={self.period})"


@dataclass  
class AttractorLandscape:
    """All attractors + basin mapping for a network."""
    model_name: str
    n_nodes: int
    attractors: List[Attractor]
    basin_map: Optional[np.ndarray] = None  # basin_map[state] = attractor_index
    
    @property
    def n_fixed_points(self) -> int:
        return sum(1 for a in self.attractors if a.is_fixed_point)
    
    @property
    def n_cycles(self) -> int:
        return sum(1 for a in self.attractors if not a.is_fixed_point)
    
    def get_attractor_for_state(self, state: int) -> int:
        """Get attractor index for a given initial state."""
        if self.basin_map is not None:
            return self.basin_map[state]
        raise ValueError("Basin map not computed")
    
    def basin_sizes(self) -> Dict[int, int]:
        """Get size of each attractor's basin."""
        if self.basin_map is None:
            raise ValueError("Basin map not computed")
        unique, counts = np.unique(self.basin_map, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def summary(self) -> Dict:
        """Summary dict for DataFrame."""
        return {
            'model': self.model_name,
            'n_nodes': self.n_nodes,
            'n_attractors': len(self.attractors),
            'n_fixed_points': self.n_fixed_points,
            'n_cycles': self.n_cycles,
            'cycle_lengths': [a.period for a in self.attractors if not a.is_fixed_point],
        }
    
    def __repr__(self):
        return f"AttractorLandscape({self.model_name}: {self.n_fixed_points} FPs, {self.n_cycles} cycles)"
    
    def __str__(self):
        lines = [f"=== {self.model_name} ({self.n_nodes} nodes) ===",
                 f"Attractors: {len(self.attractors)} ({self.n_fixed_points} FPs, {self.n_cycles} cycles)"]
        for i, a in enumerate(self.attractors):
            if a.is_fixed_point:
                lines.append(f"  FP{i}: {a.active_genes(a.states[0]) or '(none)'}")
            else:
                lines.append(f"  Cycle{i} (p={a.period}): {[a.active_genes(s) for s in a.states]}")
        return '\n'.join(lines)


# ============================================================================
# ATTRACTOR FINDING - BRUTE FORCE (≤21 nodes)
# ============================================================================

def find_attractors_bruteforce(F: List, I: List, var_names: List[str], 
                                compute_basins: bool = True,
                                max_attractors: int = 1000,
                                basin_dtype=np.int16,
                                verify_cycles: bool = True) -> AttractorLandscape:
    """
    Find all attractors via brute-force state space traversal.
    Efficient for networks up to ~21 nodes (2^21 ≈ 2.1M states) on a modern machine.
    
    Args:
        F, I: Truth tables and regulator indices
        var_names: Variable names
        compute_basins: If True, store which attractor each state reaches
    
    Returns:
        AttractorLandscape with attractors and optional basin map
    """
    n = len(F)
    total_states = 2 ** n
    
    # Basin map: which attractor does each state reach? (-1 = not yet assigned)
    basin_map = np.full(total_states, -1, dtype=basin_dtype) if compute_basins else None
    attractors = []
    
    for start in range(total_states):
        if basin_map is not None and basin_map[start] >= 0:
            continue  # Already assigned
        
        # Trace trajectory until cycle
        trajectory = []
        visited = {}
        state = start
        
        while state not in visited:
            if basin_map is not None and basin_map[state] >= 0:
                # Hit a state already assigned to an attractor
                attr_id = basin_map[state]
                for s in trajectory:
                    basin_map[s] = attr_id
                break
            visited[state] = len(trajectory)
            trajectory.append(state)
            state = sync_update(F, I, state, n)
        else:
            # Found a new attractor (cycle starting at 'state')
            cycle_start = visited[state]
            cycle_states = trajectory[cycle_start:]
            attr_id = len(attractors)
            if attr_id >= max_attractors:
                raise ValueError(
                    f"Exceeded max_attractors={max_attractors}. "
                    f"This network appears to have >{max_attractors} attractors; "
                    f"increase the cap or skip this model."
                )
            attractors.append(Attractor(cycle_states, var_names))
            
            # Assign basin
            if basin_map is not None:
                for s in trajectory:
                    basin_map[s] = attr_id
    
    
    # Basic sanity checks
    if basin_map is not None:
        if (basin_map < 0).any():
            raise ValueError("Basin map contains unassigned states (-1). This should not happen.")
        if basin_map.max() >= len(attractors):
            raise ValueError("Basin map references a non-existent attractor id.")

    # Verify that each attractor is a valid cycle under sync_update
    if verify_cycles:
        for att in attractors:
            cyc = att.states
            if not cyc:
                raise ValueError("Empty attractor encountered.")
            if len(cyc) == 1:
                nxt = sync_update(F, I, cyc[0], n)
                if nxt != cyc[0]:
                    raise ValueError("Invalid fixed point (does not map to itself).")
            else:
                for i, s in enumerate(cyc):
                    nxt = sync_update(F, I, s, n)
                    if nxt != cyc[(i + 1) % len(cyc)]:
                        raise ValueError("Invalid cycle attractor (transition mismatch).")

    return AttractorLandscape(
            model_name="", n_nodes=n, attractors=attractors, basin_map=basin_map
        )


# ============================================================================
# ATTRACTOR FINDING - PYBOOLNET (larger networks)
# ============================================================================

def _convert_to_bnet(F: List, I: List, var_names: List[str]) -> str:
    """Convert truth table format to PyBoolNet .bnet string."""
    lines = []
    for node, (f, regs) in enumerate(zip(F, I)):
        name = var_names[node]
        reg_names = [var_names[r] for r in regs]
        k = len(regs)
        
        if k == 0:
            # Self-loop (constant)
            lines.append(f"{name}, {name}")
            continue
        
        # Build DNF from truth table    
        terms = []
        for tt_idx in range(2**k):
            if f[tt_idx]:
                term = []
                for bit, rname in enumerate(reg_names):
                    val = (tt_idx >> (k - 1 - bit)) & 1
                    term.append(rname if val else f"!{rname}")
                terms.append(" & ".join(term))
        
        if not terms:
            lines.append(f"{name}, 0")
        elif len(terms) == 2**k:
            lines.append(f"{name}, 1")
        else:
            lines.append(f"{name}, " + " | ".join(f"({t})" for t in terms))
    
    return "\n".join(lines)


def find_fixed_points_sat(F: List, I: List, var_names: List[str]) -> AttractorLandscape:
    """
    Find fixed points using SAT solver (for larger networks).
    NOTE: Only finds fixed points, not cycles.
    
    Requires: pip install python-sat
    """
    try:
        from pysat.solvers import Solver
        from pysat.formula import CNF
    except ImportError:
        raise ImportError("Install python-sat: pip install python-sat")
    
    n = len(F)
    cnf = CNF()
    
    # Encode: for each node, node_value == F(regulator_values)
    for node in range(n):
        regulators = [int(r) for r in I[node]]  # Convert numpy ints to Python ints
        truth_table = F[node]
        k = len(regulators)
        
        for tt_idx in range(2**k):
            output = int(truth_table[tt_idx])
            
            # Build implication: (reg_match) => (node == output)
            # In CNF: (NOT reg_match) OR (node == output)
            reg_literals = []
            for bit_pos, reg in enumerate(regulators):
                reg_val = (tt_idx >> (k - 1 - bit_pos)) & 1
                reg_literals.append((reg + 1) if reg_val == 1 else -(reg + 1))
            
            node_lit = (node + 1) if output == 1 else -(node + 1)
            clause = [-lit for lit in reg_literals] + [node_lit]
            cnf.append(clause)
    
    # Enumerate all fixed points
    fixed_points = []
    with Solver(name='g3', bootstrap_with=cnf) as solver:
        while solver.solve():
            model = solver.get_model()
            if not model:
                break  # No more solutions
            
            # model is a list of literals (positive = True, negative = False)
            # Convert to a dict for safe lookup since model may not have all variables
            model_dict = {abs(lit): lit > 0 for lit in model}
            
            state = 0
            blocking_clause = []
            for i in range(n):
                var = i + 1  # SAT variables are 1-indexed
                is_true = model_dict.get(var, False)
                if is_true:
                    state |= (1 << i)
                    blocking_clause.append(-var)
                else:
                    blocking_clause.append(var)
            fixed_points.append(state)
            solver.add_clause(blocking_clause)
    
    attractors = [Attractor([s], var_names) for s in fixed_points]
    return AttractorLandscape(model_name="", n_nodes=n, attractors=attractors)


def find_attractors_pyboolnet(F: List, I: List, var_names: List[str], 
                               raw_rules: Optional[str] = None,
                               quiet: bool = True,
                               max_output: int = 1000) -> AttractorLandscape:
    """
    Find attractors using PyBoolNet (efficient for larger networks).
    Finds both fixed points and cycles.
    
    Args:
        F, I, var_names: Truth tables and regulator info (fallback)
        raw_rules: Original Boolean expression text (preferred if available)
        quiet: If True, suppress PyBoolNet INFO logging (default: True)
        max_output: Maximum number of attractors to find (default: 1000)
    
    NOTE: Requires working NuSMV binary with ncurses5 support.
    
    Requires: pip install pyboolnet
    """
    try:
        import pyboolnet.file_exchange as fe
        import pyboolnet.attractors as attr
        import logging
    except ImportError:
        raise ImportError("Install pyboolnet: pip install pyboolnet")
    
    # Suppress PyBoolNet logging if quiet
    # PyBoolNet uses logging.getLogger(__file__) so we need to silence pyboolnet loggers
    if quiet:
        # Suppress all loggers that contain 'pyboolnet' in their name
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if 'pyboolnet' in name.lower():
                logging.getLogger(name).setLevel(logging.ERROR)
    
    # Prefer raw rules if available (more reliable than truth table conversion)
    if raw_rules:
        bnet_str = raw_rules.replace(' AND ', ' & ').replace(' OR ', ' | ')
        bnet_str = bnet_str.replace('NOT ', '!').replace('=', ',')
        # Also handle AND( pattern (no space before parenthesis)
        bnet_str = bnet_str.replace('AND(', '& (')
    else:
        bnet_str = _convert_to_bnet(F, I, var_names)
    
    # Sanitize variable names for PyBoolNet (requires alphanumeric + underscore only)
    # Replace special characters that may appear in variable names
    # Note: Don't replace () as they're needed for boolean grouping
    char_replace = {
        '/': '_', '-': '_', '+': 'plus', '*': 'star', '.': '_'
    }
    for char, repl in char_replace.items():
        bnet_str = bnet_str.replace(char, repl)
    
    # Prefix variable names that start with digits (NuSMV requirement)
    import re
    # Match word boundaries followed by digit-starting identifiers
    bnet_str = re.sub(r'\b(\d)', r'v_\1', bnet_str)
    
    # Sanitize non-ASCII characters (e.g., Greek letters)
    greek_map = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu', 'π': 'pi', 'σ': 'sigma', 'τ': 'tau',
        'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta', 'Κ': 'Kappa'
    }
    for greek, ascii_equiv in greek_map.items():
        bnet_str = bnet_str.replace(greek, ascii_equiv)
    
    # Also create a mapping for looking up var names in results
    # Must apply same sanitization as bnet_str
    sanitized_names = []
    for name in var_names:
        sname = name
        # Apply character replacements
        for char, repl in char_replace.items():
            sname = sname.replace(char, repl)
        # Apply Greek letter replacements
        for greek, ascii_equiv in greek_map.items():
            sname = sname.replace(greek, ascii_equiv)
        # Prefix if starts with digit
        if sname and sname[0].isdigit():
            sname = 'v_' + sname
        sanitized_names.append(sname)
    
    primes = fe.bnet2primes(bnet_str)
    
    # Compute attractors (synchronous update)
    # Note: PyBoolNet has max_output=1000 default, we increase it
    result = attr.compute_attractors(primes, "synchronous", max_output=max_output)
    
    # PyBoolNet returns a dict with 'attractors' key containing tuple of attractor dicts
    # Each attractor dict has: attractor['state']['dict'] = {var_name: 0/1}
    # For cyclic attractors, there may be a 'cycle' key with list of states
    attractors_raw = result.get('attractors', ())
    
    n = len(var_names)
    attractors = []
    
    for att in attractors_raw:
        states = []
        
        # Check if it's a cyclic attractor
        if att.get('is_cyclic', False) and 'cycle' in att:
            # Cyclic attractor: multiple states in cycle
            for cycle_state in att['cycle']:
                state_dict = cycle_state.get('dict', {})
                state_int = _state_dict_to_int(state_dict, sanitized_names)
                states.append(state_int)
        else:
            # Steady state (fixed point)
            state_dict = att.get('state', {}).get('dict', {})
            state_int = _state_dict_to_int(state_dict, sanitized_names)
            states.append(state_int)
        
        if states:
            attractors.append(Attractor(states, var_names))
    
    return AttractorLandscape(model_name="", n_nodes=n, attractors=attractors)


def _state_dict_to_int(state_dict: Dict, sanitized_names: List[str]) -> int:
    """Convert state dict to integer."""
    state_int = 0
    for i, sname in enumerate(sanitized_names):
        val = state_dict.get(sname, 0)
        if isinstance(val, bool):
            val = 1 if val else 0
        elif not isinstance(val, int):
            val = int(val) if val else 0
        state_int += val << i
    return state_int


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def find_attractors(model: Dict, method: str = 'auto', 
                    max_bruteforce: int = 21, compute_basins: bool = True) -> Optional[AttractorLandscape]:
    """
    Find all attractors for a model.
    
    Args:
        model: Dict with F, I, var keys (and optionally raw_rules)
        method: 'bruteforce', 'sat', 'pyboolnet', or 'auto'
        max_bruteforce: Threshold for auto method selection
        compute_basins: Compute basin mapping (only for bruteforce)
    
    Returns:
        AttractorLandscape, or None if failed
        
    Note: SAT method only finds fixed points, not cycles.
          PyBoolNet finds both but requires ncurses5.
    """
    n = len(model['var'])
    F, I, var_names = model['F'], model['I'], model['var']
    raw_rules = model.get('raw_rules')
    
    if method == 'auto':
        method = 'bruteforce' if n <= max_bruteforce else 'sat'
    
    try:
        if method == 'bruteforce':
            result = find_attractors_bruteforce(F, I, var_names, compute_basins)
        elif method == 'pyboolnet':
            try:
                result = find_attractors_pyboolnet(F, I, var_names, raw_rules)
            except Exception as pyboolnet_error:
                # Fallback to SAT if PyBoolNet fails (e.g., special characters)
                print(f"(pyboolnet failed, using sat) ", end='')
                result = find_fixed_points_sat(F, I, var_names)
        else:  # sat
            result = find_fixed_points_sat(F, I, var_names)
        
        result.model_name = model['model']
        return result
    except Exception as e:
        print(f"Error processing {model['model']}: {e}")
        return None


def _timeout_worker(model, method, max_bruteforce, compute_basins, result_queue):
    """Module-level worker for multiprocessing (must be picklable)."""
    try:
        res = find_attractors(model, method, max_bruteforce, compute_basins)
        result_queue.put(res)
    except Exception:
        result_queue.put(None)


def find_all_attractors(models: List[Dict], method: str = 'auto',
                        max_bruteforce: int = 21, compute_basins: bool = True,
                        verbose: bool = True, timeout_seconds: int = None,
                        skip_models: List[str] = None, max_nodes: int = None) -> List[Optional[AttractorLandscape]]:
    """
    Find attractors for all models with smart method selection.
    
    Args:
        models: List of model dicts
        method: 'bruteforce', 'sat', 'pyboolnet', or 'auto'
        max_bruteforce: Threshold for auto method selection
        compute_basins: Compute basin mapping (bruteforce only)
        verbose: Print progress
        timeout_seconds: Timeout per model in seconds (None = no timeout)
        skip_models: List of model names to skip
        max_nodes: Skip models with more than this many nodes
    
    For 'auto' method:
    - Brute-force for networks ≤ max_bruteforce nodes
    - PyBoolNet for larger networks WITH valid raw_rules
    - SAT for larger networks without valid raw_rules (fixed points only)
    """
    results = []
    total_start = time.time()
    skipped = []
    timed_out = []
    
    # Track tabular files (incompatible with pyboolnet text conversion)
    tabular_files = []
    
    for i, m in enumerate(models):
        n = len(m['var'])
        n_rules = len(m['raw_rules'].strip().split('\n')) if m.get('raw_rules') else n
        
        # Check skip conditions
        if skip_models and m['model'] in skip_models:
            print(f"[{i+1}/{len(models)}] {m['model']} ({n} nodes) [SKIPPED]")
            results.append(None)
            skipped.append(m['model'])
            continue
        
        if max_nodes and n > max_nodes:
            print(f"[{i+1}/{len(models)}] {m['model']} ({n} nodes) [SKIPPED: >{max_nodes} nodes]")
            results.append(None)
            skipped.append(m['model'])
            continue
        
        # Determine method for this model
        use_method = method
        if method == 'auto':
            is_tabular = 'tabular' in m['model'].lower()
            has_valid_rules = m.get('raw_rules') and '=' in m.get('raw_rules', '')
            
            if n <= max_bruteforce:
                use_method = 'bruteforce'
            elif has_valid_rules and not is_tabular:
                use_method = 'pyboolnet'
            else:
                use_method = 'sat'
                if is_tabular:
                    tabular_files.append(m['model'])
        
        print(f"[{i+1}/{len(models)}] {m['model']} ({n} nodes, {n_rules} rules) [{use_method}]:", end=' ', flush=True)
        
        start_time = time.time()
        result = None
        
        if timeout_seconds and timeout_seconds > 0:
            # Hybrid timeout approach:
            # - Bruteforce: use signal.alarm (fast, works for pure Python)
            # - PyBoolNet/SAT: use multiprocessing (external calls block signals)
            
            if use_method == 'bruteforce':
                # Signal-based timeout for pure Python code
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Timed out after {timeout_seconds}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                try:
                    result = find_attractors(m, use_method, max_bruteforce, compute_basins)
                except TimeoutError:
                    result = None
                    timed_out.append(m['model'])
                    print(f"TIMEOUT ({timeout_seconds}s)")
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Multiprocessing for pyboolnet/sat (external calls need forcible termination)
                import multiprocessing
                try:
                    ctx = multiprocessing.get_context('fork')
                except ValueError:
                    ctx = multiprocessing
                
                result_queue = ctx.Queue()
                proc = ctx.Process(target=_timeout_worker, args=(m, use_method, max_bruteforce, compute_basins, result_queue))
                proc.start()
                proc.join(timeout=timeout_seconds)
                
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=1)
                    result = None
                    timed_out.append(m['model'])
                    print(f"TIMEOUT ({timeout_seconds}s)")
                else:
                    try:
                        result = result_queue.get_nowait()
                    except:
                        result = None
        else:
            result = find_attractors(m, use_method, max_bruteforce, compute_basins)
        
        elapsed = time.time() - start_time
        
        results.append(result)
        if result and m['model'] not in timed_out:
            if verbose:
                print(f"{result.n_fixed_points} FPs, {result.n_cycles} cycles ({elapsed:.2f}s)")
        elif verbose and m['model'] not in timed_out:
            print(f"failed ({elapsed:.2f}s)")
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.1f}s for {len(models)} models")
    
    if skipped:
        print(f"Skipped {len(skipped)} models: {skipped}")
    if timed_out:
        print(f"Timed out {len(timed_out)} models: {timed_out}")
    if tabular_files:
        print(f"Note: {len(tabular_files)} tabular files used SAT (fixed points only): {tabular_files}")
    
    return results


# ============================================================================
# LAZY BASIN QUERY (for when full basin not stored)
# ============================================================================

def trace_to_attractor(F: List, I: List, start_state: int, n: int, max_steps: int = 10000) -> Tuple[int, int]:
    """
    Trace from a state until reaching an attractor.
    
    Returns:
        (attractor_state, steps): One state in the attractor and steps taken
    """
    visited = {}
    state = start_state
    step = 0
    
    while state not in visited and step < max_steps:
        visited[state] = step
        state = sync_update(F, I, state, n)
        step += 1
    
    return state, step


# ============================================================================
# BASIN MAP PRECOMPUTATION & CACHING (n < 22)
# ============================================================================

def save_basin_cache(landscape: AttractorLandscape, filepath: str, overwrite: bool = False) -> None:
    """Save an AttractorLandscape's basin map and attractors to disk.

    The saved file is a pickle with:
      - model_name, n_nodes, var_names
      - basin_map: np.ndarray[int16/int32] of length 2^n, mapping state_int -> attractor_id
      - attractors_int: List[List[int]] (each list is a cycle, length 1 for fixed points)
    """
    if landscape.basin_map is None:
        raise ValueError("Cannot save basin cache: landscape.basin_map is None.")
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File exists: {filepath}. Set overwrite=True to replace it.")
    payload = {
        "version": 1,
        "model_name": landscape.model_name,
        "n_nodes": landscape.n_nodes,
        "var_names": list(landscape.attractors[0].var_names) if landscape.attractors else [],
        "basin_map": landscape.basin_map,
        "attractors_int": [list(att.states) for att in landscape.attractors],
        "created_at": time.time(),
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_basin_cache(filepath: str) -> AttractorLandscape:
    """Load a basin cache saved by save_basin_cache."""
    with open(filepath, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unrecognized basin cache format.")
    basin_map = payload["basin_map"]
    if not isinstance(basin_map, np.ndarray):
        raise ValueError("Invalid basin cache: basin_map is not a numpy array.")
    n = int(payload["n_nodes"])
    if basin_map.shape[0] != (1 << n):
        raise ValueError(f"Invalid basin cache: basin_map length {basin_map.shape[0]} != 2^n ({1<<n}).")
    var_names = payload.get("var_names", [])
    attractors_int = payload.get("attractors_int", [])
    attractors = [Attractor(states=list(states), var_names=var_names) for states in attractors_int]
    return AttractorLandscape(
        model_name=payload.get("model_name", ""),
        n_nodes=n,
        attractors=attractors,
        basin_map=basin_map,
    )


def basin_cache_path(cache_dir: str, model_name: str) -> str:
    """Canonical cache file path for a model."""
    return os.path.join(cache_dir, f"{model_name}.basin.pkl")


def precompute_basin_cache_for_models(models: List[Dict],
                                      cache_dir: str,
                                      max_nodes: int = 21,
                                      max_attractors: int = 1000,
                                      basin_dtype=np.int16,
                                      overwrite: bool = False,
                                      verbose: bool = True) -> List[str]:
    """Precompute and save basin caches for all models with n_nodes <= max_nodes.

    Returns:
        List[str]: filepaths successfully written.
    """
    written = []
    for m in models:
        n = len(m["var"])
        if n > max_nodes:
            continue
        outpath = basin_cache_path(cache_dir, m["model"])
        if os.path.exists(outpath) and not overwrite:
            written.append(outpath)
            continue
        landscape = find_attractors_bruteforce(
            m["F"], m["I"], m["var"],
            compute_basins=True,
            max_attractors=max_attractors,
            basin_dtype=basin_dtype,
            verify_cycles=True,
        )
        landscape.model_name = m["model"]
        save_basin_cache(landscape, outpath, overwrite=overwrite)
        if verbose:
            print(f"Saved basin cache: {outpath} (n={n}, attractors={len(landscape.attractors)})")
        written.append(outpath)
    return written


def load_or_precompute_basin_cache(model: Dict,
                                  cache_dir: str,
                                  max_nodes: int = 21,
                                  max_attractors: int = 1000,
                                  basin_dtype=np.int16,
                                  overwrite: bool = False) -> AttractorLandscape:
    """Load basin cache for a model if present, otherwise compute and save it."""
    n = len(model["var"])
    if n > max_nodes:
        raise ValueError(f"Model has n={n} nodes > max_nodes={max_nodes}; refusing brute-force basin precompute.")
    path = basin_cache_path(cache_dir, model["model"])
    if os.path.exists(path) and not overwrite:
        return load_basin_cache(path)
    landscape = find_attractors_bruteforce(
        model["F"], model["I"], model["var"],
        compute_basins=True,
        max_attractors=max_attractors,
        basin_dtype=basin_dtype,
        verify_cycles=True,
    )
    landscape.model_name = model["model"]
    save_basin_cache(landscape, path, overwrite=True)
    return landscape

# ============================================================================
# STATE TRACING & PARTIAL STATE ANALYSIS
# ============================================================================

def dict_to_state_int(state_dict: Dict[str, bool], var_names: List[str]) -> int:
    """
    Convert a state dict to integer.
    
    Args:
        state_dict: {'gene_name': True/False} or {'gene_name': 1/0}
        var_names: List of variable names in order
        
    Returns:
        Integer state representation
    """
    state_int = 0
    for i, name in enumerate(var_names):
        val = state_dict.get(name, 0)
        if val:  # Handles True, 1, or any truthy value
            state_int |= (1 << i)
    return state_int


def state_int_to_dict(state_int: int, var_names: List[str]) -> Dict[str, int]:
    """
    Convert integer state to dict.
    
    Returns:
        {'gene_name': 0 or 1}
    """
    return {name: (state_int >> i) & 1 for i, name in enumerate(var_names)}


def simulate_to_attractor(model: Dict, initial_state: Dict[str, bool], 
                          max_steps: int = 10000) -> Dict:
    """
    Simulate network dynamics from an initial state until reaching an attractor.
    
    Args:
        model: Model dict with F, I, var keys
        initial_state: Dict of {'gene_name': True/False or 1/0}
                       Missing genes default to 0 (inactive)
        max_steps: Maximum simulation steps
        
    Returns:
        Dict with:
        - 'final_state': Dict of final gene values
        - 'steps': Number of steps to reach attractor
        - 'is_fixed_point': True if attractor is fixed point (period 1)
        - 'attractor_period': Length of attractor cycle
        - 'attractor_states': List of state dicts in the attractor
    """
    F, I, var_names = model['F'], model['I'], model['var']
    n = len(var_names)
    
    # Convert initial state dict to integer
    state = dict_to_state_int(initial_state, var_names)
    
    visited = {}
    trajectory = []
    step = 0
    
    while state not in visited and step < max_steps:
        visited[state] = step
        trajectory.append(state)
        state = sync_update(F, I, state, n)
        step += 1
    
    # Find the attractor
    if state in visited:
        cycle_start = visited[state]
        attractor_states = trajectory[cycle_start:]
    else:
        attractor_states = [state]
    
    return {
        'final_state': state_int_to_dict(attractor_states[0], var_names),
        'steps': step,
        'is_fixed_point': len(attractor_states) == 1,
        'attractor_period': len(attractor_states),
        'attractor_states': [state_int_to_dict(s, var_names) for s in attractor_states]
    }


def _iterate_completions_recursive(base_state: int, free_positions: List[int], idx: int = 0):
    """Yield all states by recursively assigning free bit positions."""
    if idx >= len(free_positions):
        yield base_state
        return
    pos = free_positions[idx]
    yield from _iterate_completions_recursive(base_state, free_positions, idx + 1)
    yield from _iterate_completions_recursive(base_state | (1 << pos), free_positions, idx + 1)


def check_partial_determines_attractor(model: Dict,
                                       partial_state: Dict[str, bool],
                                       landscape: Optional[AttractorLandscape] = None,
                                       cache_dir: Optional[str] = None,
                                       max_nodes_for_autoload: int = 21,
                                       max_attractors: int = 1000,
                                       strict: bool = True,
                                       max_enumerations: Optional[int] = None) -> Dict:
    """Exact check (no random sampling) whether a partial initial state determines the final attractor.

    Requires a basin map (state_int -> attractor_id). Provide `landscape` (with basin_map),
    or `cache_dir` to load/precompute a basin cache for n<=max_nodes_for_autoload.
    """
    var_names = model["var"]
    n = len(var_names)

    if landscape is None:
        if cache_dir is None:
            if strict:
                raise ValueError("Exact check requires `landscape` or `cache_dir` for basin map lookup.")
            return {
                "determines_attractor": None,
                "attractor_id": None,
                "attractor_states": None,
                "num_enumerated": 0,
                "different_attractor_ids": [],
                "status": "error",
                "error": "missing_basin_map",
            }
        landscape = load_or_precompute_basin_cache(
            model,
            cache_dir=cache_dir,
            max_nodes=max_nodes_for_autoload,
            max_attractors=max_attractors,
            overwrite=False,
        )

    if landscape.basin_map is None:
        if strict:
            raise ValueError("landscape.basin_map is None; cannot do exact check.")
        return {
            "determines_attractor": None,
            "attractor_id": None,
            "attractor_states": None,
            "num_enumerated": 0,
            "different_attractor_ids": [],
            "status": "error",
            "error": "missing_basin_map",
        }

    basin_map = landscape.basin_map
    if basin_map.shape[0] != (1 << n):
        raise ValueError(f"Invalid basin map length {basin_map.shape[0]} for n={n}.")

    name_to_idx = {name: i for i, name in enumerate(var_names)}

    fixed_mask = 0
    fixed_value = 0
    for g, v in partial_state.items():
        if g not in name_to_idx:
            raise KeyError(f"Gene '{g}' not in model var list.")
        i = name_to_idx[g]
        fixed_mask |= (1 << i)
        if bool(v):
            fixed_value |= (1 << i)

    free_positions = [i for i in range(n) if not ((fixed_mask >> i) & 1)]
    base_state = fixed_value

    first_id = None
    num_enum = 0

    for state_int in _iterate_completions_recursive(base_state, free_positions, 0):
        num_enum += 1
        if max_enumerations is not None and num_enum > max_enumerations:
            return {
                "determines_attractor": None,
                "attractor_id": None,
                "attractor_states": None,
                "num_enumerated": num_enum - 1,
                "different_attractor_ids": [],
                "status": "incomplete",
                "error": "max_enumerations_exceeded",
            }

        a_id = int(basin_map[state_int])
        if first_id is None:
            first_id = a_id
        elif a_id != first_id:
            return {
                "determines_attractor": False,
                "attractor_id": None,
                "attractor_states": None,
                "num_enumerated": num_enum,
                "different_attractor_ids": sorted({first_id, a_id}),
                "status": "ok",
            }

    if first_id is None:
        return {
            "determines_attractor": None,
            "attractor_id": None,
            "attractor_states": None,
            "num_enumerated": 0,
            "different_attractor_ids": [],
            "status": "error",
            "error": "no_completions",
        }

    att = landscape.attractors[first_id]
    attractor_states = [state_int_to_dict(s, var_names) for s in att.states]
    return {
        "determines_attractor": True,
        "attractor_id": first_id,
        "attractor_states": attractor_states,
        "num_enumerated": num_enum,
        "different_attractor_ids": [],
        "status": "ok",
    }


def find_minimal_determining_genes(
    model: Dict,
    partial_state: Dict[str, bool],
    landscape: Optional[AttractorLandscape] = None,
    cache_dir: Optional[str] = None,
    strict: bool = True,
    max_enumerations: Optional[int] = None,
    *,
    max_k: Optional[int] = 4,
    max_sets_to_return: int = 50,
) -> Dict:
    """Find a *globally minimal* subset of genes in `partial_state` that still determines the *same* attractor.

    This is an **exact** checker (global minimality, feasibility-aware) built on top of
    `check_partial_determines_attractor(...)`, which itself enumerates *all* completions
    of the unspecified genes (Cartesian space) using a basin map lookup.

    Notes:
      - If `max_k` is not None, we only search subsets up to size `max_k`. If no subset
        of size <= max_k determines the same attractor, we return an error.
      - The returned `minimal_genes`/`minimal_state` correspond to the first minimal set
        found. `minimal_gene_sets` contains up to `max_sets_to_return` minimal sets.

    Args:
        model: Model dict as returned by `load_models_with_text` (must include F, I, var, model).
        partial_state: Dict of gene->bool values that is known to determine an attractor.
        landscape: Optional precomputed AttractorLandscape with basin_map.
        cache_dir: Optional directory for basin cache auto-load/precompute (n<=21).
        strict: If True, raise on missing basin map / incomplete enumeration.
        max_enumerations: Optional cap for enumeration (mostly for debugging).
        max_k: Optional max subset size to search (default 4). Set to None to search all sizes.
        max_sets_to_return: Cap on number of minimal sets returned.

    Returns:
        Dict with keys:
          - minimal_genes: set[str]
          - minimal_state: dict[str,bool]
          - minimal_gene_sets: list[set[str]]
          - k_min: int
          - attractor_id: int
          - attractor_states: list[dict[str,bool]]  (states of the determined attractor)
    """
    # Verify that the provided partial_state determines an attractor
    base = check_partial_determines_attractor(
        model,
        partial_state,
        landscape=landscape,
        cache_dir=cache_dir,
        strict=strict,
        max_enumerations=max_enumerations,
    )
    if not base.get("determines_attractor"):
        return {
            "minimal_genes": None,
            "minimal_state": None,
            "minimal_gene_sets": None,
            "k_min": None,
            "attractor_id": None,
            "attractor_states": None,
            "error": "Input partial_state does not uniquely determine an attractor",
            "detail": base,
        }

    target_id = int(base["attractor_id"])
    attractor_states = base["attractor_states"]

    genes = list(partial_state.keys())
    max_size = len(genes) if max_k is None else min(len(genes), int(max_k))

    # Search subsets in increasing size (global minimality by construction)
    minimal_sets: List[Tuple[str, ...]] = []
    tested = 0

    for k in range(0, max_size + 1):
        for subset in combinations(genes, k):
            tested += 1
            test_state = {g: partial_state[g] for g in subset}
            res = check_partial_determines_attractor(
                model,
                test_state,
                landscape=landscape,
                cache_dir=cache_dir,
                strict=strict,
                max_enumerations=max_enumerations,
            )
            if res.get("status") == "incomplete":
                if strict:
                    raise ValueError("Enumeration incomplete (max_enumerations exceeded) during minimality search.")
                continue
            if res.get("determines_attractor") and int(res["attractor_id"]) == target_id:
                minimal_sets.append(subset)
                if len(minimal_sets) >= max_sets_to_return:
                    break
        if minimal_sets:
            # k is minimal
            break

    if not minimal_sets:
        return {
            "minimal_genes": None,
            "minimal_state": None,
            "minimal_gene_sets": None,
            "k_min": None,
            "attractor_id": target_id,
            "attractor_states": attractor_states,
            "error": f"No determining subset found with size <= {max_k}.",
            "n_tested": tested,
        }

    first = minimal_sets[0]
    return {
        "minimal_genes": set(first),
        "minimal_state": {g: partial_state[g] for g in first},
        "minimal_gene_sets": [set(s) for s in minimal_sets],
        "k_min": len(first),
        "attractor_id": target_id,
        "attractor_states": attractor_states,
        "n_tested": tested,
    }

# ============================================================================
# DATASET CURATION HELPERS (k-sufficient tasks, k<=4)
# ============================================================================

@dataclass
class MinKResult:
    """Result of an exact (global) minimal k-sufficient query-set search."""
    k_min: int
    min_sets: List[List[int]]
    n_tested: int
    # optional diagnostics
    n_worlds: Optional[int] = None


def bit(state_int: int, idx: int) -> int:
    """Return bit idx (0/1) from an int-encoded state."""
    return (int(state_int) >> int(idx)) & 1


def state_int_to_bits(state_int: int, n: int) -> List[int]:
    return [(int(state_int) >> i) & 1 for i in range(n)]


def bits_to_state_int(bits: List[int]) -> int:
    s = 0
    for i, v in enumerate(bits):
        if int(v) == 1:
            s |= 1 << i
    return s


def attractor_representative_states(landscape: "AttractorLandscape") -> np.ndarray:
    """Return representative state (uint64) for each attractor.

    - Fixed points: the fixed point.
    - Cycles: the first state in the cycle.

    Note: only safe when states fit in uint64 (e.g. n<=64; in practice we
    only call this for basin maps with n<=21).
    """
    if landscape is None or not getattr(landscape, "attractors", None):
        return np.zeros((0,), dtype=np.uint64)
    return np.array([int(a.states[0]) for a in landscape.attractors], dtype=np.uint64)


def build_mask_value_from_idx_vals(assignments: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Build (mask,value) bitmasks from [(idx,val), ...]."""
    mask = 0
    val = 0
    for i, v in assignments:
        i = int(i)
        mask |= 1 << i
        if int(v) == 1:
            val |= 1 << i
    return mask, val


def build_mask_value_from_partial_dict(partial: Dict[str, bool], var_names: List[str]) -> Tuple[int, int]:
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    mask = 0
    val = 0
    for g, v in partial.items():
        if g not in name_to_idx:
            raise KeyError(f"Gene '{g}' not in model var list.")
        i = name_to_idx[g]
        mask |= 1 << i
        if bool(v):
            val |= 1 << i
    return mask, val


def iter_states_consistent_with_mask_value(n: int, mask: int, value: int, *, max_enumerations: Optional[int] = None):
    """Yield all int states s in {0,1}^n such that (s & mask) == value, by recursion."""
    if value & ~mask:
        raise ValueError("value has bits outside mask")
    free_positions = [i for i in range(n) if not ((mask >> i) & 1)]
    base_state = int(value)
    num = 0
    for s in _iterate_completions_recursive(base_state, free_positions, 0):
        num += 1
        if max_enumerations is not None and num > max_enumerations:
            break
        yield int(s)


def _is_y_function_of_subset_states(states: np.ndarray, y: np.ndarray, subset: List[int]) -> bool:
    """Feasibility-aware check: within each subset-pattern group, y is constant.

    states: uint64 array of candidate worlds
    y: array aligned with states
    subset: list of gene indices
    """
    if states.size == 0:
        return False
    if len(subset) == 0:
        # y must be constant over all candidates
        return int(y.min()) == int(y.max())

    mask = 0
    for gi in subset:
        mask |= 1 << int(gi)

    seen: Dict[int, int] = {}
    m = np.uint64(mask)
    for st, yy in zip(states.tolist(), y.tolist()):
        key = int(np.uint64(st) & m)
        yy = int(yy)
        prev = seen.get(key)
        if prev is None:
            seen[key] = yy
        elif prev != yy:
            return False
    return True


def find_min_k_sufficient_sets_states(
    *,
    states: np.ndarray,
    y: np.ndarray,
    candidate_genes: List[int],
    max_k: int = 4,
    max_sets_to_return: int = 50,
) -> Optional[MinKResult]:
    """Exact global minimality search over subset sizes 0..max_k.

    Returns the smallest k such that there exists a subset S (|S|=k) where y is
    determined over candidate worlds when grouped by S.
    """
    n_tested = 0

    # k = 0 check (Known without queries)
    if _is_y_function_of_subset_states(states, y, []):
        return MinKResult(k_min=0, min_sets=[[]], n_tested=1, n_worlds=int(states.size))

    for k in range(1, max_k + 1):
        min_sets: List[List[int]] = []
        exceeded_cap = False
        for subset in combinations(candidate_genes, k):
            n_tested += 1
            if _is_y_function_of_subset_states(states, y, list(subset)):
                min_sets.append(list(subset))
                if len(min_sets) >= max_sets_to_return:
                    exceeded_cap = True
                    break
        if exceeded_cap:
            # Too many sets at this k - skip this task for completeness
            return None
        if min_sets:
            return MinKResult(k_min=k, min_sets=min_sets, n_tested=n_tested, n_worlds=int(states.size))

    return None


def _is_y_function_of_subset_basin(
    *,
    n: int,
    basin_map: np.ndarray,
    attractor_rep_states: Optional[np.ndarray],
    context_mask: int,
    context_value: int,
    subset: List[int],
    target_type: str,
    marker_idx: Optional[int] = None,
    max_enumerations: Optional[int] = None,
) -> bool:
    """Feasibility-aware check using *precomputed basin_map* (no simulation).

    We enumerate all initial states consistent with context, group by subset-pattern,
    and ensure y is constant within each pattern group.

    target_type:
      - 'attractor_id': y = basin_map[state]
      - 'marker_gene': y = marker bit in representative attractor state
    """
    if target_type not in {"attractor_id", "marker_gene"}:
        raise ValueError("target_type must be 'attractor_id' or 'marker_gene'")

    if basin_map is None or basin_map.shape[0] != (1 << n):
        raise ValueError("basin_map must be a full array of length 2^n")

    if target_type == "marker_gene":
        if marker_idx is None:
            raise ValueError("marker_idx required for marker_gene target")
        if attractor_rep_states is None:
            raise ValueError("attractor_rep_states required for marker_gene target")

    qmask = 0
    for gi in subset:
        qmask |= 1 << int(gi)
    qmask = int(qmask)

    seen: Dict[int, int] = {}

    # Enumerate completions (recursive)
    for st in iter_states_consistent_with_mask_value(n, context_mask, context_value, max_enumerations=max_enumerations):
        key = st & qmask

        a_id = int(basin_map[st])
        if target_type == "attractor_id":
            yy = a_id
        else:
            rep = int(attractor_rep_states[a_id])
            yy = (rep >> int(marker_idx)) & 1

        prev = seen.get(key)
        if prev is None:
            seen[key] = yy
        elif prev != yy:
            return False

    return True


def find_min_k_sufficient_sets_basin(
    *,
    n: int,
    basin_map: np.ndarray,
    context_mask: int,
    context_value: int,
    candidate_genes: List[int],
    target_type: str,
    marker_idx: Optional[int] = None,
    attractor_rep_states: Optional[np.ndarray] = None,
    max_k: int = 4,
    max_sets_to_return: int = 50,
    max_enumerations: Optional[int] = None,
) -> Optional[MinKResult]:
    """Exact global minimal k-sufficient search using basin_map lookups (no simulation)."""
    n_tested = 0

    # k=0: Known without queries
    n_tested += 1
    if _is_y_function_of_subset_basin(
        n=n,
        basin_map=basin_map,
        attractor_rep_states=attractor_rep_states,
        context_mask=context_mask,
        context_value=context_value,
        subset=[],
        target_type=target_type,
        marker_idx=marker_idx,
        max_enumerations=max_enumerations,
    ):
        return MinKResult(k_min=0, min_sets=[[]], n_tested=n_tested, n_worlds=None)

    for k in range(1, max_k + 1):
        min_sets: List[List[int]] = []
        exceeded_cap = False
        for subset in combinations(candidate_genes, k):
            n_tested += 1
            if _is_y_function_of_subset_basin(
                n=n,
                basin_map=basin_map,
                attractor_rep_states=attractor_rep_states,
                context_mask=context_mask,
                context_value=context_value,
                subset=list(subset),
                target_type=target_type,
                marker_idx=marker_idx,
                max_enumerations=max_enumerations,
            ):
                min_sets.append(list(subset))
                if len(min_sets) >= max_sets_to_return:
                    exceeded_cap = True
                    break
        if exceeded_cap:
            # Too many sets at this k - skip this task for completeness
            return None
        if min_sets:
            return MinKResult(k_min=k, min_sets=min_sets, n_tested=n_tested, n_worlds=None)

    return None


def sample_context_reduce_candidates_fixed_points(
    *,
    omega_states: np.ndarray,
    true_state: int,
    n: int,
    forbid_genes: Optional[set] = None,
    min_candidates: int = 16,
    max_candidates: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray]]:
    """Reveal genes from true_state until candidate count is within [min,max].

    Returns: (observed_idx_vals, candidate_indices_in_omega)
    """
    forbid_genes = forbid_genes or set()
    if rng is None:
        rng = np.random.default_rng(0)

    genes = [i for i in range(n) if i not in forbid_genes]
    rng.shuffle(genes)

    observed: List[Tuple[int, int]] = []
    mask = 0
    pat = 0

    cand_idx = np.arange(omega_states.shape[0], dtype=np.int32)

    for g in genes:
        if min_candidates <= cand_idx.size <= max_candidates:
            break

        if cand_idx.size > max_candidates:
            v = bit(true_state, g)
            observed.append((g, v))
            mask |= 1 << g
            if v == 1:
                pat |= 1 << g

            m = np.uint64(mask)
            p = np.uint64(pat)
            keep = (omega_states[cand_idx] & m) == p
            cand_idx = cand_idx[keep]

            if cand_idx.size < min_candidates:
                return None
        else:
            return None

    if not (min_candidates <= cand_idx.size <= max_candidates):
        return None

    return observed, cand_idx


@dataclass
class ModelCache:
    """Per-model artifacts used by dataset curation."""
    model: str
    var_names: List[str]
    n_nodes: int

    # Regime A
    fixed_points: List[int]
    fixed_points_method: str
    fixed_points_is_capped: bool

    # Regime B
    landscape: Optional[AttractorLandscape] = None  # includes basin_map if computed


def _model_cache_path(cache_dir: str, model_name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    safe = model_name.replace('/', '_')
    return os.path.join(cache_dir, f"{safe}.model_cache.pkl")


def save_model_cache(cache: ModelCache, path: str, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_cache(path: str) -> ModelCache:
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_or_load_model_cache(
    model: Dict,
    cache_dir: str,
    *,
    regime_a_fp_cap: Optional[int] = 512,
    bruteforce_fp_max_nodes: int = 21,
    dyn_max_nodes: int = 21,
    compute_dyn: bool = True,
    max_attractors: int = 1000,
    timeout_seconds: Optional[int] = None,
    verbose: bool = False,
) -> ModelCache:
    """Compute (or load) a ModelCache.

    Notes:
      - For n<=dyn_max_nodes and compute_dyn=True, we compute/load a full basin_map cache
        and derive fixed points from the same attractor run (no duplication).
      - For larger networks, we fall back to SAT fixed-point enumeration or PyBoolNet.
      - If timeout_seconds is set, computation that exceeds the timeout will be aborted
        and the cache will have fixed_points_method='timeout'.
    """
    import time as _time
    model_name = model["model"]
    
    cache_path = _model_cache_path(cache_dir, model_name)
    if os.path.exists(cache_path):
        cache = load_model_cache(cache_path)
        if verbose:
            # Check if cached result was a failure (timeout, unavailable, etc.)
            fp_method = cache.fixed_points_method
            if fp_method in ("timeout", "unavailable", "bruteforce_failed"):
                print(f"    [{model_name}] Loaded from cache (FAILED: {fp_method})", flush=True)
            else:
                basin_info = "yes" if (cache.landscape and cache.landscape.basin_map is not None) else "no"
                print(f"    [{model_name}] Loaded from cache (fp={len(cache.fixed_points)}, basin_map={basin_info})", flush=True)
        return cache

    var_names = list(model["var"])
    n = len(var_names)
    
    if verbose:
        print(f"    [{model_name}] Computing (n={n}, timeout={timeout_seconds}s)... ", end="", flush=True)
    
    start_time = _time.time()

    landscape = None
    fixed_points: List[int] = []
    fp_method = ""
    fp_is_capped = False

    # Helper to run computation with optional timeout
    def _run_with_timeout(func, *args, **kwargs):
        """Run func with signal-based timeout."""
        if timeout_seconds is None or timeout_seconds <= 0:
            return func(*args, **kwargs)
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Computation timed out after {timeout_seconds}s")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    # If we can brute-force basins, do it once and reuse
    if compute_dyn and n <= dyn_max_nodes:
        try:
            landscape = _run_with_timeout(
                load_or_precompute_basin_cache,
                model,
                cache_dir=cache_dir,
                max_nodes=dyn_max_nodes,
                max_attractors=max_attractors,
                overwrite=False,
            )
            fps = [a.states[0] for a in landscape.attractors if a.is_fixed_point]
            fixed_points = [int(s) for s in fps]
            fp_method = "bruteforce_with_basins"
        except TimeoutError:
            fixed_points = []
            fp_method = "timeout"
            landscape = None

    else:
        # Fixed points only (no basins)
        if n <= bruteforce_fp_max_nodes:
            try:
                land = _run_with_timeout(
                    find_attractors,
                    model, method="bruteforce", max_bruteforce=bruteforce_fp_max_nodes, compute_basins=False
                )
                if land is not None:
                    fixed_points = [int(a.states[0]) for a in land.attractors if a.is_fixed_point]
                    fp_method = "bruteforce"
                else:
                    fixed_points = []
                    fp_method = "bruteforce_failed"
            except TimeoutError:
                fixed_points = []
                fp_method = "timeout"
        else:
            # SAT -> fixed points; fallback to PyBoolNet
            try:
                land = _run_with_timeout(find_fixed_points_sat, model["F"], model["I"], var_names)
                fixed_points = [int(a.states[0]) for a in land.attractors]
                fp_method = "sat"
            except TimeoutError:
                fixed_points = []
                fp_method = "timeout"
            except Exception:
                try:
                    land = _run_with_timeout(
                        find_attractors,
                        model, method="pyboolnet", compute_basins=False
                    )
                    if land is None:
                        raise RuntimeError("pyboolnet failed")
                    fixed_points = [int(a.states[0]) for a in land.attractors if a.is_fixed_point]
                    fp_method = "pyboolnet"
                except TimeoutError:
                    fixed_points = []
                    fp_method = "timeout"
                except Exception:
                    fixed_points = []
                    fp_method = "unavailable"

    # cap fixed points if requested
    if regime_a_fp_cap is not None and len(fixed_points) > regime_a_fp_cap:
        rng = np.random.default_rng(0)
        rng.shuffle(fixed_points)
        fixed_points = fixed_points[: int(regime_a_fp_cap)]
        fp_is_capped = True

    elapsed = _time.time() - start_time
    if verbose:
        basin_info = f", basin_map={'yes' if (landscape and landscape.basin_map is not None) else 'no'}"
        print(f"method={fp_method}, fp={len(fixed_points)}{basin_info}, {elapsed:.1f}s", flush=True)

    cache = ModelCache(
        model=model_name,
        var_names=var_names,
        n_nodes=n,
        fixed_points=fixed_points,
        fixed_points_method=fp_method,
        fixed_points_is_capped=fp_is_capped,
        landscape=landscape,
    )
    save_model_cache(cache, cache_path, overwrite=True)
    return cache


def feedback_core_genes(model: Dict) -> List[int]:
    """Heuristic: genes that lie in non-trivial feedback (SCCs of size>1 or self-loops).

    This is useful for identifying large networks whose fixed points may be controlled by
    a small effective core.
    """
    n = len(model["var"])
    F, I = model["F"], model["I"]
    # Build dependency graph: edge u->v if u appears as a regulator of v
    adj = [[] for _ in range(n)]
    for v in range(n):
        for u in I[v]:
            adj[int(u)].append(v)

    # Kosaraju SCC
    rev = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            rev[v].append(u)

    seen = [False]*n
    order = []

    def dfs1(u):
        seen[u]=True
        for v in adj[u]:
            if not seen[v]:
                dfs1(v)
        order.append(u)

    for u in range(n):
        if not seen[u]:
            dfs1(u)

    comp = [-1]*n
    comps = []
    def dfs2(u, cid):
        comp[u]=cid
        comps[cid].append(u)
        for v in rev[u]:
            if comp[v]==-1:
                dfs2(v,cid)

    for u in reversed(order):
        if comp[u]==-1:
            cid=len(comps)
            comps.append([])
            dfs2(u,cid)

    core = set()
    # SCC size>1
    for c in comps:
        if len(c) > 1:
            core.update(c)

    # self-loops: u regulates itself
    for v in range(n):
        if v in [int(u) for u in I[v]]:
            core.add(v)

    return sorted(core)


def enumerate_fixed_points_via_core(
    model: Dict,
    *,
    core_genes: Optional[List[int]] = None,
    max_core: int = 22,
) -> List[int]:
    """Enumerate fixed points by brute-forcing only a small feedback core.

    Works best when the feedback core size is small (<= max_core), even if n is large.

    Method:
      1) Choose a core set C (SCC/self-loop heuristic by default).
      2) Enumerate assignments to C (2^|C|).
      3) For each assignment, iterate synchronous update from the partially-fixed state,
         but **only** to settle feed-forward nodes (nodes outside C). Since outside-C can still
         depend on C, this converges quickly if the outside graph is mostly acyclic.
      4) Accept states that satisfy x = F(x) (fixed point check).

    Note: This is a heuristic acceleration; it's exact when the chosen core contains all
    feedback necessary for fixed points (often true for SCC-based cores).
    """
    n = len(model["var"])
    F, I = model["F"], model["I"]
    if core_genes is None:
        core_genes = feedback_core_genes(model)
    core_genes = list(core_genes)

    if len(core_genes) > max_core:
        return []

    # Precompute for fast update
    core_set = set(core_genes)
    # We'll brute-force core assignment and run a few steps to settle others, then verify fixed point exactly.
    fps: List[int] = []
    # positions order in core bits
    core_pos = core_genes

    # Build mask for core
    core_mask = 0
    for gi in core_pos:
        core_mask |= 1 << int(gi)

    # Enumerate all core assignments
    for a in range(1 << len(core_pos)):
        state = 0
        for j, gi in enumerate(core_pos):
            if (a >> j) & 1:
                state |= 1 << int(gi)

        # Iterate a bounded number of steps to try to settle non-core nodes
        # (If outside graph has no cycles, it converges in <=n steps.)
        for _ in range(n + 5):
            nxt = sync_update(F, I, state, n)
            # Keep core clamped to the chosen assignment
            nxt = (nxt & ~core_mask) | (state & core_mask)
            if nxt == state:
                break
            state = nxt

        # Verify exact fixed point (no clamping now)
        if sync_update(F, I, state, n) == state:
            fps.append(int(state))

    # Deduplicate
    fps = sorted(set(fps))
    return fps
