"""MILP model for low-area XOR circuits under an allowed gate set."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gurobipy is required. Please install Gurobi and gurobipy before running this script."
    ) from exc


# Library-dependent area costs for XOR gates.
ASIC_GATE_COSTS: Dict[str, Dict[int, int]] = {
    "ASIC1": {2: 200, 3: 325, 4: 500},
    "ASIC2": {2: 333, 3: 466, 4: 599},
}
DEFAULT_COST_LIBRARY = "ASIC2"
GATE_SETS: Dict[str, Tuple[int, ...]] = {
    "23": (2, 3),
    "234": (2, 3, 4),
}
DEFAULT_GATE_SET = "234"


@dataclass(frozen=True)
class FuncOption:
    """One admissible implementation option for a candidate internal vector."""

    opt_id: int
    out_mask: int
    fanin_masks: Tuple[int, ...]
    gate_size: int
    cost: int


@dataclass
class FunctionProblem:
    """MILP-model instance derived from the input XOR2 circuit."""

    n_inputs: int
    raw_code: str
    ssa_circuit: str
    output_mask_by_index: Dict[int, int]
    original_fanins_of_mask: Dict[int, Tuple[int, ...]]
    input_masks: Tuple[int, ...]
    internal_masks: Tuple[int, ...]
    all_masks: Tuple[int, ...]
    gate_set: str
    allowed_gate_sizes: Tuple[int, ...]
    cost_library: str
    xor_costs: Dict[int, int]
    options_of: Dict[int, List[FuncOption]]
    used_by: Dict[int, List[Tuple[int, int]]]
    pruned_strict_superset_count: int

    @property
    def n_outputs(self) -> int:
        return len(self.output_mask_by_index)


@dataclass
class SolveResult:
    feasible: bool
    optimal: bool
    total_cost: Optional[int]
    gate_count: Optional[int]
    kept_masks: Tuple[int, ...]
    selected_options: Dict[int, FuncOption]
    order_by_mask: Dict[int, int]
    output_mask_by_index: Dict[int, int]
    status_code: int
    status_name: str
    mip_gap: Optional[float]
    problem_build_time_sec: Optional[float]
    model_build_time_sec: Optional[float]
    pre_solve_time_sec: Optional[float]
    solve_time_sec: Optional[float]
    total_runtime_sec: Optional[float]
    best_solution_first_seen_solve_time_sec: Optional[float]
    best_solution_first_seen_total_time_sec: Optional[float]


@dataclass
class ModelContext:
    problem: FunctionProblem
    model: gp.Model
    keep: Dict[int, gp.Var]
    pick: Dict[Tuple[int, int], gp.Var]
    order: Dict[int, gp.Var]
    total_cost: gp.Var
    gate_count: gp.Var


def strip_inline_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_raw_update_line(raw_line: str) -> Optional[Tuple[int, int, int, Optional[int]]]:
    import re

    line = strip_inline_comment(raw_line)
    if not line:
        return None

    line_re = re.compile(
        r"x\[(\d+)\]\s*=\s*x\[(\d+)\]\s*\^\s*x\[(\d+)\](?:\s+y\[(\d+)\])?\s*$"
    )
    match = line_re.fullmatch(line)
    if match is None:
        raise ValueError(f"Could not parse raw line: {raw_line}")

    target, left, right, out_idx = match.groups()
    return (
        int(target),
        int(left),
        int(right),
        int(out_idx) if out_idx is not None else None,
    )


def infer_n_inputs_from_raw_code(raw_code: str) -> int:
    seen_indices = set()

    for raw_line in raw_code.strip().splitlines():
        parsed = parse_raw_update_line(raw_line)
        if parsed is None:
            continue
        target, left, right, _ = parsed
        seen_indices.update((target, left, right))

    if not seen_indices:
        raise ValueError("Could not infer the number of inputs from an empty raw circuit.")

    return max(seen_indices) + 1


def mask_to_inputs(mask: int, n_inputs: int) -> Tuple[int, ...]:
    return tuple(i for i in range(n_inputs) if (mask >> i) & 1)


def mask_to_row(mask: int, n_inputs: int) -> Tuple[int, ...]:
    return tuple(1 if (mask >> i) & 1 else 0 for i in range(n_inputs))


def mask_label(mask: int, n_inputs: int) -> str:
    bits = "".join(str(bit) for bit in mask_to_row(mask, n_inputs))
    basis = mask_to_inputs(mask, n_inputs)
    basis_text = " ^ ".join(f"x[{i}]" for i in basis) if basis else "0"
    return f"{bits} ({basis_text})"


def strictly_contains_support(container_mask: int, target_mask: int) -> bool:
    return container_mask != target_mask and (container_mask & target_mask) == target_mask


def transform_to_ssa(code: str, n_inputs: int) -> str:
    """Convert in-place XOR2 updates into SSA form before vector extraction."""

    current_map = {i: i for i in range(n_inputs)}
    next_idx = n_inputs
    output: List[str] = []

    for raw_line in code.strip().splitlines():
        parsed = parse_raw_update_line(raw_line)
        if parsed is None:
            continue

        target, left, right, out_idx = parsed

        if left not in current_map or right not in current_map:
            raise ValueError(f"Uninitialized variable referenced in line: {raw_line}")

        src1 = current_map[left]
        src2 = current_map[right]
        dest = next_idx
        suffix = f" y[{out_idx}]" if out_idx is not None else ""
        output.append(f"x[{dest}] = x[{src1}] ^ x[{src2}]{suffix}")
        current_map[target] = dest
        next_idx += 1

    return "\n".join(output)


def parse_ssa_circuit(circuit_text: str) -> Tuple[List[Tuple[int, int, int]], Dict[int, int]]:
    import re

    line_re = re.compile(
        r"x\[(\d+)\]\s*=\s*x\[(\d+)\]\s*\^\s*x\[(\d+)\](?:\s+y\[(\d+)\])?"
    )

    ops: List[Tuple[int, int, int]] = []
    y_map: Dict[int, int] = {}

    for raw_line in circuit_text.strip().splitlines():
        line = strip_inline_comment(raw_line)
        if not line:
            continue

        match = line_re.fullmatch(line)
        if match is None:
            raise ValueError(f"Could not parse SSA line: {raw_line}")

        target, left, right, out_idx = match.groups()
        ops.append((int(target), int(left), int(right)))
        if out_idx is not None:
            y_map[int(out_idx)] = int(target)

    return ops, y_map


def build_masks(ops: Iterable[Tuple[int, int, int]], n_inputs: int) -> Dict[int, int]:
    """Encode each SSA node as a coefficient-vector mask over primary inputs."""

    masks: Dict[int, int] = {i: (1 << i) for i in range(n_inputs)}

    for target, left, right in ops:
        if left not in masks or right not in masks:
            raise ValueError(f"Node x[{target}] uses undefined inputs x[{left}], x[{right}]")
        masks[target] = masks[left] ^ masks[right]

    return masks


def build_function_problem_from_ssa(
    ssa_circuit: str,
    n_inputs: int,
    *,
    gate_set: str = DEFAULT_GATE_SET,
    raw_code: Optional[str] = None,
    cost_library: str = DEFAULT_COST_LIBRARY,
) -> FunctionProblem:
    """Build the coefficient-vector MILP instance from an SSA XOR2 circuit.

    Node identifiers are replaced by coefficient-vector masks. Implementation
    options are generated by XORing allowed-arity tuples from the vector domain.
    """

    if gate_set not in GATE_SETS:
        raise ValueError(f"Unsupported gate set: {gate_set}")
    if cost_library not in ASIC_GATE_COSTS:
        raise ValueError(f"Unsupported ASIC library: {cost_library}")
    allowed_gate_sizes = GATE_SETS[gate_set]
    xor_costs = ASIC_GATE_COSTS[cost_library]

    ops, y_map = parse_ssa_circuit(ssa_circuit)
    if not y_map:
        raise ValueError("No outputs were found. Please annotate output lines with y[k].")

    masks_by_node = build_masks(ops, n_inputs)

    all_masks = tuple(sorted(set(masks_by_node.values())))
    input_masks = tuple(sorted(1 << i for i in range(n_inputs)))
    input_mask_set = set(input_masks)
    internal_masks = tuple(sorted(mask for mask in all_masks if mask not in input_mask_set))
    internal_mask_set = set(internal_masks)
    output_mask_by_index = {
        out_idx: masks_by_node[node_idx]
        for out_idx, node_idx in sorted(y_map.items())
    }

    original_fanins_of_mask: Dict[int, Tuple[int, ...]] = {}
    for target, left, right in ops:
        out_mask = masks_by_node[target]
        if out_mask in input_mask_set:
            continue
        original_fanins_of_mask.setdefault(
            out_mask,
            tuple(sorted((masks_by_node[left], masks_by_node[right]))),
        )

    options_of: Dict[int, List[FuncOption]] = {mask: [] for mask in internal_masks}
    next_opt_id: Dict[int, int] = {mask: 0 for mask in internal_masks}
    pruned_strict_superset_count = 0

    # Enumerate the implementation set under the selected allowed gate set.
    for gate_size in allowed_gate_sizes:
        for combo in combinations(all_masks, gate_size):
            out_mask = 0
            for item in combo:
                out_mask ^= item

            if out_mask not in internal_mask_set:
                continue
            if out_mask in combo:
                continue
            if any(strictly_contains_support(fanin_mask, out_mask) for fanin_mask in combo):
                pruned_strict_superset_count += 1
                continue

            fanin_masks = tuple(sorted(combo))
            options_of[out_mask].append(
                FuncOption(
                    opt_id=next_opt_id[out_mask],
                    out_mask=out_mask,
                    fanin_masks=fanin_masks,
                    gate_size=gate_size,
                    cost=xor_costs[gate_size],
                )
            )
            next_opt_id[out_mask] += 1

    used_by: Dict[int, List[Tuple[int, int]]] = {mask: [] for mask in internal_masks}
    for out_mask, options in options_of.items():
        for option in options:
            for fanin_mask in option.fanin_masks:
                if fanin_mask in internal_mask_set:
                    used_by[fanin_mask].append((out_mask, option.opt_id))

    return FunctionProblem(
        n_inputs=n_inputs,
        raw_code=raw_code if raw_code is not None else ssa_circuit,
        ssa_circuit=ssa_circuit,
        output_mask_by_index=output_mask_by_index,
        original_fanins_of_mask=original_fanins_of_mask,
        input_masks=input_masks,
        internal_masks=internal_masks,
        all_masks=all_masks,
        gate_set=gate_set,
        allowed_gate_sizes=allowed_gate_sizes,
        cost_library=cost_library,
        xor_costs=xor_costs,
        options_of=options_of,
        used_by=used_by,
        pruned_strict_superset_count=pruned_strict_superset_count,
    )


def build_function_problem_from_raw_code(
    raw_code: str,
    n_inputs: int,
    *,
    gate_set: str = DEFAULT_GATE_SET,
    cost_library: str = DEFAULT_COST_LIBRARY,
) -> FunctionProblem:
    """Convert the input XOR2 circuit to SSA and build the MILP-model instance."""

    ssa_circuit = transform_to_ssa(raw_code, n_inputs)
    return build_function_problem_from_ssa(
        ssa_circuit,
        n_inputs,
        gate_set=gate_set,
        raw_code=raw_code,
        cost_library=cost_library,
    )


def load_problem_from_file(
    circuit_file: str | Path,
    n_inputs: Optional[int] = None,
    *,
    input_format: str = "raw",
    gate_set: str = DEFAULT_GATE_SET,
    cost_library: str = DEFAULT_COST_LIBRARY,
) -> FunctionProblem:
    path = Path(circuit_file)
    text = path.read_text(encoding="utf-8")

    if input_format == "raw":
        if n_inputs is None:
            n_inputs = infer_n_inputs_from_raw_code(text)
        return build_function_problem_from_raw_code(
            text,
            n_inputs,
            gate_set=gate_set,
            cost_library=cost_library,
        )
    if input_format == "ssa":
        if n_inputs is None:
            raise ValueError("--n-inputs is required when --input-format ssa is used.")
        return build_function_problem_from_ssa(
            text,
            n_inputs,
            gate_set=gate_set,
            raw_code=text,
            cost_library=cost_library,
        )

    raise ValueError(f"Unsupported input_format: {input_format}")


def compute_original_witness_upper_bound(problem: FunctionProblem) -> int:
    needed = set(problem.output_mask_by_index.values())
    stack = list(needed)
    while stack:
        mask = stack.pop()
        if mask in problem.input_masks:
            continue
        for fanin_mask in problem.original_fanins_of_mask.get(mask, ()):
            if fanin_mask not in needed:
                needed.add(fanin_mask)
                stack.append(fanin_mask)

    internal_needed = [mask for mask in needed if mask not in problem.input_masks]
    return len(internal_needed) * problem.xor_costs[2]


def compute_loose_feasible_upper_bound(problem: FunctionProblem) -> int:
    max_gate_cost = max(problem.xor_costs[gate_size] for gate_size in problem.allowed_gate_sizes)
    return len(problem.internal_masks) * max_gate_cost


def compute_output_lower_bound(problem: FunctionProblem) -> int:
    indiv_cost: Dict[int, int] = {mask: 0 for mask in problem.input_masks}
    for mask in problem.internal_masks:
        indiv_cost[mask] = 10**18

    changed = True
    while changed:
        changed = False
        for mask in problem.internal_masks:
            best = indiv_cost[mask]
            for option in problem.options_of[mask]:
                cand = option.cost
                ok = True
                for fanin_mask in option.fanin_masks:
                    cost = indiv_cost.get(fanin_mask, 10**18)
                    if cost >= 10**18:
                        ok = False
                        break
                    cand += cost
                if ok and cand < best:
                    best = cand
            if best < indiv_cost[mask]:
                indiv_cost[mask] = best
                changed = True

    lower = 0
    for out_mask in problem.output_mask_by_index.values():
        lower = max(lower, indiv_cost.get(out_mask, 0))
    return max(0, lower)


def _status_name(status_code: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(status_code, f"STATUS_{status_code}")


def build_gurobi_model(
    problem: FunctionProblem,
    *,
    cost_bound: Optional[int] = None,
    output_flag: int = 1,
    time_limit: Optional[float] = None,
    mip_focus: Optional[int] = None,
    threads: Optional[int] = None,
    presolve: Optional[int] = None,
) -> ModelContext:
    """Create the MILP model for selecting a low-area acyclic implementation."""

    model = gp.Model("liuqun3_function_gurobi")
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_focus is not None:
        model.Params.MIPFocus = int(mip_focus)
    if threads is not None:
        model.Params.Threads = int(threads)
    if presolve is not None:
        model.Params.Presolve = int(presolve)

    internal_masks = problem.internal_masks
    internal_mask_set = set(internal_masks)
    output_masks = set(problem.output_mask_by_index.values())
    max_order = len(internal_masks)

    keep = {
        mask: model.addVar(vtype=GRB.BINARY, name=f"keep_{mask}")
        for mask in internal_masks
    }
    pick: Dict[Tuple[int, int], gp.Var] = {}
    order = {
        mask: model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_order, name=f"order_{mask}")
        for mask in internal_masks
    }
    total_cost = model.addVar(vtype=GRB.INTEGER, lb=0, name="total_cost")
    gate_count = model.addVar(vtype=GRB.INTEGER, lb=0, name="gate_count")

    for mask in internal_masks:
        for option in problem.options_of[mask]:
            pick[(mask, option.opt_id)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"pick_{mask}_{option.opt_id}",
            )

    model.update()

    for out_mask in output_masks:
        if out_mask in internal_mask_set:
            model.addConstr(keep[out_mask] == 1, name=f"output_keep_{out_mask}")

    for mask in internal_masks:
        choices = [pick[(mask, option.opt_id)] for option in problem.options_of[mask]]
        if not choices:
            model.addConstr(keep[mask] == 0, name=f"no_option_{mask}")
            continue
        model.addConstr(
            gp.quicksum(choices) == keep[mask],
            name=f"choose_one_if_keep_{mask}",
        )

    # Topological order values are used to forbid cycles.
    for mask in internal_masks:
        model.addConstr(order[mask] >= keep[mask], name=f"order_lb_{mask}")
        model.addConstr(order[mask] <= max_order * keep[mask], name=f"order_ub_{mask}")

    for out_mask in internal_masks:
        for option in problem.options_of[out_mask]:
            pick_var = pick[(out_mask, option.opt_id)]
            for fanin_mask in option.fanin_masks:
                if fanin_mask not in internal_mask_set:
                    continue
                model.addConstr(
                    pick_var <= keep[fanin_mask],
                    name=f"dep_keep_{out_mask}_{option.opt_id}_{fanin_mask}",
                )
                model.addConstr(
                    order[fanin_mask] + 1
                    <= order[out_mask] + max_order * (1 - pick_var),
                    name=f"dep_order_{out_mask}_{option.opt_id}_{fanin_mask}",
                )

    for mask in internal_masks:
        if mask in output_masks:
            continue
        users = problem.used_by.get(mask, [])
        if not users:
            model.addConstr(keep[mask] == 0, name=f"never_used_{mask}")
            continue
        model.addConstr(
            keep[mask] <= gp.quicksum(pick[(user_mask, opt_id)] for (user_mask, opt_id) in users),
            name=f"no_dead_internal_{mask}",
        )

    cost_expr = gp.quicksum(
        option.cost * pick[(mask, option.opt_id)]
        for mask in internal_masks
        for option in problem.options_of[mask]
    )
    gate_count_expr = gp.quicksum(
        pick[(mask, option.opt_id)]
        for mask in internal_masks
        for option in problem.options_of[mask]
    )
    model.addConstr(total_cost == cost_expr, name="total_cost_def")
    model.addConstr(gate_count == gate_count_expr, name="gate_count_def")

    if cost_bound is not None:
        model.addConstr(total_cost <= int(cost_bound), name="cost_bound")

    model.ModelSense = GRB.MINIMIZE
    model.setObjectiveN(total_cost, index=0, priority=2, name="min_total_cost")
    model.setObjectiveN(gate_count, index=1, priority=1, name="min_gate_count")

    return ModelContext(
        problem=problem,
        model=model,
        keep=keep,
        pick=pick,
        order=order,
        total_cost=total_cost,
        gate_count=gate_count,
    )


def extract_result(
    context: ModelContext,
    *,
    problem_build_time_sec: Optional[float] = None,
    model_build_time_sec: Optional[float] = None,
    pre_solve_time_sec: Optional[float] = None,
    solve_time_sec: Optional[float] = None,
    total_runtime_sec: Optional[float] = None,
    best_solution_first_seen_solve_time_sec: Optional[float] = None,
    best_solution_first_seen_total_time_sec: Optional[float] = None,
) -> SolveResult:
    model = context.model
    problem = context.problem
    status_code = model.Status
    status_name = _status_name(status_code)
    feasible_statuses = {
        GRB.OPTIMAL,
        GRB.SUBOPTIMAL,
        GRB.TIME_LIMIT,
        GRB.USER_OBJ_LIMIT,
        GRB.INTERRUPTED,
    }
    has_incumbent = model.SolCount > 0

    if status_code not in feasible_statuses or not has_incumbent:
        return SolveResult(
            feasible=False,
            optimal=False,
            total_cost=None,
            gate_count=None,
            kept_masks=tuple(),
            selected_options={},
            order_by_mask={},
            output_mask_by_index=dict(problem.output_mask_by_index),
            status_code=status_code,
            status_name=status_name,
            mip_gap=None,
            problem_build_time_sec=problem_build_time_sec,
            model_build_time_sec=model_build_time_sec,
            pre_solve_time_sec=pre_solve_time_sec,
            solve_time_sec=solve_time_sec,
            total_runtime_sec=total_runtime_sec,
            best_solution_first_seen_solve_time_sec=best_solution_first_seen_solve_time_sec,
            best_solution_first_seen_total_time_sec=best_solution_first_seen_total_time_sec,
        )

    kept_masks = tuple(
        sorted(mask for mask in problem.internal_masks if context.keep[mask].X > 0.5)
    )
    selected_options: Dict[int, FuncOption] = {}
    for mask in kept_masks:
        for option in problem.options_of[mask]:
            if context.pick[(mask, option.opt_id)].X > 0.5:
                selected_options[mask] = option
                break

    order_by_mask = {mask: int(round(context.order[mask].X)) for mask in kept_masks}
    mip_gap = float(model.MIPGap) if hasattr(model, "MIPGap") else None

    return SolveResult(
        feasible=True,
        optimal=status_code == GRB.OPTIMAL,
        total_cost=int(round(context.total_cost.X)),
        gate_count=int(round(context.gate_count.X)),
        kept_masks=kept_masks,
        selected_options=selected_options,
        order_by_mask=order_by_mask,
        output_mask_by_index=dict(problem.output_mask_by_index),
        status_code=status_code,
        status_name=status_name,
        mip_gap=mip_gap,
        problem_build_time_sec=problem_build_time_sec,
        model_build_time_sec=model_build_time_sec,
        pre_solve_time_sec=pre_solve_time_sec,
        solve_time_sec=solve_time_sec,
        total_runtime_sec=total_runtime_sec,
        best_solution_first_seen_solve_time_sec=best_solution_first_seen_solve_time_sec,
        best_solution_first_seen_total_time_sec=best_solution_first_seen_total_time_sec,
    )


def solve_problem(
    problem: FunctionProblem,
    *,
    cost_bound: Optional[int] = None,
    output_flag: int = 1,
    time_limit: Optional[float] = None,
    mip_focus: Optional[int] = None,
    threads: Optional[int] = None,
    presolve: Optional[int] = None,
    overall_start_time: Optional[float] = None,
    problem_build_time_sec: Optional[float] = None,
) -> SolveResult:
    if overall_start_time is None:
        overall_start_time = time.perf_counter()

    model_build_start = time.perf_counter()
    context = build_gurobi_model(
        problem,
        cost_bound=cost_bound,
        output_flag=output_flag,
        time_limit=time_limit,
        mip_focus=mip_focus,
        threads=threads,
        presolve=presolve,
    )
    model_build_end = time.perf_counter()
    model_build_time_sec = model_build_end - model_build_start
    pre_solve_time_sec = model_build_end - overall_start_time

    best_tracker: Dict[str, object] = {
        "pair": None,
        "solve_time_sec": None,
        "total_time_sec": None,
    }

    def incumbent_time_callback(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return

        try:
            total_cost_value = int(round(model.cbGetSolution(context.total_cost)))
            gate_count_value = int(round(model.cbGetSolution(context.gate_count)))
        except gp.GurobiError:
            return

        candidate_pair = (total_cost_value, gate_count_value)
        best_pair = best_tracker["pair"]
        if best_pair is None or candidate_pair < best_pair:
            now = time.perf_counter()
            best_tracker["pair"] = candidate_pair
            best_tracker["solve_time_sec"] = now - solve_start
            best_tracker["total_time_sec"] = now - overall_start_time

    solve_start = time.perf_counter()
    context.model.optimize(incumbent_time_callback)
    solve_end = time.perf_counter()

    result = extract_result(
        context,
        problem_build_time_sec=problem_build_time_sec,
        model_build_time_sec=model_build_time_sec,
        pre_solve_time_sec=pre_solve_time_sec,
        solve_time_sec=solve_end - solve_start,
        total_runtime_sec=solve_end - overall_start_time,
        best_solution_first_seen_solve_time_sec=(
            float(best_tracker["solve_time_sec"])
            if best_tracker["solve_time_sec"] is not None
            else None
        ),
        best_solution_first_seen_total_time_sec=(
            float(best_tracker["total_time_sec"])
            if best_tracker["total_time_sec"] is not None
            else None
        ),
    )

    if result.feasible:
        final_pair = (result.total_cost, result.gate_count)
        tracked_pair = best_tracker["pair"]
        if tracked_pair != final_pair and result.solve_time_sec is not None:
            result.best_solution_first_seen_solve_time_sec = result.solve_time_sec
            result.best_solution_first_seen_total_time_sec = result.total_runtime_sec

    return result


def validate_solution(problem: FunctionProblem, result: SolveResult) -> List[str]:
    errors: List[str] = []
    if not result.feasible:
        errors.append(f"Solver status is {result.status_name}, expected a feasible incumbent.")
        return errors

    internal_mask_set = set(problem.internal_masks)
    kept_mask_set = set(result.kept_masks)
    output_masks = set(problem.output_mask_by_index.values())

    for out_idx, out_mask in result.output_mask_by_index.items():
        if out_mask in internal_mask_set and out_mask not in kept_mask_set:
            errors.append(f"Output y[{out_idx}] mask {out_mask} is not kept.")

    for mask, option in result.selected_options.items():
        if mask != option.out_mask:
            errors.append(f"Selected option mismatch for mask {mask}.")
        xor_mask = 0
        for fanin_mask in option.fanin_masks:
            xor_mask ^= fanin_mask
        if xor_mask != mask:
            errors.append(f"Option for mask {mask} does not xor back to the output mask.")
        if mask in option.fanin_masks:
            errors.append(f"Self-dependency detected for mask {mask}.")
        if len(option.fanin_masks) != option.gate_size:
            errors.append(f"Gate size mismatch for mask {mask}.")
        if option.fanin_masks != tuple(sorted(option.fanin_masks)):
            errors.append(f"Fanin order is not normalized for mask {mask}.")
        if len(option.fanin_masks) != len(set(option.fanin_masks)):
            errors.append(f"Repeated fanins detected for mask {mask}.")

    for mask in kept_mask_set:
        if mask in output_masks:
            continue
        option = result.selected_options.get(mask)
        if option is None:
            errors.append(f"Kept non-output mask {mask} has no selected option.")
            continue
        for fanin_mask in option.fanin_masks:
            if fanin_mask in internal_mask_set and fanin_mask not in kept_mask_set:
                errors.append(f"Mask {mask} uses internal fanin {fanin_mask} that is not kept.")

    for mask in kept_mask_set:
        if mask in output_masks:
            continue
        used = False
        for user_mask, option in result.selected_options.items():
            if mask in option.fanin_masks:
                used = True
                if result.order_by_mask.get(mask, 0) >= result.order_by_mask.get(user_mask, 0):
                    errors.append(f"Order violation: {mask} is not before {user_mask}.")
        if not used:
            errors.append(f"Dead internal function kept: {mask}.")

    expected_rows = {
        out_idx: mask_to_row(mask, problem.n_inputs)
        for out_idx, mask in problem.output_mask_by_index.items()
    }
    actual_rows = {
        out_idx: mask_to_row(mask, problem.n_inputs)
        for out_idx, mask in result.output_mask_by_index.items()
    }
    if expected_rows != actual_rows:
        errors.append("Output matrix rows do not match the original circuit.")

    return errors


def summarize_problem(problem: FunctionProblem) -> str:
    total_options = sum(len(options) for options in problem.options_of.values())
    xor2 = sum(
        1 for options in problem.options_of.values() for option in options if option.gate_size == 2
    )
    xor3 = sum(
        1 for options in problem.options_of.values() for option in options if option.gate_size == 3
    )
    xor4 = sum(
        1 for options in problem.options_of.values() for option in options if option.gate_size == 4
    )
    original_upper = compute_original_witness_upper_bound(problem)
    loose_upper = compute_loose_feasible_upper_bound(problem)
    lower_bound = compute_output_lower_bound(problem)
    lines = [
        "pool_mode         = NONE",
        f"gate_set          = {problem.gate_set}",
        f"allowed_gates     = {problem.allowed_gate_sizes}",
        f"cost_library      = {problem.cost_library}",
        f"xor_costs         = {problem.xor_costs}",
        "pool_config       = NONE",
        "augmented_masks   = 0",
        f"n_inputs          = {problem.n_inputs}",
        f"n_outputs         = {problem.n_outputs}",
        f"input_functions   = {len(problem.input_masks)}",
        f"internal_functions= {len(problem.internal_masks)}",
        f"all_functions     = {len(problem.all_masks)}",
        f"total_options     = {total_options}",
        f"pruned_superset   = {problem.pruned_strict_superset_count}",
        f"xor2_options      = {xor2}",
        f"xor3_options      = {xor3}",
        f"xor4_options      = {xor4}",
        f"lower_bound_cost  = {lower_bound}",
        f"original_upper    = {original_upper}",
        f"loose_upper       = {loose_upper}",
    ]
    return "\n".join(lines)



def format_optional_seconds(value: Optional[float]) -> str:
    return f"{value:.6f}" if value is not None else "N/A"

def summarize_solution(problem: FunctionProblem, result: SolveResult) -> str:
    lines = [
        f"solver_status     = {result.status_name} ({result.status_code})",
        f"feasible          = {result.feasible}",
        f"optimal           = {result.optimal}",
        f"total_cost        = {result.total_cost if result.total_cost is not None else 'N/A'}",
        f"gate_count        = {result.gate_count if result.gate_count is not None else 'N/A'}",
        f"best_incumbent_time = {format_optional_seconds(result.best_solution_first_seen_solve_time_sec)}",
        f"kept_internal     = {len(result.kept_masks)}",
    ]

    if result.mip_gap is not None:
        lines.append(f"mip_gap           = {result.mip_gap:.6f}")

    lines.extend(
        [
            f"problem_build_time_sec = {format_optional_seconds(result.problem_build_time_sec)}",
            f"model_build_time_sec   = {format_optional_seconds(result.model_build_time_sec)}",
            f"pre_solve_time_sec     = {format_optional_seconds(result.pre_solve_time_sec)}",
            f"solve_time_sec         = {format_optional_seconds(result.solve_time_sec)}",
            f"total_runtime_sec      = {format_optional_seconds(result.total_runtime_sec)}",
            f"best_first_seen_solve_time_sec = {format_optional_seconds(result.best_solution_first_seen_solve_time_sec)}",
            f"best_first_seen_total_time_sec = {format_optional_seconds(result.best_solution_first_seen_total_time_sec)}",
        ]
    )

    if not result.feasible:
        return "\n".join(lines)

    xor2 = sum(1 for option in result.selected_options.values() if option.gate_size == 2)
    xor3 = sum(1 for option in result.selected_options.values() if option.gate_size == 3)
    xor4 = sum(1 for option in result.selected_options.values() if option.gate_size == 4)
    lines.extend(
        [
            f"used_xor2         = {xor2}",
            f"used_xor3         = {xor3}",
            f"used_xor4         = {xor4}",
            "",
            "Selected internal functions:",
        ]
    )

    for mask in sorted(result.kept_masks, key=lambda item: (result.order_by_mask.get(item, 0), item)):
        option = result.selected_options[mask]
        fanins = ", ".join(mask_label(fanin_mask, problem.n_inputs) for fanin_mask in option.fanin_masks)
        lines.append(
            f"  order={result.order_by_mask[mask]:>2}  "
            f"mask={mask_label(mask, problem.n_inputs)}  "
            f"<- XOR{option.gate_size}({fanins})  cost={option.cost}"
        )

    lines.append("")
    lines.append("Output matrix rows:")
    for out_idx in sorted(problem.output_mask_by_index):
        mask = problem.output_mask_by_index[out_idx]
        lines.append(f"  y[{out_idx}] = {mask_label(mask, problem.n_inputs)}")

    return "\n".join(lines)


def format_best_circuit(problem: FunctionProblem, result: SolveResult) -> str:
    allowed_gate_text = " / ".join(f"XOR{gate_size}" for gate_size in problem.allowed_gate_sizes)
    xor2 = sum(1 for option in result.selected_options.values() if option.gate_size == 2)
    xor3 = sum(1 for option in result.selected_options.values() if option.gate_size == 3)
    xor4 = sum(1 for option in result.selected_options.values() if option.gate_size == 4)

    lines = [
        f"# solver_status = {result.status_name} ({result.status_code})",
        f"# feasible = {result.feasible}",
        f"# optimal = {result.optimal}",
        f"# total_cost = {result.total_cost if result.total_cost is not None else 'N/A'}",
        f"# gate_count = {result.gate_count if result.gate_count is not None else 'N/A'}",
        f"# best_incumbent_time = {format_optional_seconds(result.best_solution_first_seen_solve_time_sec)}",
        f"# problem_build_time_sec = {format_optional_seconds(result.problem_build_time_sec)}",
        f"# model_build_time_sec = {format_optional_seconds(result.model_build_time_sec)}",
        f"# pre_solve_time_sec = {format_optional_seconds(result.pre_solve_time_sec)}",
        f"# solve_time_sec = {format_optional_seconds(result.solve_time_sec)}",
        f"# total_runtime_sec = {format_optional_seconds(result.total_runtime_sec)}",
        f"# best_first_seen_solve_time_sec = {format_optional_seconds(result.best_solution_first_seen_solve_time_sec)}",
        f"# best_first_seen_total_time_sec = {format_optional_seconds(result.best_solution_first_seen_total_time_sec)}",
        f"# used_xor2 = {xor2}",
        f"# used_xor3 = {xor3}",
        f"# used_xor4 = {xor4}",
        f"# note = internal gates may be {allowed_gate_text}",
    ]

    if not result.feasible:
        lines.append("# no feasible incumbent, so no circuit is available")
        return "\n".join(lines) + "\n"

    ordered_masks = sorted(
        result.kept_masks,
        key=lambda item: (result.order_by_mask.get(item, 0), item),
    )
    node_of_mask: Dict[int, int] = {1 << i: i for i in range(problem.n_inputs)}
    next_node = problem.n_inputs
    output_indices_by_mask: Dict[int, List[int]] = {}
    for out_idx, mask in sorted(problem.output_mask_by_index.items()):
        output_indices_by_mask.setdefault(mask, []).append(out_idx)

    lines.append("")
    lines.append("# Internal Gates")
    for mask in ordered_masks:
        option = result.selected_options[mask]
        fanin_nodes: List[int] = []
        for fanin_mask in option.fanin_masks:
            if fanin_mask not in node_of_mask:
                raise ValueError(
                    f"Internal export failed: fanin mask {fanin_mask} has not been assigned a node."
                )
            fanin_nodes.append(node_of_mask[fanin_mask])

        node_idx = next_node
        next_node += 1
        node_of_mask[mask] = node_idx

        rhs = " ^ ".join(f"x[{node}]" for node in fanin_nodes)
        lines.append(
            f"x[{node_idx}] = {rhs}    "
            f"# order={result.order_by_mask[mask]}, XOR{option.gate_size}, cost={option.cost}, "
            f"function={mask_label(mask, problem.n_inputs)}"
        )

    lines.append("")
    lines.append("# Outputs")
    for out_idx in sorted(problem.output_mask_by_index):
        mask = problem.output_mask_by_index[out_idx]
        if mask not in node_of_mask:
            raise ValueError(
                f"Internal export failed: output y[{out_idx}] mask {mask} has no assigned node."
            )
        node_idx = node_of_mask[mask]
        lines.append(
            f"y[{out_idx}] = x[{node_idx}]    # {mask_label(mask, problem.n_inputs)}"
        )

    return "\n".join(lines) + "\n"


def build_report(
    problem: FunctionProblem,
    result: SolveResult,
    requested_cost_bound: Optional[int],
) -> str:
    errors = validate_solution(problem, result) if result.feasible else []
    chunks = [
        "=== Problem Summary ===",
        summarize_problem(problem),
        f"requested_cost_bound = {requested_cost_bound if requested_cost_bound is not None else 'NONE(direct optimize)'}",
        "",
        "=== Solution Summary ===",
        summarize_solution(problem, result),
        "",
        "=== Validation ===",
    ]

    if not result.feasible:
        chunks.append("SKIP: no feasible incumbent, so validation is not applicable.")
    elif errors:
        chunks.extend(f"FAIL: {error}" for error in errors)
    else:
        chunks.append("PASS: solution masks and output rows match the original circuit.")

    return "\n".join(chunks) + "\n"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_circuit_file = script_dir / "circuit.txt"

    parser = argparse.ArgumentParser(
        description=(
            "Read a two-input XOR circuit, build a configurable multi-input XOR MILP model, "
            "solve it with Gurobi, and export the best circuit."
        )
    )
    parser.add_argument(
        "circuit_file",
        type=str,
        nargs="?",
        default=str(default_circuit_file),
        help="Path to the input circuit file. Default: circuit.txt next to this script.",
    )
    parser.add_argument(
        "--n-inputs",
        type=int,
        default=None,
        help="Number of primary inputs. Optional for raw input, required for SSA input.",
    )
    parser.add_argument(
        "--input-format",
        choices=("raw", "ssa"),
        default="raw",
        help="Input format: raw in-place update form, or SSA form.",
    )
    parser.add_argument(
        "--cost-bound",
        type=int,
        default=None,
        help="Optional upper bound on total cost. If omitted, optimize directly.",
    )
    parser.add_argument(
        "--cost-library",
        choices=tuple(ASIC_GATE_COSTS),
        default=DEFAULT_COST_LIBRARY,
        help="ASIC standard-cell cost library. Default: ASIC2.",
    )
    parser.add_argument(
        "--gate-set",
        choices=tuple(GATE_SETS),
        default=DEFAULT_GATE_SET,
        help="Allowed XOR gate set. Use 23 for XOR2/XOR3 or 234 for XOR2/XOR3/XOR4. Default: 234.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=324000.0,
        help="Optional Gurobi time limit in seconds. Default: 324000.",
    )
    parser.add_argument(
        "--mip-focus",
        type=int,
        default=None,
        help="Optional Gurobi MIPFocus parameter.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional Gurobi Threads parameter.",
    )
    parser.add_argument(
        "--presolve",
        type=int,
        default=None,
        help="Optional Gurobi Presolve parameter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Path used to save the full textual report. Default: "
            "<input_stem>_xor<gate_set>_<cost_library>_report.txt next to this script."
        ),
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help=(
            "Path used to save the best circuit. Default: "
            "<input_stem>_xor<gate_set>_<cost_library>_circuit.txt next to this script."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Gurobi console logging. By default, solver logging is enabled.",
    )
    args = parser.parse_args()
    input_stem = Path(args.circuit_file).stem
    output_prefix = f"{input_stem}_xor{args.gate_set}_{args.cost_library.lower()}"
    output_file = (
        Path(args.output_file)
        if args.output_file is not None
        else script_dir / f"{output_prefix}_report.txt"
    )
    result_file = (
        Path(args.result_file)
        if args.result_file is not None
        else script_dir / f"{output_prefix}_circuit.txt"
    )

    program_start_time = time.perf_counter()
    problem = load_problem_from_file(
        args.circuit_file,
        args.n_inputs,
        input_format=args.input_format,
        gate_set=args.gate_set,
        cost_library=args.cost_library,
    )
    problem_build_time_sec = time.perf_counter() - program_start_time

    result = solve_problem(
        problem,
        cost_bound=args.cost_bound,
        output_flag=0 if args.quiet else 1,
        time_limit=args.time_limit,
        mip_focus=args.mip_focus,
        threads=args.threads,
        presolve=args.presolve,
        overall_start_time=program_start_time,
        problem_build_time_sec=problem_build_time_sec,
    )
    report = build_report(problem, result, args.cost_bound)
    best_circuit = format_best_circuit(problem, result)

    print(report, end="")
    result_file.write_text(best_circuit, encoding="utf-8")
    print(f"Saved best circuit to: {result_file.resolve()}")

    output_file.write_text(report, encoding="utf-8")
    print(f"Saved report to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
