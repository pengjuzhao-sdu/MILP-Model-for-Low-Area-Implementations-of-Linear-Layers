"""Solve the simplified candidate-restricted bounded feasibility model."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gurobipy is required. Please install Gurobi and gurobipy before running this script."
    ) from exc


# ============================================================
# Cost library
# ============================================================
DEFAULT_COST_LIBRARY = "ASIC2"


@dataclass(frozen=True)
class CostLibrary:
    """Library-dependent area costs for the supported XOR gate arities."""

    name: str
    xor2: int
    xor3: int
    xor4: int

    def cost_of(self, gate_size: int) -> int:
        mapping = {
            2: self.xor2,
            3: self.xor3,
            4: self.xor4,
        }
        if gate_size not in mapping:
            raise ValueError(f"Unsupported gate size XOR{gate_size}.")
        return mapping[gate_size]


COST_LIBRARIES: Dict[str, CostLibrary] = {
    "ASIC1": CostLibrary(name="ASIC1", xor2=200, xor3=325, xor4=500),
    "ASIC2": CostLibrary(name="ASIC2", xor2=333, xor3=466, xor4=599),
}


def get_cost_library(name: Optional[str] = None) -> CostLibrary:
    selected_name = name or DEFAULT_COST_LIBRARY
    if selected_name not in COST_LIBRARIES:
        raise ValueError(
            f"Unknown cost library '{selected_name}'. Available: {', '.join(sorted(COST_LIBRARIES))}"
        )
    return COST_LIBRARIES[selected_name]


def resolve_input_file(file_path: str | Path) -> Path:
    path = Path(file_path)
    candidates = [path]
    if path.suffix == "":
        candidates.append(path.with_suffix(".txt"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    available = ", ".join(sorted(p.name for p in Path.cwd().glob("*.txt")))
    hint = f" Available txt files: {available}" if available else ""
    raise FileNotFoundError(f"Candidate file not found: {path}.{hint}")


def default_output_file(candidate_file: Path, cost_library: CostLibrary, cost_bound: int) -> Path:
    """Build a deterministic result filename from the input instance settings."""

    return Path(
        f"simplified_{candidate_file.stem}_{cost_library.name.lower()}_bound{cost_bound}_result.txt"
    )


# ============================================================
# Parser
# ============================================================
@dataclass(frozen=True)
class Option:
    """One pre-enumerated implementation option for an internal node."""

    node: int
    opt_id: int
    gate_size: int               # 2 / 3 / 4
    fanins: Tuple[int, ...]      # e.g. (12, 42, 44)
    cost: int
    expr: str                    # original text for printing/debugging


@dataclass
class ProblemInstance:
    """Simplified-model instance parsed from a candidate file."""

    n_inputs: int
    max_node: int
    output_map: Dict[int, int]                 # y_idx -> x_idx
    options_of: Dict[int, List[Option]]
    cost_library: CostLibrary
    gate_set: Optional[str] = None
    candidate_policy: Optional[str] = None     # e.g. strict_forward / free_topology

    @property
    def outputs(self) -> set[int]:
        return set(self.output_map.values())

    @property
    def internal_nodes(self) -> List[int]:
        return list(range(self.n_inputs, self.max_node + 1))

    @property
    def n_outputs(self) -> int:
        return len(self.output_map)



def parse_instance(
    file_path: str | Path,
    cost_library: Optional[CostLibrary] = None,
) -> ProblemInstance:
    """Parse a candidate file for the simplified candidate-restricted model.

    Expected metadata includes n_inputs, gate_set, and candidate_policy.
    Each node section lists XOR2/XOR3/XOR4 implementation options.
    """

    cost_lib = cost_library or get_cost_library()

    path = resolve_input_file(file_path)

    lines = path.read_text(encoding="utf-8").splitlines()

    n_inputs: int | None = None
    max_node = -1
    output_map: Dict[int, int] = {}
    options_of: Dict[int, List[Option]] = {}
    gate_set: Optional[str] = None
    candidate_policy: Optional[str] = None

    current_node: int | None = None
    current_gate_size: int | None = None
    opt_id_counter: Dict[int, int] = {}

    n_inputs_re = re.compile(r"n_inputs\s*=\s*(\d+)")
    gate_set_re = re.compile(r"gate_set\s*=\s*(\S+)")
    policy_re = re.compile(r"candidate_policy\s*=\s*(\S+)")
    node_header_re = re.compile(r"x\[(\d+)\]\s*=.*")
    output_tag_re = re.compile(r"\(output\s*y\[(\d+)\]\)")
    gate_header_re = re.compile(r"XOR([234])\s+implementations:\s*(\d+)")
    option_line_re = re.compile(r"\s*\[(\d+)\]\s+(.*)")
    fanin_re = re.compile(r"x\[(\d+)\]")

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue

        m = n_inputs_re.search(line)
        if m:
            n_inputs = int(m.group(1))
            continue


        m = gate_set_re.search(line)
        if m:
            gate_set = m.group(1)
            continue
        m = policy_re.search(line)
        if m:
            candidate_policy = m.group(1)
            continue

        m = node_header_re.match(line.strip())
        if m:
            current_node = int(m.group(1))
            max_node = max(max_node, current_node)
            options_of.setdefault(current_node, [])
            opt_id_counter.setdefault(current_node, 0)
            current_gate_size = None
            mo = output_tag_re.search(line)
            if mo is not None:
                output_map[int(mo.group(1))] = current_node
            continue

        m = gate_header_re.search(line)
        if m:
            current_gate_size = int(m.group(1))
            continue

        m = option_line_re.match(line)
        if m and current_node is not None and current_gate_size is not None:
            expr = m.group(2).strip()
            fanins = tuple(int(x) for x in fanin_re.findall(expr))
            if len(fanins) != current_gate_size:
                raise ValueError(
                    f"Node x[{current_node}] option '{expr}' has {len(fanins)} fanins, "
                    f"but gate size is XOR{current_gate_size}."
                )

            opt = Option(
                node=current_node,
                opt_id=opt_id_counter[current_node],
                gate_size=current_gate_size,
                fanins=fanins,
                cost=cost_lib.cost_of(current_gate_size),
                expr=expr,
            )
            options_of[current_node].append(opt)
            opt_id_counter[current_node] += 1
            continue

    if n_inputs is None:
        raise ValueError("Could not parse n_inputs from candidate file.")
    if max_node < n_inputs:
        raise ValueError("No non-input nodes were parsed from candidate file.")

    instance = ProblemInstance(
        n_inputs=n_inputs,
        max_node=max_node,
        output_map=output_map,
        options_of=options_of,
        cost_library=cost_lib,
        gate_set=gate_set,
        candidate_policy=candidate_policy,
    )
    validate_instance(instance)
    return instance



def validate_instance(instance: ProblemInstance) -> None:
    """Check the candidate-file search space before building the MILP.

    Important: in output-reuse mode, fan-ins are not required to have smaller
    node IDs. The final topological order is decided by MILP variables.
    """

    for v in instance.internal_nodes:
        if v not in instance.options_of:
            raise ValueError(f"Missing candidate list for internal node x[{v}].")
        if not instance.options_of[v]:
            raise ValueError(f"Internal node x[{v}] has an empty candidate list.")

        seen = set()
        for opt in instance.options_of[v]:
            if opt.node != v:
                raise ValueError(f"Option node mismatch: option says {opt.node}, expected {v}.")
            if opt.opt_id in seen:
                raise ValueError(f"Duplicate opt_id={opt.opt_id} for node x[{v}].")
            seen.add(opt.opt_id)

            if opt.gate_size not in (2, 3, 4):
                raise ValueError(f"Illegal gate size XOR{opt.gate_size} at x[{v}].")
            if len(opt.fanins) != opt.gate_size:
                raise ValueError(
                    f"x[{v}] opt {opt.opt_id} has len(fanins)={len(opt.fanins)} != XOR{opt.gate_size}."
                )
            if len(set(opt.fanins)) != len(opt.fanins):
                raise ValueError(f"x[{v}] opt {opt.opt_id} contains duplicate fanins: {opt.fanins}")

            for u in opt.fanins:
                if u < 0 or u > instance.max_node:
                    raise ValueError(f"x[{v}] opt {opt.opt_id} references out-of-range x[{u}].")
                if u == v:
                    raise ValueError(
                        f"x[{v}] opt {opt.opt_id} references itself directly, which is a real cycle."
                    )

    for y_idx, out_v in instance.output_map.items():
        if out_v < instance.n_inputs or out_v > instance.max_node:
            raise ValueError(f"Output y[{y_idx}] maps to invalid internal node x[{out_v}].")



def build_used_by(instance: ProblemInstance) -> Dict[int, List[Tuple[int, int]]]:
    """Reverse fan-in table used by the no-dead-node constraint.

    used_by[u] = [(v, opt_id), ...]
    means internal node u is used by implementation option opt_id of node v.
    """

    used_by: Dict[int, List[Tuple[int, int]]] = {
        u: [] for u in instance.internal_nodes
    }
    for v, opts in instance.options_of.items():
        for opt in opts:
            for u in opt.fanins:
                if u >= instance.n_inputs:
                    used_by[u].append((v, opt.opt_id))
    return used_by



def summarize_instance(instance: ProblemInstance, output_preview: int = 8) -> str:
    """Return a compact summary of the parsed candidate-file search space."""

    num_internal = len(instance.internal_nodes)
    num_options = sum(len(opts) for opts in instance.options_of.values())
    xor2 = sum(1 for opts in instance.options_of.values() for opt in opts if opt.gate_size == 2)
    xor3 = sum(1 for opts in instance.options_of.values() for opt in opts if opt.gate_size == 3)
    xor4 = sum(1 for opts in instance.options_of.values() for opt in opts if opt.gate_size == 4)

    preview_items = sorted(instance.output_map.items())[:output_preview]
    preview = ", ".join(f"y[{y}] = x[{x}]" for y, x in preview_items)
    if instance.n_outputs > output_preview:
        preview += ", ..."

    lines = [
        f"n_inputs         = {instance.n_inputs}",
        f"n_outputs        = {instance.n_outputs}",
        f"max_node         = {instance.max_node}",
        f"internal_nodes   = {num_internal}",
        f"total_options    = {num_options}",
        f"xor2_options     = {xor2}",
        f"xor3_options     = {xor3}",
        f"xor4_options     = {xor4}",
        f"gate_set         = {instance.gate_set or 'unknown'}",
        f"candidate_policy = {instance.candidate_policy or 'unknown'}",
        f"cost_library     = {instance.cost_library.name}",
        f"xor_costs        = {{2: {instance.cost_library.xor2}, 3: {instance.cost_library.xor3}, 4: {instance.cost_library.xor4}}}",
        f"output_map       = {preview}",
    ]
    return "\n".join(lines)


# ============================================================
# Solver
# ============================================================
@dataclass
class SolveResult:
    feasible: bool
    kept_nodes: List[int]
    selected_options: Dict[int, Option]
    schedule_order: List[int]
    schedule_level: Dict[int, int]
    total_cost: Optional[float]
    solve_time: Optional[float]
    status_code: int
    status_name: str



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



def build_feasibility_model(
    instance: ProblemInstance,
    cost_bound: int,
    *,
    output_flag: int = 1,
    time_limit: Optional[float] = None,
    mip_focus: Optional[int] = None,
    threads: Optional[int] = None,
    presolve: Optional[int] = None,
) -> tuple[
    gp.Model,
    Dict[int, gp.Var],
    Dict[Tuple[int, int], gp.Var],
    Dict[int, gp.Var],
    gp.LinExpr,
]:
    """Build the simplified bounded-feasibility MILP under area bound B.

    Core logic:
    - the candidate file supplies a fixed implementation-option set;
    - schedule variables t[v] choose the final topological order;
    - selected dependency edges must go from earlier t to later t;
    - the selected graph is therefore acyclic and has no dead internal nodes.
    """

    model = gp.Model("xor_rewrite_feasibility_free_topology")
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_focus is not None:
        model.Params.MIPFocus = int(mip_focus)
    if threads is not None:
        model.Params.Threads = int(threads)
    if presolve is not None:
        model.Params.Presolve = int(presolve)

    nodes = instance.internal_nodes
    M = len(nodes)

    # --------------------
    # Variables
    # --------------------
    y: Dict[int, gp.Var] = {
        v: model.addVar(vtype=GRB.BINARY, name=f"y_{v}")
        for v in nodes
    }

    z: Dict[Tuple[int, int], gp.Var] = {}
    for v in nodes:
        for opt in instance.options_of[v]:
            z[(v, opt.opt_id)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"z_{v}_{opt.opt_id}",
            )

    # t[v] = 0 if node removed; otherwise 1..M is its topological order value.
    t: Dict[int, gp.Var] = {
        v: model.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name=f"t_{v}")
        for v in nodes
    }

    model.update()

    # --------------------
    # Constraints
    # --------------------

    # (1) Output nodes must exist.
    for v in instance.outputs:
        model.addConstr(y[v] == 1, name=f"output_keep_{v}")

    # (2) keep node <=> choose exactly one option.
    for v in nodes:
        model.addConstr(
            gp.quicksum(z[(v, opt.opt_id)] for opt in instance.options_of[v]) == y[v],
            name=f"choose_one_if_keep_{v}",
        )

    # (3) Tie schedule variables to keep/remove decisions.
    for v in nodes:
        model.addConstr(t[v] <= M * y[v], name=f"time_upper_{v}")
        model.addConstr(t[v] >= y[v], name=f"time_lower_{v}")

    # (4) Dependency closure + acyclic scheduling.
    for v in nodes:
        for opt in instance.options_of[v]:
            zv = z[(v, opt.opt_id)]
            for u in opt.fanins:
                if u >= instance.n_inputs:
                    # If this option is selected, the internal predecessor must exist.
                    model.addConstr(
                        zv <= y[u],
                        name=f"dep_keep_{v}_{opt.opt_id}_{u}",
                    )
                    # If this option is selected, predecessor must appear earlier.
                    model.addConstr(
                        t[u] + 1 <= t[v] + M * (1 - zv),
                        name=f"topo_{u}_before_{v}_opt_{opt.opt_id}",
                    )

    # (5) No dead internal nodes: a non-output node can survive only if some chosen option uses it.
    used_by = build_used_by(instance)
    for u in nodes:
        if u in instance.outputs:
            continue
        users = used_by[u]
        if users:
            model.addConstr(
                y[u] <= gp.quicksum(z[(v, opt_id)] for (v, opt_id) in users),
                name=f"no_dead_node_{u}",
            )
        else:
            model.addConstr(y[u] == 0, name=f"never_used_{u}")

    # (6) Area bound B.
    total_cost = gp.quicksum(
        opt.cost * z[(v, opt.opt_id)]
        for v in nodes
        for opt in instance.options_of[v]
    )
    model.addConstr(total_cost <= int(cost_bound), name="cost_bound")

    # Pure feasibility objective; the area is controlled by the bound above.
    model.setObjective(0.0, GRB.MINIMIZE)
    return model, y, z, t, total_cost



def solve_feasibility(
    instance: ProblemInstance,
    cost_bound: int,
    *,
    output_flag: int = 1,
    time_limit: Optional[float] = None,
    mip_focus: Optional[int] = None,
    threads: Optional[int] = None,
    presolve: Optional[int] = None,
) -> SolveResult:
    """Run the bounded-feasibility MILP and extract a selected circuit if one exists."""

    model, y, z, t, total_cost_expr = build_feasibility_model(
        instance,
        cost_bound,
        output_flag=output_flag,
        time_limit=time_limit,
        mip_focus=mip_focus,
        threads=threads,
        presolve=presolve,
    )
    model.optimize()

    solve_time = float(model.Runtime)
    status = model.Status
    status_name = _status_name(status)

    feasible_statuses = {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.USER_OBJ_LIMIT}
    has_incumbent = model.SolCount > 0

    if status not in feasible_statuses or not has_incumbent:
        return SolveResult(
            feasible=False,
            kept_nodes=[],
            selected_options={},
            schedule_order=[],
            schedule_level={},
            total_cost=None,
            solve_time=solve_time,
            status_code=status,
            status_name=status_name,
        )

    kept_nodes = [v for v in instance.internal_nodes if y[v].X > 0.5]
    selected_options: Dict[int, Option] = {}
    schedule_level: Dict[int, int] = {}

    for v in kept_nodes:
        chosen = None
        for opt in instance.options_of[v]:
            if z[(v, opt.opt_id)].X > 0.5:
                chosen = opt
                break
        if chosen is None:
            raise RuntimeError(f"Node x[{v}] is kept but no option is selected.")
        selected_options[v] = chosen
        schedule_level[v] = int(round(t[v].X))

    schedule_order = sorted(kept_nodes, key=lambda v: (schedule_level[v], v))
    total_cost = float(total_cost_expr.getValue())

    return SolveResult(
        feasible=True,
        kept_nodes=kept_nodes,
        selected_options=selected_options,
        schedule_order=schedule_order,
        schedule_level=schedule_level,
        total_cost=total_cost,
        solve_time=solve_time,
        status_code=status,
        status_name=status_name,
    )


# ============================================================
# Output formatter + CLI
# ============================================================
def count_gates(result: SolveResult) -> Dict[int, int]:
    counts = {2: 0, 3: 0, 4: 0}
    for opt in result.selected_options.values():
        counts[opt.gate_size] += 1
    return counts



def format_solution(instance: ProblemInstance, result: SolveResult) -> str:
    lines: List[str] = []
    lines.append(f"Solver status : {result.status_name} ({result.status_code})")
    lines.append(f"Feasible      : {result.feasible}")
    lines.append(
        f"Solve time    : {result.solve_time:.6f} s"
        if result.solve_time is not None else
        "Solve time    : N/A"
    )

    if not result.feasible:
        return "\n".join(lines)

    counts = count_gates(result)
    inv_output_map = {x: y for y, x in instance.output_map.items()}

    lines.append(f"Cost library  : {instance.cost_library.name}")
    lines.append(
        f"Gate costs    : XOR2={instance.cost_library.xor2}, "
        f"XOR3={instance.cost_library.xor3}, XOR4={instance.cost_library.xor4}"
    )
    lines.append(f"Total cost    : {int(result.total_cost) if result.total_cost is not None else 'N/A'}")
    lines.append(f"Kept nodes    : {len(result.kept_nodes)}")
    lines.append("Used gates    :")
    lines.append(f"  XOR2        : {counts[2]}")
    lines.append(f"  XOR3        : {counts[3]}")
    lines.append(f"  XOR4        : {counts[4]}")
    lines.append(f"  Total       : {counts[2] + counts[3] + counts[4]}")
    lines.append("")
    lines.append("=== Outputs ===")
    for y_idx, x_idx in sorted(instance.output_map.items()):
        t_level = result.schedule_level.get(x_idx, None)
        t_text = f"t={t_level}" if t_level is not None else "t=?"
        lines.append(f"y[{y_idx}] = x[{x_idx}]    ({t_text})")

    lines.append("")
    lines.append("=== Final selected implementation (sorted by model-chosen topology) ===")
    for v in result.schedule_order:
        opt = result.selected_options[v]
        rhs = " ^ ".join(f"x[{u}]" for u in opt.fanins)
        extra = f"(output y[{inv_output_map[v]}])" if v in inv_output_map else ""
        lines.append(
            f"[t={result.schedule_level[v]:>2}] x[{v}] = {rhs}    {extra}    "
            f"# XOR{opt.gate_size}, cost={opt.cost}, opt_id={opt.opt_id}"
        )
    return "\n".join(lines)



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-file solver for the simplified candidate-restricted model under a "
            "specified area bound B. The final topological order is chosen by the model."
        )
    )
    parser.add_argument(
        "candidate_file",
        help="Path to candidate implementations txt. The .txt suffix may be omitted.",
    )
    parser.add_argument(
        "--cost-bound",
        type=int,
        required=True,
        help="User-specified area bound B for the selected implementation options.",
    )
    parser.add_argument(
        "--cost-library",
        dest="cost_library",
        type=str,
        default=DEFAULT_COST_LIBRARY,
        choices=sorted(COST_LIBRARIES.keys()),
        help="ASIC standard-cell cost library. Default: ASIC2.",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="Optional Gurobi time limit in seconds")
    parser.add_argument("--mip-focus", type=int, default=None, help="Optional Gurobi MIPFocus parameter")
    parser.add_argument("--threads", type=int, default=None, help="Optional Gurobi Threads parameter")
    parser.add_argument("--presolve", type=int, default=None, help="Optional Gurobi Presolve parameter")
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Where to save the textual solution summary. Defaults to a name based on the input file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Gurobi console logging. By default, solver logging is enabled.",
    )
    args = parser.parse_args()

    cost_library = get_cost_library(args.cost_library)
    candidate_file = resolve_input_file(args.candidate_file)
    instance = parse_instance(candidate_file, cost_library=cost_library)

    print("=== Parsed instance summary ===")
    print(summarize_instance(instance))
    print(f"Cost bound       = {args.cost_bound}")
    print()

    result = solve_feasibility(
        instance,
        cost_bound=args.cost_bound,
        output_flag=0 if args.quiet else 1,
        time_limit=args.time_limit,
        mip_focus=args.mip_focus,
        threads=args.threads,
        presolve=args.presolve,
    )

    text = format_solution(instance, result)
    print(text)

    output_path = Path(args.output_file) if args.output_file else default_output_file(
        candidate_file,
        cost_library,
        args.cost_bound,
    )
    output_path.write_text(text + "\n", encoding="utf-8")
    print()
    print(f"Saved solution summary to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
