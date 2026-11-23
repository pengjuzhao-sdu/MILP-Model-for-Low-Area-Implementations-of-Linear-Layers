"""Generate candidate files for the simplified candidate-restricted model."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# Default command-line configuration.
DEFAULT_CIRCUIT_FILE = "circuit.txt"
GATE_SETS: Dict[str, Tuple[int, ...]] = {
    "23": (2, 3),
    "234": (2, 3, 4),
}
DEFAULT_GATE_SET = "234"
PRINT_ALL = True
LIMIT_PER_ARITY = None

ENABLE_OUTPUT_REUSE_OPTIONS = True
OUTPUT_REUSE_OPTIONS_PER_NODE_PER_ARITY = None


@dataclass(frozen=True)
class NodeSpec:
    """One internal node from the input XOR2 circuit after SSA conversion."""

    target: int
    fanins: Tuple[int, ...]
    y_idx: Optional[int] = None


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
    raise FileNotFoundError(f"Input file not found: {path}.{hint}")


def default_output_file(circuit_file: Path, gate_set: str) -> Path:
    return Path(f"{circuit_file.stem}_candidate_xor{gate_set}.txt")


def load_raw_code(file_path: str | Path) -> str:
    """Load XOR assignments from a text file."""

    path = resolve_input_file(file_path)
    line_re = re.compile(
        r"x\[\d+\]\s*=\s*x\[\d+\]\s*\^\s*x\[\d+\](?:\s+y\[\d+\])?"
    )

    xor_lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line_re.fullmatch(line):
            xor_lines.append(line)
    if xor_lines:
        return "\n".join(xor_lines)

    raise ValueError(f"No XOR assignments were found in: {path}")


def infer_n_inputs_from_raw_code(raw_code: str) -> int:
    """Infer the primary-input width from the largest x[i] in the raw circuit."""

    indices = [int(item) for item in re.findall(r"x\[(\d+)\]", raw_code)]
    if not indices:
        raise ValueError("Could not infer the number of inputs from an empty raw circuit.")
    return max(indices) + 1


def transform_to_ssa(code: str, n_inputs: int) -> str:
    """Convert in-place XOR2 updates to SSA form while preserving output tags."""

    current_map = {i: i for i in range(n_inputs)}
    next_idx = n_inputs
    output = []

    line_re = re.compile(r"x\[(\d+)\]\s*=\s*x\[\d+\]\s*\^\s*x\[(\d+)\](.*)")

    for raw_line in code.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = line_re.fullmatch(line)
        if not m:
            raise ValueError(f"Could not parse raw line: {raw_line}")

        a = int(m.group(1))
        b = int(m.group(2))
        suffix = m.group(3)

        if a not in current_map or b not in current_map:
            raise ValueError(f"Undefined variable in raw update: x[{a}] or x[{b}]")

        src1 = current_map[a]
        src2 = current_map[b]
        dest = next_idx

        output.append(f"x[{dest}] = x[{src1}] ^ x[{src2}]{suffix}")
        current_map[a] = dest
        next_idx += 1

    return "\n".join(output)


def parse_circuit(circuit_text: str) -> Tuple[List[NodeSpec], Dict[int, int]]:
    """Parse an SSA XOR circuit and return node specs plus output mapping."""

    line_re = re.compile(r"x\[(\d+)\]\s*=\s*(.*?)(?:\s+y\[(\d+)\])?$")
    fanin_re = re.compile(r"x\[(\d+)\]")

    ops: List[NodeSpec] = []
    y_map: Dict[int, int] = {}

    for raw_line in circuit_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = line_re.fullmatch(line)
        if not m:
            raise ValueError(f"Could not parse SSA line: {raw_line}")

        target = int(m.group(1))
        rhs = m.group(2)
        y = m.group(3)
        fanins = tuple(int(x) for x in fanin_re.findall(rhs))
        if len(fanins) < 2:
            raise ValueError(f"Could not parse SSA line: {raw_line}")

        y_idx = int(y) if y is not None else None
        ops.append(NodeSpec(target=target, fanins=fanins, y_idx=y_idx))

        if y_idx is not None:
            y_map[y_idx] = target

    return ops, y_map


def build_masks(ops: Sequence[NodeSpec], n_inputs: int) -> Dict[int, int]:
    """Encode each SSA node as a coefficient-vector mask over primary inputs."""

    masks: Dict[int, int] = {i: (1 << i) for i in range(n_inputs)}

    for op in ops:
        val = 0
        for u in op.fanins:
            if u not in masks:
                raise ValueError(f"Node x[{op.target}] depends on undefined input x[{u}]")
            val ^= masks[u]
        masks[op.target] = val

    return masks


def mask_to_basis(mask: int, n_inputs: int) -> List[int]:
    return [i for i in range(n_inputs) if (mask >> i) & 1]


def combo_xor_mask(combo: Iterable[int], masks: Dict[int, int]) -> int:
    val = 0
    for node in combo:
        val ^= masks[node]
    return val


def enumerate_implementations_for_node(
    target: int,
    masks: Dict[int, int],
    max_arity: int = 4,
    *,
    candidate_nodes: Optional[Sequence[int]] = None,
) -> Dict[int, List[Tuple[int, ...]]]:
    """Enumerate admissible XOR-ell options for a target coefficient vector.

    Each node is represented by its coefficient-vector mask. An option is valid
    when the XOR of the selected fan-in vectors equals the target vector.
    """

    if target not in masks:
        raise ValueError(f"target x[{target}] is undefined")

    if candidate_nodes is None:
        available_nodes = sorted(u for u in masks if u < target)
    else:
        available_nodes = sorted(u for u in candidate_nodes if u != target)

    target_mask = masks[target]
    results: Dict[int, List[Tuple[int, ...]]] = {k: [] for k in range(2, max_arity + 1)}
    mask_to_nodes: Dict[int, List[int]] = {}
    pair_index: Dict[int, List[Tuple[int, int]]] = {}

    for node in available_nodes:
        mask_to_nodes.setdefault(masks[node], []).append(node)

    pairs: List[Tuple[int, int, int]] = []
    for a_idx, a in enumerate(available_nodes):
        for b in available_nodes[a_idx + 1:]:
            pair_mask = masks[a] ^ masks[b]
            pairs.append((a, b, pair_mask))
            pair_index.setdefault(pair_mask, []).append((a, b))
            if pair_mask == target_mask:
                results[2].append((a, b))

    if max_arity >= 3:
        triples = set()
        for a, b, pair_mask in pairs:
            needed = target_mask ^ pair_mask
            for c in mask_to_nodes.get(needed, []):
                if c != a and c != b:
                    combo = tuple(sorted((a, b, c)))
                    if len(combo) == 3:
                        triples.add(combo)
        results[3] = sorted(triples)

    if max_arity >= 4:
        quads = set()
        for a, b, pair_mask in pairs:
            needed = target_mask ^ pair_mask
            for c, d in pair_index.get(needed, []):
                if a == c or a == d or b == c or b == d:
                    continue
                combo = tuple(sorted((a, b, c, d)))
                if len(combo) == 4:
                    quads.add(combo)
        results[4] = sorted(quads)

    return results


def support_size(mask: int) -> int:
    return mask.bit_count()


def generate_output_reuse_impls(
    target: int,
    masks: Dict[int, int],
    output_nodes: Set[int],
    anchor_nodes: Sequence[int],
    n_inputs: int,
    max_arity: int,
    *,
    limit_per_arity: Optional[int],
) -> Dict[int, List[Tuple[int, ...]]]:
    """Generate output-reuse options for output nodes in the candidate file.

    Main patterns:
        target_output = output_a ^ output_b ^ raw_input ^ internal_anchor

    Required output nodes must be kept in the final circuit, so reusing them as
    fan-ins can enrich the candidate file without adding new kept output nodes.
    The residual internal node is found by coefficient-vector equality.
    """

    impls = {k: [] for k in range(2, max_arity + 1)}
    if target not in output_nodes:
        return impls

    mask_to_anchors: Dict[int, List[int]] = {}
    for anchor in anchor_nodes:
        if anchor == target:
            continue
        mask_to_anchors.setdefault(masks[anchor], []).append(anchor)

    raw_inputs = list(range(n_inputs))
    other_outputs = sorted(output_nodes - {target})
    scored: Dict[int, List[Tuple[float, Tuple[int, ...]]]] = {
        k: [] for k in range(2, max_arity + 1)
    }

    def add_combo(combo: Tuple[int, ...]) -> None:
        combo = tuple(sorted(combo))
        if len(set(combo)) != len(combo) or target in combo:
            return
        arity = len(combo)
        if arity < 2 or arity > max_arity:
            return
        if combo_xor_mask(combo, masks) != masks[target]:
            return

        output_count = sum(1 for u in combo if u in output_nodes)
        input_count = sum(1 for u in combo if u < n_inputs)
        anchors = [u for u in combo if u in anchor_nodes]
        anchor_weight = sum(support_size(masks[u]) * 10.0 + u * 0.01 for u in anchors)
        score = output_count * 1000.0 + len(anchors) * 100.0 + input_count * 10.0 + anchor_weight
        scored[arity].append((score, combo))

    if max_arity >= 3:
        for o1, o2 in combinations(other_outputs, 2):
            residual = masks[target] ^ masks[o1] ^ masks[o2]

            # XOR3: two other outputs plus one anchor or raw input.
            for anchor in mask_to_anchors.get(residual, []):
                add_combo((o1, o2, anchor))
            for inp in raw_inputs:
                if masks[inp] == residual:
                    add_combo((o1, o2, inp))

            # XOR4: three other outputs plus one anchor or raw input.
            if max_arity >= 4:
                for output_c in other_outputs:
                    if output_c == o1 or output_c == o2:
                        continue
                    anchor_mask = residual ^ masks[output_c]
                    for inp in raw_inputs:
                        if masks[inp] == anchor_mask:
                            add_combo((o1, o2, output_c, inp))
                    for anchor in mask_to_anchors.get(anchor_mask, []):
                        add_combo((o1, o2, output_c, anchor))

    for arity, rows in scored.items():
        seen = set()
        rows.sort(key=lambda row: (-row[0], row[1]))
        for _, combo in rows:
            if combo in seen:
                continue
            seen.add(combo)
            impls[arity].append(combo)
            if limit_per_arity is not None and len(impls[arity]) >= limit_per_arity:
                break

    return impls


def merge_impls(
    base_impls: Dict[int, List[Tuple[int, ...]]],
    extra_impls: Dict[int, List[Tuple[int, ...]]],
) -> Dict[int, List[Tuple[int, ...]]]:
    """Merge topological-order and output-reuse options without duplicates."""

    merged: Dict[int, List[Tuple[int, ...]]] = {}

    for arity in sorted(set(base_impls) | set(extra_impls)):
        seen = set()
        cur: List[Tuple[int, ...]] = []

        for combo in base_impls.get(arity, []) + extra_impls.get(arity, []):
            key = tuple(sorted(combo))
            if key in seen:
                continue
            seen.add(key)
            cur.append(key)

        merged[arity] = cur

    return merged


def format_combo(combo: Tuple[int, ...]) -> str:
    return " ^ ".join(f"x[{u}]" for u in combo)


def generate_report(
    circuit_text: str,
    n_inputs: int,
    gate_set: str,
    print_all: bool = True,
    limit_per_arity: Optional[int] = None,
    *,
    enable_output_reuse_options: bool = False,
    output_reuse_options_per_node_per_arity: Optional[int] = None,
) -> str:
    """Generate the candidate file consumed by the simplified model."""

    if gate_set not in GATE_SETS:
        raise ValueError(f"Unsupported gate set: {gate_set}")
    allowed_gate_sizes = GATE_SETS[gate_set]
    max_arity = max(allowed_gate_sizes)
    candidate_policy = "free_topology" if enable_output_reuse_options else "strict_forward"

    ops, y_map = parse_circuit(circuit_text)
    masks = build_masks(ops, n_inputs)
    base_nodes = list(range(n_inputs)) + [op.target for op in ops]
    output_nodes = set(y_map.values())
    output_reuse_anchor_nodes = sorted(
        op.target for op in ops if op.target not in output_nodes
    )

    lines: List[str] = []
    lines.append("=== Candidate XOR Implementations ===")
    lines.append(f"n_inputs = {n_inputs}")
    lines.append(f"gate_set = {gate_set}")
    lines.append(f"allowed_gates = {allowed_gate_sizes}")
    lines.append(f"max_arity = {max_arity}")
    lines.append(f"candidate_policy = {candidate_policy}")
    if enable_output_reuse_options:
        lines.append("output_reuse = enabled")
        lines.append(f"output_reuse_limit = {output_reuse_options_per_node_per_arity}")
    else:
        lines.append("output_reuse = disabled")
    lines.append("")

    for op in ops:
        target_basis = mask_to_basis(masks[op.target], n_inputs)
        base_impls = enumerate_implementations_for_node(
            op.target,
            masks,
            max_arity=max_arity,
            candidate_nodes=[u for u in base_nodes if u < op.target],
        )

        extra_impls = {k: [] for k in range(2, max_arity + 1)}
        if enable_output_reuse_options:
            extra_impls = generate_output_reuse_impls(
                op.target,
                masks,
                output_nodes,
                output_reuse_anchor_nodes,
                n_inputs,
                max_arity,
                limit_per_arity=output_reuse_options_per_node_per_arity,
            )

        impls = merge_impls(base_impls, extra_impls)

        header = f"x[{op.target}] = {format_combo(op.fanins)}"
        if op.y_idx is not None:
            header += f"    (output y[{op.y_idx}])"

        lines.append(header)
        lines.append("  linear form: " + " ^ ".join(f"x[{i}]" for i in target_basis))

        for k in allowed_gate_sizes:
            sols = impls[k]
            lines.append(f"  XOR{k} implementations: {len(sols)}")

            if print_all and sols:
                shown = sols if limit_per_arity is None else sols[:limit_per_arity]
                for idx, combo in enumerate(shown, 1):
                    lines.append(f"    [{idx}] {format_combo(combo)}")

                if limit_per_arity is not None and len(sols) > limit_per_arity:
                    lines.append(f"    ... {len(sols) - limit_per_arity} more not shown")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate XOR implementations from an in-place XOR circuit txt file."
    )
    parser.add_argument(
        "circuit_file",
        nargs="?",
        default=DEFAULT_CIRCUIT_FILE,
        help="Input circuit txt file. The .txt suffix may be omitted. Default: circuit.txt.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        help=(
            "Where to save the generated candidate report. Default: "
            "<input_stem>_candidate_xor<gate_set>.txt."
        ),
    )
    parser.add_argument(
        "--gate-set",
        choices=tuple(GATE_SETS),
        default=DEFAULT_GATE_SET,
        help="Allowed XOR gate set. Use 23 for XOR2/XOR3 or 234 for XOR2/XOR3/XOR4. Default: 234.",
    )
    parser.add_argument(
        "--limit-per-arity",
        type=int,
        default=LIMIT_PER_ARITY,
        help="Limit printed implementations per arity. Omit to print all.",
    )
    parser.add_argument(
        "--no-output-reuse",
        action="store_true",
        help="Disable output-reuse candidate generation.",
    )
    parser.add_argument(
        "--output-reuse-limit",
        type=int,
        default=OUTPUT_REUSE_OPTIONS_PER_NODE_PER_ARITY,
        help="Per-node, per-arity limit for output-reuse candidates. Omit to enumerate all.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the full generated candidate report to console.",
    )
    args = parser.parse_args()

    circuit_file = resolve_input_file(args.circuit_file)
    enable_output_reuse = not args.no_output_reuse
    output_file = Path(args.output_file) if args.output_file else default_output_file(
        circuit_file,
        gate_set=args.gate_set,
    )

    raw_code = load_raw_code(circuit_file)
    n_inputs = infer_n_inputs_from_raw_code(raw_code)
    ssa_circuit = transform_to_ssa(raw_code, n_inputs=n_inputs)
    report = generate_report(
        circuit_text=ssa_circuit,
        n_inputs=n_inputs,
        gate_set=args.gate_set,
        print_all=PRINT_ALL,
        limit_per_arity=args.limit_per_arity,
        enable_output_reuse_options=enable_output_reuse,
        output_reuse_options_per_node_per_arity=args.output_reuse_limit,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    if args.print_report:
        print(report)
    print(f"\nInput file: {circuit_file.resolve()}")
    print(f"Inferred n_inputs: {n_inputs}")
    print(f"Saved candidate report to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
