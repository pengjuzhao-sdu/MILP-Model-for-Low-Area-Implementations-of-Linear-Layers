# Beyond Stepwise Transformation: MILP Models for Low-Area Implementations of Linear Layers

This repository contains the code, input circuits, and implementation
results accompanying the paper *Beyond Stepwise Transformation: MILP Models for
Low-Area Implementations of Linear Layers*.

This artifact provides the source code, input circuits, and implementation
results used in the experiments of the paper. The scripts optimize XOR2
implementations of cryptographic linear layers into lower-area circuits using
multi-input XOR gates under the ASIC cost models considered in the paper.

## Requirements

The experiments were run in the following environment:

- Python 3.12.3
- Gurobi 12.0.3

## Cost Models and Gate Sets

The code implements the two GE cost models used in the paper, denoted by
`ASIC1` and `ASIC2`. Costs are stored as integer values scaled by 100 from the
GE values in Table 1 of the paper. All costs are multiplied by 100 in the code to use integer MILP coefficients;
this uniform scaling does not affect the optimization results.

| Library | XOR2 | XOR3 | XOR4 |
| --- | ---: | ---: | ---: |
| `ASIC1` | 2.00 | 3.25 | 5.00 |
| `ASIC2` | 3.33 | 4.66 | 5.99 |

The two allowed XOR gate sets are:

| Gate set | Allowed gates | Command-line value |
| --- | --- | --- |
| `S1` | XOR2, XOR3 | `--gate-set 23` |
| `S2` | XOR2, XOR3, XOR4 | `--gate-set 234` |

## Repository Layout

`Input_circuits/` contains the input XOR2 circuits used in the experiments.

`Implementations/` contains the corresponding implementation results reported
in the paper.

## Main Scripts

The commands below use the AES input XOR2 circuit, the gate set
`S2 = {2, 3, 4}`, and the `ASIC2` cost model as an example. For other
experiments, replace the input file, `--cost-library`, `--gate-set`, and the
area bound as needed.

`candidate.py` generates the precomputed candidate set used by the simplified
MILP.

```bash
python3 candidate.py example_AES.txt --gate-set 234
```

`simp_milp.py` solves the simplified MILP over a precomputed candidate set. The
lexicographic objective is replaced by a bounded feasibility condition.

```bash
python3 simp_milp.py example_AES_candidate_xor234.txt --cost-library ASIC2 --cost-bound 24000
```

`main_model.py` solves the main MILP model over the prescribed candidate space.

```bash
python3 main_model.py example_AES.txt --cost-library ASIC2 --gate-set 234
```

`main_model_with_aug.py` extends the candidate space by adding selected
coefficient vectors before solving the main MILP model.

```bash
python3 main_model_with_aug.py example_AES.txt --cost-library ASIC2 --gate-set 234 --weight2-limit 64 --local-pairs-limit 16
```