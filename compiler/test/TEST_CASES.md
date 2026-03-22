# Compiler Feature Test Cases

Small `kernel()` programs to isolate lowering paths used by `sample.c` and the full kernel.

## How to run

From repo root:

```bash
python compiler_harness.py
```

Or compile one file:

```bash
cd compiler && cargo run --quiet -- test/t10_vector_mod2.c output/t.json
```

## Files

| File | What it stresses |
|------|------------------|
| `t01_scalar_load_store.c` | Scalar `load`, `+`, `store` |
| `t02_vector_load_store.c` | `vload` / `vstore` round-trip |
| `t03_vbroadcast_and_valu.c` | `vbroadcast`, vector `+`, `^` |
| `t04_vselect.c` | Lane-wise `vselect` |
| `t05_vhash.c` | Inlined `vhash` |
| `t06_compile_time_if.c` | Folded `if` / unrolled loop |
| `t07_runtime_if.c` | Runtime scalar `if` / `else` + merge |
| `t08_lane_read_write.c` | `node[i] =` / `vec[i]` with const `i` |
| `t09_small_end_to_end.c` | Short multi-round kernel (gather + vhash + idx walk) |
| `t10_vector_mod2.c` | **`val % vbroadcast(2)`** — lane-wise `%` (sample idx update) |
| `t11_idx_val_walk.c` | **`idx*2+1` then `idx + (val%2)`** without vhash |
| `t12_else_if_r.c` | **`if` / `else if` / `else`** on scalar `r` over rounds |
| `t13_two_batches.c` | **Two batch offsets** (`batch` 0 and 8) with `vload`/`vstore` |
| `t14_nested_batch_round.c` | **Nested** batch + round loops |
| `t15_wrap_at_forest_height.c` | **`idx = vbroadcast(0)` when `r == FOREST_HEIGHT`** |
| `t16_vselect_idx_minus_one.c` | **Sample `r==1` path**: `t = idx-1`, `vselect(t, vtwo, vone)` |
| `t17_flow_pause.c` | **`flow_pause()`** → `("flow", ("pause",))` for trace alignment with reference yields |

## Coverage vs `sample.c`

- Vector `%`, multiply/add chain, `else-if`, two-batch outer loop, nested batch×round, height wrap, and `vselect` after `idx-1` are covered by **t10–t16** in addition to **t01–t09**.
- Still not duplicated in isolation: full **`r == round % (FOREST_HEIGHT+1)`** with **three-way** `r==0` / `r==1` / gather in one kernel (see `sample.c` + `t09` together).
