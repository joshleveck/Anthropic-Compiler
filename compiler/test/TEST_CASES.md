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

Advanced VLIW scheduling is on by default; pass `--no-schedule` to emit one IR op per bundle (legacy behavior).

Compiler JSON output shape (for `load_from_rust.load_kernel_from_rust`):

```json
{"instructions":[{...bundles...}],"debug_info":{"scratch_map":{"0":["name",1]}}}
```

`scratch_map` keys are base scratch addresses (words); values are `[symbol_name, length_in_words]` like `problem.DebugInfo`. A legacy top-level JSON array of bundles is still accepted.

## Files

| File | What it stresses |
|------|------------------|
| `t01_scalar_load_store.c` | Scalar `__builtin_load`, `+`, `__builtin_store` |
| `t02_vector_load_store.c` | `__builtin_vload` / `__builtin_vstore` round-trip |
| `t03_vbroadcast_and_valu.c` | `__builtin_vbroadcast`, vector `+`, `^` |
| `t04_vselect.c` | Lane-wise `__builtin_vselect` |
| `t05_vhash.c` | `__builtin_vhash` |
| `t06_compile_time_if.c` | Folded `if` / unrolled loop |
| `t07_runtime_if.c` | Runtime scalar `if` / `else` + merge |
| `t08_lane_read_write.c` | `node[i] =` / `vec[i]` with const `i` |
| `t09_small_end_to_end.c` | Short multi-round kernel (gather + `__builtin_vhash` + idx walk) |
| `t10_vector_mod2.c` | **`val % __builtin_vbroadcast(2)`** — lane-wise `%` (sample idx update) |
| `t11_idx_val_walk.c` | **`idx*2+1` then `idx + (val%2)`** without `__builtin_vhash` |
| `t12_else_if_r.c` | **`if` / `else if` / `else`** on scalar `r` over rounds |
| `t13_two_batches.c` | **Two batch offsets** (`batch` 0 and 8) with `__builtin_vload`/`__builtin_vstore` |
| `t14_nested_batch_round.c` | **Nested** batch + round loops |
| `t15_wrap_at_forest_height.c` | **`idx = __builtin_vbroadcast(0)` when `r == FOREST_HEIGHT`** |
| `t16_vselect_idx_minus_one.c` | **Sample `r==1` path**: `t = idx-1`, `__builtin_vselect(t, vtwo, vone)` |
| `t17_flow_pause.c` | **`__builtin_flow_pause()`** → `("flow", ("pause",))` for trace alignment with reference yields |
| `t18_opposite_runtime_if.c` | Runtime scalar `if` branch opposite direction (`rounds == 0`) |
| `t19_scheduler_pack_independent_alu.c` | Scheduler packs independent scalar ALU ops into one bundle |
| `t20_scheduler_store_load_order.c` | `__builtin_sync()` fences store→load on the same address (stores only schedule on register deps) |
| `t21_scheduler_debug_after_producer.c` | Scheduler keeps `__builtin_debug` compare after producing instruction |
| `t22_sync_barrier.c` | `__builtin_sync()` emits `("flow",("sync",))` and enforces scheduling barrier |
| `t23_vector_multiply_add.c` | Fused `(a*b)+c` → `multiply_add` for vectors |
| `t24_vector_multiply_add_add.c` | Chained multiply-add style fusion |
| `t25_load_offset_gather.c` | `dst[lane]=load(addr[lane])` → `load_offset` |
| `t26_spawn.c` | `spawn(B,T,fn,...)` inlines `fn` **B×T** times; `__builtin_*` grid indices |

See also **`vector_lanes.c`**: `spawn` + `execute_rounds` with `__builtin_block_idx` / `__builtin_thread_idx` / `__builtin_block_dim`.

## Coverage vs `sample.c`

- Vector `%`, multiply/add chain, `else-if`, two-batch outer loop, nested batch×round, height wrap, and `__builtin_vselect` after `idx-1` are covered by **t10–t16** in addition to **t01–t09**.
- Still not duplicated in isolation: full **`r == round % (FOREST_HEIGHT+1)`** with **three-way** `r==0` / `r==1` / gather in one kernel (see `sample.c` + `t09` together).
