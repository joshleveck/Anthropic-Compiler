"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest
from problem import (
    Engine,
    DebugInfo,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

import kernel_schedule


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.instrs_pre_schedule = None
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def build_advanced(self, slots: list[list[tuple[Engine, tuple]]]):
        instrs = []
        for adv_slot in slots:
            if isinstance(adv_slot, list):
                instr = defaultdict(list)
                for engine, slot in adv_slot:
                    instr[engine].append(slot)
            else:
                instr = {adv_slot[0]: [adv_slot[1]]}

            instrs.append(instr)

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vscratch_const(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, VLEN)
            const_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, const_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(
                [
                    ("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))),
                    ("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))),
                ]
            )

            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(
                ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
            )

        return slots

    def vbuild_hash(self, val_hash_addr, vtmp1, vtmp2, round, batch, vconst1, vconst2):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Pack first stage operations - use all 6 VALU slots if possible
            slots.append(
                [
                    ("valu", (op1, vtmp1, val_hash_addr, self.vscratch_const(val1))),
                    ("valu", (op3, vtmp2, val_hash_addr, self.vscratch_const(val3))),
                ]
            )
            # op2 combines the results
            slots.append(("valu", (op2, val_hash_addr, vtmp1, vtmp2)))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        vzero_const = self.vscratch_const(0)
        vone_const = self.vscratch_const(1)
        vtwo_const = self.vscratch_const(2)

        self.alloc_scratch("vforest_values_p", VLEN)
        self.add(
            "valu",
            (
                "vbroadcast",
                self.scratch["vforest_values_p"],
                self.scratch["forest_values_p"],
            ),
        )

        self.alloc_scratch("vn_nodes", VLEN)
        self.add(
            "valu",
            (
                "vbroadcast",
                self.scratch["vn_nodes"],
                self.scratch["n_nodes"],
            ),
        )

        self.alloc_scratch("zero_tree_value")
        self.add(
            "load",
            ("load", self.scratch["zero_tree_value"], self.scratch["forest_values_p"]),
        )

        self.alloc_scratch("vone_tree_value", VLEN)
        one_const = self.scratch_const(1)
        self.add(
            "alu",
            ("+", tmp1, self.scratch["forest_values_p"], one_const),
        )
        # Load mem[forest_values_p + 1] into tmp1, then broadcast to VLEN.
        self.add("load", ("load", tmp1, tmp1))
        self.add("valu", ("vbroadcast", self.scratch["vone_tree_value"], tmp1))

        self.alloc_scratch("vtwo_tree_value", VLEN)
        two_const = self.scratch_const(2)
        self.add(
            "alu",
            ("+", tmp1, self.scratch["forest_values_p"], two_const),
        )
        # Load mem[forest_values_p + 2] into tmp1, then broadcast to VLEN.
        self.add("load", ("load", tmp1, tmp1))
        self.add("valu", ("vbroadcast", self.scratch["vtwo_tree_value"], tmp1))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scratch register banks with explicit temp reuse across stages.
        # Live values per lane:
        #   idx, val, tmp_a, tmp_b (vectors) + addr1, addr2 (scalars)
        vec_words_per_lane = 4 * VLEN
        scalar_words_per_lane = 2
        # lanes_by_scratch = max(1, scratch_headroom // words_per_lane)
        # lanes = max(1, min(batch_size // VLEN, lanes_by_scratch))
        lanes = 39
        if kernel_schedule.SCHEDULE_MODE == "identity":
            # In identity mode we skip repartitioning bundles, so cap lanes to keep
            # the emitted bundles under per-engine slot limits.
            lanes = min(lanes, 2)
        vtmp_idx = []
        vtmp_val = []
        tmp_addr1 = []
        tmp_addr2 = []
        vtmp_a = []
        vtmp_b = []
        for lane in range(lanes):
            vtmp_idx.append(self.alloc_scratch(f"vtmp_idx_{lane}", VLEN))
            vtmp_val.append(self.alloc_scratch(f"vtmp_val_{lane}", VLEN))
            tmp_addr1.append(self.alloc_scratch(f"tmp_addr1_{lane}"))
            tmp_addr2.append(self.alloc_scratch(f"tmp_addr2_{lane}"))
            vtmp_a.append(self.alloc_scratch(f"vtmp_a_{lane}", VLEN))
            vtmp_b.append(self.alloc_scratch(f"vtmp_b_{lane}", VLEN))

        for batch in range(0, batch_size, VLEN * lanes):
            lane_count = min(lanes, (batch_size - batch + VLEN - 1) // VLEN)
            i_consts = [
                self.scratch_const(batch + lane * VLEN) for lane in range(lane_count)
            ]

            # 1) Per-lane pointers + initial vload (idx/val live in scratch for all rounds).
            for lane in range(lane_count):
                body.append(
                    [
                        (
                            "alu",
                            (
                                "+",
                                tmp_addr1[lane],
                                self.scratch["inp_indices_p"],
                                i_consts[lane],
                            ),
                        ),
                        (
                            "alu",
                            (
                                "+",
                                tmp_addr2[lane],
                                self.scratch["inp_values_p"],
                                i_consts[lane],
                            ),
                        ),
                    ]
                )
                body.append(
                    [
                        ("load", ("vload", vtmp_idx[lane], tmp_addr1[lane])),
                        ("load", ("vload", vtmp_val[lane], tmp_addr2[lane])),
                    ]
                )

                # 2) Round-major: all lanes per tree round — helps the scheduler mix lanes.
                for round in range(rounds):
                    round_in_tree = round % (forest_height + 1)
                    is_zero_round = round_in_tree == 0
                    is_one_two_round = round_in_tree == 1
                    is_wrap_around_round = round_in_tree == forest_height
                    _ = (is_zero_round, is_one_two_round)

                    # for lane in range(lane_count):
                    if is_zero_round:
                        body.append(
                            [
                                (
                                    "valu",
                                    (
                                        "vbroadcast",
                                        vtmp_a[lane],
                                        self.scratch["zero_tree_value"],
                                    ),
                                )
                            ]
                        )
                    elif is_one_two_round:
                        body.append(
                            [
                                (
                                    "valu",
                                    ("-", vtmp_a[lane], vtmp_idx[lane], vone_const),
                                )
                            ]
                        )

                        body.append(
                            [
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        vtmp_a[lane],
                                        vtmp_a[lane],
                                        self.scratch["vtwo_tree_value"],
                                        self.scratch["vone_tree_value"],
                                    ),
                                )
                            ]
                        )
                    else:
                        body.append(
                            [
                                (
                                    "valu",
                                    (
                                        "+",
                                        vtmp_b[lane],
                                        self.scratch["vforest_values_p"],
                                        vtmp_idx[lane],
                                    ),
                                ),
                            ]
                        )
                        for li in range(0, VLEN, 2):
                            body.append(
                                [
                                    (
                                        "load",
                                        (
                                            "load_offset",
                                            vtmp_a[lane],
                                            vtmp_b[lane],
                                            li,
                                        ),
                                    ),
                                    (
                                        "load",
                                        (
                                            "load_offset",
                                            vtmp_a[lane],
                                            vtmp_b[lane],
                                            li + 1,
                                        ),
                                    ),
                                ]
                            )

                    body.append(
                        [
                            (
                                "valu",
                                ("^", vtmp_val[lane], vtmp_val[lane], vtmp_a[lane]),
                            ),
                        ]
                    )
                    for op1, val1, op2, op3, val3 in HASH_STAGES:
                        body.append(
                            [
                                (
                                    "valu",
                                    (
                                        op1,
                                        vtmp_a[lane],
                                        vtmp_val[lane],
                                        self.vscratch_const(val1),
                                    ),
                                ),
                                (
                                    "valu",
                                    (
                                        op3,
                                        vtmp_b[lane],
                                        vtmp_val[lane],
                                        self.vscratch_const(val3),
                                    ),
                                ),
                            ]
                        )
                        body.append(
                            [
                                (
                                    "valu",
                                    (
                                        op2,
                                        vtmp_val[lane],
                                        vtmp_a[lane],
                                        vtmp_b[lane],
                                    ),
                                ),
                            ]
                        )

                    body.append(
                        [
                            ("valu", ("%", vtmp_a[lane], vtmp_val[lane], vtwo_const)),
                            (
                                "valu",
                                (
                                    "multiply_add",
                                    vtmp_idx[lane],
                                    vtmp_idx[lane],
                                    vtwo_const,
                                    vone_const,
                                ),
                            ),
                        ]
                    )

                    body.append(
                        [
                            (
                                "valu",
                                ("+", vtmp_idx[lane], vtmp_idx[lane], vtmp_a[lane]),
                            ),
                        ]
                    )

                    if is_wrap_around_round:
                        body.append(
                            [
                                (
                                    "valu",
                                    ("+", vtmp_idx[lane], vzero_const, vzero_const),
                                ),
                            ]
                        )

                    # for lane in range(lane_count):
                    body.append(
                        [
                            ("store", ("vstore", tmp_addr1[lane], vtmp_idx[lane])),
                            ("store", ("vstore", tmp_addr2[lane], vtmp_val[lane])),
                        ]
                    )

        print(SCRATCH_SIZE - self.scratch_ptr)

        body_instrs = self.build_advanced(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

        self.instrs_pre_schedule = list(self.instrs)
        self.instrs = kernel_schedule.apply_schedule(
            self.instrs_pre_schedule, kernel_schedule.SCHEDULE_MODE
        )
        kernel_schedule.verify_slot_limits(self.instrs)


BASELINE = 147734
# Recorded with kernel_schedule.SCHEDULE_MODE=list, seed=123, do_kernel_test(10, 16, 256)
# With SCHEDULE_MODE=list (default in kernel_schedule)
BASELINE_KERNEL_CYCLES = 2080
# With SCHEDULE_MODE=identity (instrs == instrs_pre_schedule)
IDENTITY_BASELINE_KERNEL_CYCLES = 12053
# SHA256 of pickle(instrs_pre_schedule) for build_kernel(10, 2**11-1, 256, 16)
BASELINE_PRE_SCHEDULE_SHA256 = (
    "f7dc716c764cd4a05baa5511c2dc5802a3881cfd2744af11dd1fd99bc2fd2c91"
)


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_schedule_identity_preserves_program(self):
        old = kernel_schedule.SCHEDULE_MODE
        kernel_schedule.SCHEDULE_MODE = "identity"
        try:
            kb = KernelBuilder()
            kb.build_kernel(10, 2**11 - 1, 256, 16)
            self.assertEqual(kb.instrs, kb.instrs_pre_schedule)
        finally:
            kernel_schedule.SCHEDULE_MODE = old

    def test_pre_schedule_fingerprint_stable(self):
        kb = KernelBuilder()
        kb.build_kernel(10, 2**11 - 1, 256, 16)
        self.assertEqual(
            kernel_schedule.instrs_fingerprint(kb.instrs_pre_schedule),
            BASELINE_PRE_SCHEDULE_SHA256,
        )

    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        c = do_kernel_test(10, 16, 256)
        self.assertEqual(c, BASELINE_KERNEL_CYCLES)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
