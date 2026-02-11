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
from math import ceil

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
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
        tmp2 = self.alloc_scratch("tmp2")
        vtmp1 = self.alloc_scratch("vtmp1", VLEN)
        vtmp2 = self.alloc_scratch("vtmp2", VLEN)
        vtmp3 = self.alloc_scratch("vtmp3", VLEN)
        vtmp4 = self.alloc_scratch("vtmp4", VLEN)
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

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scratch registers
        vtmp_idx = self.alloc_scratch("vtmp_idx", VLEN)
        vtmp_val = self.alloc_scratch("vtmp_val", VLEN)
        vtmp_node_val = self.alloc_scratch("vtmp_node_val", VLEN)
        tmp_addr1 = self.alloc_scratch("tmp_addr1")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")
        vtmp_addr3 = self.alloc_scratch("vtmp_addr3", VLEN)

        # Focus on aggressive instruction packing

        for round in range(rounds):
            for batch in range(0, batch_size, VLEN):
                i_const = self.scratch_const(batch)

                # Aggressively pack: compute addresses + load in parallel
                body.append(
                    [
                        (
                            "alu",
                            ("+", tmp_addr1, self.scratch["inp_indices_p"], i_const),
                        ),
                        (
                            "alu",
                            ("+", tmp_addr2, self.scratch["inp_values_p"], i_const),
                        ),
                    ]
                )
                body.append(
                    [
                        ("load", ("vload", vtmp_idx, tmp_addr1)),
                        ("load", ("vload", vtmp_val, tmp_addr2)),
                    ]
                )

                # Compute node addresses
                body.append(
                    [
                        (
                            "valu",
                            (
                                "+",
                                vtmp_addr3,
                                self.scratch["vforest_values_p"],
                                vtmp_idx,
                            ),
                        ),
                    ]
                )

                # Load node values - pack loads maximally
                for li in range(0, VLEN, 2):
                    body.append(
                        [
                            ("load", ("load_offset", vtmp_node_val, vtmp_addr3, li)),
                            (
                                "load",
                                ("load_offset", vtmp_node_val, vtmp_addr3, li + 1),
                            ),
                        ]
                    )

                # XOR - can pack with first hash stage prep
                body.append(
                    [
                        ("valu", ("^", vtmp_val, vtmp_val, vtmp_node_val)),
                    ]
                )

                # Hash - optimized
                body.extend(
                    self.vbuild_hash(vtmp_val, vtmp1, vtmp2, round, batch, vtmp3, vtmp4)
                )

                # Update indices - pack all operations
                body.append(
                    [
                        ("valu", ("%", vtmp1, vtmp_val, vtwo_const)),
                        (
                            "valu",
                            (
                                "multiply_add",
                                vtmp_idx,
                                vtmp_idx,
                                vtwo_const,
                                vone_const,
                            ),
                        ),
                    ]
                )
                body.append(
                    [
                        ("valu", ("+", vtmp_idx, vtmp_idx, vtmp1)),
                    ]
                )

                # Wrap indices
                body.append(
                    [
                        ("valu", ("<", vtmp1, vtmp_idx, self.scratch["vn_nodes"])),
                    ]
                )
                body.append(
                    [
                        ("flow", ("vselect", vtmp_idx, vtmp1, vtmp_idx, vzero_const)),
                    ]
                )

                # Store - pack stores
                body.append(
                    [
                        ("store", ("vstore", tmp_addr1, vtmp_idx)),
                        ("store", ("vstore", tmp_addr2, vtmp_val)),
                    ]
                )

        print(SCRATCH_SIZE - self.scratch_ptr)

        body_instrs = self.build_advanced(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


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
        do_kernel_test(10, 16, 256)


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
