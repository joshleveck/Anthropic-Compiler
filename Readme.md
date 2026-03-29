# Original Performance Take-Home Compiler

This repository contains a compiler-driven approach to Anthropic's "Original Performance Take-Home Machine" challenge. Rather than manually tuning assembly, this project implements a custom compiler targeting a VLIW architecture to automate high-level optimizations and instruction scheduling.

## **Project Result**

The compiler-generated solution achieves a cycle count of **1,732**, representing a **143x speedup** from the initial baseline and significantly outperforming my best manual optimization.

| Stage | Cycle Count | Improvement |
| :--- | :--- | :--- |
| **Initial Baseline (Unoptimized)** | 248,376 | — |
| **Manual Hand-Tuned Record** | 2,072 | 120x |
| **Compiler Optimized Output** | **1,732** | **143x** |

---

## **Technical Overview**

The core of the project is a restricted C-to-machine lowering pipeline built in Rust. It leverages a VLIW-aware backend to maximize instruction-level parallelism (ILP) within the simulator's hardware constraints.

### **The Pipeline**
`C Source` → `AST (lang_c)` → `AST Expansion` → `VLIR (Virtual Low-Level IR)` → `Machine Lowering` → `JSON Bundles`

### **Architecture Constraints**
The compiler targets a 23-slot VLIW machine with the following per-cycle limits:
* **6 VALU:** 8-element SIMD vector arithmetic.
* **12 ALU:** Scalar math.
* **2 Load / 2 Store:** Memory I/O.
* **1 Flow:** Control flow and jumps.

---

## **Key Optimizations**

* **AST-Level Loop Unrolling:** Eliminates jump overhead and enables aggressive constant folding for compile-time known loop bounds.
* **Instruction Combining:** Identifies and folds specific patterns (e.g., `(a + C) + (a << k)`) into single-cycle **Multiply-Accumulate** VALU instructions.
* **VLIW Scheduling:** Implements a dependency DAG with **Longest Critical Path (Depth-to-Sink)** priority. To maximize density, the scheduler deliberately bypasses WAR (Write-After-Read) hazard enforcement for this specific independent-batch kernel.
* **GPU-Inspired Concurrency:** Includes a `__builtin_spawn` primitive that allows for thread-level unrolling, managed by a custom `__builtin_sync` barrier to reduce peak register pressure.
* **Aggressive Register Allocation:** Deallocates registers within the same cycle of their last use to allow for denser scheduling within the 1,536-word scratch memory limit.

---

## **Repository Structure**

* `compiler/`: The core Rust implementation, including the parser, IR lowering, and VLIW scheduler.
* `compiler/test/`: Sample C inputs and reference test cases.
* `tools/`: Debugging utilities.
* `compiler/kernel.c`: Final C file that executes on 1732 cycles. 

---

## **Technical Deep Dive**

For a detailed breakdown of the architectural trade-offs, the "math tricks" used in the hash function, and the evolution of the scheduler, check out the full article:
**link coming soon**
