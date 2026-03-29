# Original Performance Take-Home Compiler Project

This repository contains my compiler-focused implementation of Anthropic's original performance take-home machine challenge. Instead of only hand-optimizing instructions, this project builds a restricted C-to-machine compiler and uses compiler passes to generate highly optimized programs for the simulator.

## What is in this repo

- A restricted C frontend and lowering pipeline in `compiler/` (`parse -> AST expansion -> VLIR -> machine lowering -> JSON bundles`)
- A VLIW-aware backend targeting the take-home simulator instruction model
- Optimization passes including loop unrolling, constant folding, instruction combining (including multiply-accumulate patterns), scheduling, and register-allocation improvements
- Runtime-oriented features used for optimization experiments, including `spawn` and `sync` builtins
- Debugging and analysis tooling for instruction dependencies and scratch/register pressure
- Benchmark and validation scripts (including `tests/submission_tests.py`) for cycle-count measurement

## Project result

The current optimized solution reaches **1732 cycles**, improved from an initial baseline around **248k cycles** during development, and beats my original hand-tuned target.

Compiler implementation and examples live under `compiler/` (including sample inputs in `compiler/test/`).
