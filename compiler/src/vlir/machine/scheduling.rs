use std::collections::HashSet;
use std::fs;

use petgraph::dot::{Config, Dot};
use petgraph::{Directed, Graph};

use crate::vlir::Function;

use super::ir::{EngineKind, def_regs, has_dependency, instr_engine_kind, uses_in_instruction};

pub(crate) type ScheduledBlock = Vec<Vec<crate::vlir::InstrId>>;

pub(crate) fn build_scheduled_block(insts: &[&crate::vlir::Instruction]) -> ScheduledBlock {
    let n = insts.len();
    if n <= 1 {
        return insts.iter().map(|x| vec![x.id]).collect();
    }

    let mut uses: Vec<Vec<crate::vlir::RegisterId>> = Vec::with_capacity(n);
    let mut defs: Vec<Vec<crate::vlir::RegisterId>> = Vec::with_capacity(n);
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg: Vec<usize> = vec![0; n];
    for inst in insts {
        uses.push(uses_in_instruction(inst));
        defs.push(def_regs(inst));
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if has_dependency(insts[i], insts[j], &uses[i], &defs[i], &uses[j], &defs[j]) {
                succs[i].push(j);
                indeg[j] += 1;
            }
        }
    }

    create_graph(insts, &succs);

    // Critical-path priority: max successor depth + latency.
    // let mut crit = vec![0usize; n];
    // for i in (0..n).rev() {
    //     let path = succs[i].len();
    //     crit[i] = path;
    // }

    // Compute "distance to sink" for each instruction as the maximum number
    // of dependency edges on any path from this instruction to a sink.
    //
    // This is a DAG (edges only go from lower index -> higher index), so we can
    // do a simple reverse DP instead of BFS. Using the maximum depth ensures
    // we never schedule a producer after one of its dependent sinks would
    // require us to be earlier (deadlock avoidance).
    let mut depth = vec![0usize; n];
    let mut max_depth = 0usize;
    for i in (0..n).rev() {
        if succs[i].is_empty() {
            depth[i] = 0;
        } else {
            let m = succs[i]
                .iter()
                .map(|&s| depth[s])
                .max()
                .expect("non-sink must have at least one successor");
            depth[i] = m + 1;
        }
        max_depth = max_depth.max(depth[i]);
    }

    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    ready.sort_by(|&a, &b| depth[b].cmp(&depth[a]).then(a.cmp(&b)));
    let mut scheduled = vec![false; n];
    let mut remaining = n;
    let mut cycles: ScheduledBlock = Vec::new();

    while remaining > 0 {
        let mut cap_alu = EngineKind::Alu.slot_limit();
        let mut cap_valu = EngineKind::Valu.slot_limit();
        let mut cap_load = EngineKind::Load.slot_limit();
        let mut cap_store = EngineKind::Store.slot_limit();
        let mut cap_flow = EngineKind::Flow.slot_limit();
        let mut cap_debug = EngineKind::Debug.slot_limit();

        let mut picked: Vec<usize> = Vec::new();
        let mut carry_ready: Vec<usize> = Vec::new();
        for idx in ready {
            if scheduled[idx] {
                continue;
            }
            let kind = instr_engine_kind(insts[idx]);
            let can_fit = match kind {
                EngineKind::Alu => cap_alu > 0,
                EngineKind::Valu => cap_valu > 0,
                EngineKind::Load => cap_load > 0,
                EngineKind::Store => cap_store > 0,
                EngineKind::Flow => cap_flow > 0,
                EngineKind::Debug => cap_debug > 0,
            };
            if can_fit {
                match kind {
                    EngineKind::Alu => cap_alu -= 1,
                    EngineKind::Valu => cap_valu -= 1,
                    EngineKind::Load => cap_load -= 1,
                    EngineKind::Store => cap_store -= 1,
                    EngineKind::Flow => cap_flow -= 1,
                    EngineKind::Debug => cap_debug -= 1,
                }
                picked.push(idx);
            } else {
                carry_ready.push(idx);
            }
        }
        // Ensure forward progress even if capacity accounting is somehow exhausted.
        if picked.is_empty() && !carry_ready.is_empty() {
            picked.push(carry_ready.remove(0));
        }

        let mut next_ready: Vec<usize> = carry_ready;
        for &idx in &picked {
            if scheduled[idx] {
                continue;
            }
            scheduled[idx] = true;
            remaining -= 1;
            for &s in &succs[idx] {
                indeg[s] -= 1;
                if indeg[s] == 0 && !scheduled[s] {
                    next_ready.push(s);
                }
            }
        }

        next_ready.sort_by(|&a, &b| depth[b].cmp(&depth[a]).then(a.cmp(&b)));
        next_ready.dedup();
        ready = next_ready;
        cycles.push(picked.into_iter().map(|i| insts[i].id).collect());
    }

    cycles
}

/// One emitted instruction per bundle (original lowering behavior).
fn sequential_emitted_blocks(
    func: &Function,
    emitted: &HashSet<crate::vlir::InstrId>,
) -> Vec<ScheduledBlock> {
    func.blocks
        .iter()
        .map(|block| {
            block
                .instructions
                .iter()
                .filter(|inst| emitted.contains(&inst.id))
                .map(|inst| vec![inst.id])
                .collect()
        })
        .collect()
}

pub(crate) fn schedule_emitted_blocks(
    func: &Function,
    emitted: &HashSet<crate::vlir::InstrId>,
    advanced_scheduling: bool,
) -> Vec<ScheduledBlock> {
    if !advanced_scheduling {
        return sequential_emitted_blocks(func, emitted);
    }
    let mut out = Vec::with_capacity(func.blocks.len());
    for block in &func.blocks {
        let emitted_insts: Vec<&crate::vlir::Instruction> = block
            .instructions
            .iter()
            .filter(|inst| emitted.contains(&inst.id))
            .collect();
        out.push(build_scheduled_block(&emitted_insts));
    }
    out
}
fn create_graph(insts: &[&crate::vlir::Instruction], succs: &[Vec<usize>]) {
    let mut graph = Graph::<String, (), Directed>::new();

    let mut nodes = vec![];

    for inst in insts.iter() {
        nodes.push(graph.add_node(format!("{:?} ", inst.kind)));
    }

    for (i, _inst) in insts.iter().enumerate() {
        for succ in &succs[i] {
            graph.add_edge(nodes[i], nodes[*succ], ());
        }
    }

    fs::write(
        "graph.dot",
        format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel])),
    )
    .unwrap();
}
