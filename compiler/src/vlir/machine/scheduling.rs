use std::collections::HashSet;
use std::fs;

use petgraph::dot::{Config, Dot};
use petgraph::{Directed, Graph};

use crate::vlir::Function;

use super::ir::{
    def_regs, has_dependency, instr_engine_kind, uses_in_instruction, EngineKind,
};

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
    let mut crit = vec![0usize; n];
    for i in (0..n).rev() {
        let best_succ = succs[i].iter().map(|&s| crit[s]).max().unwrap_or(0);
        crit[i] = best_succ + insts[i].latency as usize;
    }

    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut scheduled = vec![false; n];
    let mut remaining = n;
    let mut cycles: ScheduledBlock = Vec::new();

    while remaining > 0 {
        ready.sort_by(|&a, &b| crit[b].cmp(&crit[a]).then(a.cmp(&b)));
        let mut cap_alu = EngineKind::Alu.slot_limit();
        let mut cap_valu = EngineKind::Valu.slot_limit();
        let mut cap_load = EngineKind::Load.slot_limit();
        let mut cap_store = EngineKind::Store.slot_limit();
        let mut cap_flow = EngineKind::Flow.slot_limit();
        let mut cap_debug = EngineKind::Debug.slot_limit();

        let mut picked: Vec<usize> = Vec::new();
        let mut next_ready: Vec<usize> = Vec::new();
        for idx in ready.drain(..) {
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
                next_ready.push(idx);
            }
        }

        // Should not happen with positive capacities; fallback to keep progress.
        if picked.is_empty() && !next_ready.is_empty() {
            picked.push(next_ready.remove(0));
        }

        for &idx in &picked {
            if scheduled[idx] {
                continue;
            }
            scheduled[idx] = true;
            remaining -= 1;
        }

        for &idx in &picked {
            for &s in &succs[idx] {
                indeg[s] -= 1;
                if indeg[s] == 0 && !scheduled[s] {
                    next_ready.push(s);
                }
            }
        }

        next_ready.sort_unstable();
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
        nodes.push(graph.add_node(format!("inst {:?} ", inst.kind)));
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
