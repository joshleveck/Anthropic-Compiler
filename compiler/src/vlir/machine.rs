use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::fs;

use petgraph::dot::{Config, Dot};
use petgraph::graph::UnGraph;
use petgraph::{Directed, Graph};
use visgraph::settings::SettingsBuilder;
use visgraph::{Layout, graph_to_svg};

use crate::vlir::alu::{AluInst, AluOp};
use crate::vlir::flow::{FlowInst, Terminator};
use crate::vlir::load::{LoadInst, LoadKind};
use crate::vlir::store::{StoreInst, StoreKind};
use crate::vlir::valu::{ValuInst, ValuOp};
use crate::vlir::{BlockId, Function, InstrKind, Operand, RegisterId, ValueType};

/// Must match `problem.SCRATCH_SIZE` — scratch is indexed by word (32-bit), 0..this-1.
pub const MACHINE_SCRATCH_SIZE: usize = 100000; //1536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    SlotLimitExceeded(String, usize, usize),
    /// Emitted scratch words would exceed the simulator scratch space.
    ScratchOverflow {
        used: usize,
        limit: usize,
    },
    MissingRegisterType(RegisterId),
    MissingScratch(RegisterId),
    UnsupportedImmediate(&'static str),
    UnsupportedValuOp(ValuOp),
    MissingBlock(BlockId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot {
    Alu(&'static str, usize, usize, usize),
    Valu(&'static str, usize, usize, usize),
    ValuBroadcast(usize, usize),
    ValuMulAdd(usize, usize, usize, usize),
    Load(&'static str, usize, usize),
    LoadOffset(usize, usize, i32),
    Const(usize, i32),
    Store(&'static str, usize, usize),
    FlowSelect(usize, usize, usize, usize),
    FlowVSelect(usize, usize, usize, usize),
    FlowAddImm(usize, usize, i32),
    CondJump(usize, usize),
    Jump(usize),
    Halt,
    Pause,
    Sync,
    /// `("compare", scratch_addr, (round, batch, tag_str))` — matches `reference_kernel2` trace keys.
    DebugCompare(usize, i32, i32, u8),
}

impl Slot {
    fn to_python_tuple(&self) -> String {
        match self {
            Slot::Alu(op, d, a, b) => format!("(\"{op}\", {d}, {a}, {b})"),
            Slot::Valu(op, d, a, b) => format!("(\"{op}\", {d}, {a}, {b})"),
            Slot::ValuBroadcast(d, s) => format!("(\"vbroadcast\", {d}, {s})"),
            Slot::ValuMulAdd(d, a, b, c) => format!("(\"multiply_add\", {d}, {a}, {b}, {c})"),
            Slot::Load(op, d, a) => format!("(\"{op}\", {d}, {a})"),
            Slot::LoadOffset(d, a, o) => format!("(\"load_offset\", {d}, {a}, {o})"),
            Slot::Const(d, v) => format!("(\"const\", {d}, {v})"),
            Slot::Store(op, a, s) => format!("(\"{op}\", {a}, {s})"),
            Slot::FlowSelect(d, c, a, b) => format!("(\"select\", {d}, {c}, {a}, {b})"),
            Slot::FlowVSelect(d, c, a, b) => format!("(\"vselect\", {d}, {c}, {a}, {b})"),
            Slot::FlowAddImm(d, a, i) => format!("(\"add_imm\", {d}, {a}, {i})"),
            Slot::CondJump(c, p) => format!("(\"cond_jump\", {c}, {p})"),
            Slot::Jump(p) => format!("(\"jump\", {p})"),
            Slot::Halt => "(\"halt\",)".to_string(),
            Slot::Pause => "(\"pause\",)".to_string(),
            Slot::Sync => "(\"sync\",)".to_string(),
            Slot::DebugCompare(addr, r, b, tag) => {
                let ts = debug_tag_str(*tag);
                format!("(\"compare\", {addr}, ({r}, {b}, \"{ts}\"))")
            }
        }
    }

    fn to_json_array(&self) -> String {
        match self {
            Slot::Alu(op, d, a, b) => format!(r#"["{}",{},{},{}]"#, op, d, a, b),
            Slot::Valu(op, d, a, b) => format!(r#"["{}",{},{},{}]"#, op, d, a, b),
            Slot::ValuBroadcast(d, s) => format!(r#"["vbroadcast",{},{}]"#, d, s),
            Slot::ValuMulAdd(d, a, b, c) => format!(r#"["multiply_add",{},{},{},{}]"#, d, a, b, c),
            Slot::Load(op, d, a) => format!(r#"["{}",{},{}]"#, op, d, a),
            Slot::LoadOffset(d, a, o) => format!(r#"["load_offset",{},{},{}]"#, d, a, o),
            Slot::Const(d, v) => format!(r#"["const",{},{}]"#, d, v),
            Slot::Store(op, a, s) => format!(r#"["{}",{},{}]"#, op, a, s),
            Slot::FlowSelect(d, c, a, b) => format!(r#"["select",{},{},{},{}]"#, d, c, a, b),
            Slot::FlowVSelect(d, c, a, b) => format!(r#"["vselect",{},{},{},{}]"#, d, c, a, b),
            Slot::FlowAddImm(d, a, i) => format!(r#"["add_imm",{},{},{}]"#, d, a, i),
            Slot::CondJump(c, p) => format!(r#"["cond_jump",{},{}]"#, c, p),
            Slot::Jump(p) => format!(r#"["jump",{}]"#, p),
            Slot::Halt => r#"["halt"]"#.to_string(),
            Slot::Pause => r#"["pause"]"#.to_string(),
            Slot::Sync => r#"["sync"]"#.to_string(),
            Slot::DebugCompare(addr, r, b, tag) => {
                let ts = debug_tag_str(*tag);
                format!("[\"compare\",{},[{},{},\"{}\"]]", addr, r, b, ts)
            }
        }
    }
}

fn debug_tag_str(tag: u8) -> &'static str {
    match tag {
        0 => "idx",
        1 => "val",
        2 => "node_val",
        3 => "hashed_val",
        4 => "next_idx",
        5 => "wrapped_idx",
        _ => "unknown",
    }
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            c => out.push(c),
        }
    }
    out
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InstructionBundle {
    pub alu: Vec<Slot>,
    pub valu: Vec<Slot>,
    pub load: Vec<Slot>,
    pub store: Vec<Slot>,
    pub flow: Vec<Slot>,
    pub debug: Vec<Slot>,
}

impl InstructionBundle {
    pub fn to_python_dict_literal(&self) -> String {
        let mut sections = Vec::new();
        if !self.alu.is_empty() {
            sections.push(Self::engine_to_literal("alu", &self.alu));
        }
        if !self.valu.is_empty() {
            sections.push(Self::engine_to_literal("valu", &self.valu));
        }
        if !self.load.is_empty() {
            sections.push(Self::engine_to_literal("load", &self.load));
        }
        if !self.store.is_empty() {
            sections.push(Self::engine_to_literal("store", &self.store));
        }
        if !self.flow.is_empty() {
            sections.push(Self::engine_to_literal("flow", &self.flow));
        }
        if !self.debug.is_empty() {
            sections.push(Self::engine_to_literal("debug", &self.debug));
        }
        format!("{{{}}}", sections.join(", "))
    }

    fn engine_to_literal(name: &str, slots: &[Slot]) -> String {
        let joined = slots
            .iter()
            .map(Slot::to_python_tuple)
            .collect::<Vec<_>>()
            .join(", ");
        format!("\"{name}\": [{joined}]")
    }

    pub fn program_to_python_list_literal(program: &MachineProgram) -> String {
        let mut out = String::from("[\n");
        for bundle in program {
            let _ = writeln!(&mut out, "    {},", bundle.to_python_dict_literal());
        }
        out.push(']');
        out
    }

    pub fn to_json_object_literal(&self) -> String {
        let mut sections = Vec::new();
        if !self.alu.is_empty() {
            sections.push(Self::engine_to_json("alu", &self.alu));
        }
        if !self.valu.is_empty() {
            sections.push(Self::engine_to_json("valu", &self.valu));
        }
        if !self.load.is_empty() {
            sections.push(Self::engine_to_json("load", &self.load));
        }
        if !self.store.is_empty() {
            sections.push(Self::engine_to_json("store", &self.store));
        }
        if !self.flow.is_empty() {
            sections.push(Self::engine_to_json("flow", &self.flow));
        }
        if !self.debug.is_empty() {
            sections.push(Self::engine_to_json("debug", &self.debug));
        }
        format!("{{{}}}", sections.join(","))
    }

    fn engine_to_json(name: &str, slots: &[Slot]) -> String {
        let joined = slots
            .iter()
            .map(Slot::to_json_array)
            .collect::<Vec<_>>()
            .join(",");
        format!(r#""{}":[{}]"#, name, joined)
    }

    pub fn program_to_json(program: &MachineProgram) -> String {
        let body = program
            .iter()
            .map(InstructionBundle::to_json_object_literal)
            .collect::<Vec<_>>()
            .join(",");
        format!("[{}]", body)
    }

    /// Full compiler output: instruction bundles plus `debug_info.scratch_map` for `problem.DebugInfo`.
    pub fn program_to_json_with_debug(
        program: &MachineProgram,
        scratch_map: &ScratchDebugMap,
    ) -> String {
        let instr = Self::program_to_json(program);
        let mut keys: Vec<usize> = scratch_map.keys().copied().collect();
        keys.sort_unstable();
        let entries: Vec<String> = keys
            .into_iter()
            .map(|k| {
                let (name, len) = &scratch_map[&k];
                let esc = escape_json_string(name);
                format!(r#""{}":["{}",{}]"#, k, esc, len)
            })
            .collect();
        let sm = entries.join(",");
        format!(
            r#"{{"instructions":{},"debug_info":{{"scratch_map":{{{}}}}}}}"#,
            instr, sm
        )
    }

    pub fn assert_valid(&self) -> Result<(), LoweringError> {
        if self.alu.len() > 12 {
            return Err(LoweringError::SlotLimitExceeded(
                "alu".to_string(),
                self.alu.len(),
                12,
            ));
        }
        if self.valu.len() > 6 {
            return Err(LoweringError::SlotLimitExceeded(
                "valu".to_string(),
                self.valu.len(),
                6,
            ));
        }
        if self.load.len() > 2 {
            return Err(LoweringError::SlotLimitExceeded(
                "load".to_string(),
                self.load.len(),
                2,
            ));
        }
        if self.store.len() > 2 {
            return Err(LoweringError::SlotLimitExceeded(
                "store".to_string(),
                self.store.len(),
                2,
            ));
        }
        if self.flow.len() > 1 {
            return Err(LoweringError::SlotLimitExceeded(
                "flow".to_string(),
                self.flow.len(),
                1,
            ));
        }
        if self.debug.len() > 64 {
            return Err(LoweringError::SlotLimitExceeded(
                "debug".to_string(),
                self.debug.len(),
                64,
            ));
        }
        Ok(())
    }
}

pub type MachineProgram = Vec<InstructionBundle>;

/// Base scratch word -> (symbolic name, length in 32-bit words). Matches `problem.DebugInfo.scratch_map`.
pub type ScratchDebugMap = HashMap<usize, (String, usize)>;

type ScheduledBlock = Vec<Vec<crate::vlir::InstrId>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineKind {
    Alu,
    Valu,
    Load,
    Store,
    Flow,
    Debug,
}

impl EngineKind {
    fn slot_limit(self) -> usize {
        match self {
            EngineKind::Alu => 12,
            EngineKind::Valu => 6,
            EngineKind::Load => 2,
            EngineKind::Store => 2,
            EngineKind::Flow => 1,
            EngineKind::Debug => 64,
        }
    }
}

fn instr_engine_kind(inst: &crate::vlir::Instruction) -> EngineKind {
    match &inst.kind {
        InstrKind::Const { .. } | InstrKind::Load(_) => EngineKind::Load,
        InstrKind::Store(_) => EngineKind::Store,
        InstrKind::Flow(_) => EngineKind::Flow,
        InstrKind::DebugCompare { .. } => EngineKind::Debug,
        InstrKind::Alu(_)
        | InstrKind::VectorLaneStore { .. }
        | InstrKind::VectorLaneLoad { .. } => EngineKind::Alu,
        InstrKind::Valu(_) => EngineKind::Valu,
    }
}

fn def_regs(inst: &crate::vlir::Instruction) -> Vec<RegisterId> {
    match &inst.kind {
        // Partial vector write is still a write dependency for scheduling.
        InstrKind::VectorLaneStore { vec, .. } => vec![*vec],
        InstrKind::Load(li) if li.vector_gather => vec![li.dst],
        _ => match def_of_instruction(inst) {
            Some(r) => vec![r],
            None => Vec::new(),
        },
    }
}

fn has_memory_load(inst: &crate::vlir::Instruction) -> bool {
    matches!(inst.kind, InstrKind::Load(_))
}

fn has_memory_store(inst: &crate::vlir::Instruction) -> bool {
    matches!(inst.kind, InstrKind::Store(_))
}

fn has_observable_ordering(inst: &crate::vlir::Instruction) -> bool {
    is_side_effect_instruction(inst) || matches!(inst.kind, InstrKind::Flow(_))
}

fn is_sync_barrier(inst: &crate::vlir::Instruction) -> bool {
    matches!(inst.kind, InstrKind::Flow(FlowInst::Sync))
}

fn shares_reg(a: &[RegisterId], b: &[RegisterId]) -> bool {
    a.iter().any(|x| b.contains(x))
}

/// True when `a` and `b` write different 32-bit words of the same vector register
/// (`node_val[vi] = …` / `load_offset` gathers). The IR still uses one `RegisterId` for the
/// whole vector, so plain `defs` overlap is a false WAW; the simulator only touches one lane.
fn partial_vector_writes_disjoint(
    a: &crate::vlir::Instruction,
    b: &crate::vlir::Instruction,
) -> bool {
    match (&a.kind, &b.kind) {
        (InstrKind::Load(la), InstrKind::Load(lb))
            if la.vector_gather && lb.vector_gather && la.dst == lb.dst =>
        {
            la.offset != lb.offset
        }
        (InstrKind::Load(l), InstrKind::VectorLaneStore { vec, lane, .. })
        | (InstrKind::VectorLaneStore { vec, lane, .. }, InstrKind::Load(l))
            if l.vector_gather && l.dst == *vec =>
        {
            l.offset != i32::from(*lane)
        }
        (
            InstrKind::VectorLaneStore {
                vec: va, lane: la, ..
            },
            InstrKind::VectorLaneStore {
                vec: vb, lane: lb, ..
            },
        ) => va == vb && la != lb,
        _ => false,
    }
}

fn has_dependency(
    a: &crate::vlir::Instruction,
    b: &crate::vlir::Instruction,
    uses_a: &[RegisterId],
    defs_a: &[RegisterId],
    uses_b: &[RegisterId],
    defs_b: &[RegisterId],
) -> bool {
    // Full fence semantics for explicit barriers.
    if is_sync_barrier(a) || is_sync_barrier(b) {
        return true;
    }

    // Register hazards under end-of-cycle writeback:
    // - RAW: producer must precede consumer.
    // - WAW: preserve deterministic final value (except lane-disjoint partial writes below).
    // NOTE: WAR is intentionally allowed; both reads see pre-cycle state and writes commit at
    // end of cycle, so anti-dependencies do not require ordering within a bundle.
    if shares_reg(defs_a, uses_b) {
        return true;
    }
    if shares_reg(defs_a, defs_b) && !partial_vector_writes_disjoint(a, b) {
        return true;
    }

    // Conservative memory model: preserve order around any store.
    // let mem_a = has_memory_load(a) || has_memory_store(a);
    // let mem_b = has_memory_load(b) || has_memory_store(b);
    // if mem_a && mem_b && (has_memory_store(a) || has_memory_store(b)) {
    //     return true;
    // }

    // Keep externally observable side effects in original order — except lane-disjoint
    // `vector_gather` / `VectorLaneStore` pairs, which only touch different scratch words.
    if has_observable_ordering(a) && has_observable_ordering(b) {
        if partial_vector_writes_disjoint(a, b) {
            return false;
        }
        // ignore stores
        if has_memory_store(a) || has_memory_store(b) {
            return false;
        }
        return true;
    }

    false
}

fn build_scheduled_block(insts: &[&crate::vlir::Instruction]) -> ScheduledBlock {
    let n = insts.len();
    if n <= 1 {
        return insts.iter().map(|x| vec![x.id]).collect();
    }

    let mut uses: Vec<Vec<RegisterId>> = Vec::with_capacity(n);
    let mut defs: Vec<Vec<RegisterId>> = Vec::with_capacity(n);
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

fn schedule_emitted_blocks(
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

fn terminator_bundle_count(term: &Terminator) -> usize {
    // Must match `lower_terminator`: Branch uses two bundles (cond_jump, then jump)
    // because the simulator allows only one flow slot per cycle.
    match term {
        Terminator::Branch { .. } => 2,
        Terminator::Jump { .. } | Terminator::Return { .. } | Terminator::Unreachable => 1,
    }
}

fn build_scratch_debug_map(
    func: &Function,
    layout: &ScratchLayout,
) -> Result<ScratchDebugMap, LoweringError> {
    let mut out = ScratchDebugMap::new();
    for (&reg, &base) in &layout.map {
        let w = reg_width(func, reg)?;
        let name = func.reg_display_name(reg);
        out.insert(base, (name, w));
    }
    Ok(out)
}

pub fn lower_function(
    func: &Function,
    advanced_scheduling: bool,
) -> Result<(MachineProgram, ScratchDebugMap), LoweringError> {
    let emitted = build_emission_plan(func);
    let scheduled_blocks = schedule_emitted_blocks(func, &emitted, advanced_scheduling);
    let use_counts = collect_use_counts(func, &scheduled_blocks);
    let scratch = build_greedy_scratch_layout(func, &use_counts, &scheduled_blocks)?;
    let scratch_debug = build_scratch_debug_map(func, &scratch)?;

    let mut block_pc = HashMap::new();
    let mut pc = 0usize;
    for (bi, block) in func.blocks.iter().enumerate() {
        block_pc.insert(block.id, pc);
        let count = scheduled_blocks[bi].len();
        pc += count + terminator_bundle_count(&block.terminator);
    }

    let mut bundles = Vec::new();
    let mut inst_by_id: HashMap<crate::vlir::InstrId, &crate::vlir::Instruction> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            inst_by_id.insert(inst.id, inst);
        }
    }
    for (bi, block) in func.blocks.iter().enumerate() {
        for cycle in &scheduled_blocks[bi] {
            let mut b = InstructionBundle::default();
            for iid in cycle {
                let inst = inst_by_id[iid];
                match &inst.kind {
                    InstrKind::Const { dst, value } => {
                        b.load.push(Slot::Const(scratch.addr(*dst)?, *value));
                    }
                    InstrKind::VectorLaneStore {
                        vec,
                        lane,
                        src,
                        zero,
                    } => {
                        let base = scratch.addr(*vec)?;
                        let dst_slot = base + *lane as usize;
                        let a = scratch.addr(*src)?;
                        let z = scratch.addr(*zero)?;
                        b.alu.push(Slot::Alu("+", dst_slot, a, z));
                    }
                    InstrKind::VectorLaneLoad {
                        dst,
                        vec,
                        lane,
                        zero,
                    } => {
                        let base = scratch.addr(*vec)?;
                        let src_lane = base + *lane as usize;
                        let d = scratch.addr(*dst)?;
                        let z = scratch.addr(*zero)?;
                        b.alu.push(Slot::Alu("+", d, src_lane, z));
                    }
                    InstrKind::Alu(ai) => b.alu.push(lower_alu(ai, &scratch)?),
                    InstrKind::Valu(vi) => b.valu.push(lower_valu(vi, &scratch)?),
                    InstrKind::Flow(fi) => b.flow.push(lower_flow(fi, &scratch)?),
                    InstrKind::Load(li) => b.load.push(lower_load(li, &scratch)?),
                    InstrKind::Store(si) => b.store.push(lower_store(si, &scratch)?),
                    InstrKind::DebugCompare {
                        value,
                        round,
                        batch,
                        tag,
                    } => {
                        let a = scratch.addr(*value)?;
                        b.debug.push(Slot::DebugCompare(a, *round, *batch, *tag));
                    }
                }
            }
            b.assert_valid()?;
            bundles.push(b);
        }
        bundles.extend(lower_terminator(&block.terminator, &scratch, &block_pc)?);
    }
    Ok((bundles, scratch_debug))
}

#[derive(Default)]
struct ScratchLayout {
    map: HashMap<RegisterId, usize>,
}

impl ScratchLayout {
    fn addr(&self, reg: RegisterId) -> Result<usize, LoweringError> {
        self.map
            .get(&reg)
            .copied()
            .ok_or(LoweringError::MissingScratch(reg))
    }
}

fn build_greedy_scratch_layout(
    func: &Function,
    use_counts: &HashMap<RegisterId, usize>,
    scheduled_blocks: &[ScheduledBlock],
) -> Result<ScratchLayout, LoweringError> {
    let mut map = HashMap::new();
    let mut allocs: HashMap<RegisterId, (usize, usize)> = HashMap::new();
    let mut remaining_uses = use_counts.clone();
    let mut used = vec![false; MACHINE_SCRATCH_SIZE];

    let mut inst_by_id: HashMap<crate::vlir::InstrId, &crate::vlir::Instruction> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            inst_by_id.insert(inst.id, inst);
        }
    }

    for (bi, block) in func.blocks.iter().enumerate() {
        for cycle in &scheduled_blocks[bi] {
            let mut cycle_uses: Vec<RegisterId> = Vec::new();
            for iid in cycle {
                let inst = inst_by_id[iid];
                for r in uses_in_instruction(inst) {
                    ensure_allocated(func, r, &mut allocs, &mut map, &mut used)?;
                    cycle_uses.push(r);
                }

                if let Some(dst) = def_of_instruction(inst) {
                    ensure_allocated(func, dst, &mut allocs, &mut map, &mut used)?;
                } else {
                    for r in def_regs(inst) {
                        ensure_allocated(func, r, &mut allocs, &mut map, &mut used)?;
                    }
                }
            }
            // Important: all reads in a VLIW bundle happen before any writes become visible.
            // Defer use-count decrements/frees until the whole cycle has been allocated.
            for r in cycle_uses {
                decrement_and_maybe_free(r, &mut remaining_uses, &mut allocs, &mut used);
            }
        }
        for r in uses_in_terminator(&block.terminator) {
            ensure_allocated(func, r, &mut allocs, &mut map, &mut used)?;
            decrement_and_maybe_free(r, &mut remaining_uses, &mut allocs, &mut used);
        }
    }

    Ok(ScratchLayout { map })
}

fn decrement_and_maybe_free(
    reg: RegisterId,
    remaining_uses: &mut HashMap<RegisterId, usize>,
    allocs: &mut HashMap<RegisterId, (usize, usize)>,
    used: &mut [bool],
) {
    let Some(count) = remaining_uses.get_mut(&reg) else {
        return;
    };
    if *count == 0 {
        return;
    }
    *count -= 1;
    if *count == 0 {
        if let Some((base, width)) = allocs.remove(&reg) {
            release_range(used, base, width);
        }
    }
}

fn ensure_allocated(
    func: &Function,
    reg: RegisterId,
    allocs: &mut HashMap<RegisterId, (usize, usize)>,
    map: &mut HashMap<RegisterId, usize>,
    used: &mut [bool],
) -> Result<(), LoweringError> {
    if allocs.contains_key(&reg) {
        return Ok(());
    }
    let width = reg_width(func, reg)?;
    let Some(base) = alloc_first_fit(used, width) else {
        let occupied = used.iter().filter(|x| **x).count();
        return Err(LoweringError::ScratchOverflow {
            used: occupied + width,
            limit: MACHINE_SCRATCH_SIZE,
        });
    };
    for slot in used.iter_mut().skip(base).take(width) {
        *slot = true;
    }
    allocs.insert(reg, (base, width));
    map.insert(reg, base);
    Ok(())
}

fn alloc_first_fit(used: &[bool], width: usize) -> Option<usize> {
    if width == 0 || width > used.len() {
        return None;
    }
    let mut run = 0usize;
    let mut start = 0usize;
    for (idx, occupied) in used.iter().enumerate() {
        if !occupied {
            if run == 0 {
                start = idx;
            }
            run += 1;
            if run >= width {
                return Some(start);
            }
        } else {
            run = 0;
        }
    }
    None
}

fn release_range(used: &mut [bool], base: usize, width: usize) {
    for slot in used.iter_mut().skip(base).take(width) {
        *slot = false;
    }
}

fn collect_use_counts(
    func: &Function,
    scheduled_blocks: &[ScheduledBlock],
) -> HashMap<RegisterId, usize> {
    let mut uses = HashMap::new();
    let mut inst_by_id: HashMap<crate::vlir::InstrId, &crate::vlir::Instruction> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            inst_by_id.insert(inst.id, inst);
        }
    }
    for (bi, block) in func.blocks.iter().enumerate() {
        for cycle in &scheduled_blocks[bi] {
            for iid in cycle {
                let inst = inst_by_id[iid];
                for r in uses_in_instruction(inst) {
                    *uses.entry(r).or_insert(0) += 1;
                }
            }
        }
        for r in uses_in_terminator(&block.terminator) {
            *uses.entry(r).or_insert(0) += 1;
        }
    }
    uses
}

fn build_emission_plan(func: &Function) -> HashSet<crate::vlir::InstrId> {
    let mut needed: HashSet<RegisterId> = HashSet::new();
    let mut emitted: HashSet<crate::vlir::InstrId> = HashSet::new();

    for block in func.blocks.iter().rev() {
        for r in uses_in_terminator(&block.terminator) {
            needed.insert(r);
        }
        for inst in block.instructions.iter().rev() {
            match def_of_instruction(inst) {
                Some(dst) => {
                    if needed.contains(&dst) {
                        emitted.insert(inst.id);
                        for r in uses_in_instruction(inst) {
                            needed.insert(r);
                        }
                    }
                }
                None => {
                    if is_side_effect_instruction(inst) {
                        emitted.insert(inst.id);
                        for r in uses_in_instruction(inst) {
                            needed.insert(r);
                        }
                    }
                }
            }
        }
    }
    emitted
}

fn is_side_effect_instruction(inst: &crate::vlir::Instruction) -> bool {
    match &inst.kind {
        InstrKind::Load(li) if li.vector_gather => true,
        InstrKind::Store(_)
        | InstrKind::VectorLaneStore { .. }
        | InstrKind::DebugCompare { .. }
        | InstrKind::Flow(FlowInst::Pause)
        | InstrKind::Flow(FlowInst::Sync) => true,
        _ => false,
    }
}

fn reg_width(func: &Function, reg: RegisterId) -> Result<usize, LoweringError> {
    match func.reg_types.get(&reg) {
        Some(ValueType::Vector) => Ok(8),
        Some(_) => Ok(1),
        None => Err(LoweringError::MissingRegisterType(reg)),
    }
}

fn def_of_instruction(inst: &crate::vlir::Instruction) -> Option<RegisterId> {
    match &inst.kind {
        InstrKind::Const { dst, .. } => Some(*dst),
        // Partial in-place lane write: side effect, not a full vector redefinition.
        InstrKind::VectorLaneStore { .. } => None,
        InstrKind::VectorLaneLoad { dst, .. } => Some(*dst),
        InstrKind::Alu(ai) => Some(ai.dst),
        InstrKind::Valu(vi) => Some(vi.dst),
        InstrKind::Flow(FlowInst::Select { dst, .. })
        | InstrKind::Flow(FlowInst::VSelect { dst, .. })
        | InstrKind::Flow(FlowInst::AddImm { dst, .. }) => Some(*dst),
        InstrKind::Flow(FlowInst::Pause) | InstrKind::Flow(FlowInst::Sync) => None,
        InstrKind::Load(li) => {
            if li.vector_gather {
                None
            } else {
                Some(li.dst)
            }
        }
        InstrKind::Store(_) => None,
        InstrKind::DebugCompare { .. } => None,
    }
}

fn uses_in_instruction(inst: &crate::vlir::Instruction) -> Vec<RegisterId> {
    match &inst.kind {
        InstrKind::Const { .. } => Vec::new(),
        InstrKind::VectorLaneStore { vec, src, zero, .. } => vec![*vec, *src, *zero],
        InstrKind::VectorLaneLoad { vec, zero, .. } => vec![*vec, *zero],
        InstrKind::Alu(ai) => {
            let mut out = Vec::new();
            if let Operand::Reg(r) = ai.lhs {
                out.push(r);
            }
            if let Operand::Reg(r) = ai.rhs {
                out.push(r);
            }
            out
        }
        InstrKind::Valu(vi) => {
            let mut out = vec![vi.src1, vi.src2];
            if let Some(r) = vi.src3 {
                out.push(r);
            }
            out
        }
        InstrKind::Flow(FlowInst::Select { cond, a, b, .. })
        | InstrKind::Flow(FlowInst::VSelect { cond, a, b, .. }) => vec![*cond, *a, *b],
        InstrKind::Flow(FlowInst::AddImm { a, .. }) => vec![*a],
        InstrKind::Flow(FlowInst::Pause) | InstrKind::Flow(FlowInst::Sync) => Vec::new(),
        InstrKind::Load(li) => vec![li.base_ptr],
        InstrKind::Store(si) => vec![si.base_ptr, si.src],
        InstrKind::DebugCompare { value, .. } => vec![*value],
    }
}

fn uses_in_terminator(term: &Terminator) -> Vec<RegisterId> {
    match term {
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return { value: Some(r) } => vec![*r],
        Terminator::Jump { .. } | Terminator::Return { value: None } | Terminator::Unreachable => {
            Vec::new()
        }
    }
}

fn lower_alu(inst: &AluInst, layout: &ScratchLayout) -> Result<Slot, LoweringError> {
    let dst = layout.addr(inst.dst)?;
    let a = operand_as_reg(inst.lhs.clone(), layout)?;
    let b = operand_as_reg(inst.rhs.clone(), layout)?;
    Ok(Slot::Alu(alu_name(inst.op), dst, a, b))
}

fn lower_valu(inst: &ValuInst, layout: &ScratchLayout) -> Result<Slot, LoweringError> {
    let d = layout.addr(inst.dst)?;
    let a = layout.addr(inst.src1)?;
    let b = layout.addr(inst.src2)?;
    match (inst.op, inst.src3) {
        (ValuOp::Broadcast, _) => Ok(Slot::ValuBroadcast(d, a)),
        (ValuOp::Mul, Some(c)) => Ok(Slot::ValuMulAdd(d, a, b, layout.addr(c)?)),
        (
            ValuOp::Add
            | ValuOp::Sub
            | ValuOp::Mul
            | ValuOp::Div
            | ValuOp::Mod
            | ValuOp::And
            | ValuOp::Or
            | ValuOp::Xor
            | ValuOp::Shl
            | ValuOp::Shr
            | ValuOp::CmpEq
            | ValuOp::CmpLt,
            _,
        ) => Ok(Slot::Valu(valu_name(inst.op), d, a, b)),
    }
}

fn lower_load(inst: &LoadInst, layout: &ScratchLayout) -> Result<Slot, LoweringError> {
    let d = layout.addr(inst.dst)?;
    let a = layout.addr(inst.base_ptr)?;
    Ok(match inst.kind {
        LoadKind::I32 | LoadKind::U32 => {
            if inst.offset == 0 {
                Slot::Load("load", d, a)
            } else {
                Slot::LoadOffset(d, a, inst.offset)
            }
        }
        LoadKind::Vec128 => Slot::Load("vload", d, a),
    })
}

fn lower_store(inst: &StoreInst, layout: &ScratchLayout) -> Result<Slot, LoweringError> {
    let a = layout.addr(inst.base_ptr)?;
    let s = layout.addr(inst.src)?;
    match inst.kind {
        StoreKind::I32 | StoreKind::U32 => Ok(Slot::Store("store", a, s)),
        StoreKind::Vec128 => Ok(Slot::Store("vstore", a, s)),
    }
}

fn lower_flow(inst: &FlowInst, layout: &ScratchLayout) -> Result<Slot, LoweringError> {
    match inst {
        FlowInst::Select { dst, cond, a, b } => Ok(Slot::FlowSelect(
            layout.addr(*dst)?,
            layout.addr(*cond)?,
            layout.addr(*a)?,
            layout.addr(*b)?,
        )),
        FlowInst::VSelect { dst, cond, a, b } => Ok(Slot::FlowVSelect(
            layout.addr(*dst)?,
            layout.addr(*cond)?,
            layout.addr(*a)?,
            layout.addr(*b)?,
        )),
        FlowInst::AddImm { dst, a, imm } => {
            Ok(Slot::FlowAddImm(layout.addr(*dst)?, layout.addr(*a)?, *imm))
        }
        FlowInst::Pause => Ok(Slot::Pause),
        FlowInst::Sync => Ok(Slot::Sync),
    }
}

fn lower_terminator(
    term: &Terminator,
    layout: &ScratchLayout,
    block_pc: &HashMap<BlockId, usize>,
) -> Result<Vec<InstructionBundle>, LoweringError> {
    match term {
        Terminator::Jump { target } => {
            let mut b = InstructionBundle::default();
            let pc = block_pc
                .get(target)
                .copied()
                .ok_or(LoweringError::MissingBlock(*target))?;
            b.flow.push(Slot::Jump(pc));
            Ok(vec![b])
        }
        Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } => {
            let mut b1 = InstructionBundle::default();
            let mut b2 = InstructionBundle::default();
            let c = layout.addr(*cond)?;
            let then_pc = block_pc
                .get(then_bb)
                .copied()
                .ok_or(LoweringError::MissingBlock(*then_bb))?;
            let else_pc = block_pc
                .get(else_bb)
                .copied()
                .ok_or(LoweringError::MissingBlock(*else_bb))?;
            b1.flow.push(Slot::CondJump(c, then_pc));
            b2.flow.push(Slot::Jump(else_pc));
            Ok(vec![b1, b2])
        }
        Terminator::Return { .. } | Terminator::Unreachable => {
            let mut b = InstructionBundle::default();
            b.flow.push(Slot::Halt);
            Ok(vec![b])
        }
    }
}

fn operand_as_reg(op: Operand, layout: &ScratchLayout) -> Result<usize, LoweringError> {
    match op {
        Operand::Reg(r) => layout.addr(r),
        Operand::ImmI32(_) => Err(LoweringError::UnsupportedImmediate(
            "ALU immediates require explicit const materialization",
        )),
    }
}

fn alu_name(op: AluOp) -> &'static str {
    match op {
        AluOp::Add => "+",
        AluOp::Sub => "-",
        AluOp::Mul => "*",
        AluOp::Div => "//",
        AluOp::Mod => "%",
        AluOp::And => "&",
        AluOp::Or => "|",
        AluOp::Xor => "^",
        AluOp::Shl => "<<",
        AluOp::Shr => ">>",
        AluOp::CmpEq => "==",
        AluOp::CmpLt => "<",
    }
}

fn valu_name(op: ValuOp) -> &'static str {
    match op {
        ValuOp::Add => "+",
        ValuOp::Sub => "-",
        ValuOp::Mul => "*",
        ValuOp::Div => "//",
        ValuOp::Mod => "%",
        ValuOp::And => "&",
        ValuOp::Or => "|",
        ValuOp::Xor => "^",
        ValuOp::Shl => "<<",
        ValuOp::Shr => ">>",
        ValuOp::CmpEq => "==",
        ValuOp::CmpLt => "<",
        ValuOp::Broadcast => "vbroadcast",
    }
}

fn create_graph(insts: &[&crate::vlir::Instruction], succs: &Vec<Vec<usize>>) {
    let mut graph = Graph::<String, (), Directed>::new();

    let mut nodes = vec![];

    for (i, inst) in insts.iter().enumerate() {
        nodes.push(graph.add_node(format!("inst {:?} ", inst.kind)));
    }

    for (i, inst) in insts.iter().enumerate() {
        for succ in &succs[i] {
            graph.add_edge(nodes[i], nodes[*succ], ());
        }
    }

    // write to dot file "graph.dot"

    fs::write(
        "graph.dot",
        format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel])),
    )
    .unwrap();
}
