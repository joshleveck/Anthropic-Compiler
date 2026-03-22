use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::vlir::alu::{AluInst, AluOp};
use crate::vlir::flow::{FlowInst, Terminator};
use crate::vlir::load::{LoadInst, LoadKind};
use crate::vlir::store::{StoreInst, StoreKind};
use crate::vlir::valu::{ValuInst, ValuOp};
use crate::vlir::{BlockId, Function, InstrKind, Operand, RegisterId, ValueType};

/// Must match `problem.SCRATCH_SIZE` — scratch is indexed by word (32-bit), 0..this-1.
pub const MACHINE_SCRATCH_SIZE: usize = 1536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    SlotLimitExceeded(String, usize, usize),
    /// Emitted scratch words would exceed the simulator scratch space.
    ScratchOverflow { used: usize, limit: usize },
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

fn terminator_bundle_count(term: &Terminator) -> usize {
    // Must match `lower_terminator`: Branch uses two bundles (cond_jump, then jump)
    // because the simulator allows only one flow slot per cycle.
    match term {
        Terminator::Branch { .. } => 2,
        Terminator::Jump { .. } | Terminator::Return { .. } | Terminator::Unreachable => 1,
    }
}

pub fn lower_function(func: &Function) -> Result<MachineProgram, LoweringError> {
    let emitted = build_emission_plan(func);
    let use_counts = collect_use_counts(func, &emitted);
    let scratch = build_greedy_scratch_layout(func, &use_counts, &emitted)?;

    let mut block_pc = HashMap::new();
    let mut pc = 0usize;
    for block in &func.blocks {
        block_pc.insert(block.id, pc);
        let mut count = 0usize;
        for inst in &block.instructions {
            if !emitted.contains(&inst.id) {
                continue;
            }
            count += 1;
        }
        pc += count + terminator_bundle_count(&block.terminator);
    }

    let mut bundles = Vec::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if !emitted.contains(&inst.id) {
                continue;
            }
            let mut b = InstructionBundle::default();
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
            bundles.push(b);
        }
        bundles.extend(lower_terminator(&block.terminator, &scratch, &block_pc)?);
    }
    Ok(bundles)
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
    emitted: &HashSet<crate::vlir::InstrId>,
) -> Result<ScratchLayout, LoweringError> {
    let mut map = HashMap::new();
    let mut allocs: HashMap<RegisterId, (usize, usize)> = HashMap::new();
    let mut remaining_uses = use_counts.clone();
    let mut used = vec![false; MACHINE_SCRATCH_SIZE];

    for block in &func.blocks {
        for inst in &block.instructions {
            if !emitted.contains(&inst.id) {
                continue;
            }
            for r in uses_in_instruction(inst) {
                ensure_allocated(func, r, &mut allocs, &mut map, &mut used)?;
                decrement_and_maybe_free(r, &mut remaining_uses, &mut allocs, &mut used);
            }

            if let Some(dst) = def_of_instruction(inst) {
                ensure_allocated(func, dst, &mut allocs, &mut map, &mut used)?;
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
    emitted: &HashSet<crate::vlir::InstrId>,
) -> HashMap<RegisterId, usize> {
    let mut uses = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if !emitted.contains(&inst.id) {
                continue;
            }
            for r in uses_in_instruction(inst) {
                *uses.entry(r).or_insert(0) += 1;
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
    matches!(
        inst.kind,
        InstrKind::Store(_)
            | InstrKind::VectorLaneStore { .. }
            | InstrKind::DebugCompare { .. }
            | InstrKind::Flow(FlowInst::Pause)
    )
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
        InstrKind::Flow(FlowInst::Pause) => None,
        InstrKind::Load(li) => Some(li.dst),
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
        InstrKind::Flow(FlowInst::Pause) => Vec::new(),
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
