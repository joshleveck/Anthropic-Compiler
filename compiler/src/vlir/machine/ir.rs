use crate::vlir::flow::{FlowInst, Terminator};
use crate::vlir::{Function, InstrKind, Operand, RegisterId, ValueType};

use super::error::LoweringError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineKind {
    Alu,
    Valu,
    Load,
    Store,
    Flow,
    Debug,
}

impl EngineKind {
    pub(crate) fn slot_limit(self) -> usize {
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

pub(crate) fn instr_engine_kind(inst: &crate::vlir::Instruction) -> EngineKind {
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

pub(crate) fn def_regs(inst: &crate::vlir::Instruction) -> Vec<RegisterId> {
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

pub(crate) fn has_memory_store(inst: &crate::vlir::Instruction) -> bool {
    matches!(inst.kind, InstrKind::Store(_))
}

pub(crate) fn has_observable_ordering(inst: &crate::vlir::Instruction) -> bool {
    is_side_effect_instruction(inst) || matches!(inst.kind, InstrKind::Flow(_))
}

pub(crate) fn is_sync_barrier(inst: &crate::vlir::Instruction) -> bool {
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

pub(crate) fn has_dependency(
    a: &crate::vlir::Instruction,
    b: &crate::vlir::Instruction,
    _uses_a: &[RegisterId],
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

pub(crate) fn terminator_bundle_count(term: &Terminator) -> usize {
    // Must match `lower_terminator`: Branch uses two bundles (cond_jump, then jump)
    // because the simulator allows only one flow slot per cycle.
    match term {
        Terminator::Branch { .. } => 2,
        Terminator::Jump { .. } | Terminator::Return { .. } | Terminator::Unreachable => 1,
    }
}

pub(crate) fn is_side_effect_instruction(inst: &crate::vlir::Instruction) -> bool {
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

pub(crate) fn reg_width(func: &Function, reg: RegisterId) -> Result<usize, LoweringError> {
    match func.reg_types.get(&reg) {
        Some(ValueType::Vector) => Ok(8),
        Some(_) => Ok(1),
        None => Err(LoweringError::MissingRegisterType(reg)),
    }
}

pub(crate) fn def_of_instruction(inst: &crate::vlir::Instruction) -> Option<RegisterId> {
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

pub(crate) fn uses_in_instruction(inst: &crate::vlir::Instruction) -> Vec<RegisterId> {
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

pub(crate) fn uses_in_terminator(term: &Terminator) -> Vec<RegisterId> {
    match term {
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return { value: Some(r) } => vec![*r],
        Terminator::Jump { .. } | Terminator::Return { value: None } | Terminator::Unreachable => {
            Vec::new()
        }
    }
}
