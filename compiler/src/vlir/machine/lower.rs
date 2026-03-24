use std::collections::HashMap;

use crate::vlir::alu::AluInst;
use crate::vlir::flow::{FlowInst, Terminator};
use crate::vlir::load::{LoadInst, LoadKind};
use crate::vlir::store::{StoreInst, StoreKind};
use crate::vlir::valu::{ValuInst, ValuOp};
use crate::vlir::{BlockId, Function, InstrKind, Operand};

use super::MACHINE_SCRATCH_SIZE;
use super::bundle::{InstructionBundle, MachineProgram, ScratchDebugMap, ScratchLifetimeTrace};
use super::error::LoweringError;
use super::ir::{reg_width, terminator_bundle_count};
use super::scheduling::schedule_emitted_blocks;
use super::scratch::{
    ScratchLayout, build_emission_plan, build_greedy_scratch_layout, collect_use_counts,
};
use super::slot::Slot;

pub(crate) fn lower_function(
    func: &Function,
    advanced_scheduling: bool,
    trace_scratch_lifetime: bool,
) -> Result<
    (
        MachineProgram,
        ScratchDebugMap,
        Option<ScratchLifetimeTrace>,
    ),
    LoweringError,
> {
    let emitted = build_emission_plan(func);
    let scheduled_blocks = schedule_emitted_blocks(func, &emitted, advanced_scheduling);
    let use_counts = collect_use_counts(func, &scheduled_blocks);
    let (scratch, lifetime_trace) = build_greedy_scratch_layout(
        func,
        &use_counts,
        &scheduled_blocks,
        trace_scratch_lifetime,
        MACHINE_SCRATCH_SIZE,
    )?;
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
    Ok((bundles, scratch_debug, lifetime_trace))
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
        LoadKind::I32 => {
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
    }
}

fn alu_name(op: crate::vlir::alu::AluOp) -> &'static str {
    use crate::vlir::alu::AluOp;
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
