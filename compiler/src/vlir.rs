pub mod alu;
pub mod flow;
pub mod load;
pub mod lowering;
pub mod machine;
pub mod store;
pub mod valu;

use std::collections::HashMap;

use alu::AluInst;
use flow::{FlowInst, Terminator};
use load::LoadInst;
use machine::{InstructionBundle, LoweringError, MachineProgram, ScratchDebugMap};
use store::StoreInst;
use valu::ValuInst;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstrId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Scalar,
    Vector,
    Ptr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Reg(RegisterId),
    ImmI32(i32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstrKind {
    Const { dst: RegisterId, value: i32 },
    /// Write scalar `src` into lane `lane` (0..7) of vector `vec` (scratch `base(vec)+lane`).
    /// `zero` must hold 0; used as `dst = src + zero` to copy into the lane word.
    VectorLaneStore {
        vec: RegisterId,
        lane: u8,
        src: RegisterId,
        zero: RegisterId,
    },
    /// Read lane `lane` of `vec` into scalar `dst` via `dst = lane_word + zero`.
    VectorLaneLoad {
        dst: RegisterId,
        vec: RegisterId,
        lane: u8,
        zero: RegisterId,
    },
    Alu(AluInst),
    Valu(ValuInst),
    Flow(FlowInst),
    Load(LoadInst),
    Store(StoreInst),
    /// Compare `value` scratch against `reference_kernel2` trace key `(round, batch, tag)`.
    /// `tag` is 0..=5: idx, val, node_val, hashed_val, next_idx, wrapped_idx.
    DebugCompare {
        value: RegisterId,
        round: i32,
        batch: i32,
        tag: u8,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitClass {
    ScalarAlu,
    VectorAlu,
    LoadStore,
    /// Debug compare slots (see `problem.Machine` `debug` engine).
    Debug,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    pub id: InstrId,
    pub kind: InstrKind,
    pub unit: UnitClass,
    pub latency: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
    pub reg_types: HashMap<RegisterId, ValueType>,
    /// Optional display names for scratch allocation (see `Machine` / `DebugInfo.scratch_map`).
    pub reg_names: HashMap<RegisterId, String>,
    next_reg: u32,
    next_instr: u32,
    next_block: u32,
}

impl Function {
    pub fn new(name: impl Into<String>) -> Self {
        let entry = BlockId(0);
        Self {
            name: name.into(),
            entry,
            blocks: vec![BasicBlock {
                id: entry,
                instructions: Vec::new(),
                terminator: Terminator::Unreachable,
            }],
            reg_types: HashMap::new(),
            reg_names: HashMap::new(),
            next_reg: 0,
            next_instr: 0,
            next_block: 1,
        }
    }

    pub fn new_register(&mut self, ty: ValueType) -> RegisterId {
        let id = RegisterId(self.next_reg);
        self.next_reg += 1;
        self.reg_types.insert(id, ty);
        self.reg_names.insert(id, format!("r{}", id.0));
        id
    }

    pub fn set_reg_name(&mut self, reg: RegisterId, name: String) {
        self.reg_names.insert(reg, name);
    }

    pub fn reg_display_name(&self, reg: RegisterId) -> String {
        self.reg_names
            .get(&reg)
            .cloned()
            .unwrap_or_else(|| format!("r{}", reg.0))
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        self.blocks.push(BasicBlock {
            id,
            instructions: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        id
    }

    pub fn append_instruction(
        &mut self,
        block: BlockId,
        kind: InstrKind,
        unit: UnitClass,
        latency: u8,
    ) -> InstrId {
        let id = InstrId(self.next_instr);
        self.next_instr += 1;
        if let Some(bb) = self.blocks.iter_mut().find(|bb| bb.id == block) {
            bb.instructions.push(Instruction {
                id,
                kind,
                unit,
                latency,
            });
        }
        id
    }

    pub fn set_terminator(&mut self, block: BlockId, term: Terminator) {
        if let Some(bb) = self.blocks.iter_mut().find(|bb| bb.id == block) {
            bb.terminator = term;
        }
    }

    pub fn lower_to_machine(
        &self,
        advanced_scheduling: bool,
    ) -> Result<(MachineProgram, ScratchDebugMap), LoweringError> {
        machine::lower_function(self, advanced_scheduling)
    }
}

#[derive(Debug, Default, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
}

impl Program {
    pub fn lower_to_machine(
        &self,
        advanced_scheduling: bool,
    ) -> Result<Vec<(MachineProgram, ScratchDebugMap)>, LoweringError> {
        self.functions
            .iter()
            .map(|f| f.lower_to_machine(advanced_scheduling))
            .collect()
    }

    pub fn to_python_list_literal(&self) -> Result<Vec<String>, LoweringError> {
        let lowered = self.lower_to_machine(true)?;
        Ok(lowered
            .iter()
            .map(|(p, _)| InstructionBundle::program_to_python_list_literal(p))
            .collect())
    }
}
