use crate::vlir::{BlockId, RegisterId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowInst {
    Select {
        dst: RegisterId,
        cond: RegisterId,
        a: RegisterId,
        b: RegisterId,
    },
    VSelect {
        dst: RegisterId,
        cond: RegisterId,
        a: RegisterId,
        b: RegisterId,
    },
    AddImm {
        dst: RegisterId,
        a: RegisterId,
        imm: i32,
    },
    /// Pauses the core (`problem.Machine` `enable_pause`); matches reference kernel yields.
    Pause,
    /// Scheduling barrier / no-op flow op.
    /// Lowers to `("sync",)` and acts as a full issue boundary.
    Sync,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Jump {
        target: BlockId,
    },
    Branch {
        cond: RegisterId,
        then_bb: BlockId,
        else_bb: BlockId,
    },
    Return {
        value: Option<RegisterId>,
    },
    Unreachable,
}
