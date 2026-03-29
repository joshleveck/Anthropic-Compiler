use crate::vlir::valu::ValuOp;
use crate::vlir::{BlockId, RegisterId};

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
