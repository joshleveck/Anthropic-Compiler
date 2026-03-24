//! VLIR → machine program: VLIW scheduling, greedy scratch allocation, and JSON emission.

mod bundle;
mod error;
mod ir;
mod lower;
mod scheduling;
mod scratch;
mod slot;

pub use bundle::{InstructionBundle, MachineProgram, ScratchDebugMap, ScratchLifetimeTrace};
pub use error::LoweringError;
pub use lower::lower_function;

/// Must match `problem.SCRATCH_SIZE` — scratch is indexed by word (32-bit), 0..this-1.
pub const MACHINE_SCRATCH_SIZE: usize = 100000;
