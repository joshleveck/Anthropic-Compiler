use crate::vlir::{Operand, RegisterId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    CmpEq,
    CmpLt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AluInst {
    pub op: AluOp,
    pub dst: RegisterId,
    pub lhs: Operand,
    pub rhs: Operand,
}
