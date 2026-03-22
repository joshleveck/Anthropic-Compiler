use crate::vlir::RegisterId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValuOp {
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
    Broadcast,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValuInst {
    pub op: ValuOp,
    pub dst: RegisterId,
    pub src1: RegisterId,
    pub src2: RegisterId,
    pub src3: Option<RegisterId>,
}
