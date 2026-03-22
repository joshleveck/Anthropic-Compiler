use crate::vlir::RegisterId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreKind {
    I32,
    U32,
    Vec128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreInst {
    pub kind: StoreKind,
    pub base_ptr: RegisterId,
    pub offset: i32,
    pub src: RegisterId,
}
