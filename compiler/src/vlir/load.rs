use crate::vlir::RegisterId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadKind {
    I32,
    U32,
    Vec128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadInst {
    pub kind: LoadKind,
    pub dst: RegisterId,
    pub base_ptr: RegisterId,
    pub offset: i32,
}
