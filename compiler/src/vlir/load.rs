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
    /// When true, `dst` and `base_ptr` are vector registers and this load writes only
    /// `dst + offset` from `mem[scratch[base_ptr + offset]]` (matches `load_offset` in `problem.py`).
    /// Used for `dst_vec[lane] = load(addr_vec[lane])` when `lane` is the same for both vectors.
    pub vector_gather: bool,
}
