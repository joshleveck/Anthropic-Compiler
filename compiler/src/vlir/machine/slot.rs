#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot {
    Alu(&'static str, usize, usize, usize),
    Valu(&'static str, usize, usize, usize),
    ValuBroadcast(usize, usize),
    ValuMulAdd(usize, usize, usize, usize),
    Load(&'static str, usize, usize),
    LoadOffset(usize, usize, i32),
    Const(usize, i32),
    Store(&'static str, usize, usize),
    FlowSelect(usize, usize, usize, usize),
    FlowVSelect(usize, usize, usize, usize),
    FlowAddImm(usize, usize, i32),
    CondJump(usize, usize),
    Jump(usize),
    Halt,
    Pause,
    Sync,
    /// `("compare", scratch_addr, (round, batch, tag_str))` — matches `reference_kernel2` trace keys.
    DebugCompare(usize, i32, i32, u8),
}

impl Slot {
    pub(crate) fn to_python_tuple(&self) -> String {
        match self {
            Slot::Alu(op, d, a, b) => format!("(\"{op}\", {d}, {a}, {b})"),
            Slot::Valu(op, d, a, b) => format!("(\"{op}\", {d}, {a}, {b})"),
            Slot::ValuBroadcast(d, s) => format!("(\"vbroadcast\", {d}, {s})"),
            Slot::ValuMulAdd(d, a, b, c) => format!("(\"multiply_add\", {d}, {a}, {b}, {c})"),
            Slot::Load(op, d, a) => format!("(\"{op}\", {d}, {a})"),
            Slot::LoadOffset(d, a, o) => format!("(\"load_offset\", {d}, {a}, {o})"),
            Slot::Const(d, v) => format!("(\"const\", {d}, {v})"),
            Slot::Store(op, a, s) => format!("(\"{op}\", {a}, {s})"),
            Slot::FlowSelect(d, c, a, b) => format!("(\"select\", {d}, {c}, {a}, {b})"),
            Slot::FlowVSelect(d, c, a, b) => format!("(\"vselect\", {d}, {c}, {a}, {b})"),
            Slot::FlowAddImm(d, a, i) => format!("(\"add_imm\", {d}, {a}, {i})"),
            Slot::CondJump(c, p) => format!("(\"cond_jump\", {c}, {p})"),
            Slot::Jump(p) => format!("(\"jump\", {p})"),
            Slot::Halt => "(\"halt\",)".to_string(),
            Slot::Pause => "(\"pause\",)".to_string(),
            Slot::Sync => "(\"sync\",)".to_string(),
            Slot::DebugCompare(addr, r, b, tag) => {
                let ts = debug_tag_str(*tag);
                format!("(\"compare\", {addr}, ({r}, {b}, \"{ts}\"))")
            }
        }
    }

    pub(super) fn to_json_array(&self) -> String {
        match self {
            Slot::Alu(op, d, a, b) => format!(r#"["{}",{},{},{}]"#, op, d, a, b),
            Slot::Valu(op, d, a, b) => format!(r#"["{}",{},{},{}]"#, op, d, a, b),
            Slot::ValuBroadcast(d, s) => format!(r#"["vbroadcast",{},{}]"#, d, s),
            Slot::ValuMulAdd(d, a, b, c) => format!(r#"["multiply_add",{},{},{},{}]"#, d, a, b, c),
            Slot::Load(op, d, a) => format!(r#"["{}",{},{}]"#, op, d, a),
            Slot::LoadOffset(d, a, o) => format!(r#"["load_offset",{},{},{}]"#, d, a, o),
            Slot::Const(d, v) => format!(r#"["const",{},{}]"#, d, v),
            Slot::Store(op, a, s) => format!(r#"["{}",{},{}]"#, op, a, s),
            Slot::FlowSelect(d, c, a, b) => format!(r#"["select",{},{},{},{}]"#, d, c, a, b),
            Slot::FlowVSelect(d, c, a, b) => format!(r#"["vselect",{},{},{},{}]"#, d, c, a, b),
            Slot::FlowAddImm(d, a, i) => format!(r#"["add_imm",{},{},{}]"#, d, a, i),
            Slot::CondJump(c, p) => format!(r#"["cond_jump",{},{}]"#, c, p),
            Slot::Jump(p) => format!(r#"["jump",{}]"#, p),
            Slot::Halt => r#"["halt"]"#.to_string(),
            Slot::Pause => r#"["pause"]"#.to_string(),
            Slot::Sync => r#"["sync"]"#.to_string(),
            Slot::DebugCompare(addr, r, b, tag) => {
                let ts = debug_tag_str(*tag);
                format!("[\"compare\",{},[{},{},\"{}\"]]", addr, r, b, ts)
            }
        }
    }
}

fn debug_tag_str(tag: u8) -> &'static str {
    match tag {
        0 => "idx",
        1 => "val",
        2 => "node_val",
        3 => "hashed_val",
        4 => "next_idx",
        5 => "wrapped_idx",
        _ => "unknown",
    }
}
