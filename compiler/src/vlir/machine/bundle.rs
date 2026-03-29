use std::collections::HashMap;
use std::fmt::Write;

use serde::Serialize;

use super::error::LoweringError;
use super::slot::Slot;

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            c => out.push(c),
        }
    }
    out
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InstructionBundle {
    pub alu: Vec<Slot>,
    pub valu: Vec<Slot>,
    pub load: Vec<Slot>,
    pub store: Vec<Slot>,
    pub flow: Vec<Slot>,
    pub debug: Vec<Slot>,
}

impl InstructionBundle {
    pub fn to_json_object_literal(&self) -> String {
        let mut sections = Vec::new();
        if !self.alu.is_empty() {
            sections.push(Self::engine_to_json("alu", &self.alu));
        }
        if !self.valu.is_empty() {
            sections.push(Self::engine_to_json("valu", &self.valu));
        }
        if !self.load.is_empty() {
            sections.push(Self::engine_to_json("load", &self.load));
        }
        if !self.store.is_empty() {
            sections.push(Self::engine_to_json("store", &self.store));
        }
        if !self.flow.is_empty() {
            sections.push(Self::engine_to_json("flow", &self.flow));
        }
        if !self.debug.is_empty() {
            sections.push(Self::engine_to_json("debug", &self.debug));
        }
        format!("{{{}}}", sections.join(","))
    }

    fn engine_to_json(name: &str, slots: &[Slot]) -> String {
        let joined = slots
            .iter()
            .map(Slot::to_json_array)
            .collect::<Vec<_>>()
            .join(",");
        format!(r#""{}":[{}]"#, name, joined)
    }

    pub fn program_to_json(program: &MachineProgram) -> String {
        let body = program
            .iter()
            .map(InstructionBundle::to_json_object_literal)
            .collect::<Vec<_>>()
            .join(",");
        format!("[{}]", body)
    }

    /// Full compiler output: instruction bundles plus `debug_info.scratch_map` for `problem.DebugInfo`.
    pub fn program_to_json_with_debug(
        program: &MachineProgram,
        scratch_map: &ScratchDebugMap,
    ) -> String {
        Self::program_to_json_with_debug_and_trace(program, scratch_map, None)
    }

    /// Like [`program_to_json_with_debug`], optionally embedding `scratch_lifetime` when present.
    pub fn program_to_json_with_debug_and_trace(
        program: &MachineProgram,
        scratch_map: &ScratchDebugMap,
        scratch_lifetime: Option<&ScratchLifetimeTrace>,
    ) -> String {
        let instr = Self::program_to_json(program);
        let mut keys: Vec<usize> = scratch_map.keys().copied().collect();
        keys.sort_unstable();
        let entries: Vec<String> = keys
            .into_iter()
            .map(|k| {
                let (name, len) = &scratch_map[&k];
                let esc = escape_json_string(name);
                format!(r#""{}":["{}",{}]"#, k, esc, len)
            })
            .collect();
        let sm = entries.join(",");
        let lifetime = match scratch_lifetime {
            Some(t) => format!(
                r#","scratch_lifetime":{}"#,
                serde_json::to_string(t).unwrap_or_else(|_| "{}".to_string())
            ),
            None => String::new(),
        };
        format!(
            r#"{{"instructions":{},"debug_info":{{"scratch_map":{{{}}}{}}}}}"#,
            instr, sm, lifetime
        )
    }

    pub fn assert_valid(&self) -> Result<(), LoweringError> {
        if self.alu.len() > 12 {
            return Err(LoweringError::SlotLimitExceeded(
                "alu".to_string(),
                self.alu.len(),
                12,
            ));
        }
        if self.valu.len() > 6 {
            return Err(LoweringError::SlotLimitExceeded(
                "valu".to_string(),
                self.valu.len(),
                6,
            ));
        }
        if self.load.len() > 2 {
            return Err(LoweringError::SlotLimitExceeded(
                "load".to_string(),
                self.load.len(),
                2,
            ));
        }
        if self.store.len() > 2 {
            return Err(LoweringError::SlotLimitExceeded(
                "store".to_string(),
                self.store.len(),
                2,
            ));
        }
        if self.flow.len() > 1 {
            return Err(LoweringError::SlotLimitExceeded(
                "flow".to_string(),
                self.flow.len(),
                1,
            ));
        }
        if self.debug.len() > 64 {
            return Err(LoweringError::SlotLimitExceeded(
                "debug".to_string(),
                self.debug.len(),
                64,
            ));
        }
        Ok(())
    }
}

pub type MachineProgram = Vec<InstructionBundle>;

/// Base scratch word -> (symbolic name, length in 32-bit words). Matches `problem.DebugInfo.scratch_map`.
pub type ScratchDebugMap = HashMap<usize, (String, usize)>;

/// Ordered log of scratch allocator actions while simulating greedy placement.
/// `bundle_index` is the linear index of the VLIW bundle in program order (matches emitted
/// `instructions` array). Terminator-only bundles are included after each block's scheduled cycles.
#[derive(Debug, Clone, Serialize)]
pub struct ScratchLifetimeTrace {
    pub events: Vec<ScratchLifetimeEvent>,
    pub intervals: Vec<ScratchLifetimeInterval>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScratchLifetimeEvent {
    pub step: u64,
    pub bundle_index: usize,
    pub block: usize,
    pub cycle_in_block: usize,
    pub instr_id: Option<u32>,
    #[serde(rename = "kind")]
    pub event_kind: ScratchLifetimeEventKind,
    pub reg: u32,
    pub reg_name: String,
    pub scratch_base: Option<usize>,
    pub width: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScratchLifetimeEventKind {
    Alloc,
    UseRead,
    UseDef,
    Free,
}

/// One logical register's live range in (bundle_index, step) space.
#[derive(Debug, Clone, Serialize)]
pub struct ScratchLifetimeInterval {
    pub reg: u32,
    pub reg_name: String,
    pub scratch_base: usize,
    pub width: usize,
    pub alloc_bundle: usize,
    pub free_bundle: usize,
    pub alloc_step: u64,
    pub free_step: u64,
}
