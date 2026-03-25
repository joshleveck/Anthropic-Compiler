use std::collections::{HashMap, HashSet};

use crate::vlir::{Function, RegisterId};

use super::bundle::{
    ScratchLifetimeEvent, ScratchLifetimeEventKind, ScratchLifetimeInterval, ScratchLifetimeTrace,
};
use super::error::LoweringError;
use super::ir::{
    def_of_instruction, def_regs, is_side_effect_instruction, reg_width, uses_in_instruction,
    uses_in_terminator,
};
use super::scheduling::ScheduledBlock;

#[derive(Default)]
pub(crate) struct ScratchLayout {
    pub(crate) map: HashMap<RegisterId, usize>,
}

impl ScratchLayout {
    pub(crate) fn addr(&self, reg: RegisterId) -> Result<usize, LoweringError> {
        self.map
            .get(&reg)
            .copied()
            .ok_or(LoweringError::MissingScratch(reg))
    }
}

#[derive(Clone, Copy)]
struct TraceSite {
    bundle_index: usize,
    block: usize,
    cycle_in_block: usize,
    instr_id: Option<crate::vlir::InstrId>,
}

struct ScratchTraceState {
    events: Vec<ScratchLifetimeEvent>,
    intervals: Vec<ScratchLifetimeInterval>,
    /// Pending interval until `Free`: alloc_bundle, alloc_step, base, width, name
    pending: HashMap<RegisterId, (usize, u64, usize, usize, String)>,
    step: u64,
}

impl ScratchTraceState {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            intervals: Vec::new(),
            pending: HashMap::new(),
            step: 0,
        }
    }

    fn finish(self) -> ScratchLifetimeTrace {
        ScratchLifetimeTrace {
            events: self.events,
            intervals: self.intervals,
        }
    }

    fn push_event(
        &mut self,
        site: TraceSite,
        kind: ScratchLifetimeEventKind,
        func: &Function,
        reg: RegisterId,
        scratch_base: Option<usize>,
        width: usize,
    ) {
        let reg_name = func.reg_display_name(reg);
        self.events.push(ScratchLifetimeEvent {
            step: self.step,
            bundle_index: site.bundle_index,
            block: site.block,
            cycle_in_block: site.cycle_in_block,
            instr_id: site.instr_id.map(|i| i.0),
            event_kind: kind,
            reg: reg.0,
            reg_name,
            scratch_base,
            width,
        });
        self.step += 1;
    }
}

pub(crate) fn build_greedy_scratch_layout(
    func: &Function,
    use_counts: &HashMap<RegisterId, usize>,
    scheduled_blocks: &[ScheduledBlock],
    trace_scratch_lifetime: bool,
    machine_scratch_size: usize,
) -> Result<(ScratchLayout, Option<ScratchLifetimeTrace>), LoweringError> {
    let mut map = HashMap::new();
    let mut allocs: HashMap<RegisterId, (usize, usize)> = HashMap::new();
    let mut remaining_uses = use_counts.clone();
    let mut used = vec![false; machine_scratch_size];
    let mut trace_state = trace_scratch_lifetime.then(ScratchTraceState::new);

    let mut inst_by_id: HashMap<crate::vlir::InstrId, &crate::vlir::Instruction> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            inst_by_id.insert(inst.id, inst);
        }
    }

    let mut global_bundle_index = 0usize;
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ci, cycle) in scheduled_blocks[bi].iter().enumerate() {
            for iid in cycle {
                let inst = inst_by_id[iid];
                let site = TraceSite {
                    bundle_index: global_bundle_index,
                    block: bi,
                    cycle_in_block: ci,
                    instr_id: Some(inst.id),
                };
                for r in uses_in_instruction(inst) {
                    let fresh = ensure_allocated(
                        func,
                        r,
                        &mut allocs,
                        &mut map,
                        &mut used,
                        machine_scratch_size,
                    )?;
                    if let Some(ref mut t) = trace_state {
                        if fresh {
                            let (base, width) = allocs[&r];
                            let name = func.reg_display_name(r);
                            t.pending
                                .insert(r, (site.bundle_index, t.step, base, width, name));
                            t.push_event(
                                site,
                                ScratchLifetimeEventKind::Alloc,
                                func,
                                r,
                                Some(base),
                                width,
                            );
                        }
                        let base = map[&r];
                        let w = reg_width(func, r)?;
                        t.push_event(
                            site,
                            ScratchLifetimeEventKind::UseRead,
                            func,
                            r,
                            Some(base),
                            w,
                        );
                    }
                    decrement_and_maybe_free(
                        r,
                        &mut remaining_uses,
                        &mut allocs,
                        &mut used,
                        site,
                        func,
                        trace_state.as_mut(),
                    );
                }

                if let Some(dst) = def_of_instruction(inst) {
                    let fresh = ensure_allocated(
                        func,
                        dst,
                        &mut allocs,
                        &mut map,
                        &mut used,
                        machine_scratch_size,
                    )?;
                    if let Some(ref mut t) = trace_state {
                        if fresh {
                            let (base, width) = allocs[&dst];
                            let name = func.reg_display_name(dst);
                            t.pending
                                .insert(dst, (site.bundle_index, t.step, base, width, name));
                            t.push_event(
                                site,
                                ScratchLifetimeEventKind::Alloc,
                                func,
                                dst,
                                Some(base),
                                width,
                            );
                        } else {
                            let base = map[&dst];
                            let w = reg_width(func, dst)?;
                            t.push_event(
                                site,
                                ScratchLifetimeEventKind::UseDef,
                                func,
                                dst,
                                Some(base),
                                w,
                            );
                        }
                    }
                    maybe_free_dead_def(
                        dst,
                        &remaining_uses,
                        &mut allocs,
                        &mut used,
                        site,
                        func,
                        trace_state.as_mut(),
                    );
                } else {
                    for r in def_regs(inst) {
                        let fresh = ensure_allocated(
                            func,
                            r,
                            &mut allocs,
                            &mut map,
                            &mut used,
                            machine_scratch_size,
                        )?;
                        if let Some(ref mut t) = trace_state {
                            if fresh {
                                let (base, width) = allocs[&r];
                                let name = func.reg_display_name(r);
                                t.pending
                                    .insert(r, (site.bundle_index, t.step, base, width, name));
                                t.push_event(
                                    site,
                                    ScratchLifetimeEventKind::Alloc,
                                    func,
                                    r,
                                    Some(base),
                                    width,
                                );
                            } else {
                                let base = map[&r];
                                let w = reg_width(func, r)?;
                                t.push_event(
                                    site,
                                    ScratchLifetimeEventKind::UseDef,
                                    func,
                                    r,
                                    Some(base),
                                    w,
                                );
                            }
                        }
                        maybe_free_dead_def(
                            r,
                            &remaining_uses,
                            &mut allocs,
                            &mut used,
                            site,
                            func,
                            trace_state.as_mut(),
                        );
                    }
                }
            }
            global_bundle_index += 1;
        }
        let term_site = TraceSite {
            bundle_index: global_bundle_index,
            block: bi,
            cycle_in_block: scheduled_blocks[bi].len(),
            instr_id: None,
        };
        for r in uses_in_terminator(&block.terminator) {
            let fresh = ensure_allocated(
                func,
                r,
                &mut allocs,
                &mut map,
                &mut used,
                machine_scratch_size,
            )?;
            if let Some(ref mut t) = trace_state {
                if fresh {
                    let (base, width) = allocs[&r];
                    let name = func.reg_display_name(r);
                    t.pending
                        .insert(r, (term_site.bundle_index, t.step, base, width, name));
                    t.push_event(
                        term_site,
                        ScratchLifetimeEventKind::Alloc,
                        func,
                        r,
                        Some(base),
                        width,
                    );
                }
                let base = map[&r];
                let w = reg_width(func, r)?;
                t.push_event(
                    term_site,
                    ScratchLifetimeEventKind::UseRead,
                    func,
                    r,
                    Some(base),
                    w,
                );
            }
            decrement_and_maybe_free(
                r,
                &mut remaining_uses,
                &mut allocs,
                &mut used,
                term_site,
                func,
                trace_state.as_mut(),
            );
        }
        global_bundle_index += super::ir::terminator_bundle_count(&block.terminator);
    }

    let lifetime = trace_state.map(ScratchTraceState::finish);

    Ok((ScratchLayout { map }, lifetime))
}

fn decrement_and_maybe_free(
    reg: RegisterId,
    remaining_uses: &mut HashMap<RegisterId, usize>,
    allocs: &mut HashMap<RegisterId, (usize, usize)>,
    used: &mut [bool],
    site: TraceSite,
    func: &Function,
    trace: Option<&mut ScratchTraceState>,
) {
    let Some(count) = remaining_uses.get_mut(&reg) else {
        return;
    };
    if *count == 0 {
        return;
    }
    *count -= 1;
    if *count == 0 {
        if let Some((base, width)) = allocs.remove(&reg) {
            release_range(used, base, width);
            if let Some(t) = trace {
                let w = width;
                let free_step = t.step;
                t.push_event(
                    site,
                    ScratchLifetimeEventKind::Free,
                    func,
                    reg,
                    Some(base),
                    w,
                );
                if let Some((alloc_bundle, alloc_step, b, w2, name)) = t.pending.remove(&reg) {
                    t.intervals.push(ScratchLifetimeInterval {
                        reg: reg.0,
                        reg_name: name,
                        scratch_base: b,
                        width: w2,
                        alloc_bundle,
                        free_bundle: site.bundle_index,
                        alloc_step,
                        free_step,
                    });
                }
            }
        }
    }
}

fn maybe_free_dead_def(
    reg: RegisterId,
    remaining_uses: &HashMap<RegisterId, usize>,
    allocs: &mut HashMap<RegisterId, (usize, usize)>,
    used: &mut [bool],
    site: TraceSite,
    func: &Function,
    trace: Option<&mut ScratchTraceState>,
) {
    if remaining_uses.get(&reg).copied().unwrap_or(0) != 0 {
        return;
    }
    if let Some((base, width)) = allocs.remove(&reg) {
        release_range(used, base, width);
        if let Some(t) = trace {
            let w = width;
            let free_step = t.step;
            t.push_event(
                site,
                ScratchLifetimeEventKind::Free,
                func,
                reg,
                Some(base),
                w,
            );
            if let Some((alloc_bundle, alloc_step, b, w2, name)) = t.pending.remove(&reg) {
                t.intervals.push(ScratchLifetimeInterval {
                    reg: reg.0,
                    reg_name: name,
                    scratch_base: b,
                    width: w2,
                    alloc_bundle,
                    free_bundle: site.bundle_index,
                    alloc_step,
                    free_step,
                });
            }
        }
    }
}

fn ensure_allocated(
    func: &Function,
    reg: RegisterId,
    allocs: &mut HashMap<RegisterId, (usize, usize)>,
    map: &mut HashMap<RegisterId, usize>,
    used: &mut [bool],
    machine_scratch_size: usize,
) -> Result<bool, LoweringError> {
    if allocs.contains_key(&reg) {
        return Ok(false);
    }
    let width = reg_width(func, reg)?;
    let Some(base) = alloc_fit(used, width) else {
        let occupied = used.iter().filter(|x| **x).count();
        return Err(LoweringError::ScratchOverflow {
            used: occupied + width,
            limit: machine_scratch_size,
        });
    };
    for slot in used.iter_mut().skip(base).take(width) {
        *slot = true;
    }
    allocs.insert(reg, (base, width));
    map.insert(reg, base);
    Ok(true)
}

/// Width-aware allocation to reduce fragmentation:
/// - Vectors (width>1): low-to-high first-fit
/// - Scalars (width=1): high-to-low first-fit
/// This keeps contiguous low-end runs available for vec8 temporaries while scalar churn
/// tends to occupy/disrupt high-end single slots.
fn alloc_fit(used: &[bool], width: usize) -> Option<usize> {
    if width <= 1 {
        return alloc_first_fit_high(used, width).or_else(|| alloc_first_fit_low(used, width));
    }
    alloc_first_fit_low(used, width).or_else(|| alloc_first_fit_high(used, width))
}

fn alloc_first_fit_low(used: &[bool], width: usize) -> Option<usize> {
    if width == 0 || width > used.len() {
        return None;
    }
    let mut run = 0usize;
    let mut start = 0usize;
    for (idx, occupied) in used.iter().enumerate() {
        if !occupied {
            if run == 0 {
                start = idx;
            }
            run += 1;
            if run >= width {
                return Some(start);
            }
        } else {
            run = 0;
        }
    }
    None
}

fn alloc_first_fit_high(used: &[bool], width: usize) -> Option<usize> {
    if width == 0 || width > used.len() {
        return None;
    }
    let mut run = 0usize;
    let mut end = 0usize;
    for idx in (0..used.len()).rev() {
        if !used[idx] {
            if run == 0 {
                end = idx;
            }
            run += 1;
            if run >= width {
                return Some(end + 1 - width);
            }
        } else {
            run = 0;
        }
    }
    None
}

fn release_range(used: &mut [bool], base: usize, width: usize) {
    for slot in used.iter_mut().skip(base).take(width) {
        *slot = false;
    }
}

pub(crate) fn collect_use_counts(
    func: &Function,
    scheduled_blocks: &[ScheduledBlock],
) -> HashMap<RegisterId, usize> {
    let mut uses = HashMap::new();
    let mut inst_by_id: HashMap<crate::vlir::InstrId, &crate::vlir::Instruction> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            inst_by_id.insert(inst.id, inst);
        }
    }
    for (bi, block) in func.blocks.iter().enumerate() {
        for cycle in &scheduled_blocks[bi] {
            for iid in cycle {
                let inst = inst_by_id[iid];
                for r in uses_in_instruction(inst) {
                    *uses.entry(r).or_insert(0) += 1;
                }
            }
        }
        for r in uses_in_terminator(&block.terminator) {
            *uses.entry(r).or_insert(0) += 1;
        }
    }
    uses
}

pub(crate) fn build_emission_plan(func: &Function) -> HashSet<crate::vlir::InstrId> {
    let mut needed: HashSet<RegisterId> = HashSet::new();
    let mut emitted: HashSet<crate::vlir::InstrId> = HashSet::new();

    for block in func.blocks.iter().rev() {
        for r in uses_in_terminator(&block.terminator) {
            needed.insert(r);
        }
        for inst in block.instructions.iter().rev() {
            match def_of_instruction(inst) {
                Some(dst) => {
                    if needed.contains(&dst) {
                        emitted.insert(inst.id);
                        for r in uses_in_instruction(inst) {
                            needed.insert(r);
                        }
                    }
                }
                None => {
                    if is_side_effect_instruction(inst) {
                        emitted.insert(inst.id);
                        for r in uses_in_instruction(inst) {
                            needed.insert(r);
                        }
                    }
                }
            }
        }
    }
    emitted
}
