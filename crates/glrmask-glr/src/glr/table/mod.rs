use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::analysis::{AnalyzedGrammar, EOF};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};

pub(crate) mod action;
mod build;
mod compose;
mod optimize;
pub(crate) mod row;

pub use action::{Action, GuardedStackShift, StackShift, StackShiftGuard};
#[allow(unused_imports)]
pub use compose::{
    ComposedTable, SubgrammarTableInput, compose_subgrammar_tables,
    compose_subgrammar_tables_explicit, compose_subgrammar_tables_explicit_with_rules,
    compose_subgrammar_tables_with_rules, empty_terminals_in_composed_table,
    subgrammar_child_return_pop,
};

use build::{build_table, build_table_with_default_construction, Item, PendingAction};
#[allow(unused_imports)]
pub use optimize::ControlEliminationReport;
#[allow(unused_imports)]
pub use optimize::control_elimination_budget_exhausted;
#[allow(unused_imports)]
pub use optimize::control_elimination_run_count;
use optimize::merge_same_core_lr1_states;

use row::{ActionRow, GotoRow};

const DISABLE_DEFAULT_ACTION_ROWS_ENV: &str = "GLRMASK_DISABLE_DEFAULT_ACTION_ROWS";
const EMBEDDED_NULLABLE_START_SUFFIX: &str = "\0glrmask:embedded-nullable-start";
const EMBEDDED_END_TOKEN_IDS_PREFIX: &str = "\0glrmask:embedded-end-token-ids=";

fn default_action_rows_enabled() -> bool {
    !std::env::var(DISABLE_DEFAULT_ACTION_ROWS_ENV)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardedShiftCellIndex {
    pub guard_pops: Box<[u32]>,
    pub by_guard_key: FxHashMap<(u32, u32), Box<[u32]>>,
    pub guard_counts: Box<[u16]>,
    pub unguarded_indices: Box<[u32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdmissionPolicy {
    /// Cheap row-presence admission is exact for this table.
    RowPresenceExact,
    /// Dynamically simulate reductions/gotos before admitting a terminal.
    ExactSimulation,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self::RowPresenceExact
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GlrTableConstruction {
    LegacyRowBisim,
    Lalr,
    ExperimentalCoreMerged,
}

impl Default for GlrTableConstruction {
    fn default() -> Self {
        Self::LegacyRowBisim
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    pub action: Vec<ActionRow>,
    pub goto: Vec<GotoRow>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
    #[serde(default)]
    pub nonterminal_display_names: Vec<String>,
    #[serde(default)]
    pub construction: GlrTableConstruction,
    #[serde(default)]
    pub admission_policy: AdmissionPolicy,
    /// Terminal support used by cheap admission/mask queries.
    ///
    /// `action` is the optimized execution table. Some execution actions are
    /// guarded stack effects, whose guards must be evaluated when executing the
    /// action. This side table is captured before guard-producing stack-effect
    /// lowering and is kept in sync across state remapping/merging. A bit set in
    /// this vector answers only the admission question: "can a reachable parser
    /// path with this top state advance on this terminal?"  That lets
    /// `stack_may_advance_on*` be pure row-presence checks without inspecting an
    /// optimized action body.
    #[serde(default)]
    pub advance: Vec<BitSet>,
    /// Runtime-derived subset of `advance` whose current-row action is
    /// unconditionally admissible without inspecting deeper parser stack.
    /// Rebuilt after load/final table mutation; never serialized.
    #[serde(skip, default)]
    pub unconditional_advance: Vec<BitSet>,
    /// Set of (state, terminal) pairs where the shift was created by the
    /// transfer mechanism. The characterization should treat these as
    /// non-replace to avoid creating pop-0 reduces in the template NFA.
    pub forwarded_shifts: FxHashSet<(u32, TerminalID)>,
    /// Linker-internal zero-width terminal labels. These are never emitted by
    /// the lexer; parser execution computes their closure before and after a
    /// real terminal.
    #[serde(default)]
    pub control_terminals: BTreeSet<TerminalID>,
    /// Terminals whose table action is parser identity in a subset of states.
    /// This metadata preserves scoped ignore ownership across nested compiled
    /// composition; the actions themselves remain the execution authority.
    #[serde(default)]
    pub skip_terminals: BTreeSet<TerminalID>,
    #[serde(skip)]
    pub guarded_shift_index: Vec<FxHashMap<TerminalID, GuardedShiftCellIndex>>,
    /// Maximum-width direct-regular frontier descriptors captured while rows
    /// are constructed, before the expanded table would need to be rescanned.
    #[serde(default)]
    pub direct_regular_wide_frontiers: Vec<DirectRegularWideFrontierDescriptor>,
}

/// Version-scoped artifact serialization for GLR tables.
///
/// Historical constraint formats delegate to the ordinary derived GLRTable
/// serde unchanged. New sectioned artifacts can replace the in-core table with
/// a one-byte placeholder and serialize the real table independently using a
/// compact representation of the `advance` relation.
pub mod artifact_serde {
    use std::cell::Cell;

    use super::*;
    use rayon::prelude::*;
    use serde::{Deserializer, Serializer};

    thread_local! {
        static EXTERNAL_TABLE_SERDE: Cell<bool> = const { Cell::new(false) };
    }

    pub fn set_external_serde(enabled: bool) -> bool {
        EXTERNAL_TABLE_SERDE.with(|mode| mode.replace(enabled))
    }

    pub fn external_serde_enabled() -> bool {
        EXTERNAL_TABLE_SERDE.with(Cell::get)
    }

    #[derive(Serialize, Deserialize)]
    struct CompactAdvance {
        bit_len: u32,
        row_ids: Vec<u32>,
        unique_rows: Vec<Vec<u32>>,
    }

    impl CompactAdvance {
        fn from_rows(rows: &[BitSet]) -> Self {
            let bit_len = rows.first().map_or(0usize, BitSet::len);
            debug_assert!(rows.iter().all(|row| row.len() == bit_len));
            let mut row_ids = Vec::with_capacity(rows.len());
            let mut unique_rows = Vec::<Vec<u32>>::new();
            let mut by_row = FxHashMap::<BitSet, u32>::default();
            for row in rows {
                if let Some(&id) = by_row.get(row) {
                    row_ids.push(id);
                    continue;
                }
                let id = unique_rows.len() as u32;
                unique_rows.push(row.iter_ones().map(|terminal| terminal as u32).collect());
                by_row.insert(row.clone(), id);
                row_ids.push(id);
            }
            Self {
                bit_len: u32::try_from(bit_len).expect("GLR advance bitset width should fit u32"),
                row_ids,
                unique_rows,
            }
        }

        fn into_rows(self, num_terminals: u32) -> Result<Vec<BitSet>, String> {
            // Empty advance rows are a valid omitted/derived cache. Recursive
            // composition coordinators deliberately serialize a zero-state
            // grammar shell, whose terminal domain remains meaningful even
            // though it has no executable advance rows.
            if self.row_ids.is_empty() && self.unique_rows.is_empty() {
                return Ok(Vec::new());
            }
            if self.bit_len < num_terminals {
                return Err(format!(
                    "compact GLR advance width {} is smaller than {num_terminals} terminals",
                    self.bit_len,
                ));
            }
            let mut unique = Vec::with_capacity(self.unique_rows.len());
            for terminals in self.unique_rows {
                let mut row = BitSet::new(self.bit_len as usize);
                for terminal in terminals {
                    if terminal >= self.bit_len {
                        return Err(format!(
                            "compact GLR advance terminal {terminal} is out of range for width {}",
                            self.bit_len,
                        ));
                    }
                    row.set(terminal as usize);
                }
                unique.push(row);
            }
            self.row_ids
                .into_iter()
                .map(|row| {
                    unique
                        .get(row as usize)
                        .cloned()
                        .ok_or_else(|| "invalid compact GLR advance row id".to_owned())
                })
                .collect()
        }
    }

    #[derive(Serialize)]
    struct CompactTableRef<'a> {
        action: &'a [ActionRow],
        goto: &'a [GotoRow],
        num_states: u32,
        num_terminals: u32,
        num_rules: u32,
        rules: &'a [Rule],
        nonterminal_display_names: &'a [String],
        construction: GlrTableConstruction,
        admission_policy: AdmissionPolicy,
        advance: CompactAdvance,
        forwarded_shifts: &'a FxHashSet<(u32, TerminalID)>,
        control_terminals: &'a BTreeSet<TerminalID>,
        skip_terminals: &'a BTreeSet<TerminalID>,
        direct_regular_wide_frontiers: &'a [DirectRegularWideFrontierDescriptor],
    }

    #[derive(Deserialize)]
    struct CompactTable {
        action: Vec<ActionRow>,
        goto: Vec<GotoRow>,
        num_states: u32,
        num_terminals: u32,
        num_rules: u32,
        rules: Vec<Rule>,
        nonterminal_display_names: Vec<String>,
        construction: GlrTableConstruction,
        admission_policy: AdmissionPolicy,
        advance: CompactAdvance,
        forwarded_shifts: FxHashSet<(u32, TerminalID)>,
        control_terminals: BTreeSet<TerminalID>,
        skip_terminals: BTreeSet<TerminalID>,
        direct_regular_wide_frontiers: Vec<DirectRegularWideFrontierDescriptor>,
    }

    #[derive(Serialize)]
    struct CompactTableMetaRef<'a> {
        num_states: u32,
        num_terminals: u32,
        num_rules: u32,
        rules: &'a [Rule],
        nonterminal_display_names: &'a [String],
        construction: GlrTableConstruction,
        admission_policy: AdmissionPolicy,
        forwarded_shifts: &'a FxHashSet<(u32, TerminalID)>,
        control_terminals: &'a BTreeSet<TerminalID>,
        skip_terminals: &'a BTreeSet<TerminalID>,
        direct_regular_wide_frontiers: &'a [DirectRegularWideFrontierDescriptor],
    }

    #[derive(Deserialize)]
    struct CompactTableMeta {
        num_states: u32,
        num_terminals: u32,
        num_rules: u32,
        rules: Vec<Rule>,
        nonterminal_display_names: Vec<String>,
        construction: GlrTableConstruction,
        admission_policy: AdmissionPolicy,
        forwarded_shifts: FxHashSet<(u32, TerminalID)>,
        control_terminals: BTreeSet<TerminalID>,
        skip_terminals: BTreeSet<TerminalID>,
        direct_regular_wide_frontiers: Vec<DirectRegularWideFrontierDescriptor>,
    }

    #[derive(Serialize)]
    struct CompactTableMetaNoRulesRef<'a> {
        num_states: u32,
        num_terminals: u32,
        num_rules: u32,
        first_rule: Option<&'a Rule>,
        nonterminal_display_names: &'a [String],
        construction: GlrTableConstruction,
        admission_policy: AdmissionPolicy,
        forwarded_shifts: &'a FxHashSet<(u32, TerminalID)>,
        control_terminals: &'a BTreeSet<TerminalID>,
        skip_terminals: &'a BTreeSet<TerminalID>,
        direct_regular_wide_frontiers: &'a [DirectRegularWideFrontierDescriptor],
    }

    #[derive(Deserialize)]
    struct CompactTableMetaNoRules {
        num_states: u32,
        num_terminals: u32,
        num_rules: u32,
        first_rule: Option<Rule>,
        nonterminal_display_names: Vec<String>,
        construction: GlrTableConstruction,
        admission_policy: AdmissionPolicy,
        forwarded_shifts: FxHashSet<(u32, TerminalID)>,
        control_terminals: BTreeSet<TerminalID>,
        skip_terminals: BTreeSet<TerminalID>,
        direct_regular_wide_frontiers: Vec<DirectRegularWideFrontierDescriptor>,
    }

    const PARALLEL_TABLE_MAGIC_V2: &[u8; 4] = b"GTC2";
    const PARALLEL_TABLE_MAGIC: &[u8; 4] = b"GTC3";
    const PARALLEL_TABLE_HEADER_LEN: usize = 4 + 4 * 8;
    const CHUNKED_ACTION_MAGIC: &[u8; 4] = b"GTA1";
    const SIMPLE_ACTION_MAGIC: &[u8; 4] = b"GSA1";
    const CHUNKED_ACTION_HEADER_LEN: usize = 12;
    const CHUNKED_ACTION_MIN_ROWS: usize = 8_192;
    const SIMPLE_ACTION_MIN_ROWS: usize = 512;
    const DEFERRED_META_MAGIC: &[u8; 4] = b"GTM2";
    const DEFERRED_META_HEADER_LEN: usize = 4 + 4 + 8 + 8;
    const CHUNKED_RULES_MIN_ROWS: usize = 1_024;

    #[inline]
    fn put_fixed_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    #[inline]
    fn take_fixed_u32(input: &[u8], pos: &mut usize, context: &str) -> Result<u32, String> {
        let end = pos
            .checked_add(4)
            .ok_or_else(|| format!("overflowing {context} offset"))?;
        let bytes: [u8; 4] = input
            .get(*pos..end)
            .ok_or_else(|| format!("truncated {context} u32"))?
            .try_into()
            .expect("four-byte slice");
        *pos = end;
        Ok(u32::from_le_bytes(bytes))
    }

    fn decode_simple_action(input: &[u8], pos: &mut usize) -> Result<Action, String> {
        let tag = *input
            .get(*pos)
            .ok_or_else(|| "truncated simple GLR action tag".to_owned())?;
        *pos += 1;
        match tag {
            0 => {
                let target = take_fixed_u32(input, pos, "simple GLR shift target")?;
                let replace = match input.get(*pos).copied() {
                    Some(0) => false,
                    Some(1) => true,
                    Some(_) => return Err("invalid simple GLR shift replace flag".to_owned()),
                    None => return Err("truncated simple GLR shift replace flag".to_owned()),
                };
                *pos += 1;
                Ok(Action::Shift(target, replace))
            }
            1 => {
                let nonterminal = take_fixed_u32(input, pos, "simple GLR reduce nonterminal")?;
                let len = take_fixed_u32(input, pos, "simple GLR reduce length")?;
                Ok(Action::Reduce(nonterminal, len))
            }
            2 => Ok(Action::Accept),
            3 => Ok(Action::Skip),
            4 => {
                let flags = *input
                    .get(*pos)
                    .ok_or_else(|| "truncated simple GLR split flags".to_owned())?;
                *pos += 1;
                if flags & !0b111 != 0 || (flags & 0b10 != 0 && flags & 0b1 == 0) {
                    return Err("invalid simple GLR split flags".to_owned());
                }
                let shift = if flags & 1 != 0 {
                    let target = take_fixed_u32(input, pos, "simple GLR split target")?;
                    Some((target, flags & 2 != 0))
                } else {
                    None
                };
                let reduce_count =
                    take_fixed_u32(input, pos, "simple GLR split reduce count")? as usize;
                let mut reduces = Vec::with_capacity(reduce_count);
                for _ in 0..reduce_count {
                    reduces.push((
                        take_fixed_u32(input, pos, "simple GLR split nonterminal")?,
                        take_fixed_u32(input, pos, "simple GLR split reduce length")?,
                    ));
                }
                Ok(Action::Split {
                    shift,
                    reduces,
                    accept: flags & 4 != 0,
                })
            }
            _ => Err("invalid simple GLR action tag".to_owned()),
        }
    }

    #[inline]
    fn simple_action_supported(action: &Action) -> bool {
        matches!(
            action,
            Action::Shift(..)
                | Action::Reduce(..)
                | Action::Split { .. }
                | Action::Accept
                | Action::Skip
        )
    }

    fn encode_simple_action(out: &mut Vec<u8>, action: &Action) {
        match action {
            Action::Shift(target, replace) => {
                out.push(0);
                put_fixed_u32(out, *target);
                out.push(u8::from(*replace));
            }
            Action::Reduce(nonterminal, len) => {
                out.push(1);
                put_fixed_u32(out, *nonterminal);
                put_fixed_u32(out, *len);
            }
            Action::Accept => out.push(2),
            Action::Skip => out.push(3),
            Action::Split {
                shift,
                reduces,
                accept,
            } => {
                out.push(4);
                let mut flags = u8::from(shift.is_some());
                if shift.is_some_and(|(_, replace)| replace) {
                    flags |= 2;
                }
                if *accept {
                    flags |= 4;
                }
                out.push(flags);
                if let Some((target, _)) = shift {
                    put_fixed_u32(out, *target);
                }
                put_fixed_u32(out, reduces.len() as u32);
                for &(nonterminal, len) in reduces {
                    put_fixed_u32(out, nonterminal);
                    put_fixed_u32(out, len);
                }
            }
            _ => unreachable!("simple-action precheck rejected complex action"),
        }
    }

    fn encode_simple_action_rows(rows: &[ActionRow]) -> Option<Vec<u8>> {
        let simple = rows.iter().all(|row| {
            !row.is_default_compressed()
                && row.values().all(simple_action_supported)
        });
        if !simple {
            return None;
        }

        let target_chunks = (rayon::current_num_threads() * 2).clamp(2, 16);
        // Small-but-dense tables such as JS (~685 rows, mostly 50-70 actions
        // per row) are dominated by independent hash-row construction. A 512
        // row floor left only two jobs on a 16-thread host. Keep chunks large
        // enough to amortize Rayon, but expose the row-allocation parallelism.
        let encode_chunk = |chunk: &[ActionRow]| {
            let mut body = Vec::new();
            for row in chunk {
                put_fixed_u32(&mut body, row.len() as u32);
                for (terminal, action) in row.iter() {
                    put_fixed_u32(&mut body, terminal);
                    encode_simple_action(&mut body, action);
                }
            }
            body
        };
        let (chunks, chunk_rows) = if rayon::current_num_threads() == 1 {
            (vec![encode_chunk(rows)], vec![rows.len()])
        } else {
            let chunk_size = rows.len().div_ceil(target_chunks).max(64);
            (
                rows.par_chunks(chunk_size)
                    .map(encode_chunk)
                    .collect::<Vec<_>>(),
                rows.chunks(chunk_size)
                    .map(|chunk| chunk.len())
                    .collect::<Vec<_>>(),
            )
        };
        let payload_len = chunks.iter().map(Vec::len).sum::<usize>();
        let mut out = Vec::with_capacity(
            CHUNKED_ACTION_HEADER_LEN + chunks.len() * (4 + 8) + payload_len,
        );
        out.extend_from_slice(SIMPLE_ACTION_MAGIC);
        out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        for (&row_count, bytes) in chunk_rows.iter().zip(&chunks) {
            out.extend_from_slice(&(row_count as u32).to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        }
        for bytes in chunks {
            out.extend_from_slice(&bytes);
        }
        Some(out)
    }

    fn encode_action_rows(rows: &[ActionRow]) -> Vec<u8> {
        let has_indexed_rows = rows
            .iter()
            .any(|row| matches!(row, ActionRow::Indexed { .. }));
        if (has_indexed_rows
            || (rows.len() >= SIMPLE_ACTION_MIN_ROWS && rayon::current_num_threads() > 1))
            && let Some(bytes) = encode_simple_action_rows(rows)
        {
            return bytes;
        }
        if rows.len() < CHUNKED_ACTION_MIN_ROWS || rayon::current_num_threads() == 1 {
            return bincode::serialize(rows).expect("GLR action serialization should succeed");
        }
        let target_chunks = (rayon::current_num_threads() * 2).clamp(2, 16);
        let chunk_size = rows.len().div_ceil(target_chunks).max(512);
        let chunks = rows
            .par_chunks(chunk_size)
            .map(|chunk| {
                bincode::serialize(chunk).expect("GLR action chunk serialization should succeed")
            })
            .collect::<Vec<_>>();
        let chunk_rows = rows
            .chunks(chunk_size)
            .map(|chunk| chunk.len())
            .collect::<Vec<_>>();
        debug_assert_eq!(chunks.len(), chunk_rows.len());
        let payload_len = chunks.iter().map(Vec::len).sum::<usize>();
        let mut out = Vec::with_capacity(
            CHUNKED_ACTION_HEADER_LEN + chunks.len() * (4 + 8) + payload_len,
        );
        out.extend_from_slice(CHUNKED_ACTION_MAGIC);
        out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        for (&row_count, bytes) in chunk_rows.iter().zip(&chunks) {
            out.extend_from_slice(&(row_count as u32).to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        }
        for bytes in chunks {
            out.extend_from_slice(&bytes);
        }
        out
    }

    fn decode_simple_action_chunk(
        input: &[u8],
        expected_rows: usize,
    ) -> Result<Vec<ActionRow>, String> {
        let mut pos = 0usize;
        let mut rows = Vec::with_capacity(expected_rows);
        for _ in 0..expected_rows {
            let entry_count = take_fixed_u32(input, &mut pos, "simple GLR action")? as usize;
            let mut entries = Vec::with_capacity(entry_count);
            for _ in 0..entry_count {
                let terminal = take_fixed_u32(input, &mut pos, "simple GLR action terminal")?;
                let action = decode_simple_action(input, &mut pos)?;
                entries.push((terminal, action));
            }
            let row = ActionRow::from_indexed_entries(entries);
            if row.len() != entry_count {
                return Err("duplicate terminal in simple GLR action row".to_owned());
            }
            rows.push(row);
        }
        if pos != input.len() {
            return Err("trailing bytes in simple GLR action chunk".to_owned());
        }
        Ok(rows)
    }

    fn decode_action_rows(input: &[u8]) -> Result<Vec<ActionRow>, String> {
        if input.starts_with(SIMPLE_ACTION_MAGIC) {
            if input.len() < CHUNKED_ACTION_HEADER_LEN {
                return Err("truncated simple GLR action header".to_owned());
            }
            let row_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
            let chunk_count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
            if chunk_count == 0 || chunk_count > row_count.max(1) {
                return Err("invalid simple GLR action chunk count".to_owned());
            }
            let descriptor_bytes = chunk_count
                .checked_mul(12)
                .ok_or_else(|| "simple GLR action descriptor overflow".to_owned())?;
            let mut pos = CHUNKED_ACTION_HEADER_LEN;
            let payload_start = pos
                .checked_add(descriptor_bytes)
                .ok_or_else(|| "simple GLR action header overflow".to_owned())?;
            if payload_start > input.len() {
                return Err("truncated simple GLR action descriptors".to_owned());
            }
            let mut descriptors = Vec::<(usize, usize, usize)>::with_capacity(chunk_count);
            let mut payload_pos = payload_start;
            let mut described_rows = 0usize;
            for _ in 0..chunk_count {
                let rows = u32::from_le_bytes(input[pos..pos + 4].try_into().unwrap()) as usize;
                let bytes = usize::try_from(u64::from_le_bytes(
                    input[pos + 4..pos + 12].try_into().unwrap(),
                ))
                .map_err(|_| "simple GLR action length does not fit platform".to_owned())?;
                pos += 12;
                if rows == 0 {
                    return Err("empty chunk in simple GLR action section".to_owned());
                }
                described_rows = described_rows
                    .checked_add(rows)
                    .ok_or_else(|| "simple GLR action row count overflow".to_owned())?;
                let end = payload_pos
                    .checked_add(bytes)
                    .ok_or_else(|| "simple GLR action payload overflow".to_owned())?;
                if end > input.len() {
                    return Err("truncated simple GLR action payload".to_owned());
                }
                descriptors.push((rows, payload_pos, end));
                payload_pos = end;
            }
            if described_rows != row_count || payload_pos != input.len() {
                return Err("invalid simple GLR action row count or trailing bytes".to_owned());
            }
            let decoded = descriptors
                .par_iter()
                .map(|&(expected_rows, start, end)| {
                    decode_simple_action_chunk(&input[start..end], expected_rows)
                })
                .collect::<Result<Vec<_>, String>>()?;
            let mut rows = Vec::with_capacity(row_count);
            for mut chunk in decoded {
                rows.append(&mut chunk);
            }
            debug_assert_eq!(rows.len(), row_count);
            return Ok(rows);
        }
        if !input.starts_with(CHUNKED_ACTION_MAGIC) {
            return bincode::deserialize::<Vec<ActionRow>>(input).map_err(|err| err.to_string());
        }
        if input.len() < CHUNKED_ACTION_HEADER_LEN {
            return Err("truncated chunked GLR action header".to_owned());
        }
        let row_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
        let chunk_count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
        if chunk_count == 0 || chunk_count > row_count.max(1) {
            return Err("invalid chunked GLR action chunk count".to_owned());
        }
        let descriptor_bytes = chunk_count
            .checked_mul(12)
            .ok_or_else(|| "chunked GLR action descriptor overflow".to_owned())?;
        let mut pos = CHUNKED_ACTION_HEADER_LEN;
        let payload_start = pos
            .checked_add(descriptor_bytes)
            .ok_or_else(|| "chunked GLR action header overflow".to_owned())?;
        if payload_start > input.len() {
            return Err("truncated chunked GLR action descriptors".to_owned());
        }
        let mut descriptors = Vec::<(usize, usize, usize)>::with_capacity(chunk_count);
        let mut payload_pos = payload_start;
        let mut described_rows = 0usize;
        for _ in 0..chunk_count {
            let rows = u32::from_le_bytes(input[pos..pos + 4].try_into().unwrap()) as usize;
            let bytes = usize::try_from(u64::from_le_bytes(
                input[pos + 4..pos + 12].try_into().unwrap(),
            ))
            .map_err(|_| "chunked GLR action length does not fit platform".to_owned())?;
            pos += 12;
            if rows == 0 {
                return Err("empty chunk in chunked GLR action section".to_owned());
            }
            described_rows = described_rows
                .checked_add(rows)
                .ok_or_else(|| "chunked GLR action row count overflow".to_owned())?;
            let end = payload_pos
                .checked_add(bytes)
                .ok_or_else(|| "chunked GLR action payload overflow".to_owned())?;
            if end > input.len() {
                return Err("truncated chunked GLR action payload".to_owned());
            }
            descriptors.push((rows, payload_pos, end));
            payload_pos = end;
        }
        if described_rows != row_count || payload_pos != input.len() {
            return Err("invalid chunked GLR action row count or trailing bytes".to_owned());
        }
        let decoded = descriptors
            .par_iter()
            .map(|&(expected_rows, start, end)| {
                let rows = bincode::deserialize::<Vec<ActionRow>>(&input[start..end])
                    .map_err(|err| err.to_string())?;
                if rows.len() != expected_rows {
                    return Err("chunked GLR action row count mismatch".to_owned());
                }
                Ok(rows)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut rows = Vec::with_capacity(row_count);
        for mut chunk in decoded {
            rows.append(&mut chunk);
        }
        debug_assert_eq!(rows.len(), row_count);
        Ok(rows)
    }

    fn encode_meta(table: &GLRTable, source_rules: &[Rule]) -> Vec<u8> {
        if source_rules.len() < CHUNKED_RULES_MIN_ROWS {
            return bincode::serialize(&CompactTableMetaRef {
                num_states: table.num_states,
                num_terminals: table.num_terminals,
                num_rules: table.num_rules,
                rules: &source_rules,
                nonterminal_display_names: &table.nonterminal_display_names,
                construction: table.construction,
                admission_policy: table.admission_policy,
                forwarded_shifts: &table.forwarded_shifts,
                control_terminals: &table.control_terminals,
                skip_terminals: &table.skip_terminals,
                direct_regular_wide_frontiers: &table.direct_regular_wide_frontiers,
            })
            .expect("GLR metadata serialization should succeed");
        }
        let encode_rules = || {
            bincode::serialize(&source_rules).expect("GLR rules serialization should succeed")
        };
        let encode_rest = || {
            bincode::serialize(&CompactTableMetaNoRulesRef {
                num_states: table.num_states,
                num_terminals: table.num_terminals,
                num_rules: table.num_rules,
                first_rule: source_rules.first(),
                nonterminal_display_names: &table.nonterminal_display_names,
                construction: table.construction,
                admission_policy: table.admission_policy,
                forwarded_shifts: &table.forwarded_shifts,
                control_terminals: &table.control_terminals,
                skip_terminals: &table.skip_terminals,
                direct_regular_wide_frontiers: &table.direct_regular_wide_frontiers,
            })
            .expect("GLR metadata serialization should succeed")
        };
        // Deferred source rules are a wire-format choice, not a parallelism
        // choice. Even a single-threaded load should not rebuild thousands of
        // source Rule trees that runtime parsing never reads. Only the encoding
        // schedule depends on Rayon availability.
        let (rules, rest) = if rayon::current_num_threads() == 1 {
            (encode_rules(), encode_rest())
        } else {
            rayon::join(encode_rules, encode_rest)
        };
        let mut out = Vec::with_capacity(DEFERRED_META_HEADER_LEN + rules.len() + rest.len());
        out.extend_from_slice(DEFERRED_META_MAGIC);
        out.extend_from_slice(&(source_rules.len() as u32).to_le_bytes());
        out.extend_from_slice(&(rules.len() as u64).to_le_bytes());
        out.extend_from_slice(&(rest.len() as u64).to_le_bytes());
        out.extend_from_slice(&rules);
        out.extend_from_slice(&rest);
        out
    }

    #[derive(Debug, Clone)]
    pub enum DeferredRuleBytes {
        Owned(std::sync::Arc<[u8]>),
        Backed {
            backing: std::sync::Arc<Vec<u8>>,
            start: usize,
            len: usize,
        },
    }

    impl DeferredRuleBytes {
        pub fn as_slice(&self) -> &[u8] {
            match self {
                Self::Owned(bytes) => bytes,
                Self::Backed {
                    backing,
                    start,
                    len,
                } => &backing[*start..*start + *len],
            }
        }
    }

    fn decode_meta_deferred(
        input: &[u8],
        backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
    ) -> Result<(CompactTableMeta, Option<DeferredRuleBytes>), String> {
        if !input.starts_with(DEFERRED_META_MAGIC) {
            return bincode::deserialize::<CompactTableMeta>(input)
                .map(|meta| (meta, None))
                .map_err(|err| err.to_string());
        }
        if input.len() < DEFERRED_META_HEADER_LEN {
            return Err("truncated deferred GLR metadata header".to_owned());
        }
        let rule_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
        let rules_len = usize::try_from(u64::from_le_bytes(input[8..16].try_into().unwrap()))
            .map_err(|_| "deferred GLR rules length does not fit platform".to_owned())?;
        let rest_len = usize::try_from(u64::from_le_bytes(input[16..24].try_into().unwrap()))
            .map_err(|_| "deferred GLR metadata length does not fit platform".to_owned())?;
        let rules_end = DEFERRED_META_HEADER_LEN
            .checked_add(rules_len)
            .ok_or_else(|| "deferred GLR metadata rules overflow".to_owned())?;
        let end = rules_end
            .checked_add(rest_len)
            .ok_or_else(|| "deferred GLR metadata payload overflow".to_owned())?;
        if end != input.len() {
            return Err("invalid deferred GLR metadata lengths".to_owned());
        }
        let rules_bytes = &input[DEFERRED_META_HEADER_LEN..rules_end];
        let rest_bytes = &input[rules_end..end];
        let rest = bincode::deserialize::<CompactTableMetaNoRules>(rest_bytes)
            .map_err(|err| err.to_string())?;
        if rule_count != rest.num_rules as usize {
            return Err("deferred GLR metadata rule count mismatch".to_owned());
        }
        if (rule_count == 0) != rest.first_rule.is_none() {
            return Err("deferred GLR metadata first-rule mismatch".to_owned());
        }
        let rules = rest.first_rule.into_iter().collect();
        let deferred = if let Some((backing, meta_start)) = backing {
            let start = meta_start
                .checked_add(DEFERRED_META_HEADER_LEN)
                .ok_or_else(|| "deferred GLR rules backing offset overflow".to_owned())?;
            let end = start
                .checked_add(rules_len)
                .ok_or_else(|| "deferred GLR rules backing range overflow".to_owned())?;
            if backing.get(start..end) != Some(rules_bytes) {
                return Err("deferred GLR rules do not match artifact backing".to_owned());
            }
            DeferredRuleBytes::Backed {
                backing,
                start,
                len: rules_len,
            }
        } else {
            DeferredRuleBytes::Owned(std::sync::Arc::from(rules_bytes))
        };
        Ok((
            CompactTableMeta {
                num_states: rest.num_states,
                num_terminals: rest.num_terminals,
                num_rules: rest.num_rules,
                rules,
                nonterminal_display_names: rest.nonterminal_display_names,
                construction: rest.construction,
                admission_policy: rest.admission_policy,
                forwarded_shifts: rest.forwarded_shifts,
                control_terminals: rest.control_terminals,
                skip_terminals: rest.skip_terminals,
                direct_regular_wide_frontiers: rest.direct_regular_wide_frontiers,
            },
            Some(deferred),
        ))
    }

    fn assemble_table(
        action: Vec<ActionRow>,
        goto: Vec<GotoRow>,
        advance: CompactAdvance,
        meta: CompactTableMeta,
        rules_deferred: bool,
    ) -> Result<GLRTable, String> {
        let row_count_valid = |len: usize| len == 0 || len == meta.num_states as usize;
        if !row_count_valid(action.len()) || !row_count_valid(goto.len()) {
            return Err(format!(
                "compact GLR table row count does not match num_states: action={} goto={} num_states={}",
                action.len(),
                goto.len(),
                meta.num_states,
            ));
        }
        let rules_valid = if rules_deferred {
            meta.rules.len() == usize::from(meta.num_rules != 0)
        } else {
            meta.rules.len() == meta.num_rules as usize
        };
        if !rules_valid {
            return Err("compact GLR table rule count does not match num_rules".to_owned());
        }
        let advance = advance.into_rows(meta.num_terminals)?;
        if !advance.is_empty() && advance.len() != meta.num_states as usize {
            return Err("compact GLR advance row count does not match num_states".to_owned());
        }
        Ok(GLRTable {
            action,
            goto,
            num_states: meta.num_states,
            num_terminals: meta.num_terminals,
            num_rules: meta.num_rules,
            rules: meta.rules,
            nonterminal_display_names: meta.nonterminal_display_names,
            construction: meta.construction,
            admission_policy: meta.admission_policy,
            advance,
            unconditional_advance: Vec::new(),
            forwarded_shifts: meta.forwarded_shifts,
            control_terminals: meta.control_terminals,
            skip_terminals: meta.skip_terminals,
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: meta.direct_regular_wide_frontiers,
        })
    }

    pub struct DecodedCompactTable {
        pub table: GLRTable,
        pub deferred_rules: Option<DeferredRuleBytes>,
    }

    fn placeholder() -> GLRTable {
        GLRTable {
            action: Vec::new(),
            goto: Vec::new(),
            num_states: 0,
            num_terminals: 0,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::default(),
            admission_policy: AdmissionPolicy::default(),
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: BTreeSet::new(),
            skip_terminals: BTreeSet::new(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        }
    }

    pub fn serialize<S>(table: &GLRTable, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if external_serde_enabled() {
            return 0u8.serialize(serializer);
        }
        table.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<GLRTable, D::Error>
    where
        D: Deserializer<'de>,
    {
        if external_serde_enabled() {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 {
                return Err(serde::de::Error::custom("invalid external GLR table placeholder"));
            }
            return Ok(placeholder());
        }
        GLRTable::deserialize(deserializer)
    }

    pub fn to_compact_bytes(table: &GLRTable) -> Vec<u8> {
        to_compact_bytes_impl(table, &table.rules)
    }

    /// Serialize immutable runtime rows with separately retained source rules.
    /// A loaded runtime table may carry only its augmented rule, while its
    /// complete source rules are stored in an independently decoded section.
    pub fn to_compact_bytes_with_rules(table: &GLRTable, rules: &[Rule]) -> Vec<u8> {
        assert_eq!(rules.len(), table.num_rules as usize, "retained GLR rule count");
        assert_eq!(rules.first(), table.rules.first(), "retained augmented GLR rule");
        to_compact_bytes_impl(table, rules)
    }

    fn to_compact_bytes_impl(table: &GLRTable, rules: &[Rule]) -> Vec<u8> {
        let encode_action = || encode_action_rows(&table.action);
        let encode_goto = || {
            bincode::serialize(&table.goto).expect("GLR goto serialization should succeed")
        };
        let encode_advance = || {
            let compact = CompactAdvance::from_rows(&table.advance);
            bincode::serialize(&compact).expect("GLR advance serialization should succeed")
        };
        let encode_meta = || encode_meta(table, rules);
        // Scheduling four tiny bincode jobs costs more than the work for the
        // ordinary schema tables. Keep large tables parallel, but let compact
        // tables stay on one worker and in cache.
        let (action, goto, advance, meta) = if table.num_states < 1_024
            || rayon::current_num_threads() == 1
        {
            (encode_action(), encode_goto(), encode_advance(), encode_meta())
        } else {
            let ((action, goto), (advance, meta)) = rayon::join(
                || rayon::join(encode_action, encode_goto),
                || rayon::join(encode_advance, encode_meta),
            );
            (action, goto, advance, meta)
        };
        let mut out = Vec::with_capacity(
            PARALLEL_TABLE_HEADER_LEN + action.len() + goto.len() + advance.len() + meta.len(),
        );
        out.extend_from_slice(PARALLEL_TABLE_MAGIC);
        for len in [action.len(), goto.len(), advance.len(), meta.len()] {
            out.extend_from_slice(&(len as u64).to_le_bytes());
        }
        out.extend_from_slice(&action);
        out.extend_from_slice(&goto);
        out.extend_from_slice(&advance);
        out.extend_from_slice(&meta);
        out
    }

    #[cfg(test)]
    #[test]
    fn retained_rule_writer_preserves_deferred_runtime_table() {
        let mut table = placeholder();
        table.num_rules = 2048;
        table.rules = vec![Rule { lhs: 0, rhs: Vec::new() }];
        let rules = vec![table.rules[0].clone(); table.num_rules as usize];
        let bytes = to_compact_bytes_with_rules(&table, &rules);
        let deferred = from_compact_bytes_deferred(&bytes).unwrap();
        assert!(deferred.deferred_rules.is_some());
        assert_eq!(deferred.table.rules.len(), 1);
        assert_eq!(deferred.table.num_rules, 2048);
        let full = from_compact_bytes(&bytes).unwrap();
        assert_eq!(full.rules, rules);
        assert_eq!(to_compact_bytes(&full), bytes);
    }

    fn from_compact_bytes_deferred_impl(
        input: &[u8],
        backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
    ) -> Result<DecodedCompactTable, String> {
        if input.starts_with(PARALLEL_TABLE_MAGIC)
            || input.starts_with(PARALLEL_TABLE_MAGIC_V2)
        {
            let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
            if input.len() < PARALLEL_TABLE_HEADER_LEN {
                return Err("truncated parallel GLR table header".to_owned());
            }
            let mut pos = PARALLEL_TABLE_MAGIC.len();
            let mut lengths = [0usize; 4];
            for len in &mut lengths {
                let end = pos + 8;
                let encoded = u64::from_le_bytes(
                    input[pos..end]
                        .try_into()
                        .expect("parallel GLR table length has fixed width"),
                );
                *len = usize::try_from(encoded)
                    .map_err(|_| "parallel GLR table section length does not fit platform".to_owned())?;
                pos = end;
            }
            let expected = lengths.iter().try_fold(PARALLEL_TABLE_HEADER_LEN, |total, &len| {
                total.checked_add(len)
            }).ok_or_else(|| "parallel GLR table section lengths overflow".to_owned())?;
            if expected != input.len() {
                return Err("invalid parallel GLR table section lengths".to_owned());
            }
            let action = &input[pos..pos + lengths[0]];
            pos += lengths[0];
            let goto = &input[pos..pos + lengths[1]];
            pos += lengths[1];
            let advance = &input[pos..pos + lengths[2]];
            pos += lengths[2];
            let meta = &input[pos..pos + lengths[3]];
            let meta_offset = pos;

            let decode_action = || {
                let started = profile.then(std::time::Instant::now);
                let result = decode_action_rows(action);
                if let Some(started) = started {
                    eprintln!(
                        "[glrmask/profile][table_decode] name=action ms={:.3} bytes={}",
                        started.elapsed().as_secs_f64() * 1000.0,
                        action.len(),
                    );
                }
                result
            };
            let decode_goto = || {
                let started = profile.then(std::time::Instant::now);
                let result = bincode::deserialize::<Vec<GotoRow>>(goto)
                    .map_err(|err| err.to_string());
                if let Some(started) = started {
                    eprintln!(
                        "[glrmask/profile][table_decode] name=goto ms={:.3} bytes={}",
                        started.elapsed().as_secs_f64() * 1000.0,
                        goto.len(),
                    );
                }
                result
            };
            let decode_advance = || {
                let started = profile.then(std::time::Instant::now);
                let result = bincode::deserialize::<CompactAdvance>(advance)
                    .map_err(|err| err.to_string());
                if let Some(started) = started {
                    eprintln!(
                        "[glrmask/profile][table_decode] name=advance ms={:.3} bytes={}",
                        started.elapsed().as_secs_f64() * 1000.0,
                        advance.len(),
                    );
                }
                result
            };
            let decode_meta = || {
                let started = profile.then(std::time::Instant::now);
                let meta_backing = backing.as_ref().map(|(backing, table_start)| {
                    (
                        std::sync::Arc::clone(backing),
                        table_start + meta_offset,
                    )
                });
                let result = decode_meta_deferred(meta, meta_backing);
                let elapsed_ms = started.map(|started| started.elapsed().as_secs_f64() * 1000.0);
                if profile && let Ok((decoded, deferred)) = &result {
                    eprintln!(
                        "[glrmask/profile][table_meta_shape] rules={} deferred_rule_bytes={} names={} forwarded={} controls={} skips={} wide={} name_bytes={} forwarded_bytes={} control_bytes={} skip_bytes={} wide_bytes={}",
                        decoded.num_rules,
                        deferred.as_ref().map_or(0, |bytes| bytes.as_slice().len()),
                        decoded.nonterminal_display_names.len(),
                        decoded.forwarded_shifts.len(),
                        decoded.control_terminals.len(),
                        decoded.skip_terminals.len(),
                        decoded.direct_regular_wide_frontiers.len(),
                        bincode::serialized_size(&decoded.nonterminal_display_names).unwrap_or(0),
                        bincode::serialized_size(&decoded.forwarded_shifts).unwrap_or(0),
                        bincode::serialized_size(&decoded.control_terminals).unwrap_or(0),
                        bincode::serialized_size(&decoded.skip_terminals).unwrap_or(0),
                        bincode::serialized_size(&decoded.direct_regular_wide_frontiers).unwrap_or(0),
                    );
                }
                if let Some(elapsed_ms) = elapsed_ms {
                    eprintln!(
                        "[glrmask/profile][table_decode] name=meta ms={:.3} bytes={}",
                        elapsed_ms,
                        meta.len(),
                    );
                }
                result
            };
            let (action, goto, advance, meta) = if input.len() < 128 * 1024
                || rayon::current_num_threads() == 1
            {
                (decode_action(), decode_goto(), decode_advance(), decode_meta())
            } else {
                let ((action, goto), (advance, meta)) = rayon::join(
                    || rayon::join(decode_action, decode_goto),
                    || rayon::join(decode_advance, decode_meta),
                );
                (action, goto, advance, meta)
            };
            let (meta, deferred_rules) = meta?;
            let table = assemble_table(
                action?,
                goto?,
                advance?,
                meta,
                deferred_rules.is_some(),
            )?;
            return Ok(DecodedCompactTable {
                table,
                deferred_rules,
            });
        }

        let artifact: CompactTable = bincode::deserialize(input).map_err(|err| err.to_string())?;
        let table = assemble_table(
            artifact.action,
            artifact.goto,
            artifact.advance,
            CompactTableMeta {
                num_states: artifact.num_states,
                num_terminals: artifact.num_terminals,
                num_rules: artifact.num_rules,
                rules: artifact.rules,
                nonterminal_display_names: artifact.nonterminal_display_names,
                construction: artifact.construction,
                admission_policy: artifact.admission_policy,
                forwarded_shifts: artifact.forwarded_shifts,
                control_terminals: artifact.control_terminals,
                skip_terminals: artifact.skip_terminals,
                direct_regular_wide_frontiers: artifact.direct_regular_wide_frontiers,
            },
            false,
        )?;
        Ok(DecodedCompactTable {
            table,
            deferred_rules: None,
        })
    }

    pub fn from_compact_bytes_deferred(input: &[u8]) -> Result<DecodedCompactTable, String> {
        from_compact_bytes_deferred_impl(input, None)
    }

    pub fn from_compact_bytes_deferred_backed(
        input: &[u8],
        backing: std::sync::Arc<Vec<u8>>,
        section_start: usize,
    ) -> Result<DecodedCompactTable, String> {
        let section_end = section_start
            .checked_add(input.len())
            .ok_or_else(|| "GLR table backing range overflow".to_owned())?;
        let backed = backing
            .get(section_start..section_end)
            .ok_or_else(|| "GLR table section is outside artifact backing".to_owned())?;
        if backed.as_ptr() != input.as_ptr() || backed.len() != input.len() {
            return Err("GLR table section does not match artifact backing".to_owned());
        }
        from_compact_bytes_deferred_impl(input, Some((backing, section_start)))
    }

    pub fn from_compact_bytes(input: &[u8]) -> Result<GLRTable, String> {
        let mut decoded = from_compact_bytes_deferred(input)?;
        if let Some(rules) = decoded.deferred_rules.take() {
            let materialized = bincode::deserialize::<Vec<Rule>>(rules.as_slice())
                .map_err(|err| err.to_string())?;
            if materialized.len() != decoded.table.num_rules as usize {
                return Err("deferred GLR rule count does not match num_rules".to_owned());
            }
            if materialized.first() != decoded.table.rules.first() {
                return Err("deferred GLR augmented-start rule mismatch".to_owned());
            }
            decoded.table.rules = materialized;
        }
        Ok(decoded.table)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use super::super::row::SparseRow;

        #[test]
        fn simple_action_wire_roundtrips_inline_and_large_rows() {
            let inline = ActionRow::from_iter([
                (1, Action::Shift(11, false)),
                (2, Action::Shift(12, true)),
                (3, Action::Reduce(7, 4)),
                (4, Action::Accept),
                (5, Action::Skip),
            ]);
            let large = ActionRow::from_iter(
                (0..9).map(|terminal| (terminal, Action::Shift(100 + terminal, false))),
            );
            let rows = vec![inline, large];

            let bytes = encode_simple_action_rows(&rows).expect("rows are simple");
            assert!(bytes.starts_with(SIMPLE_ACTION_MAGIC));
            let decoded = decode_action_rows(&bytes).expect("simple action wire decodes");
            assert_eq!(decoded.len(), rows.len());

            for (before, after) in rows.iter().zip(&decoded) {
                assert_eq!(before.len(), after.len());
                for (terminal, action) in before.iter() {
                    assert_eq!(after.get(&terminal), Some(action));
                }
            }

            assert!(matches!(
                decoded[0],
                ActionRow::Sparse(SparseRow::Inline(_))
            ));
            assert!(matches!(
                decoded[1],
                ActionRow::Indexed { .. }
            ));

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("one-thread test pool");
            let reencoded = pool.install(|| encode_action_rows(&decoded));
            assert!(
                reencoded.starts_with(SIMPLE_ACTION_MAGIC),
                "runtime Indexed rows must re-emit the stable GSA1 wire even with one worker",
            );
            let redecoded = decode_action_rows(&reencoded).expect("re-encoded GSA1 decodes");
            assert_eq!(redecoded, decoded);
        }

        #[test]
        fn simple_action_wire_falls_back_for_default_rows() {
            let rows = vec![ActionRow::Default {
                default: Action::Accept,
                exceptions: SparseRow::default(),
                num_terminals: 4,
            }];
            assert!(encode_simple_action_rows(&rows).is_none());
        }

        #[test]
        fn simple_action_wire_rejects_trailing_or_invalid_action_bytes() {
            let rows = vec![ActionRow::from_iter([(3, Action::Accept)])];
            let bytes = encode_simple_action_rows(&rows).expect("rows are simple");

            let mut trailing = bytes.clone();
            trailing.push(0);
            assert!(decode_action_rows(&trailing).is_err());

            // One chunk: 12-byte header + 12-byte descriptor.  Its body starts
            // with row entry-count (4), terminal id (4), then the action tag.
            let mut invalid_tag = bytes;
            let tag_offset = CHUNKED_ACTION_HEADER_LEN + 12 + 4 + 4;
            invalid_tag[tag_offset] = 0xff;
            assert!(decode_action_rows(&invalid_tag).is_err());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DirectRegularWideFrontierDescriptor {
    pub source_state: u32,
    pub terminal: TerminalID,
    pub target_states: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TableAmbiguityKind {
    Split,
    StackShifts,
    GuardedStackShifts,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableAmbiguity {
    pub state: u32,
    pub terminal: TerminalID,
    pub kind: TableAmbiguityKind,
    pub alternatives: usize,
}

fn guarded_stack_shift_constraints(
    guards: &[StackShiftGuard],
) -> Option<BTreeMap<u32, BTreeSet<u32>>> {
    let mut constraints: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for guard in guards {
        let states: BTreeSet<u32> = guard.states.iter().copied().collect();
        if states.is_empty() {
            return None;
        }
        if let Some(existing) = constraints.get_mut(&guard.pop) {
            existing.retain(|state| states.contains(state));
            if existing.is_empty() {
                return None;
            }
        } else {
            constraints.insert(guard.pop, states);
        }
    }
    Some(constraints)
}

fn guarded_stack_shifts_overlap(left: &GuardedStackShift, right: &GuardedStackShift) -> bool {
    let Some(left_constraints) = guarded_stack_shift_constraints(&left.guards) else {
        return false;
    };
    let Some(right_constraints) = guarded_stack_shift_constraints(&right.guards) else {
        return false;
    };

    for (pop, left_states) in &left_constraints {
        if let Some(right_states) = right_constraints.get(pop)
            && left_states.is_disjoint(right_states)
        {
            return false;
        }
    }

    true
}

fn guarded_stack_shifts_are_ambiguous(shifts: &[GuardedStackShift]) -> bool {
    for first in 0..shifts.len() {
        for second in (first + 1)..shifts.len() {
            if guarded_stack_shifts_overlap(&shifts[first], &shifts[second]) {
                return true;
            }
        }
    }
    false
}

fn action_ambiguity(action: &Action) -> Option<(TableAmbiguityKind, usize)> {
    match action {
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            let alternatives = usize::from(shift.is_some()) + reduces.len() + usize::from(*accept);
            (alternatives > 1).then_some((TableAmbiguityKind::Split, alternatives))
        }
        Action::StackShifts(shifts) => {
            (shifts.len() > 1).then_some((TableAmbiguityKind::StackShifts, shifts.len()))
        }
        Action::GuardedStackShifts(shifts) => {
            (guarded_stack_shifts_are_ambiguous(shifts))
                .then_some((TableAmbiguityKind::GuardedStackShifts, shifts.len()))
        }
        _ => None,
    }
}

impl GLRTable {
    fn augmented_start_display_name(&self) -> Option<&str> {
        let augmented_start = self.rules.first()?.lhs;
        self.nonterminal_display_names
            .get(augmented_start as usize)
            .map(String::as_str)
    }

    fn augmented_start_display_name_mut(&mut self) -> Option<&mut String> {
        let augmented_start = self.rules.first()?.lhs;
        self.nonterminal_display_names
            .get_mut(augmented_start as usize)
    }

    /// Whether the source grammar's start symbol can derive epsilon when this
    /// compiled constraint is embedded as a subgrammar. Standalone generation
    /// intentionally does not finish without committing a token, so this bit
    /// is retained in augmented-start display metadata rather than execution
    /// rows. This preserves the existing serialized artifact shape.
    pub fn embedded_start_nullable(&self) -> bool {
        self.augmented_start_display_name()
            .is_some_and(|name| name.ends_with(EMBEDDED_NULLABLE_START_SUFFIX))
    }

    pub fn set_embedded_start_nullable(&mut self, nullable: bool) {
        let Some(name) = self.augmented_start_display_name_mut() else {
            return;
        };
        if let Some(base_len) = name.len().checked_sub(EMBEDDED_NULLABLE_START_SUFFIX.len())
            && name[base_len..] == *EMBEDDED_NULLABLE_START_SUFFIX
        {
            name.truncate(base_len);
        }
        if nullable {
            name.push_str(EMBEDDED_NULLABLE_START_SUFFIX);
        }
    }

    /// Model token IDs that were explicitly appended as grammar-level end
    /// tokens when this constraint was compiled. This private display-name
    /// metadata preserves the existing serialized table shape while allowing
    /// later compiled-constraint composition to distinguish end-token roles
    /// from an otherwise identical `@token(...)` placeholder terminal.
    pub fn embedded_end_token_ids(&self) -> Vec<u32> {
        let Some(mut name) = self.augmented_start_display_name() else {
            return Vec::new();
        };
        if let Some(base_len) = name.len().checked_sub(EMBEDDED_NULLABLE_START_SUFFIX.len())
            && name[base_len..] == *EMBEDDED_NULLABLE_START_SUFFIX
        {
            name = &name[..base_len];
        }
        let Some(marker) = name.rfind(EMBEDDED_END_TOKEN_IDS_PREFIX) else {
            return Vec::new();
        };
        name[marker + EMBEDDED_END_TOKEN_IDS_PREFIX.len()..]
            .split(',')
            .filter(|part| !part.is_empty())
            .filter_map(|part| part.parse::<u32>().ok())
            .collect()
    }

    pub fn set_embedded_end_token_ids(&mut self, token_ids: &[u32]) {
        let nullable = self.embedded_start_nullable();
        let Some(name) = self.augmented_start_display_name_mut() else {
            return;
        };
        if nullable {
            name.truncate(name.len() - EMBEDDED_NULLABLE_START_SUFFIX.len());
        }
        if let Some(marker) = name.rfind(EMBEDDED_END_TOKEN_IDS_PREFIX) {
            name.truncate(marker);
        }
        let mut token_ids = token_ids.to_vec();
        token_ids.sort_unstable();
        token_ids.dedup();
        if !token_ids.is_empty() {
            name.push_str(EMBEDDED_END_TOKEN_IDS_PREFIX);
            for (index, token_id) in token_ids.iter().enumerate() {
                if index != 0 {
                    name.push(',');
                }
                name.push_str(&token_id.to_string());
            }
        }
        if nullable {
            name.push_str(EMBEDDED_NULLABLE_START_SUFFIX);
        }
    }
    /// Minimal table metadata for a runtime that executes a retained direct-regular
    /// automaton instead of materialized LR action rows.
    pub fn direct_regular_runtime_stub(num_states: u32, num_terminals: u32) -> Self {
        Self {
            action: Vec::new(),
            goto: Vec::new(),
            num_states,
            num_terminals,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        }
    }



    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        build_table(grammar)
    }

    pub fn build_with_default_construction(
        grammar: &AnalyzedGrammar,
        default_construction: GlrTableConstruction,
    ) -> Self {
        build_table_with_default_construction(grammar, default_construction)
    }

    #[inline]
    fn terminal_bit(&self, terminal: TerminalID) -> Option<usize> {
        if terminal == EOF {
            Some(self.num_terminals as usize)
        } else if terminal < self.num_terminals {
            Some(terminal as usize)
        } else {
            None
        }
    }

    #[inline]
    fn has_advance_rows(&self) -> bool {
        self.advance.len() == self.num_states as usize
    }

    pub fn rebuild_advance_rows_from_actions(&mut self) {
        if rayon::current_num_threads() == 1 || self.action.len() < 128 {
            self.advance = action_presence_rows(&self.action, self.num_terminals);
        } else {
            use rayon::prelude::*;
            let num_terminals = self.num_terminals;
            self.advance = self
                .action
                .par_iter()
                .map(|row| action_presence_row(row, num_terminals))
                .collect();
        }
    }

    pub fn rebuild_unconditional_advance_rows(&mut self) {
        use rayon::prelude::*;

        let terminal_count = self.num_terminals as usize;
        let build_row = |row: &ActionRow| {
            let unconditional = |terminal: TerminalID, action: &Action| match action {
                Action::Shift(..) | Action::ReplaceShifts(_) | Action::Skip => true,
                Action::Split { shift, accept, .. } => {
                    shift.is_some() || (*accept && terminal == crate::glr::analysis::EOF)
                }
                Action::StackShifts(_)
                | Action::GuardedStackShifts(_)
                | Action::Reduce(..)
                | Action::Accept => false,
            };
            match row {
                ActionRow::Sparse(row) => {
                    let mut admitted = BitSet::new(terminal_count);
                    for (terminal, action) in row.iter() {
                        if (*terminal as usize) < terminal_count
                            && unconditional(*terminal, action)
                        {
                            admitted.set(*terminal as usize);
                        }
                    }
                    admitted
                }
                ActionRow::Indexed { .. } => {
                    let mut admitted = BitSet::new(terminal_count);
                    for (terminal, action) in row.iter() {
                        if (terminal as usize) < terminal_count && unconditional(terminal, action) {
                            admitted.set(terminal as usize);
                        }
                    }
                    admitted
                }
                ActionRow::Default {
                    default,
                    exceptions,
                    num_terminals,
                } if *num_terminals as usize == terminal_count => {
                    let default_unconditional = match default {
                        Action::Shift(..) | Action::ReplaceShifts(_) | Action::Skip => true,
                        Action::Split { shift, .. } => shift.is_some(),
                        Action::StackShifts(_)
                        | Action::GuardedStackShifts(_)
                        | Action::Reduce(..)
                        | Action::Accept => false,
                    };
                    let mut admitted = if default_unconditional {
                        BitSet::all(terminal_count)
                    } else {
                        BitSet::new(terminal_count)
                    };
                    for (terminal, action) in exceptions.iter() {
                        let is_unconditional = action
                            .as_ref()
                            .is_some_and(|action| unconditional(*terminal, action));
                        if is_unconditional {
                            admitted.set(*terminal as usize);
                        } else {
                            admitted.clear(*terminal as usize);
                        }
                    }
                    admitted
                }
                ActionRow::Default { .. } => {
                    // Composition can retain a default row over a component's
                    // original terminal domain inside a wider composed table.
                    // In that case the default applies only to the row's own
                    // domain; preserve the generic iterator semantics rather
                    // than extending the default to newly added terminals.
                    let mut admitted = BitSet::new(terminal_count);
                    for (terminal, action) in row.iter() {
                        if (terminal as usize) < terminal_count && unconditional(terminal, action) {
                            admitted.set(terminal as usize);
                        }
                    }
                    admitted
                }
            }
        };
        if rayon::current_num_threads() == 1 || self.action.len() < 128 {
            self.unconditional_advance = self.action.iter().map(build_row).collect();
        } else {
            self.unconditional_advance = self.action.par_iter().map(build_row).collect();
        }
    }

    #[inline]
    pub fn unconditional_advance_row(&self, state: u32) -> Option<&BitSet> {
        (self.unconditional_advance.len() == self.num_states as usize)
            .then(|| self.unconditional_advance.get(state as usize))
            .flatten()
    }

    pub fn rebuild_guarded_shift_index(&mut self) {
        let build_row = |row: &ActionRow| {
            let mut index_row = FxHashMap::default();

            for (terminal, action) in row {
                let Action::GuardedStackShifts(shifts) = action else {
                    continue;
                };

                let mut guard_pops = BTreeSet::new();
                let mut by_guard_key: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
                let mut guard_counts = vec![0u16; shifts.len()];
                let mut unguarded_indices = Vec::new();

                for (shift_index, shift) in shifts.iter().enumerate() {
                    if shift.guards.is_empty() {
                        unguarded_indices.push(shift_index as u32);
                        continue;
                    }

                    for guard in &shift.guards {
                        guard_pops.insert(guard.pop);
                        guard_counts[shift_index] += 1;
                        for &guard_state in &guard.states {
                            by_guard_key
                                .entry((guard.pop, guard_state))
                                .or_default()
                                .push(shift_index as u32);
                        }
                    }
                }

                index_row.insert(
                    terminal,
                    GuardedShiftCellIndex {
                        guard_pops: guard_pops.into_iter().collect::<Vec<_>>().into_boxed_slice(),
                        by_guard_key: by_guard_key
                            .into_iter()
                            .map(|(key, shift_indices)| (key, shift_indices.into_boxed_slice()))
                            .collect(),
                        guard_counts: guard_counts.into_boxed_slice(),
                        unguarded_indices: unguarded_indices.into_boxed_slice(),
                    },
                );
            }

            index_row
        };
        if rayon::current_num_threads() == 1 || self.action.len() < 128 {
            self.guarded_shift_index = self.action.iter().map(build_row).collect();
        } else {
            use rayon::prelude::*;
            self.guarded_shift_index = self.action.par_iter().map(build_row).collect();
        }
    }

    /// Whether any parser cell actually contains a guarded stack shift.
    ///
    /// Dynamic artifacts deliberately omit `guarded_shift_index` because it is
    /// derived. Most ordinary schema tables contain no guarded shifts at all;
    /// in that common case rebuilding an all-empty Vec<FxHashMap<..>> after
    /// every load is pure allocation work. A linear action scan is both much
    /// cheaper and sufficient to decide whether the index is needed.
    pub fn has_guarded_stack_shifts(&self) -> bool {
        self.action.iter().any(|row| {
            row.iter()
                .any(|(_, action)| matches!(action, Action::GuardedStackShifts(_)))
        })
    }

    #[inline]
    pub fn guarded_shift_index(
        &self,
        state: u32,
        terminal: TerminalID,
    ) -> Option<&GuardedShiftCellIndex> {
        self.guarded_shift_index
            .get(state as usize)
            .and_then(|row| row.get(&terminal))
    }

    #[inline]
    pub fn advance_row_allows(&self, state: u32, terminal: TerminalID) -> bool {
        if self.has_advance_rows() {
            let Some(bit) = self.terminal_bit(terminal) else {
                return false;
            };
            return self
                .advance
                .get(state as usize)
                .is_some_and(|row| row.contains(bit));
        }

        // Compatibility fallback for hand-built test tables and older serialized
        // artifacts that do not carry the side table. Newly compiled tables build
        // `advance` before guard-producing optimizations run.
        self.action(state, terminal).is_some()
    }

    #[inline]
    pub fn advance_row(&self, state: u32) -> Option<&BitSet> {
        self.has_advance_rows()
            .then(|| self.advance.get(state as usize))
            .flatten()
    }

    #[inline]
    pub fn advance_row_intersects(&self, state: u32, terminals: &BitSet) -> bool {
        if self.has_advance_rows()
            && let Some(row) = self.advance.get(state as usize)
        {
            return row
                .words()
                .iter()
                .zip(terminals.words())
                .any(|(left, right)| (*left & *right) != 0);
        }

        self.action.get(state as usize).is_some_and(|actions| {
            actions.keys().any(|terminal| {
                self.terminal_bit(terminal)
                    .is_some_and(|bit| terminals.contains(bit))
            })
        })
    }

    pub fn compress_default_action_rows(&mut self) {
        use rayon::prelude::*;
        let num_terminals = self.num_terminals;
        if rayon::current_num_threads() == 1 || self.action.len() < 128 {
            for row in &mut self.action {
                row.compress_default(num_terminals);
            }
        } else {
            self.action
                .par_iter_mut()
                .for_each(|row| row.compress_default(num_terminals));
        }
    }

    #[inline]
    pub fn action(&self, state: u32, terminal: TerminalID) -> Option<&Action> {
        self.action
            .get(state as usize)
            .and_then(|by_terminal| by_terminal.get(&terminal))
    }

    pub fn ambiguous_actions(&self) -> Vec<TableAmbiguity> {
        let mut ambiguities = Vec::new();
        for (state, row) in self.action.iter().enumerate() {
            for (terminal, action) in row {
                if let Some((kind, alternatives)) = action_ambiguity(action) {
                    ambiguities.push(TableAmbiguity {
                        state: state as u32,
                        terminal,
                        kind,
                        alternatives,
                    });
                }
            }
        }
        ambiguities
    }

    pub fn has_ambiguity(&self) -> bool {
        self.action.iter().enumerate().any(|(_, row)| {
            row.into_iter()
                .any(|(_, action)| action_ambiguity(action).is_some())
        })
    }

    #[inline]
    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<(u32, bool)> {
        self.goto
            .get(state as usize)
            .and_then(|by_nt| by_nt.get(&nt).copied())
    }

    pub(super) fn validate_structure(&self, context: &str) {
        let expected_len = self.num_states as usize;
        if self.action.len() != expected_len {
            panic!(
                "{context}: action row count {} does not match num_states {}",
                self.action.len(),
                self.num_states,
            );
        }
        if self.goto.len() != expected_len {
            panic!(
                "{context}: goto row count {} does not match num_states {}",
                self.goto.len(),
                self.num_states,
            );
        }
        if !self.advance.is_empty() && self.advance.len() != expected_len {
            panic!(
                "{context}: advance row count {} does not match num_states {}",
                self.advance.len(),
                self.num_states,
            );
        }

        let validate_target = |source_state: u32,
                               label_kind: &str,
                               label_value: u32,
                               path: &str,
                               target: u32| {
            if target >= self.num_states {
                panic!(
                    "{context}: state {} {} {} has invalid {} target {} >= num_states {}",
                    source_state,
                    label_kind,
                    label_value,
                    path,
                    target,
                    self.num_states,
                );
            }
        };

        for (source_state, row) in self.action.iter().enumerate() {
            let source_state = source_state as u32;
            for (terminal, action) in row.iter() {
                match action {
                    Action::Shift(target, _) => {
                        validate_target(source_state, "terminal", terminal, "Action::Shift", *target);
                    }
                    Action::ReplaceShifts(targets) => {
                        for (target_idx, &target) in targets.iter().enumerate() {
                            validate_target(
                                source_state,
                                "terminal",
                                terminal,
                                &format!("Action::ReplaceShifts[{target_idx}]"),
                                target,
                            );
                        }
                    }
                    Action::StackShifts(shifts) => {
                        for (shift_idx, shift) in shifts.iter().enumerate() {
                            for (push_idx, &target) in shift.pushes.iter().enumerate() {
                                validate_target(
                                    source_state,
                                    "terminal",
                                    terminal,
                                    &format!("Action::StackShifts[{shift_idx}].pushes[{push_idx}]"),
                                    target,
                                );
                            }
                        }
                    }
                    Action::GuardedStackShifts(shifts) => {
                        for (shift_idx, shift) in shifts.iter().enumerate() {
                            for (guard_idx, guard) in shift.guards.iter().enumerate() {
                                for (state_idx, &target) in guard.states.iter().enumerate() {
                                    validate_target(
                                        source_state,
                                        "terminal",
                                        terminal,
                                        &format!("Action::GuardedStackShifts[{shift_idx}].guards[{guard_idx}].states[{state_idx}]"),
                                        target,
                                    );
                                }
                            }
                            for (push_idx, &target) in shift.pushes.iter().enumerate() {
                                validate_target(
                                    source_state,
                                    "terminal",
                                    terminal,
                                    &format!("Action::GuardedStackShifts[{shift_idx}].pushes[{push_idx}]"),
                                    target,
                                );
                            }
                        }
                    }
                    Action::Split { shift: Some((target, _)), .. } => {
                        validate_target(
                            source_state,
                            "terminal",
                            terminal,
                            "Action::Split.shift",
                            *target,
                        );
                    }
                    Action::Split { shift: None, .. }
                    | Action::Reduce(..)
                    | Action::Accept
                    | Action::Skip => {}
                }
            }
        }

        for (source_state, row) in self.goto.iter().enumerate() {
            let source_state = source_state as u32;
            for (&nonterminal, &(target, _)) in row.iter() {
                validate_target(source_state, "nonterminal", nonterminal, "goto", target);
            }
        }

        for &(state, terminal) in &self.forwarded_shifts {
            if state >= self.num_states {
                panic!(
                    "{context}: forwarded_shifts has invalid state {} for terminal {} >= num_states {}",
                    state,
                    terminal,
                    self.num_states,
                );
            }
        }
    }

    #[inline]
    pub fn nonterminal_display_name(&self, nt: NonterminalID) -> Option<&str> {
        self.nonterminal_display_names
            .get(nt as usize)
            .map(String::as_str)
    }
}

fn action_presence_rows(action: &[ActionRow], num_terminals: u32) -> Vec<BitSet> {
    let mut rows = Vec::with_capacity(action.len());
    for action_row in action {
        rows.push(action_presence_row(action_row, num_terminals));
    }
    rows
}

fn action_presence_row(action_row: &ActionRow, num_terminals: u32) -> BitSet {
    let mut row = BitSet::new(num_terminals as usize + 1);
    for terminal in action_row.keys() {
        let bit = if terminal == EOF {
            num_terminals as usize
        } else if terminal < num_terminals {
            terminal as usize
        } else {
            continue;
        };
        row.set(bit);
    }
    row
}

impl GLRTable {
    pub fn extend_advance_rows_from_actions(&mut self) {
        if self.advance.is_empty() {
            return;
        }

        for action_row in self.action.iter().skip(self.advance.len()) {
            self.advance
                .push(action_presence_row(action_row, self.num_terminals));
        }
    }
}

pub mod testing {
    use super::row::{ActionRow, GotoRow};
    use super::{Action, AdmissionPolicy, GlrTableConstruction, GLRTable};
    use crate::grammar::flat::{NonterminalID, TerminalID};

    pub fn build_test_table(
        num_states: u32,
        num_terminals: u32,
        action_rows: &[&[(TerminalID, Action)]],
        goto_rows: &[&[(NonterminalID, (u32, bool))]],
    ) -> GLRTable {
        let action: Vec<_> = action_rows
            .iter()
            .map(|row| ActionRow::from_iter(row.iter().cloned()))
            .collect();
        let advance = super::action_presence_rows(&action, num_terminals);
        GLRTable {
            action,
            goto: goto_rows
                .iter()
                .map(|row| GotoRow::from_iter(row.iter().cloned()))
                .collect(),
            num_states,
            num_terminals,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance,
            unconditional_advance: Vec::new(),
            forwarded_shifts: Default::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        }
    }
}

#[cfg(test)]
mod ambiguity_tests {
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::grammar::ast::{lower, GrammarExpr, Quantifier, NamedGrammar, NamedRule};
    use crate::grammar::expr_nfa::ExprNfaBuilder;
    use crate::grammar::glrm::from_glrm;

    use super::testing::build_test_table;
    use super::{Action, GLRTable, GuardedStackShift, StackShift, TableAmbiguity, TableAmbiguityKind};

    fn build_table_from_glrm(glrm: &str) -> GLRTable {
        let named = from_glrm(glrm).unwrap();
        build_table_from_named_grammar(&named)
    }

    fn build_table_from_named_grammar(named: &NamedGrammar) -> GLRTable {
        let grammar = lower(&named).unwrap();
        let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
        GLRTable::build(&analyzed)
    }

    #[test]
    #[ignore]
    fn dump_tiny_subgrammar_splice_oracle() {
        fn dump(label: &str, source: &str) {
            let named = from_glrm(source).unwrap();
            let grammar = lower(&named).unwrap();
            let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
            let table = GLRTable::build(&analyzed);
            eprintln!("\n=== {label} ===");
            eprintln!(
                "terminals={:?}",
                analyzed.terminal_display_names,
            );
            eprintln!(
                "nonterminals={:?}",
                analyzed.nonterminal_display_names,
            );
            eprintln!("rules={:?}", table.rules);
            for state in 0..table.num_states as usize {
                let actions = table.action[state]
                    .iter()
                    .map(|(terminal, action)| (terminal, action.clone()))
                    .collect::<Vec<_>>();
                let gotos = table.goto[state]
                    .iter()
                    .map(|(nonterminal, target)| (*nonterminal, *target))
                    .collect::<Vec<_>>();
                eprintln!("state {state}: action={actions:?} goto={gotos:?}");
            }
        }

        dump(
            "child",
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
        );
        dump(
            "parent-pseudo",
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">";
            "#,
        );
        dump(
            "flattened",
            r#"
                start document;
                g inner ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "<" inner ">";
            "#,
        );
    }

    fn build_expr_nfa_optional_pair_suffix_grammar_with_value(
        value_symbol: GrammarExpr,
        extra_rules: Vec<NamedRule>,
    ) -> NamedGrammar {
        let mut builder = ExprNfaBuilder::new();

        let start = builder.start_state();
        let after_a = builder.add_state();
        let after_av = builder.add_state();
        let before_b_or_c = builder.add_state();
        let after_b = builder.add_state();
        let after_bv = builder.add_state();
        let before_c = builder.add_state();
        let after_c = builder.add_state();
        let after_cv = builder.add_state();
        let accept = builder.add_state();

        builder.add_epsilon(start, before_b_or_c);
        builder.add_transition(start, GrammarExpr::Literal(b"a".to_vec()), after_a);
        builder.add_transition(after_a, value_symbol.clone(), after_av);
        builder.add_transition(after_av, GrammarExpr::Literal(b",".to_vec()), before_b_or_c);

        builder.add_epsilon(before_b_or_c, before_c);
        builder.add_transition(before_b_or_c, GrammarExpr::Literal(b"b".to_vec()), after_b);
        builder.add_transition(after_b, value_symbol.clone(), after_bv);
        builder.add_transition(after_bv, GrammarExpr::Literal(b",".to_vec()), before_c);

        builder.add_transition(before_c, GrammarExpr::Literal(b"c".to_vec()), after_c);
        builder.add_transition(after_c, value_symbol, after_cv);
        builder.add_transition(after_cv, GrammarExpr::Literal(b"$".to_vec()), accept);
        builder.set_accepting(accept);

        let expr_nfa = builder.build().into_determinized_and_minimized();
        let mut rules = vec![NamedRule {
            name: "start".into(),
            expr: GrammarExpr::ExprNFA(Box::new(expr_nfa)),
            is_terminal: false,
            is_internal: false,
        }];
        rules.extend(extra_rules);
        NamedGrammar {
            rules,
            start: "start".into(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        }
    }

    fn glrm_recursive_array_colorpalette_minimal_stackshift_mre() -> &'static str {
        r#"
            start start;
            t JSON_ITEM_SEPARATOR ::= /(?:, )/;
            nt json_array ::= "[" (json_value (JSON_ITEM_SEPARATOR json_value)*)? "]";
            nt json_value ::= json_array;
            nt start ::= "{" "\"icons\": " "{" "\"ColorPalette\": " "{" ("\"k\": " json_value)* "}" "}" "}";
        "#
    }

    fn build_direct_recursive_array_colorpalette_minimal_grammar() -> NamedGrammar {
        let start_expr = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"{".to_vec()),
            GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"\"icons\": ".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"{".to_vec()),
                    GrammarExpr::SeparatedSequence {
                        items: vec![(
                            GrammarExpr::Sequence(vec![
                                GrammarExpr::Literal(b"\"ColorPalette\": ".to_vec()),
                                GrammarExpr::Sequence(vec![
                                    GrammarExpr::Literal(b"{".to_vec()),
                                    GrammarExpr::SeparatedSequence {
                                        items: vec![(
                                            GrammarExpr::Sequence(vec![
                                                GrammarExpr::Literal(b"\"k\": ".to_vec()),
                                                GrammarExpr::Ref("json_value".into()),
                                            ]),
                                            Some(Quantifier::ZeroPlus),
                                        )],
                                        separator: Box::new(GrammarExpr::Ref("JSON_ITEM_SEPARATOR".into())),
                                        allow_empty: true,
                                    },
                                    GrammarExpr::Literal(b"}".to_vec()),
                                ]),
                            ]),
                            Some(Quantifier::Optional),
                        )],
                        separator: Box::new(GrammarExpr::Ref("JSON_ITEM_SEPARATOR".into())),
                        allow_empty: true,
                    },
                    GrammarExpr::Literal(b"}".to_vec()),
                ]),
            ]),
            GrammarExpr::Literal(b"}".to_vec()),
        ]);

        NamedGrammar {
            rules: vec![
                NamedRule {
                    name: "JSON_ITEM_SEPARATOR".into(),
                    expr: GrammarExpr::RawRegex("(?:, )".into()),
                    is_terminal: true,
                    is_internal: false,
                },
                NamedRule {
                    name: "json_array".into(),
                    expr: GrammarExpr::Sequence(vec![
                        GrammarExpr::Literal(b"[".to_vec()),
                        GrammarExpr::Quantified(Box::new(GrammarExpr::Sequence(vec![
                            GrammarExpr::Ref("json_value".into()),
                            GrammarExpr::Quantified(Box::new(GrammarExpr::Sequence(vec![
                                GrammarExpr::Ref("JSON_ITEM_SEPARATOR".into()),
                                GrammarExpr::Ref("json_value".into()),
                            ])), Quantifier::ZeroPlus),
                        ])), Quantifier::Optional),
                        GrammarExpr::Literal(b"]".to_vec()),
                    ]),
                    is_terminal: false,
                    is_internal: false,
                },
                NamedRule {
                    name: "json_value".into(),
                    expr: GrammarExpr::Ref("json_array".into()),
                    is_terminal: false,
                    is_internal: false,
                },
                NamedRule {
                    name: "start".into(),
                    expr: start_expr,
                    is_terminal: false,
                    is_internal: false,
                },
            ],
            start: "start".into(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        }
    }

    fn assert_table_has_all_pop1_stack_shift_ambiguity(table: &GLRTable) {
        let ambiguities = table.ambiguous_actions();
        let Some(ambiguity) = ambiguities.iter().find(|ambiguity| {
            ambiguity.kind == TableAmbiguityKind::StackShifts
                && matches!(
                    table.action(ambiguity.state, ambiguity.terminal),
                    Some(Action::StackShifts(shifts))
                        if shifts.len() > 1 && shifts.iter().all(|shift| shift.pop == 1)
                )
        }) else {
            panic!("expected all-pop1 StackShifts ambiguity, but table had none: {:?}", table.ambiguous_actions());
        };

        let action = table
            .action(ambiguity.state, ambiguity.terminal)
            .expect("ambiguous action should still be present in table");
        match action {
            Action::StackShifts(shifts) => {
                assert!(shifts.iter().all(|shift| shift.pop == 1));
            }
            other => panic!("expected StackShifts action, found {:?}", other),
        }
    }

    fn assert_table_has_no_ambiguity(table: &GLRTable) {
        let ambiguities = table.ambiguous_actions();
        if let Some(TableAmbiguity {
            state,
            terminal,
            kind,
            alternatives,
        }) = ambiguities.first()
        {
            let action = table
                .action(*state, *terminal)
                .expect("ambiguous action should still be present in table");
            panic!(
                "expected unambiguous table, found ambiguity at state={} terminal={} kind={:?} alternatives={} action={:?}",
                state,
                terminal,
                kind,
                alternatives,
                action
            );
        }
    }

    fn assert_table_has_no_all_pop1_stack_shift_ambiguity(table: &GLRTable) {
        let ambiguities = table.ambiguous_actions();
        if let Some(ambiguity) = ambiguities.iter().find(|ambiguity| {
            ambiguity.kind == TableAmbiguityKind::StackShifts
                && matches!(
                    table.action(ambiguity.state, ambiguity.terminal),
                    Some(Action::StackShifts(shifts))
                        if shifts.len() > 1 && shifts.iter().all(|shift| shift.pop == 1)
                )
        }) {
            let action = table
                .action(ambiguity.state, ambiguity.terminal)
                .expect("ambiguous action should still be present in table");
            panic!(
                "expected unambiguous table, found all-pop1 StackShifts ambiguity at state={} terminal={} alternatives={} action={:?}",
                ambiguity.state,
                ambiguity.terminal,
                ambiguity.alternatives,
                action,
            );
        }
    }

    #[test]
    fn ambiguous_actions_reports_split_and_stack_shift_fanout() {
        let token = 0;
        let table = build_test_table(
            4,
            1,
            &[
                &[(token, Action::Shift(1, false))],
                &[(
                    token,
                    Action::Split {
                        shift: Some((2, false)),
                        reduces: vec![(0, 1)],
                        accept: false,
                    },
                )],
                &[(
                    token,
                    Action::StackShifts(vec![
                        StackShift {
                            pop: 1,
                            pushes: vec![2],
                        },
                        StackShift {
                            pop: 2,
                            pushes: vec![3],
                        },
                    ]),
                )],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![
                        GuardedStackShift {
                            guards: Vec::new(),
                            pop: 1,
                            pushes: vec![2],
                        },
                        GuardedStackShift {
                            guards: Vec::new(),
                            pop: 1,
                            pushes: vec![3],
                        },
                    ]),
                )],
            ],
            &[&[], &[], &[], &[]],
        );

        let ambiguities = table.ambiguous_actions();
        assert!(table.has_ambiguity());
        assert_eq!(ambiguities.len(), 3);
        assert_eq!(ambiguities[0].kind, TableAmbiguityKind::Split);
        assert_eq!(ambiguities[0].alternatives, 2);
        assert_eq!(ambiguities[1].kind, TableAmbiguityKind::StackShifts);
        assert_eq!(ambiguities[1].alternatives, 2);
        assert_eq!(ambiguities[2].kind, TableAmbiguityKind::GuardedStackShifts);
        assert_eq!(ambiguities[2].alternatives, 2);
    }

    #[test]
    fn guarded_stack_shifts_with_disjoint_guards_are_not_ambiguous() {
        let token = 0;
        let table = build_test_table(
            1,
            1,
            &[&[(
                token,
                Action::GuardedStackShifts(vec![
                    GuardedStackShift {
                        guards: vec![super::StackShiftGuard {
                            pop: 1,
                            states: vec![7],
                        }],
                        pop: 2,
                        pushes: vec![5],
                    },
                    GuardedStackShift {
                        guards: vec![super::StackShiftGuard {
                            pop: 1,
                            states: vec![10],
                        }],
                        pop: 2,
                        pushes: vec![8],
                    },
                ]),
            )]],
            &[&[]],
        );

        assert!(!table.has_ambiguity());
        assert!(table.ambiguous_actions().is_empty());
    }

    #[test]
    #[should_panic(expected = "validator test action target")]
    fn validate_structure_panics_on_invalid_action_target() {
        let token = 0;
        let table = build_test_table(
            2,
            1,
            &[&[(token, Action::Shift(2, false))], &[]],
            &[&[], &[]],
        );

        table.validate_structure("validator test action target");
    }

    #[test]
    fn expr_nfa_optional_pair_suffixes_with_value_ref_have_no_table_ambiguity() {
        let grammar = build_expr_nfa_optional_pair_suffix_grammar_with_value(
            GrammarExpr::Ref("value".into()),
            vec![NamedRule {
                name: "value".into(),
                expr: GrammarExpr::Literal(b"v".to_vec()),
                is_terminal: false,
                is_internal: false,
            }],
        );

        let table = build_table_from_named_grammar(&grammar);

        assert_table_has_no_ambiguity(&table);
    }

    #[test]
    fn glrm_recursive_array_colorpalette_minimal_does_not_reproduce_all_pop1_stack_shifts() {
        let table = build_table_from_glrm(glrm_recursive_array_colorpalette_minimal_stackshift_mre());

        assert_table_has_no_all_pop1_stack_shift_ambiguity(&table);
    }

    #[test]
    fn direct_recursive_array_colorpalette_minimal_avoids_all_pop1_stack_shifts() {
        let grammar = build_direct_recursive_array_colorpalette_minimal_grammar();
        let table = build_table_from_named_grammar(&grammar);

        assert_table_has_no_all_pop1_stack_shift_ambiguity(&table);
    }

}
