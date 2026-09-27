//! Current GLR-table wire codec, including chunked rows and deferred rules.

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

fn external_serde_enabled() -> bool {
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
struct CompactTableMetaRef<'a> {
    num_states: u32,
    num_terminals: u32,
    num_rules: u32,
    rules: &'a [Rule],
    nonterminal_display_names: &'a [String],
    embedded_start: &'a EmbeddedStart,
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
    embedded_start: EmbeddedStart,
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
    embedded_start: &'a EmbeddedStart,
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
    embedded_start: EmbeddedStart,
    construction: GlrTableConstruction,
    admission_policy: AdmissionPolicy,
    forwarded_shifts: FxHashSet<(u32, TerminalID)>,
    control_terminals: BTreeSet<TerminalID>,
    skip_terminals: BTreeSet<TerminalID>,
    direct_regular_wide_frontiers: Vec<DirectRegularWideFrontierDescriptor>,
}

const PARALLEL_TABLE_MAGIC: &[u8; 4] = b"GTC4";
const PARALLEL_TABLE_HEADER_LEN: usize = 4 + 4 * 8;
const CHUNKED_ACTION_MAGIC: &[u8; 4] = b"GTA1";
const SIMPLE_ACTION_MAGIC: &[u8; 4] = b"GSA1";
const CHUNKED_ACTION_HEADER_LEN: usize = 12;
const CHUNKED_ACTION_MIN_ROWS: usize = 8_192;
const SIMPLE_ACTION_MIN_ROWS: usize = 512;
const DEFERRED_META_MAGIC: &[u8; 4] = b"GTM3";
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
            embedded_start: &table.embedded_start,
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
            embedded_start: &table.embedded_start,
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
            embedded_start: rest.embedded_start,
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
        embedded_start: meta.embedded_start,
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
        embedded_start: Default::default(),
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
    if !input.starts_with(PARALLEL_TABLE_MAGIC) {
        return Err("unsupported GLR table artifact format".to_owned());
    }
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
    Ok(DecodedCompactTable {
        table,
        deferred_rules,
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
