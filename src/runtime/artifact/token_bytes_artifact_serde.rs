use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

const LEGACY_MAGIC: &[u8; 4] = b"TBP1";
const INDEXED_MAGIC: &[u8; 4] = b"TBP2";
const INDEXED_HEADER_LEN: usize = INDEXED_MAGIC.len() + 1 + 4;

thread_local! {
    static PACKED: Cell<bool> = const { Cell::new(false) };
    static DEFER_UNPACK: Cell<bool> = const { Cell::new(false) };
    static EXTERNAL: Cell<bool> = const { Cell::new(false) };
    static DEFERRED: RefCell<Option<Arc<PackedTokenBytes>>> = const { RefCell::new(None) };
}

pub(crate) fn set_packed(enabled: bool) -> bool {
    PACKED.with(|mode| mode.replace(enabled))
}

pub(crate) fn set_defer_unpack(enabled: bool) -> bool {
    DEFER_UNPACK.with(|mode| mode.replace(enabled))
}

pub(crate) fn set_external(enabled: bool) -> bool {
    EXTERNAL.with(|mode| mode.replace(enabled))
}

pub(crate) fn take_deferred() -> Option<Arc<PackedTokenBytes>> {
    DEFERRED.with(|slot| slot.borrow_mut().take())
}

#[derive(Debug)]
pub(crate) struct PackedTokenBytes {
    wire: Arc<Vec<u8>>,
    wire_start: usize,
    wire_len: usize,
    indexed: Option<PackedTokenBytesIndexed>,
    spans: Box<[(u32, u32)]>,
    sparse_ids: Option<Box<[u32]>>,
}

#[derive(Debug, Clone, Copy)]
struct PackedTokenBytesIndexed {
    count: usize,
    sparse_ids_start: Option<usize>,
    offsets_start: usize,
    data_start: usize,
}

impl PackedTokenBytes {
    pub(crate) fn from_runtime_entries(value: &BTreeMap<u32, Vec<u8>>) -> Result<Self, String> {
        // The indexed representation is useful runtime state in its own
        // right: token lookup/iteration reads it directly. Build it once
        // when a compiler-created Constraint is finalized rather than
        // rebuilding the same index inside every save().
        Self::parse(pack(value))
    }

    fn parse(wire: Vec<u8>) -> Result<Self, String> {
        let wire_len = wire.len();
        Self::parse_backed(Arc::new(wire), 0, wire_len)
    }

    pub(crate) fn parse_backed(
        wire: Arc<Vec<u8>>,
        wire_start: usize,
        wire_len: usize,
    ) -> Result<Self, String> {
        let wire_end = wire_start
            .checked_add(wire_len)
            .ok_or_else(|| "overflowing packed token-byte backing range".to_owned())?;
        let input = wire
            .get(wire_start..wire_end)
            .ok_or_else(|| "packed token-byte backing range is out of bounds".to_owned())?;
        if input.starts_with(INDEXED_MAGIC) {
            return Self::parse_indexed_backed(wire, wire_start, wire_len);
        }
        if !input.starts_with(LEGACY_MAGIC) {
            return Err("invalid packed token-byte header".to_owned());
        }
        let mut pos = LEGACY_MAGIC.len();
        let sparse = match input.get(pos).copied() {
            Some(0) => false,
            Some(1) => true,
            _ => return Err("invalid packed token-byte mode".to_owned()),
        };
        pos += 1;
        let count = take_var_u32(input, &mut pos)? as usize;
        let mut spans = Vec::with_capacity(count);
        let mut sparse_ids = sparse.then(|| Vec::with_capacity(count));
        let mut previous_end = 0u64;
        for dense_id in 0..count {
            let id = if sparse {
                let gap = take_var_u32(input, &mut pos)? as u64;
                let id = previous_end
                    .checked_add(gap)
                    .ok_or_else(|| "overflowing packed token id".to_owned())?;
                let id = u32::try_from(id)
                    .map_err(|_| "overflowing packed token id".to_owned())?;
                previous_end = id as u64 + 1;
                sparse_ids.as_mut().expect("sparse ids enabled").push(id);
                id
            } else {
                u32::try_from(dense_id)
                    .map_err(|_| "dense packed token id exceeds u32".to_owned())?
            };
            let _ = id;
            let len = take_var_u32(input, &mut pos)? as usize;
            let start = pos;
            let end = start
                .checked_add(len)
                .ok_or_else(|| "overflowing packed token-byte length".to_owned())?;
            if end > input.len() {
                return Err("truncated packed token bytes".to_owned());
            }
            spans.push((
                u32::try_from(start)
                    .map_err(|_| "packed token byte offset exceeds u32".to_owned())?,
                u32::try_from(len)
                    .map_err(|_| "packed token byte length exceeds u32".to_owned())?,
            ));
            pos = end;
        }
        if pos != input.len() {
            return Err("trailing bytes in packed token-byte vocabulary".to_owned());
        }
        Ok(Self {
            wire,
            wire_start,
            wire_len,
            indexed: None,
            spans: spans.into_boxed_slice(),
            sparse_ids: sparse_ids.map(Vec::into_boxed_slice),
        })
    }

    fn parse_indexed_backed(
        wire: Arc<Vec<u8>>,
        wire_start: usize,
        wire_len: usize,
    ) -> Result<Self, String> {
        let wire_end = wire_start
            .checked_add(wire_len)
            .ok_or_else(|| "overflowing indexed token-byte backing range".to_owned())?;
        let input = wire
            .get(wire_start..wire_end)
            .ok_or_else(|| "indexed token-byte backing range is out of bounds".to_owned())?;
        if input.len() < INDEXED_HEADER_LEN || !input.starts_with(INDEXED_MAGIC) {
            return Err("invalid indexed token-byte header".to_owned());
        }
        let sparse = match input[INDEXED_MAGIC.len()] {
            0 => false,
            1 => true,
            _ => return Err("invalid indexed token-byte mode".to_owned()),
        };
        let count_start = INDEXED_MAGIC.len() + 1;
        let count = u32::from_le_bytes(
            input[count_start..count_start + 4]
                .try_into()
                .expect("indexed token count has fixed width"),
        ) as usize;
        let sparse_ids_start = sparse.then_some(INDEXED_HEADER_LEN);
        let ids_bytes = if sparse {
            count
                .checked_mul(4)
                .ok_or_else(|| "indexed token-id table overflows".to_owned())?
        } else {
            0
        };
        let offsets_start = INDEXED_HEADER_LEN
            .checked_add(ids_bytes)
            .ok_or_else(|| "indexed token-byte offsets start overflows".to_owned())?;
        let offsets_bytes = count
            .checked_add(1)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| "indexed token-byte offset table overflows".to_owned())?;
        let data_start = offsets_start
            .checked_add(offsets_bytes)
            .ok_or_else(|| "indexed token-byte data start overflows".to_owned())?;
        if data_start > input.len() {
            return Err("truncated indexed token-byte tables".to_owned());
        }
        let first_offset = read_u32_at(input, offsets_start)? as usize;
        let final_offset = read_u32_at(input, offsets_start + count * 4)? as usize;
        if first_offset != 0 || final_offset != input.len() - data_start {
            return Err("invalid indexed token-byte offset bounds".to_owned());
        }
        Ok(Self {
            wire,
            wire_start,
            wire_len,
            indexed: Some(PackedTokenBytesIndexed {
                count,
                sparse_ids_start,
                offsets_start,
                data_start,
            }),
            spans: Box::new([]),
            sparse_ids: None,
        })
    }

    #[inline]
    pub(crate) fn wire(&self) -> &[u8] {
        &self.wire[self.wire_start..self.wire_start + self.wire_len]
    }

    pub(crate) fn whole_wire_arc(&self) -> Option<Arc<Vec<u8>>> {
        (self.wire_start == 0 && self.wire_len == self.wire.len())
            .then(|| Arc::clone(&self.wire))
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.indexed.map_or(self.spans.len(), |indexed| indexed.count)
    }

    #[inline]
    pub(crate) fn get(&self, token_id: u32) -> Option<&[u8]> {
        if let Some(indexed) = self.indexed {
            let index = match indexed.sparse_ids_start {
                None => usize::try_from(token_id)
                    .ok()
                    .filter(|&index| index < indexed.count)?,
                Some(_) => self.indexed_sparse_token_index(indexed, token_id)?,
            };
            return self.indexed_bytes_at(indexed, index);
        }
        let index = match &self.sparse_ids {
            None => usize::try_from(token_id).ok().filter(|&index| index < self.spans.len())?,
            Some(ids) => ids.binary_search(&token_id).ok()?,
        };
        let (start, len) = self.spans[index];
        let start = start as usize;
        self.wire().get(start..start + len as usize)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, &[u8])> + '_ {
        (0..self.len()).map(|index| {
            let token_id = self
                .token_id_at(index)
                .expect("validated packed token index should have an id");
            let bytes = self
                .bytes_at_index(index)
                .expect("validated packed token index should have bytes");
            (token_id, bytes)
        })
    }

    pub(crate) fn max_token_id(&self) -> Option<u32> {
        if self.indexed.is_some() {
            return self.len().checked_sub(1).and_then(|index| self.token_id_at(index));
        }
        match &self.sparse_ids {
            Some(ids) => ids.last().copied(),
            None => self.spans.len().checked_sub(1).map(|id| id as u32),
        }
    }

    pub(crate) fn materialize(&self) -> Arc<BTreeMap<u32, Vec<u8>>> {
        Arc::new(
            self.iter()
                .map(|(token_id, bytes)| (token_id, bytes.to_vec()))
                .collect(),
        )
    }

    fn token_id_at(&self, index: usize) -> Option<u32> {
        if let Some(indexed) = self.indexed {
            if index >= indexed.count {
                return None;
            }
            return indexed.sparse_ids_start.map_or_else(
                || u32::try_from(index).ok(),
                |start| read_u32_at(self.wire(), start + index * 4).ok(),
            );
        }
        self.sparse_ids
            .as_ref()
            .map_or_else(|| u32::try_from(index).ok(), |ids| ids.get(index).copied())
    }

    fn bytes_at_index(&self, index: usize) -> Option<&[u8]> {
        if let Some(indexed) = self.indexed {
            return self.indexed_bytes_at(indexed, index);
        }
        let &(start, len) = self.spans.get(index)?;
        let start = start as usize;
        self.wire().get(start..start + len as usize)
    }

    fn indexed_bytes_at(
        &self,
        indexed: PackedTokenBytesIndexed,
        index: usize,
    ) -> Option<&[u8]> {
        if index >= indexed.count {
            return None;
        }
        let wire = self.wire();
        let start = read_u32_at(wire, indexed.offsets_start + index * 4).ok()? as usize;
        let end = read_u32_at(wire, indexed.offsets_start + (index + 1) * 4).ok()? as usize;
        if start > end {
            return None;
        }
        wire.get(indexed.data_start + start..indexed.data_start + end)
    }

    fn indexed_sparse_token_index(
        &self,
        indexed: PackedTokenBytesIndexed,
        token_id: u32,
    ) -> Option<usize> {
        let start = indexed.sparse_ids_start?;
        let wire = self.wire();
        let mut lo = 0usize;
        let mut hi = indexed.count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let candidate = read_u32_at(wire, start + mid * 4).ok()?;
            match candidate.cmp(&token_id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }
}

#[inline]
fn read_u32_at(input: &[u8], pos: usize) -> Result<u32, String> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| "packed token-byte u32 offset overflows".to_owned())?;
    let bytes = input
        .get(pos..end)
        .ok_or_else(|| "truncated packed token-byte u32".to_owned())?;
    Ok(u32::from_le_bytes(
        bytes.try_into().expect("packed token u32 has fixed width"),
    ))
}

#[inline]
fn put_var_u32(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[inline]
fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
    let mut value = 0u32;
    let mut shift = 0u32;
    for _ in 0..5 {
        let byte = *input
            .get(*pos)
            .ok_or_else(|| "truncated packed token-byte varint".to_owned())?;
        *pos += 1;
        if shift == 28 && byte > 0x0f {
            return Err("overflowing packed token-byte varint".to_owned());
        }
        value |= ((byte & 0x7f) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    Err("overflowing packed token-byte varint".to_owned())
}

fn pack_legacy(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
    let dense = value
        .keys()
        .copied()
        .enumerate()
        .all(|(expected, actual)| actual as usize == expected);
    let mut out = Vec::new();
    out.extend_from_slice(LEGACY_MAGIC);
    out.push(u8::from(!dense));
    put_var_u32(
        &mut out,
        u32::try_from(value.len()).expect("token vocabulary should fit u32"),
    );
    let mut previous_end = 0u64;
    for (&id, bytes) in value {
        if !dense {
            let gap = (id as u64)
                .checked_sub(previous_end)
                .expect("token ids are sorted");
            put_var_u32(
                &mut out,
                u32::try_from(gap).expect("token-id gap should fit u32"),
            );
            previous_end = id as u64 + 1;
        }
        put_var_u32(
            &mut out,
            u32::try_from(bytes.len()).expect("token byte length should fit u32"),
        );
        out.extend_from_slice(bytes);
    }
    out
}

fn pack(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
    let dense = value
        .keys()
        .copied()
        .enumerate()
        .all(|(expected, actual)| actual as usize == expected);
    let count = value.len();
    let data_len = value.values().try_fold(0usize, |total, bytes| {
        total.checked_add(bytes.len())
    });
    let Some(data_len) = data_len.filter(|&len| u32::try_from(len).is_ok()) else {
        return pack_legacy(value);
    };
    let ids_len = if dense { 0 } else { count.saturating_mul(4) };
    let Some(offsets_len) = count.checked_add(1).and_then(|count| count.checked_mul(4)) else {
        return pack_legacy(value);
    };
    let capacity = INDEXED_HEADER_LEN
        .checked_add(ids_len)
        .and_then(|len| len.checked_add(offsets_len))
        .and_then(|len| len.checked_add(data_len));
    let Some(capacity) = capacity else {
        return pack_legacy(value);
    };
    // Reserve the complete indexed representation up front and fill the
    // id/offset/data regions in one BTreeMap traversal.  The previous
    // implementation walked the pointer-heavy map once for offsets and a
    // second time for bytes; on 100k+ token vocabularies that dominated a
    // genuinely fresh Constraint::save().
    let mut out = vec![0u8; capacity];
    out[..INDEXED_MAGIC.len()].copy_from_slice(INDEXED_MAGIC);
    out[INDEXED_MAGIC.len()] = u8::from(!dense);
    out[INDEXED_MAGIC.len() + 1..INDEXED_HEADER_LEN].copy_from_slice(
        &u32::try_from(count)
            .expect("token vocabulary should fit u32")
            .to_le_bytes(),
    );

    let ids_start = INDEXED_HEADER_LEN;
    let offsets_start = ids_start + ids_len;
    let data_start = offsets_start + offsets_len;
    out[offsets_start..offsets_start + 4].copy_from_slice(&0u32.to_le_bytes());

    let mut offset = 0u32;
    let mut data_pos = data_start;
    for (index, (&token_id, bytes)) in value.iter().enumerate() {
        if !dense {
            let id_pos = ids_start + index * 4;
            out[id_pos..id_pos + 4].copy_from_slice(&token_id.to_le_bytes());
        }
        let next_offset = offset
            .checked_add(bytes.len() as u32)
            .expect("indexed token-byte data length was prevalidated");
        let offset_pos = offsets_start + (index + 1) * 4;
        out[offset_pos..offset_pos + 4].copy_from_slice(&next_offset.to_le_bytes());
        let data_end = data_pos + bytes.len();
        out[data_pos..data_end].copy_from_slice(bytes);
        data_pos = data_end;
        offset = next_offset;
    }
    debug_assert_eq!(data_pos, capacity);
    out
}

pub(crate) fn pack_external(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
    pack(value)
}

fn unpack(input: &[u8]) -> Result<Arc<BTreeMap<u32, Vec<u8>>>, String> {
    if input.starts_with(INDEXED_MAGIC) {
        return PackedTokenBytes::parse(input.to_vec()).map(|packed| packed.materialize());
    }
    if !input.starts_with(LEGACY_MAGIC) {
        return Err("invalid packed token-byte header".to_owned());
    }
    let mut pos = LEGACY_MAGIC.len();
    let sparse = match input.get(pos).copied() {
        Some(0) => false,
        Some(1) => true,
        _ => return Err("invalid packed token-byte mode".to_owned()),
    };
    pos += 1;
    let count = take_var_u32(input, &mut pos)? as usize;
    let mut map = BTreeMap::new();
    let mut previous_end = 0u64;
    for dense_id in 0..count {
        let id = if sparse {
            let gap = take_var_u32(input, &mut pos)? as u64;
            let id = previous_end
                .checked_add(gap)
                .ok_or_else(|| "overflowing packed token id".to_owned())?;
            let id = u32::try_from(id)
                .map_err(|_| "overflowing packed token id".to_owned())?;
            previous_end = id as u64 + 1;
            id
        } else {
            u32::try_from(dense_id)
                .map_err(|_| "dense packed token id exceeds u32".to_owned())?
        };
        let len = take_var_u32(input, &mut pos)? as usize;
        let end = pos
            .checked_add(len)
            .ok_or_else(|| "overflowing packed token-byte length".to_owned())?;
        let bytes = input
            .get(pos..end)
            .ok_or_else(|| "truncated packed token bytes".to_owned())?
            .to_vec();
        pos = end;
        if map.insert(id, bytes).is_some() {
            return Err("duplicate packed token id".to_owned());
        }
    }
    if pos != input.len() {
        return Err("trailing bytes in packed token-byte vocabulary".to_owned());
    }
    Ok(Arc::new(map))
}

pub fn serialize<S>(
    value: &Arc<BTreeMap<u32, Vec<u8>>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if !PACKED.with(Cell::get) {
        return value.serialize(serializer);
    }
    if EXTERNAL.with(Cell::get) {
        return Vec::<u8>::new().serialize(serializer);
    }
    pack(value).serialize(serializer)
}

pub fn deserialize<'de, D>(
    deserializer: D,
) -> Result<Arc<BTreeMap<u32, Vec<u8>>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    if !PACKED.with(Cell::get) {
        return Arc::<BTreeMap<u32, Vec<u8>>>::deserialize(deserializer);
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let total = profile.then(std::time::Instant::now);
    let packed_started = profile.then(std::time::Instant::now);
    let packed = Vec::<u8>::deserialize(deserializer)?;
    let packed_len = packed.len();
    let packed_ms = packed_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
    if EXTERNAL.with(Cell::get) {
        if !packed.is_empty() {
            return Err(serde::de::Error::custom(
                "external packed token-byte placeholder must be empty",
            ));
        }
        return Ok(Arc::new(BTreeMap::new()));
    }
    let unpack_started = profile.then(std::time::Instant::now);
    let result = if DEFER_UNPACK.with(Cell::get) {
        let deferred = PackedTokenBytes::parse(packed)
            .map(Arc::new)
            .map_err(serde::de::Error::custom)?;
        DEFERRED.with(|slot| *slot.borrow_mut() = Some(deferred));
        Ok(Arc::new(BTreeMap::new()))
    } else {
        unpack(&packed).map_err(serde::de::Error::custom)
    };
    if let Some(total) = total {
        eprintln!(
            "[glrmask/profile][token_bytes_decode] wire_bytes={} vec_ms={packed_ms:.3} unpack_ms={:.3} total_ms={:.3}",
            packed_len,
            unpack_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0),
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_roundtrip(value: BTreeMap<u32, Vec<u8>>) {
        let packed = pack(&value);
        assert!(packed.starts_with(INDEXED_MAGIC));
        let view = PackedTokenBytes::parse(packed).unwrap();
        assert_eq!(view.len(), value.len());
        assert_eq!(
            view.iter()
                .map(|(id, bytes)| (id, bytes.to_vec()))
                .collect::<BTreeMap<_, _>>(),
            value
        );
        for (&id, bytes) in &value {
            assert_eq!(view.get(id), Some(bytes.as_slice()));
        }
        assert_eq!(view.max_token_id(), value.keys().next_back().copied());

        let legacy = pack_legacy(&value);
        let legacy_view = PackedTokenBytes::parse(legacy).unwrap();
        assert_eq!(
            legacy_view
                .iter()
                .map(|(id, bytes)| (id, bytes.to_vec()))
                .collect::<BTreeMap<_, _>>(),
            value
        );
    }

    #[test]
    fn indexed_token_bytes_roundtrip_dense_and_sparse() {
        check_roundtrip(BTreeMap::from([
            (0, b"a".to_vec()),
            (1, b"bc".to_vec()),
            (2, Vec::new()),
        ]));
        check_roundtrip(BTreeMap::from([
            (2, b"a".to_vec()),
            (9, b"bc".to_vec()),
            (1000, b"xyz".to_vec()),
        ]));
    }
}
