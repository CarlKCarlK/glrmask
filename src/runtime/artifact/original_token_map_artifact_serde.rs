use std::cell::Cell;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

const VARINT_MAGIC: &[u8; 4] = b"OTM1";
const FIXED_MAGIC: &[u8; 4] = b"OTM2";
const FIXED_HEADER_LEN: usize = FIXED_MAGIC.len() + 1 + 4;

#[derive(Debug)]
pub(crate) struct PackedOriginalTokenMap {
    backing: Arc<Vec<u8>>,
    payload_start: usize,
    count: usize,
    width: usize,
}

impl PackedOriginalTokenMap {
    pub(crate) fn parse_backed(
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    ) -> Result<Self, String> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| "fixed original-token map range overflows".to_owned())?;
        let input = backing
            .get(start..end)
            .ok_or_else(|| "fixed original-token map is outside artifact backing".to_owned())?;
        if input.len() < FIXED_HEADER_LEN || !input.starts_with(FIXED_MAGIC) {
            return Err("invalid fixed original-token map header".to_owned());
        }
        let width = input[FIXED_MAGIC.len()] as usize;
        if !matches!(width, 1 | 2 | 4) {
            return Err("invalid fixed original-token map width".to_owned());
        }
        let count_start = FIXED_MAGIC.len() + 1;
        let count = u32::from_le_bytes(
            input[count_start..count_start + 4]
                .try_into()
                .expect("fixed original-token count has fixed width"),
        ) as usize;
        let payload_len = count
            .checked_mul(width)
            .ok_or_else(|| "fixed original-token map payload overflows".to_owned())?;
        if FIXED_HEADER_LEN
            .checked_add(payload_len)
            .is_none_or(|expected| expected != input.len())
        {
            return Err("invalid fixed original-token map length".to_owned());
        }
        Ok(Self {
            backing,
            payload_start: start + FIXED_HEADER_LEN,
            count,
            width,
        })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<u32> {
        if index >= self.count {
            return None;
        }
        let start = self.payload_start + index * self.width;
        let bytes = self.backing.get(start..start + self.width)?;
        Some(match self.width {
            1 => {
                let value = bytes[0];
                if value == u8::MAX { u32::MAX } else { value as u32 }
            }
            2 => {
                let value = u16::from_le_bytes([bytes[0], bytes[1]]);
                if value == u16::MAX { u32::MAX } else { value as u32 }
            }
            4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            _ => unreachable!(),
        })
    }

    pub(crate) fn materialize(&self) -> Vec<u32> {
        (0..self.count)
            .map(|index| self.get(index).expect("validated packed original-token map index"))
            .collect()
    }
}

thread_local! {
    static PACKED: Cell<bool> = const { Cell::new(false) };
    static EXTERNAL: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn set_packed(enabled: bool) -> bool {
    PACKED.with(|mode| mode.replace(enabled))
}

pub(crate) fn set_external(enabled: bool) -> bool {
    EXTERNAL.with(|mode| mode.replace(enabled))
}

pub(crate) fn to_fast_bytes(value: &[u32]) -> Vec<u8> {
    pack_fixed(value)
}

pub(crate) fn from_fast_bytes(input: &[u8]) -> Result<Vec<u32>, String> {
    unpack(input)
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
            .ok_or_else(|| "truncated packed original-token map".to_owned())?;
        *pos += 1;
        if shift == 28 && byte > 0x0f {
            return Err("overflowing packed original-token map".to_owned());
        }
        value |= ((byte & 0x7f) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    Err("overflowing packed original-token map".to_owned())
}

fn pack_varint(value: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(value.len().saturating_mul(2));
    out.extend_from_slice(VARINT_MAGIC);
    put_var_u32(
        &mut out,
        u32::try_from(value.len()).expect("token vocabulary should fit u32"),
    );
    for &internal in value {
        let encoded = if internal == u32::MAX {
            0
        } else {
            internal
                .checked_add(1)
                .expect("u32::MAX is reserved as the unmapped-token sentinel")
        };
        put_var_u32(&mut out, encoded);
    }
    out
}

fn unpack_varint(input: &[u8]) -> Result<Vec<u32>, String> {
    if !input.starts_with(VARINT_MAGIC) {
        return Err("invalid varint original-token map header".to_owned());
    }
    let mut pos = VARINT_MAGIC.len();
    let count = take_var_u32(input, &mut pos)? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let encoded = take_var_u32(input, &mut pos)?;
        out.push(if encoded == 0 { u32::MAX } else { encoded - 1 });
    }
    if pos != input.len() {
        return Err("trailing bytes in packed original-token map".to_owned());
    }
    Ok(out)
}

fn pack_fixed(value: &[u32]) -> Vec<u8> {
    let max_internal = value
        .iter()
        .copied()
        .filter(|&internal| internal != u32::MAX)
        .max()
        .unwrap_or(0);
    let width = if max_internal < u8::MAX as u32 {
        1u8
    } else if max_internal < u16::MAX as u32 {
        2u8
    } else {
        4u8
    };
    let payload_len = value
        .len()
        .checked_mul(width as usize)
        .expect("original-token map payload should fit usize");
    let mut out = Vec::with_capacity(FIXED_HEADER_LEN + payload_len);
    out.extend_from_slice(FIXED_MAGIC);
    out.push(width);
    out.extend_from_slice(
        &u32::try_from(value.len())
            .expect("token vocabulary should fit u32")
            .to_le_bytes(),
    );
    match width {
        1 => {
            for &internal in value {
                out.push(if internal == u32::MAX {
                    u8::MAX
                } else {
                    internal as u8
                });
            }
        }
        2 => {
            for &internal in value {
                let encoded = if internal == u32::MAX {
                    u16::MAX
                } else {
                    internal as u16
                };
                out.extend_from_slice(&encoded.to_le_bytes());
            }
        }
        4 => {
            for &internal in value {
                out.extend_from_slice(&internal.to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    out
}

fn unpack_fixed(input: &[u8]) -> Result<Vec<u32>, String> {
    if input.len() < FIXED_HEADER_LEN || !input.starts_with(FIXED_MAGIC) {
        return Err("invalid fixed original-token map header".to_owned());
    }
    let width = input[FIXED_MAGIC.len()] as usize;
    if !matches!(width, 1 | 2 | 4) {
        return Err("invalid fixed original-token map width".to_owned());
    }
    let count_start = FIXED_MAGIC.len() + 1;
    let count = u32::from_le_bytes(
        input[count_start..count_start + 4]
            .try_into()
            .expect("fixed original-token count has fixed width"),
    ) as usize;
    let payload_len = count
        .checked_mul(width)
        .ok_or_else(|| "fixed original-token map payload overflows".to_owned())?;
    if FIXED_HEADER_LEN
        .checked_add(payload_len)
        .is_none_or(|expected| expected != input.len())
    {
        return Err("invalid fixed original-token map length".to_owned());
    }
    let payload = &input[FIXED_HEADER_LEN..];
    let mut out = Vec::with_capacity(count);
    match width {
        1 => out.extend(payload.iter().map(|&encoded| {
            if encoded == u8::MAX {
                u32::MAX
            } else {
                encoded as u32
            }
        })),
        2 => out.extend(payload.chunks_exact(2).map(|bytes| {
            let encoded = u16::from_le_bytes([bytes[0], bytes[1]]);
            if encoded == u16::MAX {
                u32::MAX
            } else {
                encoded as u32
            }
        })),
        4 => out.extend(payload.chunks_exact(4).map(|bytes| {
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        })),
        _ => unreachable!(),
    }
    Ok(out)
}

fn unpack(input: &[u8]) -> Result<Vec<u32>, String> {
    if input.starts_with(FIXED_MAGIC) {
        unpack_fixed(input)
    } else if input.starts_with(VARINT_MAGIC) {
        unpack_varint(input)
    } else {
        Err("invalid packed original-token map header".to_owned())
    }
}

pub fn serialize<S>(value: &[u32], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if EXTERNAL.with(Cell::get) {
        return 0u8.serialize(serializer);
    }
    if !PACKED.with(Cell::get) {
        return value.serialize(serializer);
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let total_started = profile.then(std::time::Instant::now);
    let pack_started = profile.then(std::time::Instant::now);
    let packed = pack_fixed(value);
    let pack_ms = pack_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
    let wire_bytes = packed.len();
    let result = packed.serialize(serializer);
    if let Some(started) = total_started {
        eprintln!(
            "[glrmask/profile][original_token_map_encode] entries={} wire_bytes={} pack_ms={:.3} total_ms={:.3}",
            value.len(),
            wire_bytes,
            pack_ms,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    if EXTERNAL.with(Cell::get) {
        let marker = u8::deserialize(deserializer)?;
        if marker != 0 {
            return Err(serde::de::Error::custom(
                "invalid external original-token map placeholder",
            ));
        }
        return Ok(Vec::new());
    }
    if !PACKED.with(Cell::get) {
        return Vec::<u32>::deserialize(deserializer);
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let total_started = profile.then(std::time::Instant::now);
    let packed = Vec::<u8>::deserialize(deserializer)?;
    let packed_len = packed.len();
    let result = unpack(&packed).map_err(serde::de::Error::custom);
    if let Some(started) = total_started {
        eprintln!(
            "[glrmask/profile][original_token_map_decode] wire_bytes={} ms={:.3}",
            packed_len,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_original_token_map_roundtrips_all_widths_and_legacy() {
        for values in [
            vec![0, 12, u32::MAX, 254],
            vec![0, 255, 4096, u32::MAX, 65534],
            vec![0, 65535, 1_000_000, u32::MAX],
        ] {
            let packed = pack_fixed(&values);
            assert_eq!(unpack(&packed).unwrap(), values);

            let legacy = pack_varint(&values);
            assert_eq!(unpack(&legacy).unwrap(), values);
        }
    }
}
