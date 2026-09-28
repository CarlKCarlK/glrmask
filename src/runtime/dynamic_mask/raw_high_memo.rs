//! Query-local bounded memo for large raw coordinates.
//!
//! Only complete keys may hit. Replacement loses cached work, never semantic
//! state; a miss takes the caller's existing exact transition/config path.
//! No storage grows with the numeric value of a raw state ID.

const EMPTY: u64 = u64::MAX;

pub(super) struct TaggedMemo<const N: usize> {
    entries: Vec<(u64, u64)>,
}

impl<const N: usize> TaggedMemo<N> {
    pub(super) fn new() -> Self {
        assert!(N.is_power_of_two());
        Self { entries: Vec::new() }
    }

    #[inline(always)]
    fn index(key: u64) -> usize {
        // Mix the state and byte parts; correctness never depends on this hash.
        ((key ^ (key >> 8) ^ (key >> 24)) as usize).wrapping_mul(0x9e37_79b1) & (N - 1)
    }

    #[inline]
    pub(super) fn get(&self, key: u64) -> Option<u64> {
        let &(stored, value) = self.entries.get(Self::index(key))?;
        (stored == key && stored != EMPTY).then_some(value)
    }

    #[inline]
    pub(super) fn insert(&mut self, key: u64, value: u64) {
        // Runtime keys are at most 40 bits (u32 state plus one byte).
        assert_ne!(key, EMPTY);
        if self.entries.is_empty() {
            self.entries.resize(N, (EMPTY, 0));
        }
        self.entries[Self::index(key)] = (key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn unallocated_misses_and_collisions_never_alias_answers() {
        let mut memo=TaggedMemo::<64>::new();
        assert_eq!(memo.entries.capacity(),0);
        for key in [0,1,u32::MAX as u64,(u32::MAX as u64)<<8] { assert_eq!(memo.get(key),None); }
        assert_eq!(memo.entries.capacity(),0);
        let first=12345;
        let second=(first+1..first+10000).find(|&k|TaggedMemo::<64>::index(k)==TaggedMemo::<64>::index(first)).unwrap();
        memo.insert(first,17);assert_eq!(memo.get(first),Some(17));
        memo.insert(second,81);assert_eq!(memo.get(first),None);assert_eq!(memo.get(second),Some(81));
        memo.insert(first,19);assert_eq!(memo.get(first),Some(19));assert_eq!(memo.get(second),None);
        assert_eq!(memo.entries.len(),64);assert_eq!(memo.entries.capacity(),64);
    }

    #[test]
    fn all_bytes_preserve_dead_targets_and_finalizer_bits_at_large_states() {
        let mut memo=TaggedMemo::<256>::new();
        for state in [1024u32,19094,1<<28,u32::MAX-1] {
            for byte in 0u16..256 {
                let key=(u64::from(state)<<8)|u64::from(byte);
                let target=if byte%3==0 {u32::MAX} else {state.wrapping_add(byte as u32)};
                let packed=u64::from(target)|((u64::from(byte)&1)<<32);
                memo.insert(key,packed);
            }
            for byte in 0u16..256 {
                let key=(u64::from(state)<<8)|u64::from(byte);
                let target=if byte%3==0 {u32::MAX} else {state.wrapping_add(byte as u32)};
                assert_eq!(memo.get(key),Some(u64::from(target)|((u64::from(byte)&1)<<32)));
            }
        }
        assert_eq!(memo.entries.len(),256);assert_eq!(memo.entries.capacity(),256);
    }

    #[test]
    fn generated_eviction_and_recomputation_agree_with_exact_map() {
        let mut memo=TaggedMemo::<64>::new();let mut exact=HashMap::new();let mut seed=0x29_2840u64;
        for i in 0..20000u64 {
            seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let key=(seed>>12)&((1u64<<40)-1);
            if let Some(value)=memo.get(key) { assert_eq!(Some(&value),exact.get(&key)); }
            let value=key.rotate_left(19)^i;exact.insert(key,value);memo.insert(key,value);
            assert_eq!(memo.get(key),Some(value));
            if i%97==0 { for (&k,&v) in &exact { if let Some(hit)=memo.get(k){assert_eq!(hit,v);} } }
        }
        assert_eq!(memo.entries.len(),64);assert_eq!(memo.entries.capacity(),64);
    }

    #[test]
    fn independent_owners_and_target_state_keys_do_not_share_entries() {
        let mut a=TaggedMemo::<64>::new();let mut b=TaggedMemo::<64>::new();
        for state in [0u32,1023,1024,65536,u32::MAX-1] {
            let key=u64::from(state);a.insert(key,7);b.insert(key,19);
            assert_eq!(a.get(key),Some(7));assert_eq!(b.get(key),Some(19));
        }
        let c=TaggedMemo::<64>::new();assert_eq!(c.get(u64::from(u32::MAX-1)),None);
    }
}
