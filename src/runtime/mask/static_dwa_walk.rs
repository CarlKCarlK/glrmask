//! Shared single-stack parser-DWA traversal.
//!
//! State-coordinate mapping and shard selection happen before entry. The row
//! provider and mask operations are monomorphized closures: the ordinary path
//! does not test a composition flag or call through a trait object per edge.
//! Weight representation is independent of traversal (packed/materialized
//! weights and the small u64 boundary representation use the same loop).

#[derive(Clone, Copy)]
pub(super) enum StackWalkEvent<W> {
    Final(W),
    Top(u32),
    Intersect(W),
}

/// Visit every accepting prefix of a top-first parser stack. `visit` returns
/// false when its path mask becomes empty, is fully accepted, or a caller's
/// planning budget ends.
/// In either case already-emitted contributions remain valid. Row providers
/// own positive/domain/default-label lookup; the kernel never remaps IDs.
#[inline(always)]
pub(super) fn walk_single_stack<const VISIT_TOP: bool, W: Copy>(
    start_state: u32,
    top_first: &[u32],
    mut final_weight: impl FnMut(u32) -> Option<W>,
    mut transition: impl FnMut(u32, u32) -> Option<(u32, W)>,
    mut visit: impl FnMut(StackWalkEvent<W>) -> bool,
) {
    let mut state = start_state;
    let mut stack_index = 0;
    loop {
        if let Some(weight) = final_weight(state)
            && !visit(StackWalkEvent::Final(weight))
        {
            break;
        }
        let Some(&parser_state) = top_first.get(stack_index) else {
            break;
        };
        stack_index += 1;
        if VISIT_TOP && stack_index == 1 && !visit(StackWalkEvent::Top(parser_state)) {
            break;
        }
        let Some((target, weight)) = transition(state, parser_state) else {
            break;
        };
        if !visit(StackWalkEvent::Intersect(weight)) {
            break;
        }
        state = target;
    }
}

#[cfg(test)]
mod tests {
    use super::{StackWalkEvent, walk_single_stack};

    #[test]
    fn accepting_prefixes_survive_a_dead_edge_and_mapping_is_provider_owned() {
        let mut path = 0b111_u64;
        let mut accepted = 0;
        let mut tops = Vec::new();
        walk_single_stack::<true, _>(
            0,
            &[100, 101, 102],
            |state| [Some(0b001), Some(0b110), Some(0b100)].get(state as usize).copied().flatten(),
            |state, parser| match (state, parser.checked_sub(100)) {
                (0, Some(0)) => Some((1, 0b110)),
                (1, Some(1)) => Some((2, 0)),
                _ => panic!("must not traverse below the dead edge"),
            },
            |event| {
                match event {
                    StackWalkEvent::Final(weight) => accepted |= path & weight,
                    StackWalkEvent::Top(state) => tops.push(state),
                    StackWalkEvent::Intersect(weight) => {
                        path &= weight;
                        return path != 0;
                    }
                }
                true
            },
        );
        assert_eq!(accepted, 0b111);
        assert_eq!(tops, [100]);
    }

    #[test]
    fn boundary_specialization_accepts_empty_stack_without_top_dispatch() {
        let mut accepted = 0_u64;
        walk_single_stack::<false, _>(
            7, &[],
            |state| { assert_eq!(state, 7); Some(0b101) },
            |_, _| panic!("empty stack must not look up an edge"),
            |event| {
                match event {
                    StackWalkEvent::Final(weight) => accepted |= weight,
                    _ => panic!("no top/edge event for empty stack"),
                }
                true
            },
        );
        assert_eq!(accepted, 0b101);
    }

    #[test]
    fn full_final_shortcut_matches_literal_all_prefix_union() {
        fn random(seed: &mut u64) -> u64 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        for case in 0..512 {
            let depth = case % 65;
            let stack: Vec<u32> = (0..depth as u32).collect();
            let edges: Vec<u64> = (0..depth).map(|_| random(&mut seed) & 0xff).collect();
            let finals: Vec<Option<u64>> = (0..=depth).map(|_| {
                let bits = random(&mut seed);
                match bits % 4 {
                    0 => None,
                    1 => Some(u64::MAX),
                    _ => Some(bits & 0xff),
                }
            }).collect();
            let top_accept = random(&mut seed) & 0xff;
            let mut expected_union = 0;
            let mut actual_union = 0;
            // Different seeds model independently correlated path exclusions
            // and lexer coordinates. Finishing one path never finishes all.
            for path_seed in [0, 1, 0x55, 0xaa, 0xff] {
                let mut expected_path = path_seed;
                for index in 0..=depth {
                    if let Some(final_weight) = finals[index] {
                        expected_union |= expected_path & final_weight;
                    }
                    if index == depth { break; }
                    if index == 0 { expected_union |= expected_path & top_accept; }
                    expected_path &= edges[index];
                }
                let mut actual_path = path_seed;
                walk_single_stack::<true, _>(
                    0, &stack,
                    |state| finals[state as usize],
                    |state, parser| {
                        assert_eq!(state, parser);
                        Some((state + 1, edges[state as usize]))
                    },
                    |event| {
                        match event {
                            StackWalkEvent::Final(weight) => {
                                actual_union |= actual_path & weight;
                                return weight != u64::MAX;
                            }
                            StackWalkEvent::Top(_) => actual_union |= actual_path & top_accept,
                            StackWalkEvent::Intersect(weight) => {
                                actual_path &= weight;
                                return actual_path != 0;
                            }
                        }
                        true
                    },
                );
                assert_eq!(actual_union, expected_union, "case={case} seed={path_seed}");
            }
        }
    }
}
