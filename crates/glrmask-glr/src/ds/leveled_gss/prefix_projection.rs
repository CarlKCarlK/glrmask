//! Exact top-prefix projection without enumerating the represented stacks.
use super::*;

fn merge_acc<A: Merge>(target: &mut Option<A>, source: Option<&A>) {
    if let Some(source) = source {
        *target = Some(match target.as_ref() {
            Some(current) => current.merge(source),
            None => source.clone(),
        });
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> LeveledGSS<T, A> {
    /// Map each maximal top-first prefix for which `map` returns `Some`.
    /// The first unmapped value and the entire suffix below it are discarded.
    /// Paths with an unmapped top, and empty input stacks, contribute nothing.
    ///
    /// The original path accumulator is retained even when its interface lies
    /// below the discarded suffix. Equal projected stacks use the ordinary
    /// accumulator merge. Shared suffixes are processed once, so an exponential
    /// concrete stack language remains a shared graph rather than a path list.
    pub fn map_nonempty_top_prefixes<U, F>(&self, mut map: F) -> LeveledGSS<U, A>
    where
        U: Clone + Eq + Hash,
        F: FnMut(&T) -> Option<U>,
    {
        let dag = self.indexed_dag();
        let count = dag.nodes.len();
        let mut lower: Vec<Option<Arc<Lower<U>>>> = vec![None; count];
        let mut upper: Vec<Option<Arc<Upper<U, A>>>> = vec![None; count];
        // Reduction over the ORIGINAL suffix, before projection. A cutoff
        // above an interface must not lose that interface's accumulator.
        let mut suffix_acc: Vec<Option<A>> = vec![None; count];
        let mut value_memo = StdHashMap::<T, Option<U>>::new();
        let mut mapped = |value: &T| value_memo.entry(value.clone())
            .or_insert_with(|| map(value)).clone();
        let epsilon = new_lower_precanonicalized(Children::new(), true);

        // indexed_dag emits children before parents. No recursive transform or
        // concrete stack walk is required, including across shared interfaces.
        for (id, node) in dag.nodes.into_iter().enumerate() {
            match node {
                IndexedLeveledGssNode::LowerGeneral { empty, children, .. } => {
                    let mut empty = empty;
                    let mut output = Children::<U, Lower<U>>::new();
                    for (value, child) in children {
                        let Some(suffix) = lower[child as usize].as_ref() else { continue; };
                        match mapped(&value) {
                            Some(value) => insert_lower_child_shared(&mut output, value,
                                suffix.max_depth(), Arc::clone(suffix)),
                            None => empty = true,
                        }
                    }
                    if empty || !output.is_empty() {
                        lower[id] = Some(new_lower_precanonicalized(output, empty));
                    }
                }
                IndexedLeveledGssNode::LowerSegment { values, next, .. } => {
                    let Some(suffix) = lower[next as usize].as_ref() else { continue; };
                    let mut output = Vec::with_capacity(values.len());
                    let mut cutoff = false;
                    for value in values.iter().rev() {
                        match mapped(value) {
                            Some(value) => output.push(value),
                            None => { cutoff = true; break; }
                        }
                    }
                    output.reverse();
                    let base = if cutoff { Arc::clone(&epsilon) } else { Arc::clone(suffix) };
                    lower[id] = Some(if output.is_empty() { base }
                        else { new_segment(SV::from_vec(output), base) });
                }
                IndexedLeveledGssNode::Interface { accumulator, lower: child } => {
                    if let Some(suffix) = lower[child as usize].as_ref() {
                        upper[id] = Some(try_promote(&new_interface(Arc::clone(suffix), accumulator.clone())));
                        suffix_acc[id] = Some(accumulator);
                    }
                }
                IndexedLeveledGssNode::UpperBranch { empty, children } => {
                    let mut output = Children::<U, Upper<U, A>>::new();
                    let mut summary = empty.clone();
                    let mut empty = empty;
                    for (value, child) in children {
                        let child = child as usize;
                        merge_acc(&mut summary, suffix_acc[child].as_ref());
                        match mapped(&value) {
                            Some(value) => if let Some(suffix) = upper[child].as_ref() {
                                insert_upper_child_shared(&mut output, value, Arc::clone(suffix));
                            },
                            None => merge_acc(&mut empty, suffix_acc[child].as_ref()),
                        }
                    }
                    if empty.is_some() || !output.is_empty() {
                        upper[id] = Some(try_promote(&new_branch(output, empty)));
                    }
                    suffix_acc[id] = summary;
                }
            }
        }

        // Cutting a foreign TOP is not an activation of an empty local stack.
        // Remove only root epsilon; epsilon below a retained top is required.
        let Some(root) = upper[dag.root as usize].take() else { return LeveledGSS::empty(); };
        let inner = match root.as_ref() {
            Upper::Branch(branch) => new_branch(branch.children.clone(), None),
            Upper::Interface(interface) => {
                let projected = match interface.inner.as_ref() {
                    Lower::General { children, .. } => new_lower_precanonicalized(children.clone(), false),
                    Lower::Segment(_) => Arc::clone(&interface.inner),
                };
                new_interface(projected, interface.acc.clone())
            }
        };
        LeveledGSS { inner: try_promote(&inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Tags(u32);
    impl Merge for Tags { fn merge(&self, other: &Self) -> Self { Self(self.0 | other.0) } }

    fn literal(source: &LeveledGSS<u32, Tags>, map: impl Fn(u32) -> Option<u32>)
        -> BTreeMap<Vec<u32>, u32>
    {
        let mut expected = BTreeMap::new();
        for (stack, acc) in source.to_stacks(100_000).unwrap() {
            let mut prefix = stack.into_iter().rev().map_while(&map).collect::<Vec<_>>();
            if prefix.is_empty() { continue; }
            prefix.reverse();
            *expected.entry(prefix).or_insert(0) |= acc.0;
        }
        expected
    }

    fn actual(source: &LeveledGSS<u32, Tags>) -> BTreeMap<Vec<u32>, u32> {
        let mut values = BTreeMap::new();
        for (stack, acc) in source.to_stacks(100_000).unwrap() {
            *values.entry(stack).or_insert(0) |= acc.0;
        }
        values
    }

    #[test]
    fn prefix_projection_preserves_suffix_annotations_and_foreign_top_rejection() {
        let source = LeveledGSS::from_stacks(&[
            (vec![90, 1, 2], Tags(1)), (vec![91, 1, 2], Tags(2)),
            (vec![80, 3], Tags(4)), (vec![1, 99], Tags(8)),
            (vec![], Tags(16)), (vec![2], Tags(32)),
        ]);
        let map = |value| (value < 4).then_some(value % 2);
        let projected = source.map_nonempty_top_prefixes(|&value| map(value));
        assert_eq!(actual(&projected), literal(&source, map));
        assert!(projected.isolate(None).is_empty());
        assert_eq!(LeveledGSS::<u32, Tags>::empty().map_nonempty_top_prefixes(|v| Some(*v)).max_depth(), 0);
    }

    #[test]
    fn prefix_projection_matches_literal_random_annotated_graphs() {
        let mut seed = 0x4716_5928_u64;
        let mut random = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 32) as u32 };
        for _ in 0..256 {
            let words = (0..48).map(|_| {
                let length = random() % 12;
                let stack = (0..length).map(|_| random() % 7).collect();
                (stack, Tags(1 << (random() % 8)))
            }).collect::<Vec<_>>();
            let source = LeveledGSS::from_stacks(&words);
            for ceiling in 0..=7 {
                let map = |value| (value < ceiling).then_some(value % 3);
                assert_eq!(actual(&source.map_nonempty_top_prefixes(|&v| map(v))), literal(&source, map));
            }
        }
    }

    #[test]
    fn prefix_projection_keeps_exponential_languages_as_shared_graphs() {
        let mut source = LeveledGSS::from_single_stack(vec![99_u32], Tags(1));
        for _ in 0..32 { source = source.clone().push(0).merge(&source.push(1)); }
        let result = source.map_nonempty_top_prefixes(|&v| (v < 2).then_some(v));
        assert_eq!(result.max_depth(), 32);
        assert_eq!(result.path_count_at_most(1_000_000), 1_000_000);
        assert!(result.node_count_at_most(1_000) < 1_000);
        assert!(result.isolate(None).is_empty());
        let merged = source.map_nonempty_top_prefixes(|&v| (v < 2).then_some(0_u32));
        assert_eq!(actual(&merged), BTreeMap::from([(vec![0; 32], 1)]));
        assert_eq!(source.max_depth(), 33);
    }
}
