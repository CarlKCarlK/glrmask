use super::*;

#[test]
fn automatic_policy_keeps_small_low_core_and_unsupported_inputs_serial() {
    for threads in [0, 1, 2, 3] {
        assert!(Policy::for_configuration(true, 24_043, 7_947, threads, true).is_none());
    }
    for states in [0, 1, 4095, 200_001] {
        assert!(Policy::for_configuration(true, states, 7_947, 10, true).is_none());
    }
    assert!(Policy::for_configuration(false, 24_043, 7_947, 10, true).is_none());
    assert!(Policy::for_configuration(true, 24_043, 7_947, 10, false).is_none());
    assert!(Policy::for_configuration(true, 24_043, 32_769, 10, true).is_none());
    for threads in [4, 6, 10, 64] {
        let p = Policy::for_configuration(true, 4096, 32_768, threads, true).unwrap();
        assert_eq!((p.threshold, p.batch, p.chunk), (64, 4096, 256));
        assert!(p.live_import);
    }
}

#[test]
fn importing_private_values_obeys_global_limit_without_publishing_a_row() {
    let mut p = pool(1);
    let a = p.intern(smallvec::smallvec![0b11011]);
    let b = p.intern(smallvec::smallvec![0b10101]);
    assert!(!p.ids.contains_key([0b10001u64].as_slice()));
    p.limits.as_mut().unwrap().weights = p.values.len();
    let source = vec![
        FastBoundaryNwaState { final_weight: 0, epsilons: vec![],
            transitions: vec![(0, smallvec::smallvec![(1, b)])] },
        FastBoundaryNwaState { final_weight: 1, epsilons: vec![], transitions: vec![] },
    ];
    let mut out = vec![FastBoundaryDwaState::default()];
    let mut supports = vec![vec![0]];
    let mut pending = VecDeque::from([(0, vec![(0, a)])]);
    let mut singleton_states = vec![u32::MAX; 2];
    let mut subsets = FxHashMap::default();
    let mut singletons = FxHashMap::default();
    let mut closures = FxHashMap::default();
    let result = batch(Policy { threshold: 1, batch: 1, chunk: 1, live_import: true }, &source, 6,
        &mut p, &mut singleton_states, &mut subsets, &mut singletons, &mut closures,
        &mut out, &mut supports, &mut pending, &mut 0, &mut Profile::default());
    assert!(result.is_none(), "a resource failure is not a successful empty mask");
    assert!(p.failed);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].final_weight, 0);
    assert!(out[0].transitions.is_empty());
    assert!(subsets.is_empty() && singletons.is_empty() && closures.is_empty());
}

fn pool(words: usize) -> FastBoundaryWeightInterner {
    let mut p = FastBoundaryWeightInterner::new(words, 64).unwrap();
    p.limits = Some(FiniteCompileLimits::default());
    for pattern in [1u64, 2, 3, 5, 7, 13, 31, 0x5555, 0xaaaa, 0xffff] {
        p.intern((0..words).map(|q| pattern.rotate_left((q % 64) as u32)).collect());
    }
    // Overlay reads the actual pre-existing cache contents as well as values.
    p.intersection(4, 7);
    p.union(5, 6);
    p
}

#[test]
fn immutable_overlay_matches_boolean_values_and_limits() {
    for words in [1, 2, 17, 44, 64] {
        let base = pool(words);
        let mut reference = pool(words);
        let mut overlay = Overlay::new(&base).unwrap();
        let mut pairs = (0..base.values.len() as u32).map(|id| (id, id)).collect::<Vec<_>>();
        let mut seed = 401626u64;
        for index in 0..1024 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let (a, ar) = pairs[(seed >> 32) as usize % pairs.len()];
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let (b, br) = pairs[(seed >> 32) as usize % pairs.len()];
            assert_eq!(overlay.is_subset(a, b), reference.is_subset(ar, br));
            let (actual, expected) = if index % 2 == 0 {
                (overlay.intersection(a, b), reference.intersection(ar, br))
            } else { (overlay.union(a, b), reference.union(ar, br)) };
            assert!(!overlay.failed);
            assert_eq!(overlay.value(actual), reference.values[expected as usize].as_slice());
            pairs.push((actual, expected));
        }
        let mut limited = Overlay::new(&base).unwrap();
        limited.limit = 0;
        assert_eq!(limited.intern(&base.values[2]), 2, "existing values do not spend a local slot");
        let novel = vec![0x123456789abcdef0; words];
        assert!(!base.ids.contains_key(novel.as_slice()));
        assert_eq!(limited.intern(&novel), 0);
        assert!(limited.failed, "zero is an unwind aid, never a successful new result");
    }
}

fn finite(
    source: &[FastBoundaryNwaState], starts: &[u32],
    p: &mut FastBoundaryWeightInterner, policy: Option<Policy>,
) -> Option<FiniteBoundaryDwa> {
    match determinize_preconverted_small_boundary_output_with_parallel_policy(
        source, starts, 6, p, p.values.len(), 0.0, Instant::now(), false, true, policy,
    )? {
        SmallBoundaryDeterminizeOutput::Finite(output) => Some(output),
        _ => panic!("finite-output test must not switch backend"),
    }
}

fn same_native(a: &FiniteBoundaryDwa, b: &FiniteBoundaryDwa, case: usize) {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.token_count, b.token_count);
    assert_eq!(a.states.len(), b.states.len(), "case{case}: deterministic node IDs");
    for (q, (left, right)) in a.states.iter().zip(&b.states).enumerate() {
        assert_eq!(a.weights[left.final_weight as usize], b.weights[right.final_weight as usize],
            "case{case} q{q}: final coefficient");
        assert_eq!(left.transitions.len(), right.transitions.len(), "case{case} q{q}: edge count");
        for (&(la, ta, wa), &(lb, tb, wb)) in left.transitions.iter().zip(&right.transitions) {
            assert_eq!((la, ta), (lb, tb), "case{case} q{q}: label/target identities");
            assert_eq!(a.weights[wa as usize], b.weights[wb as usize],
                "case{case} q{q} label{la}: exact coefficient");
        }
    }
}

#[test]
fn parallel_packets_preserve_native_targets_guards_and_coefficients() {
    let threads = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
    threads.install(|| {
        let policy = Policy { threshold: 1, batch: 9, chunk: 2, live_import: false };
        let mut seed = 192637u64;
        let mut next = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 32) as usize };
        let mut exercised = 0usize;
        for case in 0..256 {
            let words = [1, 2, 17, 44, 64][case % 5];
            let weights = pool(words).values.len();
            let n = 5 + next() % 13;
            let mut source = Vec::with_capacity(n);
            for q in 0..n {
                let mut groups = BTreeMap::<i32, SmallVec<[(u32,u32);1]>>::new();
                let mut epsilons = Vec::new();
                for target in q + 1..n {
                    let weight = (next() % weights) as u32;
                    let selection = next() % 9;
                    if selection == 0 { epsilons.push((target as u32, weight)); }
                    else if selection < 7 {
                        let label = if selection == 6 { DEFAULT_LABEL } else { (selection - 1) as i32 };
                        let group = groups.entry(label).or_default();
                        group.push((target as u32, weight));
                        if next() % 4 == 0 { group.push((target as u32, (next() % weights) as u32)); }
                    }
                }
                if case % 3 == 0 { groups.entry(5).or_default(); }
                if case % 7 == 0 { groups.entry(DEFAULT_LABEL).or_default().push((0,0)); }
                source.push(FastBoundaryNwaState { final_weight: (next() % weights) as u32,
                    epsilons, transitions: groups.into_iter().collect() });
            }
            let starts = if case % 4 == 0 { vec![0,1] } else { vec![0] };
            let reference = finite(&source, &starts, &mut pool(words), None).unwrap();
            exercised += reference.states.len();
            for live_import in [false, true] {
                let candidate = finite(&source, &starts, &mut pool(words),
                    Some(Policy { live_import, ..policy })).unwrap();
                same_native(&reference, &candidate, case);
            }
        }
        assert!(exercised > 1000, "exercise many weighted registered frontiers");
    });
}

#[test]
fn parallel_packets_preserve_wide_convergence_and_state_budget_decline() {
    let mut source = vec![FastBoundaryNwaState { final_weight: 0, epsilons: vec![], transitions: vec![] }; 193];
    // A broad layer, with repeated structures and weighted joins, genuinely
    // reaches multi-packet execution rather than only the single-row fallback.
    for q in 1..97 {
        source[0].transitions.push((q as i32, smallvec::smallvec![(q as u32, 2 + q as u32 % 9)]));
        source[q].transitions.push((0, smallvec::smallvec![(97 + (q % 48) as u32, 3), (145 + (q % 48) as u32, 5)]));
    }
    for row in &mut source[97..] { row.final_weight = 7; }
    let policy = Some(Policy { threshold: 2, batch: 64, chunk: 8, live_import: true });
    let threads = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    threads.install(|| {
        let reference = finite(&source, &[0], &mut pool(17), None).unwrap();
        let candidate = finite(&source, &[0], &mut pool(17), policy).unwrap();
        same_native(&reference, &candidate, 999);
        for mode in [None, policy] {
            let mut p = pool(17);
            p.limits.as_mut().unwrap().states = 1;
            assert!(finite(&source, &[0], &mut p, mode).is_none());
        }
    });
}

#[test]
fn escaping_values_cover_every_published_field_and_reject_bad_references() {
    let mut packet = Packet {
        values: (0..8).map(|i| smallvec::smallvec![1u64 << i]).collect(),
        recipes: vec![Recipe {
            incoming: vec![(10, LOCAL)], closure: vec![(11, LOCAL | 1)], weight: LOCAL | 2,
        }],
        rows: vec![PreparedRow {
            id: 0, members: 1, final_weight: LOCAL | 3,
            edges: vec![(0, 1, LOCAL | 4), (1, 1, 1)],
            pending: vec![(0, Target::Singleton(1, LOCAL | 5)),
                (1, Target::Known(1, LOCAL | 6)), (2, Target::Recipe(0))],
            local_coefficients: true,
        }],
        base_hits: 0, local_hits: 0, worker: 1,
    };
    assert_eq!(packet.escaping_values().unwrap(), vec![true, true, true, true, true, true, true, false]);
    packet.rows[0].pending.push((3, Target::Recipe(99)));
    assert!(packet.escaping_values().is_none());
    packet.rows[0].pending.pop();
    packet.rows[0].final_weight = LOCAL | 8;
    assert!(packet.escaping_values().is_none());
}

#[test]
fn live_import_skips_only_unreferenced_values_and_preserves_exact_translation() {
    let mut reference = pool(1);
    let mut candidate = pool(1);
    let values = vec![smallvec::smallvec![0x123456789abcdef0],
        smallvec::smallvec![0x123456789abcdef1], smallvec::smallvec![u64::MAX]];
    let all = import_private_values(values.clone(), None, &mut reference).unwrap();
    let selected = import_private_values(values, Some(&[true, false, true]), &mut candidate).unwrap();
    assert_eq!(selected[1], u32::MAX);
    for index in [0usize, 2] {
        let id = translated(LOCAL | index as u32, &selected);
        assert_eq!(candidate.values[id as usize], reference.values[all[index] as usize]);
    }
    assert_eq!(translated(1, &selected), 1);
    assert!(!candidate.ids.contains_key([0x123456789abcdef1u64].as_slice()));
    assert_eq!(candidate.values.len() + 1, reference.values.len());
}
