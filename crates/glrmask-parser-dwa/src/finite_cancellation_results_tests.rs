use super::*;

fn pool(rows: usize) -> FastBoundaryWeightInterner {
    let mut pool = FastBoundaryWeightInterner::new(rows,64).unwrap();
    pool.limits = Some(Default::default());
    for pattern in [1u64,2,3,5,7,13,33] {
        pool.intern((0..rows).map(|row|pattern.rotate_left((row%64)as u32))
            .collect::<FastBoundaryWeightValue>());
    }
    pool
}

fn canonical(rows: &[FastBoundaryDerivedRow]) -> Vec<Vec<(u32,u32)>> {
    rows.iter().map(|row| {
        let mut result=Vec::new();row.for_each(|q,w|result.push((q,w)));
        result.sort_unstable_by_key(|&(q,_)|q);result
    }).collect()
}

#[test]
fn packed_handles_preserve_nested_empty_answers_and_shared_joins() {
    fn execute<R:ResultRows>() -> (Vec<Vec<(u32,u32)>>,SummaryStats,Vec<FastBoundaryWeightValue>) {
        let source=vec![
            FastBoundaryNwaState{final_weight:0,epsilons:vec![(1,1),(2,1)],transitions:vec![]},
            FastBoundaryNwaState{final_weight:0,epsilons:vec![(3,1)],transitions:vec![]},
            FastBoundaryNwaState{final_weight:0,epsilons:vec![(3,1)],transitions:vec![]},
            FastBoundaryNwaState{final_weight:0,epsilons:vec![(4,1)],transitions:vec![]},
            FastBoundaryNwaState{final_weight:0,epsilons:vec![],transitions:vec![(7,smallvec::smallvec![(5,2)])]},
            FastBoundaryNwaState{final_weight:1,epsilons:vec![],transitions:vec![]},
        ];
        let mut weights=pool(2);
        let mut solver=Solver::<_,R>{states:source.as_slice(),interner:&mut weights,
            effective:source.iter().map(|row|row.epsilons.clone()).collect(),
            known:FxHashMap::default(),results:R::new(),stack:Vec::new(),inflight_pairs:0,
            stats:Default::default(),may_read:Vec::new(),filter:false};
        let mut answers=Vec::new();
        for (state,label) in [(0,7),(0,7),(2,7),(0,11),(0,11)] {
            let handle=solver.query(state,label).unwrap();
            answers.push((0..solver.results.row_len(handle)).map(|i|solver.results.pair(handle,i)).collect());
            assert!(solver.stack.is_empty());assert_eq!(solver.inflight_pairs,0);
        }
        let stats=std::mem::take(&mut solver.stats);drop(solver);
        (answers,stats,weights.values)
    }
    let expected=execute::<VectorRows>();
    assert_eq!(expected,execute::<PackedRows>());
    assert_eq!(expected.0[0].len(),1);assert!(expected.0[3].is_empty());
    assert!(expected.1.max_stack>=4 && expected.1.cache_hits>2);
}

#[test]
fn packed_results_preserve_derived_rows_counters_and_numeric_weights() {
    let mut seed=647098u64;
    let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
    let mut queries=0;
    for case in 0..512 {
        let words=[1,2,17,64][case%4];
        let weight_count=pool(words).values.len();
        let n=4+next()%12;
        let mut source=Vec::new();
        for q in 0..n {
            let mut groups=BTreeMap::<i32,SmallVec<[(u32,u32);1]>>::new();
            let mut epsilons=Vec::new();
            for target in q+1..n {
                let weight=(next()%weight_count)as u32;
                let label=match next()%5 {
                    0=>{epsilons.push((target as u32,weight));continue;},
                    1=>crate::compiler::glr::labels::encode_negative_label((next()%5)as u32),
                    2=>DEFAULT_LABEL,_=>(next()%5)as i32,
                };
                let branches=groups.entry(label).or_default();
                branches.push((target as u32,weight));
                if next()%4==0{branches.push((target as u32,(next()%weight_count)as u32));}
            }
            if case%3==0{groups.entry(23).or_default();}
            if case%7==0{groups.entry(DEFAULT_LABEL).or_default().push((0,0));}
            source.push(FastBoundaryNwaState{epsilons,transitions:groups.into_iter().collect(),
                final_weight:(next()%weight_count)as u32});
        }
        let topo=fast_boundary_topological_order(&source).unwrap();
        for filter in [false,true] {
            let mut original=pool(words);
            let(expected,expected_stats)=compute_on_graph_policy(source.as_slice(),&mut original,
                &topo,filter,ResultPolicy::Vector).unwrap();
            queries+=expected_stats.queries;
            for policy in [ResultPolicy::Packed] {
                let mut candidate=pool(words);
                let(actual,stats)=compute_on_graph_policy(source.as_slice(),&mut candidate,
                    &topo,filter,policy).unwrap();
                assert_eq!(expected_stats,stats,"case={case} policy={policy:?}");
                assert_eq!(canonical(&expected),canonical(&actual));
                assert_eq!(original.values,candidate.values,"numeric weight sequence case={case}");
                assert_eq!(original.work,candidate.work);
                assert_eq!(original.failed,candidate.failed);
            }
            if case<24 {
                for work in [0,1,20,100] {
                    let mut reference=pool(words);
                    reference.limits.as_mut().unwrap().work=work;
                    let a=compute_on_graph_policy(source.as_slice(),&mut reference,&topo,filter,ResultPolicy::Vector);
                    for policy in [ResultPolicy::Packed] {
                        let mut candidate=pool(words);candidate.limits.as_mut().unwrap().work=work;
                        let b=compute_on_graph_policy(source.as_slice(),&mut candidate,&topo,filter,policy);
                        assert_eq!(a.is_some(),b.is_some());
                        assert_eq!(reference.values,candidate.values);
                        assert_eq!(reference.work,candidate.work);
                        assert_eq!(reference.failed,candidate.failed);
                    }
                }
            }
        }
    }
    assert!(queries>1000,"corpus must exercise cancellation queries");
}
