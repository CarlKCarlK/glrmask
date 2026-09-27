use crate::grammar::flat::Symbol as MbFactorSourceSymbol;

// Test-only exact bounded grammar-factor oracle.
//
// The factor/sub-string language of a CFG is context-free in general, not a
// finite-state language.  The exact B-A boundary residual used by this oracle
// is acyclic, however, with a tiny finite maximum terminal length.  We exploit
// that: compute the CFG's exact Full/Prefix/Suffix/Factor languages only for
// terminal words that can occur as fragments of a residual path, truncated at
// the residual's maximum length.  The final filter is then a lazy DFA whose
// state is simply the residual state plus the already-read terminal prefix.
// This is the requested "n = infinity" substring filter for every word that the
// boundary DWA can actually contain, without materializing an unbounded factor
// automaton.

const MB_FACTOR_WORD_CAP: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct MbFactorWord {
    len: u8,
    labels: [u16; MB_FACTOR_WORD_CAP],
}

impl MbFactorWord {
    const EMPTY: Self = Self {
        len: 0,
        labels: [0; MB_FACTOR_WORD_CAP],
    };

    fn singleton(terminal: u32) -> Self {
        assert!(terminal <= u16::MAX as u32);
        let mut labels = [0; MB_FACTOR_WORD_CAP];
        labels[0] = terminal as u16;
        Self { len: 1, labels }
    }

    fn pushed(self, terminal: u32, max_len: usize) -> Option<Self> {
        if self.len as usize >= max_len || self.len as usize >= MB_FACTOR_WORD_CAP {
            return None;
        }
        assert!(terminal <= u16::MAX as u32);
        let mut out = self;
        out.labels[out.len as usize] = terminal as u16;
        out.len += 1;
        Some(out)
    }

    fn concat(self, other: Self, max_len: usize) -> Option<Self> {
        let len = self.len as usize + other.len as usize;
        if len > max_len || len > MB_FACTOR_WORD_CAP {
            return None;
        }
        let mut out = self;
        for index in 0..other.len as usize {
            out.labels[self.len as usize + index] = other.labels[index];
        }
        out.len = len as u8;
        Some(out)
    }
}

fn mb_factor_residual_max_depth(residual: &DWA) -> usize {
    fn visit(state: u32, residual: &DWA, memo: &mut [Option<usize>]) -> usize {
        if let Some(depth) = memo[state as usize] {
            return depth;
        }
        let row = &residual.states()[state as usize];
        let mut best = if row
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            0
        } else {
            0
        };
        for (_, (target, _)) in &row.transitions {
            best = best.max(1 + visit(*target, residual, memo));
        }
        memo[state as usize] = Some(best);
        best
    }
    let mut memo = vec![None; residual.num_states() as usize];
    visit(residual.start_state(), residual, &mut memo)
}

fn mb_factor_candidate_fragments(
    residual: &DWA,
    max_len: usize,
) -> FxHashSet<MbFactorWord> {
    fn coreachable(state: u32, residual: &DWA, memo: &mut [Option<bool>]) -> bool {
        if let Some(value) = memo[state as usize] {
            return value;
        }
        let row = &residual.states()[state as usize];
        let value = row
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
            || row
                .transitions
                .values()
                .any(|(target, _)| coreachable(*target, residual, memo));
        memo[state as usize] = Some(value);
        value
    }

    let mut coreach_memo = vec![None; residual.num_states() as usize];
    let mut coreach = vec![false; residual.num_states() as usize];
    for state in 0..residual.num_states() {
        coreach[state as usize] = coreachable(state, residual, &mut coreach_memo);
    }
    let mut reachable = vec![false; residual.num_states() as usize];
    let mut queue = VecDeque::from([residual.start_state()]);
    while let Some(state) = queue.pop_front() {
        if std::mem::replace(&mut reachable[state as usize], true) {
            continue;
        }
        for (target, _) in residual.states()[state as usize].transitions.values() {
            if coreach[*target as usize] && !reachable[*target as usize] {
                queue.push_back(*target);
            }
        }
    }

    fn enumerate(
        residual: &DWA,
        state: u32,
        word: MbFactorWord,
        max_len: usize,
        coreach: &[bool],
        out: &mut FxHashSet<MbFactorWord>,
    ) {
        if word.len as usize >= max_len {
            return;
        }
        for (&label, (target, _)) in &residual.states()[state as usize].transitions {
            if label < 0 || !coreach[*target as usize] {
                continue;
            }
            let Some(next) = word.pushed(label as u32, max_len) else {
                continue;
            };
            out.insert(next);
            enumerate(residual, *target, next, max_len, coreach, out);
        }
    }

    let mut fragments = FxHashSet::default();
    fragments.insert(MbFactorWord::EMPTY);
    for state in 0..residual.num_states() {
        if reachable[state as usize] && coreach[state as usize] {
            enumerate(
                residual,
                state,
                MbFactorWord::EMPTY,
                max_len,
                &coreach,
                &mut fragments,
            );
        }
    }
    fragments
}

fn mb_factor_source_properties(
    grammar: &AnalyzedGrammar,
    zero_width: &std::collections::BTreeSet<u32>,
) -> (Vec<bool>, Vec<bool>, Vec<bool>, Vec<bool>) {
    let mut productive = vec![false; grammar.num_nonterminals as usize];
    let mut productive_rules = vec![false; grammar.rules.len()];
    loop {
        let mut changed = false;
        for (rule_index, rule) in grammar.rules.iter().enumerate() {
            let yes = rule.rhs.iter().all(|symbol| match symbol {
                MbFactorSourceSymbol::Terminal(_) => true,
                MbFactorSourceSymbol::Nonterminal(nonterminal) => productive
                    .get(*nonterminal as usize)
                    .copied()
                    .unwrap_or(false),
            });
            if yes && !productive_rules[rule_index] {
                productive_rules[rule_index] = true;
                changed = true;
            }
            if yes
                && let Some(slot) = productive.get_mut(rule.lhs as usize)
                && !*slot
            {
                *slot = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut nullable = vec![false; grammar.num_nonterminals as usize];
    loop {
        let mut changed = false;
        for (rule_index, rule) in grammar.rules.iter().enumerate() {
            if !productive_rules[rule_index] {
                continue;
            }
            let yes = rule.rhs.iter().all(|symbol| match symbol {
                MbFactorSourceSymbol::Terminal(terminal) => zero_width.contains(terminal),
                MbFactorSourceSymbol::Nonterminal(nonterminal) => nullable
                    .get(*nonterminal as usize)
                    .copied()
                    .unwrap_or(false),
            });
            if yes
                && let Some(slot) = nullable.get_mut(rule.lhs as usize)
                && !*slot
            {
                *slot = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let start = grammar.rules.first().expect("augmented start").lhs;
    let mut reachable = vec![false; grammar.num_nonterminals as usize];
    let mut queue = VecDeque::from([start]);
    while let Some(nonterminal) = queue.pop_front() {
        let Some(slot) = reachable.get_mut(nonterminal as usize) else {
            continue;
        };
        if std::mem::replace(slot, true) {
            continue;
        }
        for &rule_index in grammar
            .rules_by_lhs
            .get(nonterminal as usize)
            .into_iter()
            .flatten()
        {
            if !productive_rules.get(rule_index as usize).copied().unwrap_or(false) {
                continue;
            }
            for symbol in &grammar.rules[rule_index as usize].rhs {
                if let MbFactorSourceSymbol::Nonterminal(child) = symbol
                    && productive.get(*child as usize).copied().unwrap_or(false)
                    && !reachable.get(*child as usize).copied().unwrap_or(false)
                {
                    queue.push_back(*child);
                }
            }
        }
    }
    (productive, nullable, productive_rules, reachable)
}

#[derive(Clone, Default)]
struct MbFactorSummary {
    productive: bool,
    nullable: bool,
    complete: FxHashSet<MbFactorWord>,
    prefix: FxHashSet<MbFactorWord>,
    suffix: FxHashSet<MbFactorWord>,
    factor: FxHashSet<MbFactorWord>,
}

impl MbFactorSummary {
    fn productive_base(nullable: bool) -> Self {
        let mut result = Self {
            productive: true,
            nullable,
            ..Default::default()
        };
        if nullable {
            result.complete.insert(MbFactorWord::EMPTY);
        }
        result.prefix.insert(MbFactorWord::EMPTY);
        result.suffix.insert(MbFactorWord::EMPTY);
        result.factor.insert(MbFactorWord::EMPTY);
        result
    }
}

fn mb_factor_terminal_summary(
    terminal: u32,
    zero_width: &std::collections::BTreeSet<u32>,
    fragments: &FxHashSet<MbFactorWord>,
) -> MbFactorSummary {
    if zero_width.contains(&terminal) {
        return MbFactorSummary::productive_base(true);
    }
    let mut result = MbFactorSummary::productive_base(false);
    let word = MbFactorWord::singleton(terminal);
    if fragments.contains(&word) {
        result.complete.insert(word);
        result.prefix.insert(word);
        result.suffix.insert(word);
        result.factor.insert(word);
    }
    result
}

fn mb_factor_insert_concat(
    out: &mut FxHashSet<MbFactorWord>,
    left: &FxHashSet<MbFactorWord>,
    right: &FxHashSet<MbFactorWord>,
    max_len: usize,
    fragments: &FxHashSet<MbFactorWord>,
) {
    for &a in left {
        for &b in right {
            let Some(word) = a.concat(b, max_len) else {
                continue;
            };
            if word.len == 0 || fragments.contains(&word) {
                out.insert(word);
            }
        }
    }
}

fn mb_factor_concat_summary(
    left: &MbFactorSummary,
    right: &MbFactorSummary,
    max_len: usize,
    fragments: &FxHashSet<MbFactorWord>,
) -> MbFactorSummary {
    if !left.productive || !right.productive {
        return MbFactorSummary::default();
    }
    let mut result = MbFactorSummary::productive_base(left.nullable && right.nullable);
    mb_factor_insert_concat(
        &mut result.complete,
        &left.complete,
        &right.complete,
        max_len,
        fragments,
    );

    result.prefix.extend(left.prefix.iter().copied());
    mb_factor_insert_concat(
        &mut result.prefix,
        &left.complete,
        &right.prefix,
        max_len,
        fragments,
    );

    result.suffix.extend(right.suffix.iter().copied());
    mb_factor_insert_concat(
        &mut result.suffix,
        &left.suffix,
        &right.complete,
        max_len,
        fragments,
    );

    result.factor.extend(left.factor.iter().copied());
    result.factor.extend(right.factor.iter().copied());
    mb_factor_insert_concat(
        &mut result.factor,
        &left.suffix,
        &right.prefix,
        max_len,
        fragments,
    );
    result
}

fn mb_factor_union_into(target: &mut MbFactorSummary, source: MbFactorSummary) -> bool {
    let before = (
        target.complete.len(),
        target.prefix.len(),
        target.suffix.len(),
        target.factor.len(),
    );
    target.productive |= source.productive;
    target.nullable |= source.nullable;
    target.complete.extend(source.complete);
    target.prefix.extend(source.prefix);
    target.suffix.extend(source.suffix);
    target.factor.extend(source.factor);
    before
        != (
            target.complete.len(),
            target.prefix.len(),
            target.suffix.len(),
            target.factor.len(),
        )
}

fn mb_exact_bounded_factor_words(
    grammar: &AnalyzedGrammar,
    zero_width: &std::collections::BTreeSet<u32>,
    fragments: &FxHashSet<MbFactorWord>,
    max_len: usize,
) -> FxHashSet<MbFactorWord> {
    let started = Instant::now();
    let (productive, nullable, productive_rules, reachable) =
        mb_factor_source_properties(grammar, zero_width);
    let mut summaries = (0..grammar.num_nonterminals as usize)
        .map(|nonterminal| {
            if productive[nonterminal] {
                MbFactorSummary::productive_base(nullable[nonterminal])
            } else {
                MbFactorSummary::default()
            }
        })
        .collect::<Vec<_>>();
    let mut terminal_cache = FxHashMap::<u32, MbFactorSummary>::default();
    let mut iterations = 0usize;
    loop {
        iterations += 1;
        let mut changed = false;
        for (rule_index, rule) in grammar.rules.iter().enumerate() {
            if !productive_rules[rule_index]
                || !reachable.get(rule.lhs as usize).copied().unwrap_or(false)
            {
                continue;
            }
            let mut sequence = MbFactorSummary::productive_base(true);
            for symbol in &rule.rhs {
                let symbol_summary = match symbol {
                    MbFactorSourceSymbol::Terminal(terminal) => terminal_cache
                        .entry(*terminal)
                        .or_insert_with(|| {
                            mb_factor_terminal_summary(*terminal, zero_width, fragments)
                        })
                        .clone(),
                    MbFactorSourceSymbol::Nonterminal(nonterminal) => {
                        summaries[*nonterminal as usize].clone()
                    }
                };
                sequence = mb_factor_concat_summary(
                    &sequence,
                    &symbol_summary,
                    max_len,
                    fragments,
                );
                if !sequence.productive {
                    break;
                }
            }
            changed |= mb_factor_union_into(&mut summaries[rule.lhs as usize], sequence);
        }
        if !changed {
            break;
        }
        assert!(iterations < 256, "bounded factor summary did not converge");
        if iterations <= 4 || iterations % 10 == 0 {
            let factor_cells = summaries.iter().map(|summary| summary.factor.len()).sum::<usize>();
            eprintln!(
                "MINBOUND bounded_factor_iteration iteration={} factor_cells={} ms={:.3}",
                iterations,
                factor_cells,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let start = grammar.rules.first().expect("augmented start").lhs as usize;
    let result = summaries[start].factor.clone();
    eprintln!(
        "MINBOUND bounded_factor_summary max_len={} fragments={} valid_factors={} iterations={} productive_nts={} reachable_nts={} productive_rules={} complete_cells={} prefix_cells={} suffix_cells={} factor_cells={} ms={:.3}",
        max_len,
        fragments.len(),
        result.len(),
        iterations,
        productive.iter().filter(|&&yes| yes).count(),
        reachable.iter().filter(|&&yes| yes).count(),
        productive_rules.iter().filter(|&&yes| yes).count(),
        summaries.iter().map(|summary| summary.complete.len()).sum::<usize>(),
        summaries.iter().map(|summary| summary.prefix.len()).sum::<usize>(),
        summaries.iter().map(|summary| summary.suffix.len()).sum::<usize>(),
        summaries.iter().map(|summary| summary.factor.len()).sum::<usize>(),
        started.elapsed().as_secs_f64() * 1000.0,
    );
    result
}

fn mb_filter_exact_factor_lazy(
    residual: &DWA,
    grammar: &AnalyzedGrammar,
    zero_width: &std::collections::BTreeSet<u32>,
) -> DWA {
    let started = Instant::now();
    let max_len = mb_factor_residual_max_depth(residual);
    assert!(
        max_len <= MB_FACTOR_WORD_CAP,
        "bounded factor oracle needs a larger word cap: {max_len}",
    );
    let fragment_started = Instant::now();
    let fragments = mb_factor_candidate_fragments(residual, max_len);
    let fragment_ms = fragment_started.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "MINBOUND bounded_factor_fragments max_len={} fragments={} ms={fragment_ms:.3}",
        max_len,
        fragments.len(),
    );
    let valid = mb_exact_bounded_factor_words(grammar, zero_width, &fragments, max_len);

    let mut states = vec![DWAState::default()];
    let mut payloads = vec![(residual.start_state(), MbFactorWord::EMPTY)];
    let mut ids = FxHashMap::<(u32, MbFactorWord), u32>::default();
    ids.insert(payloads[0], 0);
    let mut queue = VecDeque::from([0u32]);
    let mut rejected = 0usize;
    while let Some(output_state) = queue.pop_front() {
        let (residual_state, word) = payloads[output_state as usize];
        let row = &residual.states()[residual_state as usize];
        if row
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
            && (word.len == 0 || valid.contains(&word))
        {
            states[output_state as usize].final_weight = row.final_weight.clone();
        }
        for (&label, (target, weight)) in &row.transitions {
            assert!(label >= 0 && label != DEFAULT_LABEL);
            let Some(next_word) = word.pushed(label as u32, max_len) else {
                continue;
            };
            if !valid.contains(&next_word) {
                // Factor languages are prefix-closed, so no extension of an
                // invalid prefix can later become a valid factor.
                rejected += 1;
                continue;
            }
            let key = (*target, next_word);
            let target_output = if let Some(&existing) = ids.get(&key) {
                existing
            } else {
                let id = states.len() as u32;
                ids.insert(key, id);
                states.push(DWAState::default());
                payloads.push(key);
                queue.push_back(id);
                id
            };
            states[output_state as usize]
                .transitions
                .insert(label, (target_output, weight.clone()));
        }
    }
    let raw = DWA::from_parts(states, 0);
    let raw_states = raw.num_states();
    let raw_transitions = raw.num_transitions();
    let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
    eprintln!(
        "MINBOUND lazy_factor_filter raw_states={} raw_transitions={} states={} transitions={} rejected_prefix_edges={} total_ms={:.3}",
        raw_states,
        raw_transitions,
        minimized.num_states(),
        minimized.num_transitions(),
        rejected,
        started.elapsed().as_secs_f64() * 1000.0,
    );
    minimized
}
