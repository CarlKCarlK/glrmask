//! Necessary terminal adjacency with component-local ignores made explicit.
//!
//! This is an analysis grammar, never the runtime parser grammar. Each local
//! production is padded with an ignore-star nonterminal belonging to that
//! production's component. A linked child remains one nonterminal in its
//! parent's rule, so parent ignores can occur before or after the child, not
//! arbitrarily inside its expansion. Ignored text inside the child is supplied
//! only by the child's own padding. Every real scoped-ignore execution has a
//! derivation in this over-approximation; adjacent terminal pairs absent from
//! it therefore cannot occur in that execution.
//!
//! The ordinary transparent-follow relation must still be enforced as well:
//! pairwise adjacency alone forgets a predecessor across an ignore sequence.
//! This additional relation is a rejection filter, not a mask-admission rule.

use std::collections::BTreeMap;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::grammar::flat::{Rule, Symbol};
use crate::ds::bitset::BitSet;
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::compute_ever_allowed_follows;

pub fn scoped_follow_relation(
    grammar: &AnalyzedGrammar,
    component_nonterminals: &[usize],
    ignores: &[Vec<u32>],
) -> Option<BTreeMap<u32, BitSet>> {
    if component_nonterminals.len() != ignores.len()
        || component_nonterminals.iter().try_fold(0usize, |a,b| a.checked_add(*b))?
            != grammar.num_nonterminals as usize
        || grammar.rules.is_empty()
    { return None; }
    if ignores.iter().flatten().any(|&t| t >= grammar.num_terminals) { return None; }
    let mut owner = Vec::with_capacity(grammar.num_nonterminals as usize);
    for (component,&count) in component_nonterminals.iter().enumerate() {
        owner.extend(std::iter::repeat_n(component,count));
    }
    let mut names = grammar.nonterminal_display_names.clone();
    let mut stars = Vec::with_capacity(ignores.len());
    for (component,labels) in ignores.iter().enumerate() {
        if labels.is_empty() { stars.push(None); }
        else {
            let id = u32::try_from(names.len()).ok()?;
            names.push(format!("<boundary-ignore-star:{component}>")); stars.push(Some(id));
        }
    }
    let mut rules = Vec::with_capacity(grammar.rules.len()+ignores.iter().map(Vec::len).sum::<usize>()+ignores.len());
    for rule in &grammar.rules {
        let component = *owner.get(rule.lhs as usize)?;
        let Some(star) = stars[component] else { rules.push(rule.clone()); continue; };
        let mut rhs = Vec::with_capacity(rule.rhs.len()*2+1);
        rhs.push(Symbol::Nonterminal(star));
        for symbol in &rule.rhs { rhs.push(symbol.clone()); rhs.push(Symbol::Nonterminal(star)); }
        rules.push(Rule { lhs:rule.lhs, rhs });
    }
    for (component,&star) in stars.iter().enumerate() {
        let Some(star) = star else { continue; };
        rules.push(Rule { lhs:star, rhs:Vec::new() });
        for &ignore in &ignores[component] {
            rules.push(Rule { lhs:star, rhs:vec![Symbol::Terminal(ignore),Symbol::Nonterminal(star)] });
        }
    }
    let analyzed = AnalyzedGrammar::from_composed_rules(rules,grammar.num_terminals,
        grammar.terminal_display_names.clone(),names,grammar.rules[0].lhs);
    let mut result = BTreeMap::new();
    for (terminal, allowed) in compute_ever_allowed_follows(&analyzed).into_iter().enumerate() {
        let mut bits = BitSet::new(grammar.num_terminals as usize);
        for successor in allowed {
            if successor < grammar.num_terminals { bits.set(successor as usize); }
        }
        let blocked = bits.complement();
        if !blocked.is_zero() { result.insert(terminal as u32, blocked); }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn allows(map:&BTreeMap<u32,BitSet>, a:u32,b:u32)->bool {
        !map.get(&a).is_some_and(|blocked|blocked.get(b as usize))
    }
    #[test]
    fn parent_ignore_is_not_inserted_inside_child_but_can_surround_it() {
        // Parent: '(' child ')'. Child: 'a' 'b'. Terminal zero is EOF.
        let rules=vec![
            Rule{lhs:0,rhs:vec![Symbol::Terminal(1),Symbol::Nonterminal(1),Symbol::Terminal(2)]},
            Rule{lhs:1,rhs:vec![Symbol::Terminal(3),Symbol::Terminal(4)]},
        ];
        let grammar=AnalyzedGrammar::from_composed_rules(rules,7,(0..7).map(|t|format!("T{t}")).collect(),vec!["P".into(),"C".into()],0);
        let relation=scoped_follow_relation(&grammar,&[1,1],&[vec![5],vec![6]]).unwrap();
        assert!(!allows(&relation,3,5)); // parent ignore cannot split a,b
        assert!(allows(&relation,3,6));
        assert!(allows(&relation,4,5)); // but it may follow child completion
        assert!(allows(&relation,1,5));
        assert!(allows(&relation,5,3));
        assert!(allows(&relation,6,4));
        assert!(allows(&relation,5,5));
        assert!(allows(&relation,6,6));
    }
    #[test]
    fn ignore_between_two_child_calls_needs_no_real_parent_terminal() {
        let rules=vec![
            Rule{lhs:0,rhs:vec![Symbol::Nonterminal(1),Symbol::Nonterminal(2)]},
            Rule{lhs:1,rhs:vec![Symbol::Terminal(1)]},
            Rule{lhs:2,rhs:vec![Symbol::Terminal(2)]},
        ];
        let grammar=AnalyzedGrammar::from_composed_rules(rules,4,(0..4).map(|t|format!("T{t}")).collect(),vec!["P".into(),"A".into(),"B".into()],0);
        let relation=scoped_follow_relation(&grammar,&[1,1,1],&[vec![3],Vec::new(),Vec::new()]).unwrap();
        assert!(allows(&relation,1,3)); assert!(allows(&relation,3,2));
        assert!(allows(&relation,1,2));
    }
}
