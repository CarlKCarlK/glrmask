//! Deterministic predecessor-domain representation shared with the exact diagnostic checker.
use std::collections::{BTreeMap,BTreeSet};
#[derive(Clone,Debug,Default)]
pub struct Node {
    pub transitions:BTreeMap<u32,BTreeSet<u32>>,
    pub epsilons:BTreeSet<u32>,
    pub accepting:bool,
}

#[derive(Clone,Debug)]
pub struct StackDomain {
    pub nodes:Vec<Node>,
    pub start:u32,
    pub alphabet:u32,
}

impl StackDomain {
    pub fn from_words(words:&[Vec<u32>],alphabet:u32)->Result<Self,String> {
        if alphabet==0{return Err("empty parser-state alphabet".into());}
        let mut domain=Self{nodes:vec![Node::default()],start:0,alphabet};
        for word in words {
            let mut current=0usize;
            for &label in word {
                if label>=alphabet{return Err("initial stack symbol outside alphabet".into());}
                // An initial trie is sufficient; saturation may introduce NFA
                // branching later, but this constructor does not mutate it.
                let target=match domain.nodes[current].transitions.get(&label).and_then(|row|row.first()).copied(){
                    Some(target)=>target,
                    None=>{let target=domain.nodes.len() as u32;domain.nodes.push(Node::default());domain.nodes[current].transitions.entry(label).or_default().insert(target);target},
                };
                current=target as usize;
            }
            domain.nodes[current].accepting=true;
        }
        Ok(domain)
    }

    pub fn closure(&self,states:impl IntoIterator<Item=u32>)->Vec<u32> {
        let mut seen=BTreeSet::new();let mut todo=Vec::new();
        for state in states{if seen.insert(state){todo.push(state);}}
        while let Some(state)=todo.pop(){
            for &target in &self.nodes[state as usize].epsilons{if seen.insert(target){todo.push(target);}}
        }
        seen.into_iter().collect()
    }

    pub fn step(&self,states:&[u32],label:u32)->Vec<u32> {
        let mut next=BTreeSet::new();
        for state in self.closure(states.iter().copied()) {
            if let Some(targets)=self.nodes[state as usize].transitions.get(&label){next.extend(targets.iter().copied());}
        }
        self.closure(next)
    }

    pub fn accepts(&self,word:&[u32])->bool {
        let mut states=self.closure([self.start]);
        for &label in word{states=self.step(&states,label);}
        states.iter().any(|&q|self.nodes[q as usize].accepting)
    }

    pub fn coaccessible(&self)->Vec<bool> {
        let mut reverse=vec![Vec::new();self.nodes.len()];
        for (source,node) in self.nodes.iter().enumerate(){
            for &target in node.epsilons.iter().chain(node.transitions.values().flatten()){
                reverse[target as usize].push(source);
            }
        }
        let mut live=vec![false;self.nodes.len()];let mut todo=Vec::new();
        for (q,node) in self.nodes.iter().enumerate(){if node.accepting{live[q]=true;todo.push(q);}}
        while let Some(target)=todo.pop(){for &source in &reverse[target]{if !live[source]{live[source]=true;todo.push(source);}}}
        live
    }

    pub fn validate(&self)->Result<(),String> {
        if self.alphabet==0||self.nodes.is_empty()||self.start as usize>=self.nodes.len(){return Err("invalid domain header".into());}
        for node in &self.nodes {
            if node.transitions.keys().any(|&label|label>=self.alphabet){return Err("domain label outside alphabet".into());}
            if node.epsilons.iter().chain(node.transitions.values().flatten()).any(|&target|target as usize>=self.nodes.len()){
                return Err("domain transition outside state set".into());
            }
        }
        Ok(())
    }
}
