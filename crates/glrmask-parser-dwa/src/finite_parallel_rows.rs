//! Pure weighted-row preparation against an immutable interner snapshot.
//!
//! Workers never allocate deterministic state IDs. FIFO publication below keeps
//! the serial normalizer's frontier identity and DEFAULT observations intact.
use super::*;
use rayon::prelude::*;

const LOCAL: u32 = 1 << 31;
const MAX_LOCAL_VALUES: usize = 8192;
const MAX_PACKET_MEMBERS: usize = 250_000;
const MAX_PACKET_TRANSITIONS: usize = 65_536;
const MAX_PACKET_WORK: usize = 4_000_000;

type Pairs = Vec<(u32, FastBoundaryWeightId)>;
type ClosureCache = FxHashMap<Pairs, (u32, FastBoundaryWeightId)>;
type SingletonCache = FxHashMap<(u32, FastBoundaryWeightId), (u32, FastBoundaryWeightId)>;

#[derive(Clone, Copy)]
pub(super) struct Policy {
    pub threshold: usize,
    pub batch: usize,
    pub chunk: usize,
    pub live_import: bool,
}

impl Policy {
    pub(super) fn from_environment(finite: bool, states: usize, alphabet: usize) -> Option<Self> {
        let mut policy = Self::for_configuration(finite, states, alphabet, rayon::current_num_threads(),
            crate::optimized_env_flag("GLRMASK_BOUNDARY_PARALLEL_NATIVE_ROWS"))?;
        // Import only values referenced by the completed packet. The exact
        // all-values reference remains available with the ordinary override.
        policy.live_import = crate::optimized_env_flag("GLRMASK_BOUNDARY_PARALLEL_LIVE_IMPORT");
        Some(policy)
    }

    // The two-worker real-fixture screen regressed. Small inputs also cannot
    // amortize worker-local scratch and packet handoff. Keep their exact serial
    // implementation rather than changing the caller's Rayon pool.
    fn for_configuration(finite: bool, states: usize, alphabet: usize,
                         threads: usize, enabled: bool) -> Option<Self> {
        (enabled && finite && threads >= 4 && (4096..=200_000).contains(&states)
            && alphabet <= 32_768)
            .then_some(Self { threshold: 64, batch: 4096, chunk: 256, live_import: true })
    }
}

#[derive(Default, Debug)]
pub(super) struct Profile {
    pub batches: usize,
    pub packets: usize,
    pub rows: usize,
    pub recipes: usize,
    pub base_cache_hits: usize,
    pub local_cache_hits: usize,
    pub local_values: usize,
    pub imported_values: usize,
    pub ready_edges: usize,
    pub worker_mask: u64,
    pub prepare_ms: f64,
    pub publish_ms: f64,
    pub import_ms: f64,
}

struct Overlay<'a> {
    base: &'a FastBoundaryWeightInterner,
    values: Vec<FastBoundaryWeightValue>,
    ids: FxHashMap<FastBoundaryWeightValue, u32>,
    intersections: FxHashMap<(u32, u32), u32>,
    unions: FxHashMap<(u32, u32), u32>,
    limit: usize,
    failed: bool,
}

impl<'a> Overlay<'a> {
    fn new(base: &'a FastBoundaryWeightInterner) -> Option<Self> {
        if base.values.len() >= LOCAL as usize || base.tsid_count > 64 { return None; }
        Some(Self { base, values: Vec::new(), ids: FxHashMap::default(),
            intersections: FxHashMap::default(), unions: FxHashMap::default(),
            limit: MAX_LOCAL_VALUES, failed: false })
    }

    #[inline]
    fn value(&self, id: u32) -> &[u64] {
        if id & LOCAL == 0 { &self.base.values[id as usize] }
        else { &self.values[(id & !LOCAL) as usize] }
    }

    fn intern(&mut self, bits: &[u64]) -> u32 {
        if let Some(&id) = self.base.ids.get(bits) { return id; }
        if let Some(&id) = self.ids.get(bits) { return id; }
        if self.values.len() >= self.limit { self.failed = true; return 0; }
        let id = LOCAL | self.values.len() as u32;
        let owned = FastBoundaryWeightValue::from_slice(bits);
        self.ids.insert(owned.clone(), id);
        self.values.push(owned);
        id
    }

    #[inline]
    fn intersection(&mut self, a: u32, b: u32) -> u32 {
        if a == 0 || b == 0 { return 0; }
        if a == 1 { return b; }
        if b == 1 || a == b { return a; }
        let key = if a <= b { (a, b) } else { (b, a) };
        if b & LOCAL == 0 && a & LOCAL == 0 {
            if let Some(&id) = self.base.intersections.get(&key) { return id; }
        }
        if let Some(&id) = self.intersections.get(&key) { return id; }
        let mut bits = [0u64; 64];
        for (slot, (&x, &y)) in bits.iter_mut().zip(self.value(a).iter().zip(self.value(b))) {
            *slot = x & y;
        }
        let id = self.intern(&bits[..self.base.tsid_count]);
        if self.intersections.len() < 262_144 { self.intersections.insert(key, id); }
        id
    }

    #[inline]
    fn union(&mut self, a: u32, b: u32) -> u32 {
        if a == 0 { return b; }
        if b == 0 || a == b { return a; }
        if a == 1 || b == 1 { return 1; }
        let key = if a <= b { (a, b) } else { (b, a) };
        if b & LOCAL == 0 && a & LOCAL == 0 {
            if let Some(&id) = self.base.unions.get(&key) { return id; }
        }
        if let Some(&id) = self.unions.get(&key) { return id; }
        let mut bits = [0u64; 64];
        for (slot, (&x, &y)) in bits.iter_mut().zip(self.value(a).iter().zip(self.value(b))) {
            *slot = x | y;
        }
        let id = self.intern(&bits[..self.base.tsid_count]);
        if self.unions.len() < 262_144 { self.unions.insert(key, id); }
        id
    }

    #[inline]
    fn is_subset(&self, a: u32, b: u32) -> bool {
        if a == 0 || b == 1 || a == b { return true; }
        if b == 0 { return false; }
        self.value(a).iter().zip(self.value(b)).all(|(&x, &y)| x & !y == 0)
    }
}

enum Target {
    Known(u32, u32),
    Singleton(u32, u32),
    Recipe(usize),
}

struct Recipe { incoming: Pairs, closure: Pairs, weight: u32 }
struct PreparedRow {
    id: u32,
    members: usize,
    final_weight: u32,
    edges: Vec<(i32, u32, u32)>,
    pending: Vec<(usize, Target)>,
    local_coefficients: bool,
}
struct Packet {
    values: Vec<FastBoundaryWeightValue>,
    recipes: Vec<Recipe>,
    rows: Vec<PreparedRow>,
    base_hits: usize,
    local_hits: usize,
    worker: u64,
}

impl Packet {
    /// Mark every coefficient that can escape into the ordered publisher.
    /// Values are materialized bitvectors, so arithmetic operands do not need
    /// recursive tracing. Keeping every recipe is conservative when a later
    /// cache hit makes some of them unnecessary.
    fn escaping_values(&self) -> Option<Vec<bool>> {
        let mut live = vec![false; self.values.len()];
        let mut mark = |id: u32| -> Option<()> {
            if id & LOCAL != 0 { *live.get_mut((id & !LOCAL) as usize)? = true; }
            Some(())
        };
        for recipe in &self.recipes {
            mark(recipe.weight)?;
            for &(_, weight) in recipe.incoming.iter().chain(&recipe.closure) { mark(weight)?; }
        }
        for row in &self.rows {
            mark(row.final_weight)?;
            for &(_, _, weight) in &row.edges { mark(weight)?; }
            for (_, target) in &row.pending {
                match target {
                    Target::Known(_, weight) | Target::Singleton(_, weight) => mark(*weight)?,
                    Target::Recipe(index) => { self.recipes.get(*index)?; }
                }
            }
        }
        Some(live)
    }
}

fn import_private_values(
    values: Vec<FastBoundaryWeightValue>, live: Option<&[bool]>,
    interner: &mut FastBoundaryWeightInterner,
) -> Option<Vec<u32>> {
    if live.is_some_and(|marks| marks.len() != values.len()) { return None; }
    let mut translated = Vec::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        if live.is_none_or(|marks| marks[index]) {
            let id = interner.intern(value);
            if interner.failed { return None; }
            translated.push(id);
        } else {
            // Unmarked temporaries must never be confused with EMPTY.
            translated.push(u32::MAX);
        }
    }
    Some(translated)
}

struct Worker<'a> {
    nwa: &'a [FastBoundaryNwaState],
    pool: Overlay<'a>,
    singletons: &'a SingletonCache,
    singleton_states: &'a [u32],
    closures: &'a ClosureCache,
    local_closures: FxHashMap<Pairs, usize>,
    recipes: Vec<Recipe>,
    by_state: Vec<u32>,
    queue: VecDeque<u32>,
    touched: Vec<u32>,
    retained_members: usize,
    work: usize,
    base_hits: usize,
    local_hits: usize,
}

impl Worker<'_> {
    fn closure(&mut self, seed: &[(u32, u32)]) -> Option<Pairs> {
        self.touched.clear(); self.queue.clear();
        for &(q, weight) in seed {
            if weight == 0 { continue; }
            let old = self.by_state[q as usize];
            if old == 0 {
                self.by_state[q as usize] = weight;
                self.touched.push(q); self.queue.push_back(q);
            } else {
                let merged = self.pool.union(old, weight);
                if merged != old { self.by_state[q as usize] = merged; self.queue.push_back(q); }
            }
        }
        while let Some(q) = self.queue.pop_front() {
            self.work += 1 + self.nwa[q as usize].epsilons.len();
            if self.work > MAX_PACKET_WORK || self.pool.failed { return None; }
            let current = self.by_state[q as usize];
            for &(target, weight) in &self.nwa[q as usize].epsilons {
                let add = self.pool.intersection(current, weight);
                if add == 0 { continue; }
                let old = self.by_state[target as usize];
                if old == 0 {
                    self.by_state[target as usize] = add;
                    self.touched.push(target); self.queue.push_back(target);
                } else if !self.pool.is_subset(add, old) {
                    let merged = self.pool.union(old, add);
                    if merged != old {
                        self.by_state[target as usize] = merged; self.queue.push_back(target);
                    }
                }
            }
        }
        self.touched.sort_unstable();
        let mut result = Vec::with_capacity(self.touched.len());
        for &q in &self.touched {
            let weight = std::mem::replace(&mut self.by_state[q as usize], 0);
            if weight != 0 { result.push((q, weight)); }
        }
        Some(result)
    }

    fn target(&mut self, input: &mut FastBoundaryContribs) -> Option<Option<Target>> {
        if input.is_empty() { return Some(None); }
        input.sort_unstable_by_key(|&(q, _)| q);
        let mut written = 0usize;
        for index in 0..input.len() {
            let (q, weight) = input[index];
            if written != 0 && input[written - 1].0 == q {
                input[written - 1].1 = self.pool.union(input[written - 1].1, weight);
            } else { input[written] = (q, weight); written += 1; }
        }
        input.truncate(written);
        if let [(q, weight)] = input.as_slice() {
            if self.nwa[*q as usize].epsilons.is_empty() {
                let existing = self.singleton_states[*q as usize];
                if existing != u32::MAX {
                    return Some(Some(Target::Known(existing, *weight)));
                }
                return Some(Some(Target::Singleton(*q, *weight)));
            }
            if weight & LOCAL == 0 {
                if let Some(&(target, coefficient)) = self.singletons.get(&(*q, *weight)) {
                    self.base_hits += 1; return Some(Some(Target::Known(target, coefficient)));
                }
            }
        } else if input.iter().all(|&(_, weight)| weight & LOCAL == 0) {
            if let Some(&(target, coefficient)) = self.closures.get(input.as_slice()) {
                self.base_hits += 1; return Some(Some(Target::Known(target, coefficient)));
            }
        }
        if let Some(&id) = self.local_closures.get(input.as_slice()) {
            self.local_hits += 1; return Some(Some(Target::Recipe(id)));
        }
        let mut weight = 0;
        for &(_, w) in input.iter() { weight = self.pool.union(weight, w); }
        if weight == 0 { return Some(None); }
        let closure = self.closure(input.as_slice())?;
        if closure.is_empty() { return Some(None); }
        self.retained_members += input.len() + closure.len();
        if self.retained_members > MAX_PACKET_MEMBERS || self.pool.failed { return None; }
        let id = self.recipes.len();
        let incoming = input.to_vec();
        self.local_closures.insert(incoming.clone(), id);
        self.recipes.push(Recipe { incoming, closure, weight });
        Some(Some(Target::Recipe(id)))
    }
}

fn prepare_packet(
    batch: &[(u32, Pairs)], nwa: &[FastBoundaryNwaState],
    base: &FastBoundaryWeightInterner, singletons: &SingletonCache,
    closures: &ClosureCache, singleton_states: &[u32], dense_limit: usize,
) -> Option<Packet> {
    let mut worker = Worker { nwa, pool: Overlay::new(base)?, singletons, singleton_states, closures,
        local_closures: FxHashMap::default(), recipes: Vec::new(),
        by_state: vec![0; nwa.len()], queue: VecDeque::new(), touched: Vec::new(),
        retained_members: 0, work: 0, base_hits: 0, local_hits: 0 };
    let mut dense = vec![FastBoundaryContribs::new(); dense_limit];
    let mut marked = vec![false; dense_limit];
    let mut touched = Vec::new();
    let mut default = FastBoundaryContribs::new();
    let mut sparse = FxHashMap::<i32, FastBoundaryContribs>::default();
    let mut rows = Vec::with_capacity(batch.len());
    let mut transitions = 0usize;
    for (id, subset) in batch {
        let mut final_weight = 0;
        for &(q, coefficient) in subset {
            let final_mask = nwa[q as usize].final_weight;
            if final_mask != 0 {
                let add = worker.pool.intersection(coefficient, final_mask);
                final_weight = worker.pool.union(final_weight, add);
            }
        }
        for &(q, coefficient) in subset {
            for (label, branches) in &nwa[q as usize].transitions {
                worker.work += branches.len();
                for &(target, weight) in branches {
                    let add = worker.pool.intersection(coefficient, weight);
                    if add == 0 { continue; }
                    if *label >= 0 && (*label as usize) < dense_limit {
                        let index = *label as usize;
                        if !marked[index] { marked[index] = true; touched.push(index); }
                        dense[index].push((target, add));
                    } else if *label == DEFAULT_LABEL { default.push((target, add)); }
                    else { sparse.entry(*label).or_default().push((target, add)); }
                }
            }
        }
        if worker.pool.failed || worker.work > MAX_PACKET_WORK { return None; }
        touched.sort_unstable();
        let mut edges = Vec::with_capacity(touched.len() + sparse.len() + usize::from(!default.is_empty()));
        let mut pending = Vec::new();
        let mut local_coefficients = false;
        // Construct the final vector in the worker. Only unresolved identities
        // need serial handling; known edge payloads are moved, not rebuilt.
        let mut append = |label, target| {
            let index = edges.len();
            if let Target::Known(q, weight) = target {
                local_coefficients |= weight & LOCAL != 0;
                edges.push((label, q, weight));
            } else {
                edges.push((label, u32::MAX, 0));
                pending.push((index, target));
            }
        };
        for label in touched.drain(..) {
            marked[label] = false;
            if let Some(target) = worker.target(&mut dense[label])? { append(label as i32, target); }
            recycle_fast_boundary_contribs(&mut dense[label]);
        }
        if let Some(target) = worker.target(&mut default)? { append(DEFAULT_LABEL, target); }
        recycle_fast_boundary_contribs(&mut default);
        let mut others = sparse.drain().collect::<Vec<_>>();
        others.sort_unstable_by_key(|&(label, _)| label);
        for (label, mut input) in others {
            if let Some(target) = worker.target(&mut input)? { append(label, target); }
        }
        transitions += edges.len();
        if transitions > MAX_PACKET_TRANSITIONS || worker.pool.failed { return None; }
        rows.push(PreparedRow { id: *id, members: subset.len(), final_weight,
            edges, pending, local_coefficients });
    }
    let thread = rayon::current_thread_index().unwrap_or(63).min(63);
    Some(Packet { values: worker.pool.values, recipes: worker.recipes, rows,
        base_hits: worker.base_hits, local_hits: worker.local_hits, worker: 1u64 << thread })
}

#[inline]
fn translated(id: u32, local: &[u32]) -> u32 {
    if id & LOCAL == 0 { id } else {
        let translated = local[(id & !LOCAL) as usize];
        assert_ne!(translated, u32::MAX, "unmarked private coefficient escaped its worker");
        translated
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batch(
    policy: Policy, nwa: &[FastBoundaryNwaState], dense_limit: usize,
    interner: &mut FastBoundaryWeightInterner, singleton_states: &mut [u32],
    subset_map: &mut FxHashMap<Pairs, u32>, singletons: &mut SingletonCache,
    closures: &mut ClosureCache, out: &mut Vec<FastBoundaryDwaState>,
    supports: &mut Vec<Vec<u32>>, worklist: &mut VecDeque<(u32, Pairs)>,
    edge_count: &mut usize, profile: &mut Profile,
) -> Option<()> {
    let started = Instant::now();
    let count = worklist.len().min(policy.batch);
    let inputs = worklist.drain(..count).collect::<Vec<_>>();
    let prepared = inputs.par_chunks(policy.chunk).map(|chunk|
        prepare_packet(chunk, nwa, interner, singletons, closures, singleton_states, dense_limit))
        .collect::<Vec<_>>();
    profile.prepare_ms += started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    profile.batches += 1;
    for packet in prepared {
        let packet = packet?;
        profile.packets += 1; profile.rows += packet.rows.len();
        profile.recipes += packet.recipes.len();
        profile.base_cache_hits += packet.base_hits; profile.local_cache_hits += packet.local_hits;
        profile.local_values += packet.values.len(); profile.worker_mask |= packet.worker;
        let import_started = Instant::now();
        let live = if policy.live_import { Some(packet.escaping_values()?) } else { None };
        profile.imported_values += live.as_ref().map_or(packet.values.len(),
            |marks| marks.iter().filter(|&&marked| marked).count());
        let local = import_private_values(packet.values, live.as_deref(), interner)?;
        profile.import_ms += import_started.elapsed().as_secs_f64() * 1000.0;
        if interner.failed { return None; }
        let mut recipes = packet.recipes.into_iter().map(Some).collect::<Vec<_>>();
        let mut registered = vec![None; recipes.len()];
        for mut row in packet.rows {
            if !interner.allow_work(1, out.len(), *edge_count) { return None; }
            out[row.id as usize].final_weight = translated(row.final_weight, &local);
            profile.ready_edges += row.edges.len() - row.pending.len();
            if row.local_coefficients {
                for (_, _, weight) in &mut row.edges { *weight = translated(*weight, &local); }
            }
            for (slot, target) in row.pending {
                let (next, weight) = match target {
                    Target::Known(q, weight) => (q, translated(weight, &local)),
                    Target::Singleton(q, weight) => (fast_boundary_singleton_state(q,
                        singleton_states, subset_map, out, supports, worklist, interner.all_id()),
                        translated(weight, &local)),
                    Target::Recipe(index) => {
                        if let Some(result) = registered[index] { result }
                        else {
                            let recipe = recipes[index].take()?;
                            let incoming = recipe.incoming.into_iter()
                                .map(|(q,w)| (q,translated(w,&local))).collect::<Vec<_>>();
                            let singleton = if let [(q,w)] = incoming.as_slice() { Some((*q,*w)) } else { None };
                            let cached = if let Some(key) = singleton { singletons.get(&key).copied() }
                                else { closures.get(incoming.as_slice()).copied() };
                            let result = if let Some(result) = cached { result }
                            else {
                                let closure = recipe.closure.into_iter()
                                    .map(|(q,w)| (q,translated(w,&local))).collect::<Vec<_>>();
                                let result = if let [(q,w)] = closure.as_slice() {
                                    (fast_boundary_singleton_state(*q, singleton_states, subset_map,
                                        out, supports, worklist, interner.all_id()), *w)
                                } else if let Some(&existing) = subset_map.get(closure.as_slice()) {
                                    (existing, translated(recipe.weight,&local))
                                } else {
                                    let q = out.len() as u32;
                                    subset_map.insert(closure.clone(),q);
                                    supports.push(closure.iter().map(|&(s,_)|s).collect());
                                    out.push(FastBoundaryDwaState::default());
                                    worklist.push_back((q,closure));
                                    (q,translated(recipe.weight,&local))
                                };
                                if let Some(key) = singleton { singletons.insert(key,result); }
                                else { closures.insert(incoming,result); }
                                result
                            };
                            registered[index] = Some(result);
                            result
                        }
                    }
                };
                row.edges[slot].1 = next;
                row.edges[slot].2 = weight;
            }
            out[row.id as usize].transitions = row.edges;
            *edge_count = edge_count.saturating_add(out[row.id as usize].transitions.len());
            if !interner.allow_work(row.members, out.len(), *edge_count) { return None; }
        }
    }
    profile.publish_ms += started.elapsed().as_secs_f64() * 1000.0;
    Some(())
}

#[cfg(test)]
#[path = "finite_parallel_rows_tests.rs"]
mod tests;
