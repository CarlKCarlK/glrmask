//! Exact comparison of internal native boundary dumps, not runtime artifacts.
//! Build with: cargo build --release --features internal-api
//!             --example compare_boundary_native_rows
//!
//! Only the numeric IDs of equal coefficient values may differ. State IDs,
//! ordered labels, target identities, zero guards and the decoder must agree.
use glrmask_weight::__private::Weight;
use std::{error::Error, fs::File, io::Read, path::Path};

type Wire = (
    u32, u32, usize, usize,
    Vec<(Vec<(i32, u32, u32)>, u32)>, Vec<Box<[u64]>>, Vec<Weight>,
);
const MAX_DECOMPRESSED_BYTES: u64 = 128 * 1024 * 1024;

fn validate(wire: &Wire) -> Result<(), String> {
    if !(1..=64).contains(&wire.2) || !(1..=64).contains(&wire.3) {
        return Err("invalid finite coordinate dimensions".into());
    }
    if wire.5.iter().any(|row| row.len() != wire.2) {
        return Err("invalid coefficient width".into());
    }
    for (q, (edges, final_weight)) in wire.4.iter().enumerate() {
        if *final_weight as usize >= wire.5.len() {
            return Err(format!("invalid final coefficient at state {q}"));
        }
        for &(_, target, weight) in edges {
            if target as usize >= wire.4.len() || weight as usize >= wire.5.len() {
                return Err(format!("invalid target/coefficient at state {q}"));
            }
        }
    }
    Ok(())
}

fn load(path: &Path) -> Result<Wire, Box<dyn Error>> {
    let decoder = zstd::stream::read::Decoder::new(File::open(path)?)?;
    let mut bytes = Vec::new();
    decoder.take(MAX_DECOMPRESSED_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_DECOMPRESSED_BYTES {
        return Err("internal native dump exceeds the 128 MiB diagnostic limit".into());
    }
    let wire: Wire = bincode::deserialize(&bytes)?;
    validate(&wire)?;
    Ok(wire)
}

fn compare(a: &Wire, b: &Wire) -> Result<usize, String> {
    validate(a)?;
    validate(b)?;
    if (a.0, a.1, a.2, a.3) != (b.0, b.1, b.2, b.3) {
        return Err("coordinate header changed".into());
    }
    if a.6 != b.6 { return Err("original-coordinate decoder changed".into()); }
    if a.4.len() != b.4.len() { return Err("deterministic states merged/split".into()); }
    let mut count = 0;
    for (q, (left, right)) in a.4.iter().zip(&b.4).enumerate() {
        if a.5[left.1 as usize] != b.5[right.1 as usize] {
            return Err(format!("final coefficient differs at state {q}"));
        }
        if left.0.len() != right.0.len() {
            return Err(format!("outgoing degree differs at state {q}"));
        }
        for (&(la, ta, wa), &(lb, tb, wb)) in left.0.iter().zip(&right.0) {
            if (la, ta) != (lb, tb) {
                return Err(format!("ordered label/target identity differs at {q}/{la}"));
            }
            if a.5[wa as usize] != b.5[wb as usize] {
                return Err(format!("coefficient differs at {q}/{la}"));
            }
            count += 1;
        }
    }
    Ok(count)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args_os().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: compare_boundary_native_rows reference.bin.zst candidate.bin.zst".into());
    }
    let a = load(Path::new(&args[1]))?;
    let b = load(Path::new(&args[2]))?;
    let edges = compare(&a, &b)?;
    println!("NATIVE_ROWS_EXACT component={} states={} edges={} reference_weights={} candidate_weights={} atoms={}",
        a.1, a.4.len(), edges, a.5.len(), b.5.len(), a.6.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Wire {
        (6, 1, 1, 64, vec![
            (vec![(2, 1, 0), (i32::MAX - 1, 1, 2)], 0),
            (vec![], 2),
        ], vec![vec![0].into_boxed_slice(), vec![u64::MAX].into_boxed_slice(),
            vec![5].into_boxed_slice()], vec![])
    }
    #[test]
    fn accepts_exact_value_renaming_without_merging_targets() {
        let a = sample();
        let mut b = sample();
        b.5.push(vec![5].into_boxed_slice());
        b.4[0].0[1].2 = 3;
        b.4[1].1 = 3;
        assert_eq!(compare(&a, &b), Ok(2));
    }
    #[test]
    fn detects_deleted_zero_guard_and_changed_target() {
        let a = sample();
        let mut b = sample(); b.4[0].0.remove(0);
        assert!(compare(&a, &b).is_err());
        let mut b = sample(); b.4[0].0[1].1 = 0;
        assert!(compare(&a, &b).is_err());
    }
    #[test]
    fn detects_coefficient_order_and_decoder_corruption() {
        let a = sample();
        let mut b = sample(); b.5[2][0] = 4;
        assert!(compare(&a, &b).is_err());
        let mut b = sample(); b.4[0].0.reverse();
        assert!(compare(&a, &b).is_err());
        let mut b = sample(); b.6.push(Weight::empty());
        assert!(compare(&a, &b).is_err());
    }
    #[test]
    fn malformed_references_are_errors_not_index_panics() {
        let a = sample();
        let mut b = sample(); b.4[0].0[0].1 = u32::MAX;
        assert!(compare(&a, &b).is_err());
        let mut b = sample(); b.4[1].1 = u32::MAX;
        assert!(compare(&a, &b).is_err());
    }
}
