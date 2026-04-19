//! Workload generation and shared predicates for the graph benchmarks.
//!
//! All contenders consume the same [`Workload`] so that every measured
//! operation does byte-for-byte equivalent work (see the sanity gate in
//! `main.rs`). Node payloads are the ordinals `0..n` and edge payloads are
//! the positions `0..m` — unique values, required because the map-backed
//! `sg_btree` contender uses the payload as its index key.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Fixed seed so workloads are reproducible across runs and contenders.
pub const SEED: u64 = 0x5AFE_6EA9;

/// `(nodes, edges)` problem sizes. The sanity gate only validates the first
/// two (the 10k case is skipped there to keep startup fast).
pub const SIZES: &[(usize, usize)] = &[(100, 500), (1_000, 5_000), (10_000, 50_000)];

/// A directed multigraph workload: `n` nodes and a list of `(from, to)`
/// endpoint ordinals (parallel edges and self-loops allowed).
#[derive(Clone)]
pub struct Workload {
    pub n: usize,
    pub edges: Vec<(u32, u32)>,
}

/// Builds a workload of `n` nodes and `m` edges with endpoints drawn
/// uniformly from `0..n` using a seeded RNG.
pub fn generate_workload(n: usize, m: usize, seed: u64) -> Workload {
    let mut rng = StdRng::seed_from_u64(seed);
    let edges = (0..m)
        .map(|_| {
            let f = rng.gen_range(0..n) as u32;
            let t = rng.gen_range(0..n) as u32;
            (f, t)
        })
        .collect();
    Workload { n, edges }
}

/// Returns `0..n` shuffled — the shared access order for the random-access
/// group, so every contender chases the same permutation of indices.
pub fn shuffled_ordinals(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xABCD_EF01);
    let mut v: Vec<usize> = (0..n).collect();
    v.shuffle(&mut rng);
    v
}

/// Edge-removal predicate keyed on the edge payload: removes half the edges.
pub fn edge_is_victim(e: usize) -> bool {
    e.is_multiple_of(2)
}

/// Node-removal predicate keyed on the node payload: removes a quarter of
/// the nodes (and, by cascade, every edge incident to one).
pub fn node_is_victim(v: usize) -> bool {
    v.is_multiple_of(4)
}
