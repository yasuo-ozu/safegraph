//! Maximum flow on unit-capacity networks, where Dinic's bound matters most.
//!
//! - `max_flow_shared_leaf`: type-leak's shape — `n` roots, each a tuple
//!   holding its own leaf and one shared hub — as a minimum vertex cut. On its
//!   node-split network Edmonds-Karp needs one BFS over the whole network per
//!   unit of flow (quadratic in `n`), Dinic a few phases (near-linear), so
//!   Edmonds-Karp only runs at the small sizes.
//! - `max_flow_bipartite`: a random bipartite matching network.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use safegraph::algo::max_flow::{dinic, edmonds_karp, min_vertex_cut};
use safegraph::graph::Graph;
use safegraph::VecGraph;

use crate::common::SEED;

/// The type graph: node data is what cutting the node costs. Returns the
/// graph, the roots and the leaves (the hub first).
fn shared_leaf(n: u32) -> (VecGraph<Option<u32>, ()>, Vec<u32>, Vec<u32>) {
    let mut g = VecGraph::<Option<u32>, ()>::default();
    unsafe {
        let hub = g.insert_node_unchecked(Some(1)).unwrap();
        let mut roots = Vec::new();
        let mut leaves = vec![hub];
        for _ in 0..n {
            let root = g.insert_node_unchecked(Some(1)).unwrap();
            let leaf = g.insert_node_unchecked(Some(1)).unwrap();
            g.insert_edge_unchecked((), [root, leaf]).unwrap();
            g.insert_edge_unchecked((), [root, hub]).unwrap();
            roots.push(root);
            leaves.push(leaf);
        }
        (g, roots, leaves)
    }
}

/// The node-split network of [`shared_leaf`] as edge capacities, built the
/// way type-leak builds it by hand: each type is an in-node and an out-node
/// joined by an arc of capacity 1, every other arc "infinite" (more than all
/// node capacities together). Returns the network, its source and its sink.
fn shared_leaf_network(n: u32) -> (VecGraph<(), u32>, u32, u32) {
    let (types, roots, leaves) = shared_leaf(n);
    let count = Graph::len_node(&types) as u32;
    let infinite = count + 1;
    let mut g = VecGraph::<(), u32>::default();
    unsafe {
        // Type `v` is `2v` (in) and `2v + 1` (out); then the source and sink.
        for _ in 0..2 * count + 2 {
            g.insert_node_unchecked(()).unwrap();
        }
        let (s, t) = (2 * count, 2 * count + 1);
        for v in 0..count {
            g.insert_edge_unchecked(1, [2 * v, 2 * v + 1]).unwrap();
        }
        // Root `i` holds leaf `i` and the hub, which is `leaves[0]`.
        for (i, &root) in roots.iter().enumerate() {
            for &child in &[leaves[i + 1], leaves[0]] {
                g.insert_edge_unchecked(infinite, [2 * root + 1, 2 * child])
                    .unwrap();
            }
        }
        for &root in &roots {
            g.insert_edge_unchecked(infinite, [s, 2 * root]).unwrap();
        }
        for &leaf in &leaves {
            g.insert_edge_unchecked(infinite, [2 * leaf + 1, t])
                .unwrap();
        }
        (g, s, t)
    }
}

/// A bipartite matching network with `k` nodes per side and `degree` random
/// edges out of each left node, all of capacity 1. Returns the network, its
/// source and its sink.
fn bipartite(k: u32, degree: u32) -> (VecGraph<(), u32>, u32, u32) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut g = VecGraph::<(), u32>::default();
    unsafe {
        // `0` is the source, `1` the sink, then the left and the right side.
        for _ in 0..2 + 2 * k {
            g.insert_node_unchecked(()).unwrap();
        }
        for i in 0..k {
            g.insert_edge_unchecked(1, [0, 2 + i]).unwrap();
            g.insert_edge_unchecked(1, [2 + k + i, 1]).unwrap();
            for _ in 0..degree {
                let j = rng.gen_range(0..k);
                g.insert_edge_unchecked(1, [2 + i, 2 + k + j]).unwrap();
            }
        }
    }
    (g, 0, 1)
}

pub fn bench_max_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_flow_shared_leaf");
    group.sample_size(10);
    for &n in &[1_000u32, 2_000, 4_000] {
        let (g, s, t) = shared_leaf_network(n);
        assert_eq!(dinic(&g, s, t, |&c| c), n, "shared_leaf_network({})", n);
        group.bench_with_input(BenchmarkId::new("dinic", n), &n, |b, _| {
            b.iter(|| dinic(black_box(&g), s, t, |&c| c))
        });

        let (types, roots, leaves) = shared_leaf(n);
        let types = unsafe { types.unsafe_assert_stable_node() };
        let cut = min_vertex_cut(types, roots.iter().copied(), leaves.iter().copied(), |&c| c);
        assert_eq!(cut.map(|(cost, _)| cost), Some(n), "shared_leaf({})", n);
        group.bench_with_input(BenchmarkId::new("min_vertex_cut", n), &n, |b, _| {
            b.iter(|| {
                min_vertex_cut(
                    black_box(types),
                    roots.iter().copied(),
                    leaves.iter().copied(),
                    |&c| c,
                )
            })
        });
    }
    // Quadratic: the small sizes show the trend without taking minutes.
    for &n in &[250u32, 500, 1_000] {
        let (g, s, t) = shared_leaf_network(n);
        assert_eq!(
            edmonds_karp(&g, s, t, |&c| c),
            n,
            "shared_leaf_network({})",
            n
        );
        group.bench_with_input(BenchmarkId::new("edmonds_karp", n), &n, |b, _| {
            b.iter(|| edmonds_karp(black_box(&g), s, t, |&c| c))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("max_flow_bipartite");
    group.sample_size(10);
    for &k in &[250u32, 1_000] {
        let (g, s, t) = bipartite(k, 4);
        let flow = dinic(&g, s, t, |&c| c);
        assert_eq!(edmonds_karp(&g, s, t, |&c| c), flow, "bipartite({})", k);
        group.bench_with_input(BenchmarkId::new("dinic", k), &k, |b, _| {
            b.iter(|| dinic(black_box(&g), s, t, |&c| c))
        });
        group.bench_with_input(BenchmarkId::new("edmonds_karp", k), &k, |b, _| {
            b.iter(|| edmonds_karp(black_box(&g), s, t, |&c| c))
        });
    }
    group.finish();
}
