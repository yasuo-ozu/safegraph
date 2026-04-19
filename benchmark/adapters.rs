//! One module per contender. Most of the surface is uniform:
//!
//! ```text
//! build(&Workload) -> G
//! traverse_sum(&G) -> usize          // Σ node payloads + Σ edge payloads (wrapping)
//! out_degree_sum(&G) -> usize        // Σ outgoing degree (sanity coverage)
//! remove_edge_set(&mut G)
//! remove_node_set(&mut G)            // cascades incident edges
//! counts(&G) -> (usize, usize)
//! ```
//!
//! Random access differs by family, because safegraph's scoped indices cannot
//! escape a `scope()`:
//!
//! - Scoped contenders (`sg_vec_scoped` / `sg_flat` / `sg_btree`) expose
//!   `bench_random_access(&G, &[usize], &mut Bencher)`, which runs the untimed
//!   index prep AND the timed lookups inside one `scope()` so the checked
//!   `ctx.node()` accessor benefits from the `Context`'s always-true
//!   `contains_node_index` (no bounds-check branch).
//! - The others (`sg_vec_stabilized` / `sg_vec_checked` / `pg` / `pg_stable`)
//!   expose the precompute pair `access_indices(&G, &[usize]) -> Vec<NodeIx>`
//!   (untimed) and `access_sum(&G, &[NodeIx]) -> usize` (timed).
//!
//! The three VecGraph rows compare safe-access strategies: `sg_vec_scoped`
//! (scope, free `contains`), `sg_vec_stabilized` ([`Graph::stabilize`],
//! versioned `contains`), and `sg_vec_checked` (graph-level `node()`, a
//! bounds-check `contains`).
//!
//! The three scoped safegraph contenders are stamped out by the
//! [`scoped_adapter!`] macro with concrete types — not a generic `fn<G>` —
//! to sidestep the heavy `G::Endpoints: for<'scope> Map<…>` HRTB bounds.

use std::collections::BTreeMap;

use criterion::Bencher;
use safegraph::graph::capability::InsertNode;
use safegraph::graph::{Graph, GraphOperation, GraphProperty};
use safegraph::raw_graph::flat_adj_edge::{FlatAdjEdgeGraph, NodeRepr as FlatNodeRepr, TNone};
use safegraph::raw_graph::linked_adj_edge::{EdgeRepr, LinkedAdjEdgeGraph, NodeRepr as LinkedNodeRepr};

use crate::common::{edge_is_victim, node_is_victim, Workload};

/// Stamps out a contender module exercising the safe scoped (`scope`/
/// `scope_mut`) API for the concrete graph type `$G`.
macro_rules! scoped_adapter {
    ($name:ident, $G:ty) => {
        pub mod $name {
            use super::*;

            pub type G = $G;

            pub fn build(w: &Workload) -> G {
                let mut g = G::default();
                g.scope_mut(|mut ctx| {
                    let ixs: Vec<_> = (0..w.n)
                        .map(|i| ctx.insert_node(i).expect("insert_node"))
                        .collect();
                    for (j, &(f, t)) in w.edges.iter().enumerate() {
                        ctx.insert_edge(j, [ixs[f as usize], ixs[t as usize]])
                            .expect("insert_edge");
                    }
                });
                g
            }

            pub fn traverse_sum(g: &G) -> usize {
                g.scope(|ctx| {
                    let mut total = 0usize;
                    for nix in Graph::node_indices(ctx) {
                        total = total.wrapping_add(*ctx.node(nix));
                        for (_, e, _) in ctx.walks_from(nix).map(|w| w.get()) {
                            total = total.wrapping_add(*e);
                        }
                    }
                    total
                })
            }

            pub fn out_degree_sum(g: &G) -> usize {
                g.scope(|ctx| {
                    let mut total = 0usize;
                    for nix in Graph::node_indices(ctx) {
                        total += ctx.walks_from(nix).count();
                    }
                    total
                })
            }

            /// Random-access lookups through the scoped API. Scoped indices
            /// cannot escape a `scope()`, so the untimed index prep and the
            /// timed `b.iter` loop both run inside one scope. Inside it,
            /// `Context::contains_node_index` is unconditionally `true`, so the
            /// checked `ctx.node()` accessor's assert folds away — none of the
            /// bounds-check cost the graph-level `g.node(ix)` would pay.
            pub fn bench_random_access(g: &G, order: &[usize], b: &mut Bencher) {
                g.scope(|ctx| {
                    // Untimed setup: scoped indices in the requested order.
                    let all: Vec<_> = Graph::node_indices(ctx).collect();
                    let ixs: Vec<_> = order.iter().map(|&i| all[i]).collect();
                    b.iter(|| {
                        let mut total = 0usize;
                        for &ix in &ixs {
                            total = total.wrapping_add(*ctx.node(ix));
                        }
                        std::hint::black_box(total)
                    });
                });
            }

            pub fn remove_edge_set(g: &mut G) {
                g.scope_mut(|ctx| {
                    let victims: Vec<_> = Graph::edge_indices(&*ctx)
                        .filter(|&e| edge_is_victim(*ctx.edge(e)))
                        .collect();
                    ctx.remove_nodes_edges(None, victims);
                });
            }

            pub fn remove_node_set(g: &mut G) {
                g.scope_mut(|ctx| {
                    let victims: Vec<_> = Graph::node_indices(&*ctx)
                        .filter(|&n| node_is_victim(*ctx.node(n)))
                        .collect();
                    ctx.remove_nodes_edges(victims, None);
                });
            }

            pub fn counts(g: &G) -> (usize, usize) {
                g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()))
            }
        }
    };
}

// Vec-backed, safe scoped API (scope()/scope_mut()). NodeIx = u32.
scoped_adapter!(sg_vec_scoped, safegraph::VecGraph<usize, usize>);

// Flat nested-collection adjacency, no reverse index (IS = TNone).
// EdgeIx = EdgeIx<u32, u32>.
scoped_adapter!(
    sg_flat,
    FlatAdjEdgeGraph<Vec<(usize, FlatNodeRepr<Vec<(usize, u32)>, TNone>)>>
);

// BTreeMap-backed, payload-keyed stable indices. NodeIx = usize (the key).
scoped_adapter!(
    sg_btree,
    LinkedAdjEdgeGraph<
        BTreeMap<usize, LinkedNodeRepr<Option<usize>>>,
        BTreeMap<usize, EdgeRepr<usize, Option<usize>>>,
    >
);

/// Vec-backed safegraph wrapped by [`Graph::stabilize`] — tombstone-versioned
/// stable indices give a fully safe, scope-free API (no `unsafe`, no
/// `scope()`). `contains_*_index` does a version/liveness check.
pub mod sg_vec_stabilized {
    use super::*;
    use safegraph::graph::stabilized::{EdgeIx as StabEdgeIx, NodeIx as StabNodeIx, Stabilized};

    pub type G = Stabilized<
        LinkedAdjEdgeGraph<
            Vec<(StabNodeIx<usize>, LinkedNodeRepr<u32>)>,
            Vec<(StabEdgeIx<usize>, EdgeRepr<u32, u32>)>,
        >,
        usize,
        usize,
    >;
    pub type NodeIx = <G as GraphProperty>::NodeIx;

    pub fn build(w: &Workload) -> G {
        let mut g: G = safegraph::VecGraph::<usize, usize>::default().stabilize();
        let ixs: Vec<NodeIx> = (0..w.n)
            .map(|i| g.insert_node(i).expect("insert_node"))
            .collect();
        for (j, &(f, t)) in w.edges.iter().enumerate() {
            g.insert_edge(j, [ixs[f as usize], ixs[t as usize]])
                .expect("insert_edge");
        }
        g
    }

    pub fn traverse_sum(g: &G) -> usize {
        let mut total = 0usize;
        for nix in Graph::node_indices(g) {
            total = total.wrapping_add(*g.node(nix));
            for (_, e, _) in g.walks_from(nix).map(|w| w.get()) {
                total = total.wrapping_add(*e);
            }
        }
        total
    }

    pub fn out_degree_sum(g: &G) -> usize {
        Graph::node_indices(g).map(|nix| g.walks_from(nix).count()).sum()
    }

    pub fn access_indices(g: &G, order: &[usize]) -> Vec<NodeIx> {
        let all: Vec<NodeIx> = Graph::node_indices(g).collect();
        order.iter().map(|&i| all[i]).collect()
    }

    pub fn access_sum(g: &G, ixs: &[NodeIx]) -> usize {
        let mut total = 0usize;
        for &ix in ixs {
            total = total.wrapping_add(*g.node(ix));
        }
        total
    }

    pub fn remove_edge_set(g: &mut G) {
        let victims: Vec<_> = Graph::edge_indices(g)
            .filter(|&e| edge_is_victim(*g.edge(e)))
            .collect();
        g.remove_nodes_edges(None, victims);
    }

    pub fn remove_node_set(g: &mut G) {
        let victims: Vec<_> = Graph::node_indices(g)
            .filter(|&nix| node_is_victim(*g.node(nix)))
            .collect();
        g.remove_nodes_edges(victims, None);
    }

    pub fn counts(g: &G) -> (usize, usize) {
        (Graph::node_indices(g).count(), Graph::edge_indices(g).count())
    }
}

/// Plain Vec-backed safegraph that pays a real `contains_*_index` bounds check
/// in advance of *every* index-taking graph operation, with NO `scope()` — the
/// consistent "fully checked" VecGraph contender. Plain `VecGraph` is not
/// `StableNode`, so enumeration and edge walks have no safe form and use
/// `*_unstable`, but each index-taking op is still gated by a real check:
/// accesses via the checked `node`/`edge` accessors; edge insertion via an
/// explicit endpoint `contains_node_index` assert before
/// `insert_edge_unchecked`; and batch removal via `remove_nodes_edges`, which
/// asserts every index before the unchecked sweep. Unlike the scoped contender
/// (whose `contains` is unconditionally `true`, so the checks fold away), these
/// are genuine bounds checks.
pub mod sg_vec_checked {
    use super::*;

    pub type G = safegraph::VecGraph<usize, usize>;
    pub type NodeIx = <G as GraphProperty>::NodeIx;
    type EdgeIx = <G as GraphProperty>::EdgeIx;

    pub fn build(w: &Workload) -> G {
        let mut g = G::default();
        let ixs: Vec<NodeIx> = (0..w.n)
            .map(|i| unsafe { InsertNode::insert_node_unchecked(&mut g, i) }.expect("insert_node"))
            .collect();
        for (j, &(f, t)) in w.edges.iter().enumerate() {
            let endpoints = [ixs[f as usize], ixs[t as usize]];
            // Emulate the checked insert: assert both endpoints are valid (the
            // same per-edge bounds check `insert_edge_unchecked` runs) before the
            // unchecked insert. `VecGraph` is not `StableEdge`, so the safe
            // `insert_edge` is unavailable — this models its index check.
            assert!(endpoints
                .iter()
                .all(|&n| GraphOperation::contains_node_index(&g, n)));
            // SAFETY: endpoints validated immediately above.
            unsafe { Graph::insert_edge_unchecked(&mut g, j, endpoints) }.expect("insert_edge");
        }
        g
    }

    pub fn traverse_sum(g: &G) -> usize {
        let mut total = 0usize;
        // SAFETY: not mutated during iteration. Enumeration / walks have no
        // safe form for a non-`StableNode` graph; node access is the safe
        // checked `g.node`.
        for nix in GraphOperation::node_indices(g) {
            total = total.wrapping_add(*g.node(nix));
            for (_, e, _) in unsafe { GraphOperation::walks_from_unchecked(g, nix) }.map(|w| w.get()) {
                total = total.wrapping_add(*e);
            }
        }
        total
    }

    pub fn out_degree_sum(g: &G) -> usize {
        let mut total = 0usize;
        // SAFETY: not mutated during iteration.
        for nix in GraphOperation::node_indices(g) {
            total += unsafe { GraphOperation::walks_from_unchecked(g, nix) }.count();
        }
        total
    }

    pub fn access_indices(g: &G, order: &[usize]) -> Vec<NodeIx> {
        // SAFETY: setup only; graph unchanged before the timed access.
        let all: Vec<NodeIx> = GraphOperation::node_indices(g).collect();
        order.iter().map(|&i| all[i]).collect()
    }

    pub fn access_sum(g: &G, ixs: &[NodeIx]) -> usize {
        let mut total = 0usize;
        for &ix in ixs {
            // Safe checked accessor: asserts `contains_node_index` (bounds check).
            total = total.wrapping_add(*g.node(ix));
        }
        total
    }

    pub fn remove_edge_set(g: &mut G) {
        // SAFETY: not mutated while collecting victims (enumeration only).
        let victims: Vec<EdgeIx> = GraphOperation::edge_indices(&*g)
            .filter(|&e| edge_is_victim(*g.edge(e)))
            .collect();
        g.remove_nodes_edges(None, victims);
    }

    pub fn remove_node_set(g: &mut G) {
        // SAFETY: not mutated while collecting victims (enumeration only).
        let victims: Vec<NodeIx> = GraphOperation::node_indices(&*g)
            .filter(|&n| node_is_victim(*g.node(n)))
            .collect();
        g.remove_nodes_edges(victims, None);
    }

    pub fn counts(g: &G) -> (usize, usize) {
        (GraphOperation::len_node(g), GraphOperation::len_edge(g))
    }
}

/// petgraph `DiGraph` — indices invalidate on the last-removed slot.
pub mod pg {
    use petgraph::graph::{DiGraph, NodeIndex};

    use crate::common::{edge_is_victim, node_is_victim, Workload};

    pub type G = DiGraph<usize, usize>;
    pub type NodeIx = NodeIndex;

    pub fn build(w: &Workload) -> G {
        let mut g = DiGraph::new();
        let ixs: Vec<NodeIndex> = (0..w.n).map(|i| g.add_node(i)).collect();
        for (j, &(f, t)) in w.edges.iter().enumerate() {
            g.add_edge(ixs[f as usize], ixs[t as usize], j);
        }
        g
    }

    pub fn traverse_sum(g: &G) -> usize {
        let mut total = 0usize;
        for n in g.node_indices() {
            total = total.wrapping_add(*g.node_weight(n).unwrap());
            for er in g.edges(n) {
                total = total.wrapping_add(*er.weight());
            }
        }
        total
    }

    pub fn out_degree_sum(g: &G) -> usize {
        g.node_indices().map(|n| g.edges(n).count()).sum()
    }

    pub fn access_indices(g: &G, order: &[usize]) -> Vec<NodeIx> {
        let _ = g;
        order.iter().map(|&i| NodeIndex::new(i)).collect()
    }

    pub fn access_sum(g: &G, ixs: &[NodeIx]) -> usize {
        let mut total = 0usize;
        for &ix in ixs {
            total = total.wrapping_add(*g.node_weight(ix).unwrap());
        }
        total
    }

    pub fn remove_edge_set(g: &mut G) {
        g.retain_edges(|gg, e| !edge_is_victim(gg[e]));
    }

    pub fn remove_node_set(g: &mut G) {
        g.retain_nodes(|gg, n| !node_is_victim(gg[n]));
    }

    pub fn counts(g: &G) -> (usize, usize) {
        (g.node_count(), g.edge_count())
    }
}

/// petgraph `StableDiGraph` — stable indices with tombstones. Additionally
/// exposes one-at-a-time removal rows (valid only because its indices survive
/// individual removals).
pub mod pg_stable {
    use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

    use crate::common::{edge_is_victim, node_is_victim, Workload};

    pub type G = StableDiGraph<usize, usize>;
    pub type NodeIx = NodeIndex;

    pub fn build(w: &Workload) -> G {
        let mut g = StableDiGraph::new();
        let ixs: Vec<NodeIndex> = (0..w.n).map(|i| g.add_node(i)).collect();
        for (j, &(f, t)) in w.edges.iter().enumerate() {
            g.add_edge(ixs[f as usize], ixs[t as usize], j);
        }
        g
    }

    pub fn traverse_sum(g: &G) -> usize {
        let mut total = 0usize;
        for n in g.node_indices() {
            total = total.wrapping_add(*g.node_weight(n).unwrap());
            for er in g.edges(n) {
                total = total.wrapping_add(*er.weight());
            }
        }
        total
    }

    pub fn out_degree_sum(g: &G) -> usize {
        g.node_indices().map(|n| g.edges(n).count()).sum()
    }

    pub fn access_indices(g: &G, order: &[usize]) -> Vec<NodeIx> {
        let all: Vec<NodeIndex> = g.node_indices().collect();
        order.iter().map(|&i| all[i]).collect()
    }

    pub fn access_sum(g: &G, ixs: &[NodeIx]) -> usize {
        let mut total = 0usize;
        for &ix in ixs {
            total = total.wrapping_add(*g.node_weight(ix).unwrap());
        }
        total
    }

    pub fn remove_edge_set(g: &mut G) {
        g.retain_edges(|gg, e| !edge_is_victim(gg[e]));
    }

    pub fn remove_node_set(g: &mut G) {
        g.retain_nodes(|gg, n| !node_is_victim(gg[n]));
    }

    pub fn remove_edge_loop(g: &mut G) {
        let victims: Vec<EdgeIndex> =
            g.edge_indices().filter(|&e| edge_is_victim(g[e])).collect();
        for e in victims {
            g.remove_edge(e);
        }
    }

    pub fn remove_node_loop(g: &mut G) {
        let victims: Vec<NodeIndex> =
            g.node_indices().filter(|&n| node_is_victim(g[n])).collect();
        for n in victims {
            g.remove_node(n);
        }
    }

    pub fn counts(g: &G) -> (usize, usize) {
        (g.node_count(), g.edge_count())
    }
}
