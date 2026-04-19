//! Parametric test harness for `raw_graph` backends.
//!
//! Three harness functions exercise the safe `Graph` / `Context` API surface:
//!
//! * [`run_all`] — read/write operations every binary backend supports
//!   (insert, traverse, bulk-remove, clear, extend, stabilize).
//! * [`run_update`] — extra checks for backends that implement
//!   `UpdateNode` / `UpdateEdge` (Vec-backed only).
//! * [`run_unique`] — extra checks for backends that implement
//!   `UniqueNode` / `UniqueEdge` (map-backed only).
//!
//! Each backend is wired up through the [`backend!`], [`backend_update!`],
//! [`backend_unique!`] macros, which expand to a single `#[test]` delegating
//! to the corresponding harness. A separate [`linked_adj_edge_swap_remove`]
//! test stresses the swap-remove + index-rewrite path in
//! `LinkedAdjEdgeGraph<Vec, Vec>` directly.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use safegraph::VecGraph;
use safegraph::graph::capability::{
    Bigraph, Directed, InsertEdge, InsertNode, RemoveNode, StableEdge, StableNode, UniqueEdge,
    UniqueNode, UpdateEdge, UpdateNode,
};
use safegraph::graph::context::NodeIx as ScopedNIx;
use safegraph::graph::edge::Map;
use safegraph::graph::prelude::*;
use safegraph::raw_graph::flat_adj_edge::{
    EdgeIx as FlatEdgeIx, FlatAdjEdgeGraph, NodeRepr as FlatNodeRepr, TNone,
};
use safegraph::raw_graph::linked_adj_edge::{
    EdgeRepr, LinkedAdjEdgeGraph, NodeRepr as LinkedNodeRepr,
};
#[cfg(feature = "matrix")]
use safegraph::raw_graph::matrix::{CsMatGraph, EdgeIx as MatrixEdgeIx};

// ---------------------------------------------------------------------------
// Generic read-only checks. Operate on any safe `Graph` view (in practice a
// `Context` produced by `scope_mut` or `scope`).
// ---------------------------------------------------------------------------

fn check_read_only<C>(ctx: &C, n: [C::NodeIx; 3], e: [C::EdgeIx; 2])
where
    C: Graph + StableNode + StableEdge,
    C: Bigraph + GraphProperty<Node = u32, Edge = u32>,
    C: for<'r> Directed<'r>,
    C::NodeIx: Debug + Ord,
    C::EdgeIx: Debug + Ord,
{
    let [a, b, c] = n;
    let [e0, e1] = e;

    // -- counts / iteration over the whole graph
    assert_eq!(ctx.nodes().count(), 3);
    assert_eq!(ctx.edges().count(), 2);
    assert_eq!(Graph::node_indices(ctx).count(), 3);
    assert_eq!(Graph::edge_indices(ctx).count(), 2);
    let mut node_values: Vec<u32> = ctx.nodes().copied().collect();
    node_values.sort();
    assert_eq!(node_values, vec![10, 20, 30]);
    let mut edge_values: Vec<u32> = ctx.edges().copied().collect();
    edge_values.sort();
    assert_eq!(edge_values, vec![100, 200]);

    // -- direct lookup
    assert_eq!(*ctx.node(a), 10);
    assert_eq!(*ctx.node(b), 20);
    assert_eq!(*ctx.node(c), 30);
    assert_eq!(*ctx.edge(e0), 100);
    assert_eq!(*ctx.edge(e1), 200);

    // -- endpoints + endpoint_nodes
    let ep0: Vec<_> = ctx.endpoints(e0).into_iter().collect();
    assert_eq!(ep0, vec![a, b]);
    let ep0_nodes: Vec<u32> = ctx.endpoint_nodes(e0).copied().collect();
    assert_eq!(ep0_nodes, vec![10, 20]);

    // -- Bigraph endpoint indices / nodes
    assert_eq!(ctx.edge_tail_index(e0), a);
    assert_eq!(ctx.edge_head_index(e0), b);
    assert_eq!(*ctx.edge_tail(e0), 10);
    assert_eq!(*ctx.edge_head(e0), 20);

    // -- outgoing chain (edge indices, edge refs, walks)
    let from_a_eix: Vec<_> = ctx.edge_indices_from(a).collect();
    assert_eq!(from_a_eix, vec![e0]);
    let from_a_edges: Vec<u32> = ctx.edges_from(a).copied().collect();
    assert_eq!(from_a_edges, vec![100]);
    let walks_a: Vec<_> = ctx.walks_from(a).map(|w| w.get()).map(|(eix, _, nix)| (eix, nix)).collect();
    assert_eq!(walks_a, vec![(e0, b)]);
    let from_c: Vec<_> = ctx.edge_indices_from(c).collect();
    assert!(from_c.is_empty(), "expected no outgoing from c: {:?}", from_c);

    // -- incoming chain
    let to_c_eix: Vec<_> = ctx.edge_indices_to(c).collect();
    assert_eq!(to_c_eix, vec![e1]);
    let to_c_edges: Vec<u32> = ctx.edges_to(c).copied().collect();
    assert_eq!(to_c_edges, vec![200]);
    let walks_c: Vec<_> = ctx.walks_to(c).map(|w| w.get()).map(|(src, eix, _)| (src, eix)).collect();
    assert_eq!(walks_c, vec![(b, e1)]);
    let to_a: Vec<_> = ctx.edge_indices_to(a).collect();
    assert!(to_a.is_empty(), "expected no incoming to a: {:?}", to_a);

    // -- all incident edges of a middle node
    let mut incident_eix: Vec<_> = ctx.edge_indices_of(b).collect();
    incident_eix.sort();
    let mut expected_eix = vec![e0, e1];
    expected_eix.sort();
    assert_eq!(incident_eix, expected_eix);
    let mut incident_edges: Vec<u32> = ctx.edges_of(b).copied().collect();
    incident_edges.sort();
    assert_eq!(incident_edges, vec![100, 200]);

    // -- tail/head iterators
    let tails: Vec<_> = ctx.edge_tail_indices(e0).collect();
    let heads: Vec<_> = ctx.edge_head_indices(e0).collect();
    assert_eq!(tails, vec![a]);
    assert_eq!(heads, vec![b]);
    let tail_nodes: Vec<u32> = ctx.edge_tails(e0).copied().collect();
    let head_nodes: Vec<u32> = ctx.edge_heads(e0).copied().collect();
    assert_eq!(tail_nodes, vec![10]);
    assert_eq!(head_nodes, vec![20]);

    // -- neighbor iterators (indices + refs, in all three directions)
    let neigh_from_a: Vec<_> = ctx.neighbor_indices_from(a).collect();
    assert_eq!(neigh_from_a, vec![b]);
    let neigh_to_c: Vec<_> = ctx.neighbor_indices_to(c).collect();
    assert_eq!(neigh_to_c, vec![b]);
    let mut neigh_of_b: Vec<_> = ctx.neighbor_indices_of(b).collect();
    neigh_of_b.sort();
    let mut expected_of_b = vec![a, c];
    expected_of_b.sort();
    assert_eq!(neigh_of_b, expected_of_b);
    let neigh_from_a_nodes: Vec<u32> = ctx.neighbors_from(a).copied().collect();
    assert_eq!(neigh_from_a_nodes, vec![20]);
    let neigh_to_c_nodes: Vec<u32> = ctx.neighbors_to(c).copied().collect();
    assert_eq!(neigh_to_c_nodes, vec![20]);
    let mut neigh_of_b_nodes: Vec<u32> = ctx.neighbors_of(b).copied().collect();
    neigh_of_b_nodes.sort();
    assert_eq!(neigh_of_b_nodes, vec![10, 30]);

    // -- contains_*_index (the GraphProperty-level safe predicates)
    // UFCS: `C: Graph` makes the supertrait `GraphOperation`'s same-named method
    // a second candidate, so plain method syntax would be ambiguous.
    assert!(Graph::contains_node_index(ctx, a));
    assert!(Graph::contains_edge_index(ctx, e0));

    // -- walks_of yields both incidents of `b`
    let mut walks_b: Vec<u32> = ctx.walks_of(b).map(|w| w.get()).map(|(_, _, nix)| *ctx.node(nix)).collect();
    walks_b.sort();
    assert_eq!(walks_b, vec![10, 30]);
}

// ---------------------------------------------------------------------------
// Common run_all harness: every binary backend.
// ---------------------------------------------------------------------------

fn run_all<G>()
where
    G: Default + 'static,
    G: Graph + InsertNode + InsertEdge + RemoveNode,
    G: Bigraph + GraphProperty<Node = u32, Edge = u32>,
    G: for<'r> Directed<'r>,
    G::Endpoints: for<'scope> Map<
        ScopedNIx<'scope, <G as GraphProperty>::NodeIx>,
        Mapped = [ScopedNIx<'scope, <G as GraphProperty>::NodeIx>; 2],
    >,
    G::NodeIx: Debug + Ord,
    G::EdgeIx: Debug + Ord,
{
    // 1. Read-only: counts, lookups, traversals, neighbors, Bigraph helpers.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                a {10u32} -- e0 {100u32} --> b {20u32} -- e1 {200u32} --> c {30u32}
            );
            check_read_only(&*ctx, [a, b, c], [e0, e1]);
        });
    }

    // 2. `push` / `push_edge` build the graph without binding indices.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            ctx.push(7u32).unwrap();
            ctx.push(8u32).unwrap();
            assert_eq!(ctx.nodes().count(), 2);
        });
    }

    // 3. Removing a single edge: bulk path via `Some(eix)`.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                _a {10u32} -- e0 {100u32} --> _b {20u32} -- _e1 {200u32} --> _c {30u32}
            );
            ctx.remove_nodes_edges(None, Some(e0));
        });
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (3, 1), "after removing one edge");
    }

    // 4. Removing a middle node cascades to every incident edge.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                _a {10u32} -- _e0 {100u32} --> b {20u32} -- _e1 {200u32} --> _c {30u32}
            );
            ctx.remove_nodes_edges(Some(b), None);
        });
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (2, 0), "after removing middle node");
    }

    // 5a. `take_nodes_edges` with a single edge: returns its value.
    {
        let mut g = G::default();
        let (taken_n, taken_e): (Vec<u32>, Vec<u32>) = g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                _a {10u32} -- e0 {100u32} --> _b {20u32} -- _e1 {200u32} --> _c {30u32}
            );
            ctx.take_nodes_edges::<Vec<u32>, Vec<u32>>(None, Some(e0))
        });
        assert!(taken_n.is_empty());
        assert_eq!(taken_e, vec![100]);
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (3, 1));
    }

    // 5b. `take_nodes_edges` with a single node: returns the node's value and
    // cascades through every incident edge.
    {
        let mut g = G::default();
        let (taken_n, taken_e): (Vec<u32>, Vec<u32>) = g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                _a {10u32} -- _e0 {100u32} --> b {20u32} -- _e1 {200u32} --> _c {30u32}
            );
            ctx.take_nodes_edges::<Vec<u32>, Vec<u32>>(Some(b), None)
        });
        assert_eq!(taken_n, vec![20]);
        assert!(taken_e.is_empty(), "cascade-dropped edges don't come back");
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (2, 0));
    }

    // 5c. Multi-edge `take_nodes_edges` returns exactly the requested edges'
    // payloads (a permutation/multiset check — the trait does not specify
    // output order, and backends differ: `LinkedAdjEdgeGraph` returns input
    // order, `FlatAdjEdgeGraph` returns descending index order). Pins that the
    // single-kind fast path and the phased path remove the right edges and
    // leave the right surviving graph.
    {
        let mut g = G::default();
        let (taken_n, mut taken_e): (Vec<u32>, Vec<u32>) = g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                _a {1u32} -- e0 {100u32} --> _b {2u32} -- e1 {200u32} --> _c {3u32}
                    -- e2 {300u32} --> _d {4u32} -- e3 {400u32} --> _e {5u32}
            );
            ctx.take_nodes_edges::<Vec<u32>, Vec<u32>>(None, [e2, e0, e3, e1])
        });
        taken_e.sort_unstable();
        assert!(taken_n.is_empty());
        assert_eq!(taken_e, vec![100, 200, 300, 400], "every requested edge returned once");
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (5, 0));
    }

    // 5d. Mixed batch (phased path): nodes and explicit edges all come back
    // exactly once. An explicit edge incident on an explicitly removed node is
    // returned by its slot (not double-removed).
    {
        let mut g = G::default();
        let (mut taken_n, mut taken_e): (Vec<u32>, Vec<u32>) = g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                a {1u32} -- e0 {100u32} --> _b {2u32} -- _e1 {200u32} --> c {3u32}
                    -- _e2 {300u32} --> _d {4u32} -- e3 {400u32} --> _e {5u32}
            );
            ctx.take_nodes_edges::<Vec<u32>, Vec<u32>>([c, a], [e3, e0])
        });
        taken_n.sort_unstable();
        taken_e.sort_unstable();
        assert_eq!(taken_n, vec![1, 3], "both requested nodes returned");
        assert_eq!(taken_e, vec![100, 400], "both explicit edges returned once");
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (3, 0), "c,a removed; all edges gone (cascade + explicit)");
    }

}

// ---------------------------------------------------------------------------
// Clear harness. Both `Context::clear` and `Graph::clear` route through the
// backend's `take_nodes_edges_unchecked`. The default impl in the
// `RemoveNode` trait isn't safe under Vec-backed `swap_remove` — backends
// must override it. We only register this harness against backends that do.
// ---------------------------------------------------------------------------

fn run_clear<G>()
where
    G: Default + 'static,
    G: Graph + InsertNode + InsertEdge + RemoveNode,
    G: Bigraph + GraphProperty<Node = u32, Edge = u32>,
    G: for<'r> Directed<'r>,
    G::Endpoints: for<'scope> Map<
        ScopedNIx<'scope, <G as GraphProperty>::NodeIx>,
        Mapped = [ScopedNIx<'scope, <G as GraphProperty>::NodeIx>; 2],
    >,
    G::NodeIx: Debug,
    G::EdgeIx: Debug,
{
    // `Context::clear` empties the graph from inside `scope_mut`.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                _a {10u32} -- _e0 {100u32} --> _b {20u32} -- _e1 {200u32} --> _c {30u32}
            );
            ctx.clear();
        });
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (0, 0), "after Context::clear");
    }

    // `Graph::clear` is the same routine on the bare graph.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            safegraph::graph!(
                &mut *ctx =>
                _a {10u32} -- _e0 {100u32} --> _b {20u32} -- _e1 {200u32} --> _c {30u32}
            );
        });
        g.clear();
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (0, 0), "after Graph::clear");
    }
}

// ---------------------------------------------------------------------------
// Mutation harness: requires `UpdateNode` / `UpdateEdge` on the inner graph.
// ---------------------------------------------------------------------------

fn run_update<G>()
where
    G: Default + 'static,
    G: Graph + InsertNode + InsertEdge,
    G: for<'r> UpdateNode<'r> + UpdateEdge,
    G: Bigraph + GraphProperty<Node = u32, Edge = u32>,
    G: for<'r> Directed<'r>,
    G::Endpoints: for<'scope> Map<
        ScopedNIx<'scope, <G as GraphProperty>::NodeIx>,
        Mapped = [ScopedNIx<'scope, <G as GraphProperty>::NodeIx>; 2],
    >,
    G::NodeIx: Debug,
    G::EdgeIx: Debug,
{
    // `node_mut` / `edge_mut` write through to the underlying data.
    let mut g = G::default();
    g.scope_mut(|mut ctx| {
        safegraph::graph!(
            &mut *ctx =>
            a {10u32} -- e0 {100u32} --> b {20u32} -- _e1 {200u32} --> _c {30u32}
        );
        *ctx.node_mut(b) = 999;
        *ctx.edge_mut(e0) = 555;
        assert_eq!(*ctx.node(b), 999);
        assert_eq!(*ctx.edge(e0), 555);
        // `walks_from_mut` / `walks_of_mut` currently yield empty iterators on
        // every raw_graph backend (the mutable walk machinery is a TODO).
        // Verify the methods exist, can be called, and don't yield anything.
        let walks_from: Vec<_> = ctx.walks_from_mut(a).map(|w| w.into_parts().0).collect();
        assert!(walks_from.is_empty(), "walks_from_mut is unimplemented (yields empty)");
        let walks_of: Vec<_> = ctx.walks_of_mut(b).map(|w| w.into_parts().0).collect();
        assert!(walks_of.is_empty(), "walks_of_mut is unimplemented (yields empty)");
    });
}

// ---------------------------------------------------------------------------
// Unique-index harness: requires `UniqueNode` / `UniqueEdge` (map-backed).
// ---------------------------------------------------------------------------

fn run_unique<G>()
where
    G: Default + 'static,
    G: Graph + InsertNode + InsertEdge + UniqueNode + UniqueEdge,
    G: Bigraph + GraphProperty<Node = u32, Edge = u32>,
    G: for<'r> Directed<'r>,
    G::Endpoints: for<'scope> Map<
        ScopedNIx<'scope, <G as GraphProperty>::NodeIx>,
        Mapped = [ScopedNIx<'scope, <G as GraphProperty>::NodeIx>; 2],
    >,
    G::NodeIx: Debug,
    G::EdgeIx: Debug,
{
    let mut g = G::default();
    g.scope_mut(|mut ctx| {
        safegraph::graph!(
            &mut *ctx =>
            a {10u32} -- e0 {100u32} --> b {20u32} -- _e1 {200u32} --> _c {30u32}
        );
        // node_index / edge_index look up by value.
        assert_eq!(ctx.node_index(10u32), Some(a));
        assert_eq!(ctx.node_index(999u32), None);
        assert_eq!(ctx.edge_index(100u32), Some(e0));
        assert_eq!(ctx.edge_index(999u32), None);

        // get_or_insert_node returns the existing index when present.
        let existing = ctx.get_or_insert_node(10u32);
        assert_eq!(existing, a);
        let fresh = ctx.get_or_insert_node(40u32);
        assert_eq!(*ctx.node(fresh), 40);
        // get_or_insert_edge has the same semantics.
        let ep_e0 = ctx.endpoints(e0);
        let existing_e = ctx.get_or_insert_edge(100u32, ep_e0);
        assert_eq!(existing_e, e0);
    });
}

// ---------------------------------------------------------------------------
// extend_graph harness: builds two graphs, merges, asserts on the result.
// ---------------------------------------------------------------------------

fn run_extend<G>()
where
    G: Default + 'static,
    G: Graph + InsertNode + InsertEdge,
    G: Bigraph + GraphProperty<Node = u32, Edge = u32>,
    G: for<'r> Directed<'r>,
    G::Endpoints: for<'scope> Map<
        ScopedNIx<'scope, <G as GraphProperty>::NodeIx>,
        Mapped = [ScopedNIx<'scope, <G as GraphProperty>::NodeIx>; 2],
    >,
    G::NodeIx: Debug,
    G::EdgeIx: Debug,
{
    let mut g = G::default();
    g.scope_mut(|mut ctx| {
        safegraph::graph!(
            &mut *ctx =>
            _a {1u32} -- _e {10u32} --> _b {2u32}
        );
    });
    let mut other = G::default();
    other.scope_mut(|mut ctx| {
        safegraph::graph!(
            &mut *ctx =>
            _x {3u32} -- _e {20u32} --> _y {4u32}
        );
    });
    g.extend_graph(other);
    let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
    assert_eq!((n, e), (4, 2), "extend_graph merges nodes and edges");
    let mut node_values: Vec<u32> = g.scope(|ctx| ctx.nodes().copied().collect());
    node_values.sort();
    assert_eq!(node_values, vec![1, 2, 3, 4]);
    let mut edge_values: Vec<u32> = g.scope(|ctx| ctx.edges().copied().collect());
    edge_values.sort();
    assert_eq!(edge_values, vec![10, 20]);
}

// ---------------------------------------------------------------------------
// Backend registration macros.
// ---------------------------------------------------------------------------

macro_rules! backend {
    ($name:ident, $G:ty) => {
        #[test]
        fn $name() {
            run_all::<$G>();
        }
    };
}

macro_rules! backend_extend {
    ($name:ident, $G:ty) => {
        #[test]
        fn $name() {
            run_extend::<$G>();
        }
    };
}

macro_rules! backend_update {
    ($name:ident, $G:ty) => {
        #[test]
        fn $name() {
            run_update::<$G>();
        }
    };
}

macro_rules! backend_unique {
    ($name:ident, $G:ty) => {
        #[test]
        fn $name() {
            run_unique::<$G>();
        }
    };
}

macro_rules! backend_clear {
    ($name:ident, $G:ty) => {
        #[test]
        fn $name() {
            run_clear::<$G>();
        }
    };
}

// Aliases used by the per-backend tests below.
type BTreeLinkedG = LinkedAdjEdgeGraph<
    BTreeMap<u32, LinkedNodeRepr<Option<u32>>>,
    BTreeMap<u32, EdgeRepr<u32, Option<u32>>>,
>;
// `FlatAdjEdgeGraph` instantiations. `IS = TNone` keeps only outgoing
// adjacency (incoming queries scan); a set `IS` maintains an O(in-degree)
// reverse index.
type VecFlatG = FlatAdjEdgeGraph<Vec<(u32, FlatNodeRepr<Vec<(u32, u32)>, TNone>)>>;
type BTreeFlatG = FlatAdjEdgeGraph<BTreeMap<u32, FlatNodeRepr<BTreeMap<u32, u32>, TNone>>>;
// Maintained reverse index: IS = Vec (insertion order) and IS = BTreeSet (ordered).
type VecFlatVecISG =
    FlatAdjEdgeGraph<Vec<(u32, FlatNodeRepr<Vec<(u32, u32)>, Vec<FlatEdgeIx<u32, u32>>>)>>;
type VecFlatBSetG =
    FlatAdjEdgeGraph<Vec<(u32, FlatNodeRepr<Vec<(u32, u32)>, BTreeSet<FlatEdgeIx<u32, u32>>>)>>;

backend!(linked_adj_edge_vec, VecGraph<u32, u32>);
backend!(linked_adj_edge_btree, BTreeLinkedG);
backend!(flat_adj_edge_vec, VecFlatG);
backend!(flat_adj_edge_btree, BTreeFlatG);
backend!(flat_adj_edge_vec_vecis, VecFlatVecISG);
backend!(flat_adj_edge_vec_bset, VecFlatBSetG);

// Update tests: only Vec-backed inner graphs implement `UpdateNode` /
// `UpdateEdge` (map keys are immutable).
backend_update!(linked_adj_edge_vec_update, VecGraph<u32, u32>);
backend_update!(flat_adj_edge_vec_update, VecFlatG);
backend_update!(flat_adj_edge_vec_vecis_update, VecFlatVecISG);
backend_update!(flat_adj_edge_vec_bset_update, VecFlatBSetG);

// Unique-index tests: only `LinkedAdjEdgeGraph` and `HyperGraph` implement
// `UniqueEdge` (FlatAdjEdge map-backed lacks the impl entirely).
backend_unique!(linked_adj_edge_btree_unique, BTreeLinkedG);

// `clear` tests: backends that override `take_nodes_edges_unchecked` handle
// Vec-backed swap_remove correctly during a full graph wipe.
// `FlatAdjEdgeGraph` carries that override (descending-order removal), so
// every `IS` instantiation is covered here.
backend_clear!(linked_adj_edge_vec_clear, VecGraph<u32, u32>);
backend_clear!(linked_adj_edge_btree_clear, BTreeLinkedG);
backend_clear!(flat_adj_edge_vec_clear, VecFlatG);
backend_clear!(flat_adj_edge_btree_clear, BTreeFlatG);
backend_clear!(flat_adj_edge_vec_vecis_clear, VecFlatVecISG);
backend_clear!(flat_adj_edge_vec_bset_clear, VecFlatBSetG);

// `extend_graph` tests exercise `drain`. `LinkedAdjEdge` and `FlatAdjEdge`
// both drain correctly across node/edge pairings; `FlatAdjBothEdge` currently
// loses edges during the merge, so it stays excluded.
backend_extend!(linked_adj_edge_vec_extend, VecGraph<u32, u32>);
backend_extend!(linked_adj_edge_btree_extend, BTreeLinkedG);
backend_extend!(flat_adj_edge_vec_extend, VecFlatG);
backend_extend!(flat_adj_edge_btree_extend, BTreeFlatG);

// `CsMatGraph` is constructed up-front from triplets, so it skips
// insert/remove and reuses only `check_read_only`.
#[cfg(feature = "matrix")]
#[test]
fn matrix_csr() {
    let g: CsMatGraph<u32, u32> = CsMatGraph::from_triplets(
        vec![10u32, 20u32, 30u32],
        vec![(0u32, 1u32, 100u32), (1u32, 2u32, 200u32)],
    );
    check_read_only(
        &g,
        [0u32, 1u32, 2u32],
        [MatrixEdgeIx(0, 1), MatrixEdgeIx(1, 2)],
    );
}

// Collect every edge of a raw backend as a sorted `(from, to, value)` list,
// reading directly through the unchecked API (the graph is concrete and valid).
// Fully-qualifies `GraphOperation` rather than importing it, so the file's
// existing method-syntax call sites stay unambiguous against the `Graph` facade.
fn directed_edge_triples<G>(g: &G) -> Vec<(u32, u32, u32)>
where
    G: GraphProperty<NodeIx = u32, Edge = u32, Endpoints = [u32; 2]>
        + for<'r> safegraph::graph::GraphOperation<'r>,
{
    let mut out: Vec<(u32, u32, u32)> = Vec::new();
    for n in safegraph::graph::GraphOperation::node_indices(g) {
        for e in unsafe { safegraph::graph::GraphOperation::edge_indices_from_unchecked(g, n) } {
            let [from, to] = unsafe { safegraph::graph::GraphOperation::endpoints_unchecked(g, e) };
            let val = *unsafe { safegraph::graph::GraphOperation::edge_unchecked(g, e) };
            out.push((from, to, val));
        }
    }
    out.sort_unstable();
    out
}

#[cfg(feature = "matrix")]
#[test]
fn matrix_reverse_transposes_edges() {
    let mut g: CsMatGraph<u32, u32> = CsMatGraph::from_triplets(
        vec![10u32, 20u32, 30u32],
        vec![(0u32, 1u32, 100u32), (1u32, 2u32, 200u32)],
    );
    assert_eq!(directed_edge_triples(&g), vec![(0, 1, 100), (1, 2, 200)]);
    Graph::reverse(&mut g);
    // 0->1 becomes 1->0, 1->2 becomes 2->1; values move (no clone/add needed).
    assert_eq!(directed_edge_triples(&g), vec![(1, 0, 100), (2, 1, 200)]);
}

#[test]
fn flat_reverse_rebuilds_reversed_adjacency() {
    let mut g = VecFlatG::default();
    // VecFlat is not StableNode, so build via the unchecked insert API.
    let a = unsafe { InsertNode::insert_node_unchecked(&mut g, 10).unwrap() };
    let b = unsafe { InsertNode::insert_node_unchecked(&mut g, 20).unwrap() };
    let c = unsafe { InsertNode::insert_node_unchecked(&mut g, 30).unwrap() };
    unsafe { InsertEdge::insert_edge_unchecked(&mut g, 100, [a, b]).unwrap() };
    unsafe { InsertEdge::insert_edge_unchecked(&mut g, 200, [b, c]).unwrap() };
    assert_eq!(directed_edge_triples(&g), vec![(0, 1, 100), (1, 2, 200)]);
    Graph::reverse(&mut g);
    assert_eq!(directed_edge_triples(&g), vec![(1, 0, 100), (2, 1, 200)]);
}

// `Graph::stabilize` produces a tombstone-versioned wrapper that itself
// implements every safe op needed by `run_all`. This test exercises that
// alternate path (in addition to the `scope_mut` path used elsewhere) on a
// Vec-backed `LinkedAdjEdgeGraph`.
#[test]
fn stabilize_vec_backed() {
    let mut g = VecGraph::<u32, u32>::default().stabilize();
    let a = g.insert_node(10).unwrap();
    let b = g.insert_node(20).unwrap();
    let c = g.insert_node(30).unwrap();
    let e0 = g.insert_edge(100, [a, b]).unwrap();
    let e1 = g.insert_edge(200, [b, c]).unwrap();
    assert_eq!(g.nodes().count(), 3);
    assert_eq!(g.edges().count(), 2);
    assert_eq!(*g.node(a), 10);
    assert_eq!(*g.edge(e0), 100);
    assert_eq!(g.endpoints(e0), [a, b]);
    assert_eq!(g.endpoints(e1), [b, c]);

    // remove_edge (singular, from the Graph trait — Stabilized exposes it).
    g.remove_edge(e0);
    assert!(!g.contains_edge_index(e0));
    assert_eq!(g.edges().count(), 1);

    // take_node returns the previously-stored node value.
    let removed = g.take_node(c);
    assert_eq!(removed, 30);
    assert_eq!(g.nodes().count(), 2);
    assert_eq!(g.edges().count(), 0); // e1 cascaded
}

// ---------------------------------------------------------------------------
// Complex edges: self-loops and parallel edges.
//
// Graph shape (5 edges):
//   - self-loop on a:    e_aa  (value 10)
//   - parallel a -> b:   e_ab1 (value 20), e_ab2 (value 30)
//   - reverse b -> a:    e_ba  (value 40)
//   - simple a -> c:     e_ac  (value 50)
//
// We assert `edge_indices_from` / `edge_indices_of` both as the directed view
// (the bare graph) and as the [`Undirected`](safegraph::graph::undirected)
// wrapper. The wrapper redefines `edge_indices_from(nix)` to behave like
// `edge_indices_of(nix)` — i.e. yield every incident edge regardless of
// direction.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct ComplexIndices<S: GraphProperty> {
    a: S::NodeIx,
    b: S::NodeIx,
    c: S::NodeIx,
    e_aa: S::EdgeIx,
    e_ab1: S::EdgeIx,
    e_ab2: S::EdgeIx,
    e_ba: S::EdgeIx,
    e_ac: S::EdgeIx,
}

fn edge_values<S, I>(g: &S, eixs: I) -> BTreeSet<u32>
where
    S: Graph + StableNode + StableEdge + GraphProperty<Edge = u32>,
    I: IntoIterator<Item = S::EdgeIx>,
{
    eixs.into_iter().map(|eix| *g.edge(eix)).collect()
}

fn assert_directed_complex<S>(g: &S, ix: &ComplexIndices<S>)
where
    S: Graph + StableNode + StableEdge + GraphProperty<Node = u32, Edge = u32>,
    S: for<'r> Directed<'r>,
    S::EdgeIx: Ord,
{
    // Outgoing edges of `a`: the self-loop, both parallel a->b edges, and a->c.
    let from_a = edge_values(g, g.edge_indices_from(ix.a));
    assert_eq!(
        from_a,
        BTreeSet::from([10, 20, 30, 50]),
        "directed edge_indices_from(a)"
    );
    let from_b = edge_values(g, g.edge_indices_from(ix.b));
    assert_eq!(from_b, BTreeSet::from([40]), "directed edge_indices_from(b)");
    let from_c = edge_values(g, g.edge_indices_from(ix.c));
    assert!(from_c.is_empty(), "directed edge_indices_from(c)");

    // Incoming edges (Directed::edge_indices_to).
    let to_a = edge_values(g, g.edge_indices_to(ix.a));
    assert_eq!(
        to_a,
        BTreeSet::from([10, 40]),
        "directed edge_indices_to(a): self-loop + reverse"
    );
    let to_b = edge_values(g, g.edge_indices_to(ix.b));
    assert_eq!(
        to_b,
        BTreeSet::from([20, 30]),
        "directed edge_indices_to(b): both parallel edges"
    );
    let to_c = edge_values(g, g.edge_indices_to(ix.c));
    assert_eq!(to_c, BTreeSet::from([50]));

    // edge_indices_of: every incident edge, self-loop yielded exactly once.
    // Use a Vec (not a set) so we can verify the self-loop isn't double-counted.
    let of_a_vec: Vec<u32> = g.edge_indices_of(ix.a).map(|eix| *g.edge(eix)).collect();
    let mut of_a_sorted = of_a_vec.clone();
    of_a_sorted.sort();
    assert_eq!(
        of_a_sorted,
        vec![10, 20, 30, 40, 50],
        "directed edge_indices_of(a): self-loop yielded once"
    );
    let of_b = edge_values(g, g.edge_indices_of(ix.b));
    assert_eq!(of_b, BTreeSet::from([20, 30, 40]));
    let of_c = edge_values(g, g.edge_indices_of(ix.c));
    assert_eq!(of_c, BTreeSet::from([50]));
}

fn assert_undirected_complex<S>(und: &S, ix: &ComplexIndices<S>)
where
    S: Graph + StableNode + StableEdge + GraphProperty<Node = u32, Edge = u32>,
    S::EdgeIx: Ord,
{
    // In the undirected view `edge_indices_from(nix)` should yield every
    // incident edge — exactly what the directed `edge_indices_of(nix)` returns.
    // Self-loops appear once, both parallel edges show up regardless of
    // direction.
    let from_a_vec: Vec<u32> = und.edge_indices_from(ix.a).map(|eix| *und.edge(eix)).collect();
    let mut from_a_sorted = from_a_vec.clone();
    from_a_sorted.sort();
    assert_eq!(
        from_a_sorted,
        vec![10, 20, 30, 40, 50],
        "undirected edge_indices_from(a)"
    );
    let from_b = edge_values(und, und.edge_indices_from(ix.b));
    assert_eq!(from_b, BTreeSet::from([20, 30, 40]));
    let from_c = edge_values(und, und.edge_indices_from(ix.c));
    assert_eq!(from_c, BTreeSet::from([50]));

    // `edge_indices_of` and `edge_indices_from` should agree on the undirected
    // view (the wrapper aliases them).
    let of_a = edge_values(und, und.edge_indices_of(ix.a));
    assert_eq!(of_a, BTreeSet::from([10, 20, 30, 40, 50]));
    let of_b = edge_values(und, und.edge_indices_of(ix.b));
    assert_eq!(of_b, BTreeSet::from([20, 30, 40]));
    let of_c = edge_values(und, und.edge_indices_of(ix.c));
    assert_eq!(of_c, BTreeSet::from([50]));
}

fn run_complex<S, F>(build: F)
where
    S: Graph + StableNode + StableEdge + GraphProperty<Node = u32, Edge = u32> + 'static,
    S: for<'r> Directed<'r>,
    S::EdgeIx: Ord,
    F: FnOnce() -> (S, ComplexIndices<S>),
{
    let (g, ix) = build();
    assert_directed_complex(&g, &ix);
    // `.undirected()` consumes the graph but `ix` (a copy of `NodeIx`/`EdgeIx`)
    // is still valid because `Undirected` preserves both index types.
    let und = g.undirected();
    let ix_und = ComplexIndices::<safegraph::graph::undirected::Undirected<S>> {
        a: ix.a,
        b: ix.b,
        c: ix.c,
        e_aa: ix.e_aa,
        e_ab1: ix.e_ab1,
        e_ab2: ix.e_ab2,
        e_ba: ix.e_ba,
        e_ac: ix.e_ac,
    };
    assert_undirected_complex(&und, &ix_und);
}

#[test]
fn linked_adj_edge_vec_complex() {
    run_complex(|| {
        let mut g = VecGraph::<u32, u32>::default().stabilize();
        let a = g.insert_node(1).unwrap();
        let b = g.insert_node(2).unwrap();
        let c = g.insert_node(3).unwrap();
        let e_aa = g.insert_edge(10, [a, a]).unwrap();
        let e_ab1 = g.insert_edge(20, [a, b]).unwrap();
        let e_ab2 = g.insert_edge(30, [a, b]).unwrap();
        let e_ba = g.insert_edge(40, [b, a]).unwrap();
        let e_ac = g.insert_edge(50, [a, c]).unwrap();
        (
            g,
            ComplexIndices {
                a,
                b,
                c,
                e_aa,
                e_ab1,
                e_ab2,
                e_ba,
                e_ac,
            },
        )
    });
}

#[test]
fn linked_adj_edge_btree_complex() {
    run_complex(|| {
        let mut g = BTreeLinkedG::default();
        // BTreeMap-backed: node value IS the key.
        let a = g.insert_node(1u32).unwrap();
        let b = g.insert_node(2u32).unwrap();
        let c = g.insert_node(3u32).unwrap();
        let e_aa = g.insert_edge(10u32, [a, a]).unwrap();
        let e_ab1 = g.insert_edge(20u32, [a, b]).unwrap();
        let e_ab2 = g.insert_edge(30u32, [a, b]).unwrap();
        let e_ba = g.insert_edge(40u32, [b, a]).unwrap();
        let e_ac = g.insert_edge(50u32, [a, c]).unwrap();
        (
            g,
            ComplexIndices {
                a,
                b,
                c,
                e_aa,
                e_ab1,
                e_ab2,
                e_ba,
                e_ac,
            },
        )
    });
}

#[test]
fn flat_adj_edge_vec_complex() {
    run_complex(|| {
        let mut g = VecFlatG::default().stabilize();
        let a = g.insert_node(1).unwrap();
        let b = g.insert_node(2).unwrap();
        let c = g.insert_node(3).unwrap();
        let e_aa = g.insert_edge(10, [a, a]).unwrap();
        let e_ab1 = g.insert_edge(20, [a, b]).unwrap();
        let e_ab2 = g.insert_edge(30, [a, b]).unwrap();
        let e_ba = g.insert_edge(40, [b, a]).unwrap();
        let e_ac = g.insert_edge(50, [a, c]).unwrap();
        (
            g,
            ComplexIndices {
                a,
                b,
                c,
                e_aa,
                e_ab1,
                e_ab2,
                e_ba,
                e_ac,
            },
        )
    });
}

#[test]
fn flat_adj_edge_btree_complex() {
    run_complex(|| {
        let mut g = BTreeFlatG::default();
        let a = g.insert_node(1u32).unwrap();
        let b = g.insert_node(2u32).unwrap();
        let c = g.insert_node(3u32).unwrap();
        let e_aa = g.insert_edge(10u32, [a, a]).unwrap();
        let e_ab1 = g.insert_edge(20u32, [a, b]).unwrap();
        let e_ab2 = g.insert_edge(30u32, [a, b]).unwrap();
        let e_ba = g.insert_edge(40u32, [b, a]).unwrap();
        let e_ac = g.insert_edge(50u32, [a, c]).unwrap();
        (
            g,
            ComplexIndices {
                a,
                b,
                c,
                e_aa,
                e_ab1,
                e_ab2,
                e_ba,
                e_ac,
            },
        )
    });
}

// Directed-view checks for a `FlatAdjEdgeGraph` with a maintained reverse
// index (`IS = Vec`): `edge_indices_from` / `edge_indices_to` /
// `edge_indices_of` over a graph with self-loops and parallel edges,
// exercised inside `scope_mut` (which provides `StableNode + StableEdge`
// via the `Context`).
#[test]
fn flat_adj_edge_vec_vecis_complex_directed_only() {
    let mut g = VecFlatVecISG::default();
    g.scope_mut(|mut ctx| {
        let a = ctx.insert_node(1u32).unwrap();
        let b = ctx.insert_node(2u32).unwrap();
        let c = ctx.insert_node(3u32).unwrap();
        let _e_aa = ctx.insert_edge(10u32, [a, a]).unwrap();
        let _e_ab1 = ctx.insert_edge(20u32, [a, b]).unwrap();
        let _e_ab2 = ctx.insert_edge(30u32, [a, b]).unwrap();
        let _e_ba = ctx.insert_edge(40u32, [b, a]).unwrap();
        let _e_ac = ctx.insert_edge(50u32, [a, c]).unwrap();

        let from_a: BTreeSet<u32> =
            ctx.edge_indices_from(a).map(|eix| *ctx.edge(eix)).collect();
        assert_eq!(from_a, BTreeSet::from([10, 20, 30, 50]));
        let to_a: BTreeSet<u32> = ctx.edge_indices_to(a).map(|eix| *ctx.edge(eix)).collect();
        assert_eq!(to_a, BTreeSet::from([10, 40]));
        let mut of_a: Vec<u32> = ctx.edge_indices_of(a).map(|eix| *ctx.edge(eix)).collect();
        of_a.sort();
        assert_eq!(of_a, vec![10, 20, 30, 40, 50]);
    });
}

// `CsMatGraph` cannot store parallel edges between the same nodes because a
// sparse matrix has one cell per `(row, col)` pair. `from_triplets` panics
// when handed duplicate triplets.
#[cfg(feature = "matrix")]
#[test]
#[should_panic(expected = "duplicate edge")]
fn matrix_duplicate_edge_panics() {
    let _: CsMatGraph<u32, u32> = CsMatGraph::from_triplets(
        vec![10u32, 20u32],
        vec![(0u32, 1u32, 100u32), (0u32, 1u32, 200u32)],
    );
}

// Self-loops are fine for matrix (one cell, one edge). Matrix is natively
// `StableNode + StableEdge`, so the safe `Graph` trait methods apply.
#[cfg(feature = "matrix")]
#[test]
fn matrix_self_loop_and_unique_edges_complex() {
    let g: CsMatGraph<u32, u32> = CsMatGraph::from_triplets(
        vec![1u32, 2u32, 3u32],
        vec![
            (0, 0, 10), // self-loop on node 0
            (0, 1, 20),
            (1, 0, 40),
            (0, 2, 50),
        ],
    );
    // Directed view.
    let from_0: BTreeSet<u32> = g.edge_indices_from(0).map(|eix| *g.edge(eix)).collect();
    assert_eq!(from_0, BTreeSet::from([10, 20, 50]));
    let to_0: BTreeSet<u32> = g.edge_indices_to(0).map(|eix| *g.edge(eix)).collect();
    assert_eq!(to_0, BTreeSet::from([10, 40]));
    let mut of_0_vec: Vec<u32> = g.edge_indices_of(0).map(|eix| *g.edge(eix)).collect();
    of_0_vec.sort();
    assert_eq!(
        of_0_vec,
        vec![10, 20, 40, 50],
        "self-loop yielded once in edge_indices_of"
    );

    // Undirected wrapper: from = of.
    let und = g.undirected();
    let from_0_und: BTreeSet<u32> = und.edge_indices_from(0).map(|eix| *und.edge(eix)).collect();
    assert_eq!(from_0_und, BTreeSet::from([10, 20, 40, 50]));
    let of_0_und: BTreeSet<u32> = und.edge_indices_of(0).map(|eix| *und.edge(eix)).collect();
    assert_eq!(of_0_und, BTreeSet::from([10, 20, 40, 50]));
}

// ---------------------------------------------------------------------------
// LinkedAdjEdge<Vec, Vec> swap-remove correctness.
//
// `take_edge_unchecked` on the Vec-backed edge store does `swap_remove`: the
// last edge in the storage is relocated into the freed slot, then every node
// adjacency list that referenced the old last slot has to be rewritten to
// point at the new slot. The same is true for `take_node_unchecked`. These
// tests build graphs containing several edges (including self-loops and
// reciprocal directed pairs) and verify after every removal that:
//
// * `len_node` / `len_edge` match,
// * the surviving edges' endpoints (encoded as node-value pairs) are exactly
//   what we expect,
// * every node's outgoing / incoming adjacency list yields the right edges.
// ---------------------------------------------------------------------------

/// Snapshot the graph as `(sorted node values, sorted (edge value, [tail value,
/// head value]) triples)`. Comparing snapshots is invariant under any internal
/// reindexing the backend performs. Specialised to `VecGraph<u32, u32>` so the
/// helper does not need to wrestle with `Map`/`Bigraph` HRTB bounds for
/// `Context`.
fn snapshot(g: &VecGraph<u32, u32>) -> (Vec<u32>, Vec<(u32, [u32; 2])>) {
    g.scope(|ctx| {
        let mut nodes: Vec<u32> = ctx.nodes().copied().collect();
        nodes.sort();
        let mut edges: Vec<(u32, [u32; 2])> = ctx
            .edge_indices()
            .map(|eix| {
                let ep: Vec<_> = ctx.endpoints(eix).into_iter().collect();
                let tail = *ctx.node(ep[0]);
                let head = *ctx.node(ep[1]);
                (*ctx.edge(eix), [tail, head])
            })
            .collect();
        edges.sort();
        (nodes, edges)
    })
}

#[test]
fn linked_adj_edge_swap_remove_edges_one_at_a_time() {
    // Build five edges including a self-loop and a reciprocal pair so the
    // adjacency-list rewrites need to touch every direction.
    fn fresh() -> VecGraph<u32, u32> {
        let mut g = VecGraph::<u32, u32>::default();
        g.scope_mut(|mut ctx| {
            let a = ctx.insert_node(1u32).unwrap();
            let b = ctx.insert_node(2u32).unwrap();
            let c = ctx.insert_node(3u32).unwrap();
            ctx.insert_edge(10u32, [a, b]).unwrap();
            ctx.insert_edge(20u32, [b, c]).unwrap();
            ctx.insert_edge(30u32, [c, a]).unwrap();
            ctx.insert_edge(40u32, [a, a]).unwrap(); // self-loop
            ctx.insert_edge(50u32, [a, b]).unwrap(); // parallel
        });
        g
    }

    // Initial snapshot — five edges total.
    let g = fresh();
    let (nodes, edges) = snapshot(&g);
    assert_eq!(nodes, vec![1, 2, 3]);
    assert_eq!(
        edges,
        vec![
            (10, [1, 2]), // a -> b
            (20, [2, 3]), // b -> c
            (30, [3, 1]), // c -> a
            (40, [1, 1]), // self-loop on a
            (50, [1, 2]), // parallel a -> b
        ]
    );

    // Remove each edge in turn (each remove triggers swap_remove + relink).
    // Helper that builds a fresh graph, locates the edge with the given
    // payload, removes it, and snapshots the result.
    let remove_by_value = |target: u32| -> (Vec<u32>, Vec<(u32, [u32; 2])>) {
        let mut g = fresh();
        g.scope_mut(|ctx| {
            let eix = ctx
                .edge_indices()
                .find(|&eix| *ctx.edge(eix) == target)
                .expect("edge with target value");
            ctx.remove_nodes_edges(None, Some(eix));
        });
        snapshot(&g)
    };

    let (_, after_10) = remove_by_value(10);
    assert_eq!(
        after_10,
        vec![(20, [2, 3]), (30, [3, 1]), (40, [1, 1]), (50, [1, 2])]
    );

    let (_, after_20) = remove_by_value(20);
    assert_eq!(
        after_20,
        vec![(10, [1, 2]), (30, [3, 1]), (40, [1, 1]), (50, [1, 2])]
    );

    let (_, after_30) = remove_by_value(30);
    assert_eq!(
        after_30,
        vec![(10, [1, 2]), (20, [2, 3]), (40, [1, 1]), (50, [1, 2])]
    );

    let (_, after_40) = remove_by_value(40);
    assert_eq!(
        after_40,
        vec![(10, [1, 2]), (20, [2, 3]), (30, [3, 1]), (50, [1, 2])]
    );

    let (_, after_50) = remove_by_value(50);
    assert_eq!(
        after_50,
        vec![(10, [1, 2]), (20, [2, 3]), (30, [3, 1]), (40, [1, 1])]
    );
}

#[test]
fn linked_adj_edge_swap_remove_drains_in_sequence() {
    // Remove every edge from the fresh graph one at a time. After each pass
    // the surviving set must shrink by exactly one entry and the graph must
    // remain consistent (snapshot survives a scope read each time).
    let mut g = VecGraph::<u32, u32>::default();
    g.scope_mut(|mut ctx| {
        safegraph::graph!(
            &mut *ctx =>
            a {1u32} -- _e {10u32} --> b {2u32} -- _e2 {20u32} --> c {3u32}
        );
        let _ = ctx.insert_edge(30u32, [c, a]).unwrap();
        let _ = ctx.insert_edge(40u32, [a, a]).unwrap();
        let _ = ctx.insert_edge(50u32, [a, b]).unwrap();
    });

    // Drain edges one at a time; after the last removal the edge count must
    // be 0 and node count unchanged.
    for expected_remaining in (0..5usize).rev() {
        g.scope_mut(|ctx| {
            // Pop the lowest-indexed edge (arbitrary but deterministic).
            let eix = Graph::edge_indices(&*ctx).next().unwrap();
            ctx.remove_nodes_edges(None, Some(eix));
        });
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!(n, 3, "nodes preserved through edge removal");
        assert_eq!(e, expected_remaining, "edge count decreases by one each pass");
    }
}

#[test]
fn linked_adj_edge_swap_remove_node_with_many_edges() {
    // Removing a node should drop every incident edge. Make `b` the hub of
    // four edges (incoming, outgoing, and a self-loop) so that removing `b`
    // forces the backend to walk and rewrite each adjacency list.
    let mut g = VecGraph::<u32, u32>::default();
    let b_value: u32 = 2;
    g.scope_mut(|mut ctx| {
        let a = ctx.insert_node(1u32).unwrap();
        let b = ctx.insert_node(b_value).unwrap();
        let c = ctx.insert_node(3u32).unwrap();
        let d = ctx.insert_node(4u32).unwrap();
        let _ = ctx.insert_edge(10u32, [a, b]).unwrap();
        let _ = ctx.insert_edge(20u32, [b, c]).unwrap();
        let _ = ctx.insert_edge(30u32, [d, b]).unwrap();
        let _ = ctx.insert_edge(40u32, [b, b]).unwrap(); // self-loop
        let _ = ctx.insert_edge(50u32, [a, c]).unwrap(); // not incident on b
        // Verify the build looks right before we tear it down.
        assert_eq!(ctx.nodes().count(), 4);
        assert_eq!(ctx.edges().count(), 5);
        let b_incident: BTreeSet<u32> = ctx.edges_of(b).copied().collect();
        assert_eq!(b_incident, [10, 20, 30, 40].into_iter().collect());

        ctx.remove_nodes_edges(Some(b), None);
    });
    // Only edge 50 (a→c) should survive, untouched.
    let (nodes, edges) = snapshot(&g);
    assert_eq!(nodes, vec![1, 3, 4]);
    assert_eq!(edges, vec![(50, [1, 3])]);
}

// ---------------------------------------------------------------------------
// Randomized model-based stress for the batched `take_nodes_edges_unchecked`
// override on `LinkedAdjEdgeGraph`. Every payload value is unique, so a
// reference model can track the expected graph by value. After every batch
// the full incidence structure is re-derived through `walks_from` /
// `walks_to` / `walks_of` — these traverse the adjacency chains, so a stale
// or mis-spliced pointer surfaces here even when the flat edge list still
// looks correct. Repeated rounds with fresh insertions in between exercise
// chained swap-remove relocations and re-use of patched chains.
// ---------------------------------------------------------------------------

/// Tiny deterministic xorshift64* PRNG so the test needs no dev-dependency.
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn run_batch_removal_stress<G>(seed: u64)
where
    G: Default + 'static,
    G: Graph + InsertNode + InsertEdge + RemoveNode,
    G: Bigraph + GraphProperty<Node = u32, Edge = u32>,
    G: for<'r> Directed<'r>,
    G::Endpoints: for<'scope> Map<
        ScopedNIx<'scope, <G as GraphProperty>::NodeIx>,
        Mapped = [ScopedNIx<'scope, <G as GraphProperty>::NodeIx>; 2],
    >,
    G::NodeIx: Debug,
    G::EdgeIx: Debug,
{
    let mut rng = XorShift(seed);

    // Reference model, keyed by payload value.
    let n_nodes = 2 + rng.below(14);
    let n_edges = rng.below(40);
    let mut model_nodes: BTreeSet<u32> = (0..n_nodes as u32).map(|i| 1000 + i).collect();
    let mut model_edges: BTreeMap<u32, [u32; 2]> = BTreeMap::new();
    let mut next_edge_value = 2000u32;

    let mut g = G::default();
    g.scope_mut(|mut ctx| {
        let values: Vec<u32> = model_nodes.iter().copied().collect();
        let nixs: Vec<_> = values
            .iter()
            .map(|&v| ctx.insert_node(v).unwrap())
            .collect();
        for _ in 0..n_edges {
            let t = rng.below(nixs.len());
            let h = rng.below(nixs.len());
            ctx.insert_edge(next_edge_value, [nixs[t], nixs[h]]).unwrap();
            model_edges.insert(next_edge_value, [values[t], values[h]]);
            next_edge_value += 1;
        }
    });

    for round in 0..8 {
        // Pick victims by value: ~25% of nodes, ~25% of edges. Edge victims
        // may be incident on node victims — the implementation must
        // deduplicate them against the cascade.
        let node_victims: Vec<u32> = model_nodes
            .iter()
            .copied()
            .filter(|_| rng.below(100) < 25)
            .collect();
        let edge_victims: Vec<u32> = model_edges
            .keys()
            .copied()
            .filter(|_| rng.below(100) < 25)
            .collect();

        g.scope_mut(|ctx| {
            let node_ixs: Vec<_> = node_victims
                .iter()
                .map(|&v| {
                    Graph::node_indices(&*ctx)
                        .find(|&nix| *ctx.node(nix) == v)
                        .expect("victim node present")
                })
                .collect();
            let edge_ixs: Vec<_> = edge_victims
                .iter()
                .map(|&v| {
                    Graph::edge_indices(&*ctx)
                        .find(|&eix| *ctx.edge(eix) == v)
                        .expect("victim edge present")
                })
                .collect();
            let (taken_nodes, taken_edges): (Vec<u32>, Vec<u32>) =
                ctx.take_nodes_edges(node_ixs, edge_ixs);
            assert_eq!(
                taken_nodes, node_victims,
                "seed {seed} round {round}: node payloads in input order"
            );
            assert_eq!(
                taken_edges, edge_victims,
                "seed {seed} round {round}: edge payloads in input order"
            );
        });

        for &v in &node_victims {
            model_nodes.remove(&v);
        }
        model_edges
            .retain(|_, ep| !node_victims.contains(&ep[0]) && !node_victims.contains(&ep[1]));
        for &v in &edge_victims {
            model_edges.remove(&v);
        }

        // Full structural verification against the model.
        g.scope(|ctx| {
            let mut got_nodes: Vec<u32> = ctx.nodes().copied().collect();
            got_nodes.sort();
            let want_nodes: Vec<u32> = model_nodes.iter().copied().collect();
            assert_eq!(got_nodes, want_nodes, "seed {seed} round {round}: nodes");

            let mut got_edges: Vec<(u32, u32, u32)> = ctx
                .edge_indices()
                .map(|eix| {
                    let ep: Vec<_> = ctx.endpoints(eix).into_iter().collect();
                    (*ctx.edge(eix), *ctx.node(ep[0]), *ctx.node(ep[1]))
                })
                .collect();
            got_edges.sort();
            let mut want_edges: Vec<(u32, u32, u32)> = model_edges
                .iter()
                .map(|(&e, &[t, h])| (e, t, h))
                .collect();
            want_edges.sort();
            assert_eq!(got_edges, want_edges, "seed {seed} round {round}: edges");

            for nix in Graph::node_indices(ctx) {
                let v = *ctx.node(nix);

                let mut got_out: Vec<(u32, u32)> = ctx
                    .walks_from(nix)
                    .map(|w| w.get())
                    .map(|(_, e, head)| (*e, *ctx.node(head)))
                    .collect();
                got_out.sort();
                let mut want_out: Vec<(u32, u32)> = model_edges
                    .iter()
                    .filter(|(_, ep)| ep[0] == v)
                    .map(|(&e, ep)| (e, ep[1]))
                    .collect();
                want_out.sort();
                assert_eq!(
                    got_out, want_out,
                    "seed {seed} round {round}: outgoing chain of {v}"
                );

                let mut got_in: Vec<(u32, u32)> = ctx
                    .walks_to(nix)
                    .map(|w| w.get())
                    .map(|(tail, _, e)| (*e, *ctx.node(tail)))
                    .collect();
                got_in.sort();
                let mut want_in: Vec<(u32, u32)> = model_edges
                    .iter()
                    .filter(|(_, ep)| ep[1] == v)
                    .map(|(&e, ep)| (e, ep[0]))
                    .collect();
                want_in.sort();
                assert_eq!(
                    got_in, want_in,
                    "seed {seed} round {round}: incoming chain of {v}"
                );

                // `walks_of` yields self-loops once (outgoing pass only).
                let mut got_of: Vec<(u32, u32)> = ctx
                    .walks_of(nix)
                    .map(|w| w.get())
                    .map(|(_, e, peer)| (*e, *ctx.node(peer)))
                    .collect();
                got_of.sort();
                let mut want_of: Vec<(u32, u32)> = model_edges
                    .iter()
                    .filter_map(|(&e, ep)| match (ep[0] == v, ep[1] == v) {
                        (true, _) => Some((e, ep[1])),
                        (false, true) => Some((e, ep[0])),
                        _ => None,
                    })
                    .collect();
                want_of.sort();
                assert_eq!(
                    got_of, want_of,
                    "seed {seed} round {round}: incident chain of {v}"
                );
            }
        });

        // Splice a few fresh edges among the survivors so the next round
        // also exercises insertion into freshly patched chains.
        if !model_nodes.is_empty() {
            let values: Vec<u32> = model_nodes.iter().copied().collect();
            let extra = rng.below(5);
            g.scope_mut(|mut ctx| {
                let nixs: Vec<_> = values
                    .iter()
                    .map(|&v| {
                        Graph::node_indices(&*ctx)
                            .find(|&nix| *ctx.node(nix) == v)
                            .expect("survivor present")
                    })
                    .collect();
                for _ in 0..extra {
                    let t = rng.below(nixs.len());
                    let h = rng.below(nixs.len());
                    ctx.insert_edge(next_edge_value, [nixs[t], nixs[h]]).unwrap();
                    model_edges.insert(next_edge_value, [values[t], values[h]]);
                    next_edge_value += 1;
                }
            });
        }
    }
}

#[test]
fn linked_adj_edge_vec_batch_removal_stress() {
    for seed in 1..=32u64 {
        run_batch_removal_stress::<VecGraph<u32, u32>>(seed);
    }
}

#[test]
fn linked_adj_edge_btree_batch_removal_stress() {
    for seed in 101..=116u64 {
        run_batch_removal_stress::<BTreeLinkedG>(seed);
    }
}

// Hybrid collections hit the asymmetric paths: Vec nodes + BTree edges has
// node relocations but never edge relocations; BTree nodes + Vec edges the
// reverse.
#[test]
fn linked_adj_edge_hybrid_batch_removal_stress() {
    type VecNodesBTreeEdges = LinkedAdjEdgeGraph<
        Vec<(u32, LinkedNodeRepr<Option<u32>>)>,
        BTreeMap<u32, EdgeRepr<u32, Option<u32>>>,
    >;
    type BTreeNodesVecEdges = LinkedAdjEdgeGraph<
        BTreeMap<u32, LinkedNodeRepr<u32>>,
        Vec<(u32, EdgeRepr<u32, u32>)>,
    >;
    for seed in 201..=216u64 {
        run_batch_removal_stress::<VecNodesBTreeEdges>(seed);
        run_batch_removal_stress::<BTreeNodesVecEdges>(seed);
    }
}

// ---------------------------------------------------------------------------
// Regression: `FlatAdjEdgeGraph::take_node_unchecked` stale-snapshot bug.
//
// A node with >=2 incoming edges from the SAME head over a `Vec` inner
// collection used to take its incoming `EdgeIx` snapshot once, then remove
// them one at a time; the first swap-remove relocated a still-pending entry,
// causing an out-of-range `swap_remove` (panic) or wrong-edge removal. The
// fix sorts the snapshot descending. Here the parallel incoming edges sit at
// inner indices 0 and 2 with a survivor at index 1 — the order that trips it.
// ---------------------------------------------------------------------------
fn run_take_node_parallel_incoming<G>()
where
    G: Default + 'static,
    G: Graph + InsertNode + InsertEdge + RemoveNode,
    G: Bigraph + GraphProperty<Node = u32, Edge = u32>,
    G: for<'r> Directed<'r>,
    G::Endpoints: for<'scope> Map<
        ScopedNIx<'scope, <G as GraphProperty>::NodeIx>,
        Mapped = [ScopedNIx<'scope, <G as GraphProperty>::NodeIx>; 2],
    >,
    G::NodeIx: Debug + Ord,
    G::EdgeIx: Debug + Ord,
{
    let mut g = G::default();
    g.scope_mut(|mut ctx| {
        let h = ctx.insert_node(1).unwrap();
        let s = ctx.insert_node(2).unwrap();
        let v = ctx.insert_node(3).unwrap();
        // head `h` outgoing order: v (idx 0), s (idx 1, survivor), v (idx 2).
        ctx.insert_edge(100, [h, v]).unwrap();
        ctx.insert_edge(200, [h, s]).unwrap();
        ctx.insert_edge(300, [h, v]).unwrap();
        ctx.remove_nodes_edges(Some(v), None);
    });
    g.scope(|ctx| {
        let mut nodes: Vec<u32> = ctx.nodes().copied().collect();
        nodes.sort();
        assert_eq!(nodes, vec![1, 2], "node v removed; h and s remain");
        let edges: Vec<(u32, u32, u32)> = ctx
            .edge_indices()
            .map(|eix| {
                let ep: Vec<_> = ctx.endpoints(eix).into_iter().collect();
                (*ctx.edge(eix), *ctx.node(ep[0]), *ctx.node(ep[1]))
            })
            .collect();
        assert_eq!(
            edges,
            vec![(200, 1, 2)],
            "only the h->s survivor remains, with its endpoints intact"
        );
    });
}

#[test]
fn flat_vec_vecis_take_node_parallel_incoming() {
    // Maintained reverse index (IS = Vec).
    run_take_node_parallel_incoming::<VecFlatVecISG>();
}

#[test]
fn flat_vec_tnone_take_node_parallel_incoming() {
    // No reverse index (IS = TNone) — incoming discovered by scan.
    run_take_node_parallel_incoming::<VecFlatG>();
}
