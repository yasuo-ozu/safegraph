//! # Graph Algorithms
//!
//! A comprehensive suite of graph algorithms implemented as free functions.
//!
//! ## Categories
//!
//! | Category | Modules |
//! |----------|---------|
//! | **Traversal** | [`bfs`], [`dfs`] |
//! | **Ordering** | [`toposort`] |
//! | **Shortest Paths** | [`shortest_path`] |
//! | **Connectivity** | [`connectivity`], [`bridges`], [`bipartite`] |
//! | **Dominators** | [`dominators`] |
//! | **Path Enumeration** | [`simple_paths`] |
//! | **Ranking** | [`page_rank`] |
//! | **DAG Analysis** | [`tred`], [`feedback_arc_set`] |
//! | **Spanning Trees** | [`min_spanning_tree`] |
//! | **Network Flow** | [`max_flow`] |
//! | **Matching** | [`matching`] |
//!
//! ## API Pattern
//!
//! Each algorithm provides up to two variants:
//!
//! - **`foo(graph, ...)`** — safe version; requires `StableNode` and/or
//!   `StableEdge` and validates its index arguments (panics on an invalid index).
//! - **`foo_unchecked(graph, ...)`** — `unsafe` counterpart carrying the *same*
//!   `Stable*` bound(s) as `foo`, but skipping the argument validation. To run an
//!   algorithm on a non-`Stable*` graph (e.g. a plain `VecGraph`), wrap it via
//!   [`Graph::unsafe_assert_stable_node`](crate::graph::Graph::unsafe_assert_stable_node)
//!   / [`unsafe_assert_stable_edge`](crate::graph::Graph::unsafe_assert_stable_edge)
//!   (the two wrappers compose for algorithms needing both markers).
//!
//! Edge weights are extracted via closures `F: Fn(&G::Edge) -> W` rather than
//! requiring traits on the edge type.

pub mod bfs;
pub mod bipartite;
pub mod bridges;
pub mod connectivity;
pub mod dfs;
pub mod dominators;
pub mod feedback_arc_set;
pub mod matching;
pub mod max_flow;
pub mod min_spanning_tree;
pub mod page_rank;
pub mod shortest_path;
pub mod simple_paths;
pub mod toposort;
pub mod tred;

use std::borrow::Borrow;

use crate::graph::{GraphOperation, GraphProperty};

/// [`node_indices`](GraphOperation::node_indices) with every item cloned out
/// of its [`NodeIxRef`](GraphOperation::NodeIxRef): algorithms keep node
/// indices in sets, maps and stacks, so they need owned values.
pub(crate) type OwnedNodeIndices<'r, G> = std::iter::Map<
    <G as GraphOperation<'r>>::NodeIndices,
    fn(<G as GraphOperation<'r>>::NodeIxRef) -> <G as GraphProperty>::NodeIx,
>;

pub(crate) fn owned_node_indices<'r, G>(graph: &'r G) -> OwnedNodeIndices<'r, G>
where
    G: GraphOperation<'r> + ?Sized,
{
    <G as GraphOperation<'r>>::node_indices(graph)
        .map((|ix: G::NodeIxRef| ix.borrow().clone()) as fn(_) -> _)
}

/// [`edge_indices`](GraphOperation::edge_indices) with every item cloned out
/// of its [`EdgeIxRef`](GraphOperation::EdgeIxRef).
pub(crate) type OwnedEdgeIndices<'r, G> = std::iter::Map<
    <G as GraphOperation<'r>>::EdgeIndices,
    fn(<G as GraphOperation<'r>>::EdgeIxRef) -> <G as GraphProperty>::EdgeIx,
>;

pub(crate) fn owned_edge_indices<'r, G>(graph: &'r G) -> OwnedEdgeIndices<'r, G>
where
    G: GraphOperation<'r> + ?Sized,
{
    <G as GraphOperation<'r>>::edge_indices(graph)
        .map((|ix: G::EdgeIxRef| ix.borrow().clone()) as fn(_) -> _)
}

/// The two endpoints of a binary edge, cloned out of
/// [`endpoints_unchecked`](GraphOperation::endpoints_unchecked) without an
/// intermediate `Vec`.
///
/// # Safety
/// `edge_ix` must be a valid edge index of `graph`.
pub(crate) unsafe fn edge_pair<'r, G>(graph: &'r G, edge_ix: &G::EdgeIx) -> (G::NodeIx, G::NodeIx)
where
    G: GraphOperation<'r> + ?Sized,
{
    let mut ends = unsafe { <G as GraphOperation<'r>>::endpoints_unchecked(graph, edge_ix) }
        .into_iter()
        .map(|n| -> G::NodeIx { n.borrow().clone() });
    let a = ends.next().expect("a binary edge has two endpoints");
    let b = ends.next().expect("a binary edge has two endpoints");
    (a, b)
}
