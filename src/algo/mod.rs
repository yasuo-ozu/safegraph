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
