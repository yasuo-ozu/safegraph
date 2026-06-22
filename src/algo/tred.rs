//! # Transitive Reduction and Closure
//!
//! Provides algorithms for computing the transitive reduction and the
//! transitive closure of a directed acyclic graph (DAG).
//!
//! The *transitive reduction** removes every edge whose reachability is
//! already implied by other paths, yielding the sparsest graph that preserves
//! reachability. The *transitive closure** adds an edge for every pair of
//! nodes connected by a directed path, yielding the densest such graph.
//!
//! ## Components
//!
//! - [`DagTransitiveReduction`] -- iterator yielding edges to keep* (non-redundant edges)
//! - [`dag_transitive_reduction`] -- safe constructor (requires `StableEdge`)
//! - [`DagTransitiveClosure`] -- iterator yielding all `(source, target)` reachability pairs
//! - [`dag_transitive_closure`] -- safe constructor (requires `StableNode`)
//!
//! ## Algorithm
//!
//! ```text
//!   Transitive reduction:
//!     For each edge (u, v):
//!       Remove (u, v) temporarily and check if v is still reachable
//!       from u via DFS. If reachable, the edge is redundant; otherwise keep it.
//!
//!   Transitive closure:
//!     For each node u:
//!       Run DFS from u. For every reachable node v, emit (u, v).
//!
//!   Example (diamond with shortcut):
//!
//!     0 ---> 1 ---> 3       0 ---> 1 ---> 3
//!     |             ^        |             ^
//!     +----> 2 -----+        +----> 2 -----+
//!     |             ^
//!     +------+------+  <-- shortcut 0->3 is redundant
//!
//!   Reduction removes the direct 0->3 edge.
//! ```
//!
//! ## Example
//!
//! ```rust
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::tred::{dag_transitive_reduction, dag_transitive_closure};
//!
//! let mut g = BTreeGraph::<u32, &str>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(3).unwrap();
//! g.insert_edge("a", [0, 1]).unwrap();
//! g.insert_edge("b", [1, 3]).unwrap();
//! g.insert_edge("shortcut", [0, 3]).unwrap(); // redundant
//!
//! // Transitive reduction: shortcut edge is omitted
//! let kept: Vec<_> = dag_transitive_reduction(&g).collect();
//! // kept contains edges "a" and "b", but not "shortcut"
//!
//! // Transitive closure: all reachable pairs
//! let pairs: Vec<_> = dag_transitive_closure(&g).collect();
//! // pairs include (0,1), (0,3), (1,3)
//! ```

use std::collections::HashSet;

use crate::graph::capability::{Bigraph, Directed, StableEdge, StableNode};
use crate::graph::Graph;

/// Iterator that lazily yields edges forming the transitive reduction of a DAG.
///
/// Each yielded edge is one that should be kept* — edges not yielded are redundant
/// (their reachability is implied by other paths).
/// `E` is the edge-index iterator type (`<G as GraphOperation<'r>>::EdgeIndices`),
/// a separate type parameter so the struct carries no `Graph` bound.
pub struct DagTransitiveReduction<'r, G: ?Sized, E> {
    graph: &'r G,
    edges: E,
}

/// Returns an iterator over the edges in the transitive reduction of a DAG.
pub fn dag_transitive_reduction<'r, G>(
    graph: &'r G,
) -> DagTransitiveReduction<'r, G, <G as crate::graph::GraphOperation<'r>>::EdgeIndices>
where
    G: Graph + Directed<'r> + Bigraph + StableEdge + ?Sized,
{
    DagTransitiveReduction {
        graph,
        edges: <_ as crate::graph::GraphOperation<'_>>::edge_indices(graph),
    }
}

impl<'r, G> Iterator
    for DagTransitiveReduction<'r, G, <G as crate::graph::GraphOperation<'r>>::EdgeIndices>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
{
    type Item = G::EdgeIx;

    fn next(&mut self) -> Option<G::EdgeIx> {
        loop {
            let eix = self.edges.next()?;
            let tail = unsafe { self.graph.edge_tail_index_unchecked(eix) };
            let head = unsafe { self.graph.edge_head_index_unchecked(eix) };

            let mut reachable_via_other = false;
            for other_eix in unsafe {
                <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(
                    self.graph, tail,
                )
            } {
                if other_eix == eix {
                    continue;
                }
                let other_head = unsafe { self.graph.edge_head_index_unchecked(other_eix) };
                if unsafe { can_reach(self.graph, other_head, head) } {
                    reachable_via_other = true;
                    break;
                }
            }

            if !reachable_via_other {
                return Some(eix);
            }
        }
    }
}

/// Iterator that lazily yields `(source, target)` pairs forming the transitive closure.
///
/// Each pair represents a reachability relationship in the DAG.
/// `N` is the node-index type (`G::NodeIx`); it is a separate type parameter
/// so the struct carries no `Graph` bound.
pub struct DagTransitiveClosure<'r, G: ?Sized, N> {
    graph: &'r G,
    nodes: Vec<N>,
    node_idx: usize,
    // DFS state for current source node
    dfs_stack: Vec<N>,
    dfs_visited: HashSet<N>,
    current_source: Option<N>,
    // Buffer of pending pairs to yield
    pending: Vec<(N, N)>,
}

/// Returns an iterator over all `(source, target)` reachability pairs in the DAG.
pub fn dag_transitive_closure<'r, G>(graph: &'r G) -> DagTransitiveClosure<'r, G, G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    let nodes: Vec<G::NodeIx> =
        <_ as crate::graph::GraphOperation<'_>>::node_indices(graph).collect();
    DagTransitiveClosure {
        graph,
        nodes,
        node_idx: 0,
        dfs_stack: Vec::new(),
        dfs_visited: HashSet::new(),
        current_source: None,
        pending: Vec::new(),
    }
}

impl<'r, G> Iterator for DagTransitiveClosure<'r, G, G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    type Item = (G::NodeIx, G::NodeIx);

    fn next(&mut self) -> Option<(G::NodeIx, G::NodeIx)> {
        loop {
            // Yield from pending buffer first
            if let Some(pair) = self.pending.pop() {
                return Some(pair);
            }

            // Try to advance current DFS
            if let Some(current) = self.dfs_stack.pop() {
                let source = self.current_source.unwrap();
                let succs: Vec<G::NodeIx> =
                    unsafe { self.graph.neighbor_indices_from_unchecked(current) }.collect();
                for succ in succs {
                    if self.dfs_visited.insert(succ) {
                        self.dfs_stack.push(succ);
                        self.pending.push((source, succ));
                    }
                }
                continue;
            }

            // Move to next source node
            if self.node_idx >= self.nodes.len() {
                return None;
            }
            let source = self.nodes[self.node_idx];
            self.node_idx += 1;
            self.current_source = Some(source);
            self.dfs_visited.clear();
            self.dfs_visited.insert(source);
            self.dfs_stack.clear();
            self.dfs_stack.push(source);
        }
    }
}

/// Helper: check if `target` is reachable from `source` via DFS.
unsafe fn can_reach<'r, G>(graph: &'r G, source: G::NodeIx, target: G::NodeIx) -> bool
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    if source == target {
        return true;
    }
    let mut visited = HashSet::new();
    let mut stack = vec![source];
    visited.insert(source);

    while let Some(node) = stack.pop() {
        let succs: Vec<G::NodeIx> = graph.neighbor_indices_from_unchecked(node).collect();
        for succ in succs {
            if succ == target {
                return true;
            }
            if visited.insert(succ) {
                stack.push(succ);
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BTreeGraph;

    #[test]
    fn transitive_reduction_diamond() {
        // Diamond: 0->1, 0->2, 1->3, 2->3, 0->3
        // Edge 0->3 is redundant (reachable via 0->1->3 or 0->2->3)
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("1->3", [1, 3]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();
        g.insert_edge("0->3", [0, 3]).unwrap();

        let keep: HashSet<_> = dag_transitive_reduction(&g).collect();
        // Should keep 0->1, 0->2, 1->3, 2->3 but NOT 0->3
        assert_eq!(keep.len(), 4);
        assert!(keep.contains(&"0->1"));
        assert!(keep.contains(&"0->2"));
        assert!(keep.contains(&"1->3"));
        assert!(keep.contains(&"2->3"));
        assert!(!keep.contains(&"0->3"));
    }

    #[test]
    fn transitive_reduction_linear() {
        // Linear: 0->1->2->3 with shortcut 0->2
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();

        let keep: HashSet<_> = dag_transitive_reduction(&g).collect();
        assert_eq!(keep.len(), 3);
        assert!(!keep.contains(&"0->2"));
    }

    #[test]
    fn transitive_closure_diamond() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("1->3", [1, 3]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();

        let closure: HashSet<_> = dag_transitive_closure(&g).collect();
        // Direct edges
        assert!(closure.contains(&(0, 1)));
        assert!(closure.contains(&(0, 2)));
        assert!(closure.contains(&(1, 3)));
        assert!(closure.contains(&(2, 3)));
        // Transitive: 0 -> 3
        assert!(closure.contains(&(0, 3)));
        // No reverse paths
        assert!(!closure.contains(&(3, 0)));
        assert!(!closure.contains(&(1, 0)));
    }

    #[test]
    fn transitive_closure_linear() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();

        let closure: HashSet<_> = dag_transitive_closure(&g).collect();
        assert!(closure.contains(&(0, 1)));
        assert!(closure.contains(&(1, 2)));
        assert!(closure.contains(&(0, 2))); // transitive
        assert_eq!(closure.len(), 3);
    }
}
