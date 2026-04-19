//! # Breadth-First Search
//!
//! Queue-based level-order graph traversal. Starting from one or more root
//! nodes, BFS visits every reachable node layer by layer: first all neighbors
//! of the root, then all neighbors of those neighbors, and so on.
//!
//! ## Components
//!
//! - [`Bfs`] -- iterator struct that yields `G::NodeIx` in BFS (level-order) discovery order.
//!   - [`Bfs::new`] -- safe constructor, requires `StableNode`.
//!   - [`Bfs::new_unchecked`] -- `unsafe`, no bounds checks.
//!   - [`Bfs::add_start`] / [`Bfs::add_start_unchecked`] -- add extra root nodes for multi-source BFS.
//!
//! ## Algorithm
//!
//! ```text
//!   Queue: [0]           Visited: {0}
//!
//!   dequeue 0            visit 0, enqueue neighbors 1, 2
//!   Queue: [1, 2]        Visited: {0, 1, 2}
//!
//!   dequeue 1            visit 1, enqueue neighbor 3
//!   Queue: [2, 3]        Visited: {0, 1, 2, 3}
//!
//!   dequeue 2            visit 2, neighbor 3 already visited
//!   Queue: [3]           Visited: {0, 1, 2, 3}
//!
//!   dequeue 3            visit 3, no new neighbors
//!   Queue: []            done  -->  yield order: 0, 1, 2, 3
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::bfs::Bfs;
//!
//! let mut g = BTreeGraph::<u32, &str>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_edge("a", [0, 1]).unwrap();
//! g.insert_edge("b", [1, 2]).unwrap();
//!
//! let order: Vec<_> = Bfs::new(&g, 0).collect();
//! assert_eq!(order, vec![0, 1, 2]);
//! ```

use std::collections::{HashSet, VecDeque};

use crate::graph::capability::StableNode;
use crate::graph::Graph;

/// Breadth-first search iterator.
///
/// Yields node indices in BFS order starting from the given root(s).
/// `N` is the node-index type (`G::NodeIx`); it is a separate type parameter
/// so the struct itself carries no `Graph` bound.
pub struct Bfs<'r, G: ?Sized, N> {
    graph: &'r G,
    queue: VecDeque<N>,
    visited: HashSet<N>,
}

impl<'r, G> Bfs<'r, G, G::NodeIx>
where
    G: Graph + ?Sized,
{
    /// Creates a new BFS iterator without requiring `StableNode`.
    ///
    /// # Safety
    /// `start` is helded by `graph`
    pub unsafe fn new_unchecked(graph: &'r G, start: G::NodeIx) -> Self
    where
        G: StableNode,
    {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);
        Bfs {
            graph,
            queue,
            visited,
        }
    }

    /// Creates a new BFS iterator starting from `start`.
    ///
    /// Requires [`StableNode`]. Panics if `start` is not a valid node index.
    pub fn new(graph: &'r G, start: G::NodeIx) -> Self
    where
        G: StableNode,
    {
        assert!(Graph::contains_node_index(graph, start));
        // SAFETY: StableNode guarantees index stability; start checked above.
        unsafe { Self::new_unchecked(graph, start) }
    }

    /// Adds another root to the BFS frontier (for multi-source BFS).
    ///
    /// # Safety
    /// `start` must be a valid node index for `self.graph`.
    pub unsafe fn add_start_unchecked(&mut self, start: G::NodeIx) {
        if self.visited.insert(start) {
            self.queue.push_back(start);
        }
    }

    /// Adds another root to the BFS frontier (for multi-source BFS).
    ///
    /// Panics if `start` is not a valid node index.
    pub fn add_start(&mut self, start: G::NodeIx) {
        assert!(Graph::contains_node_index(self.graph, start));
        // SAFETY: checked in precondition
        unsafe { self.add_start_unchecked(start) }
    }
}

impl<'r, G> Iterator for Bfs<'r, G, G::NodeIx>
where
    G: Graph + StableNode + ?Sized,
{
    type Item = G::NodeIx;

    fn next(&mut self) -> Option<G::NodeIx> {
        let node = self.queue.pop_front()?;

        // SAFETY: `node` came from the graph (either the initial start that
        // passed has_node_index, or a neighbor yielded by the graph).
        // The caller guarantees the graph is not modified (via StableNode or
        // the unsafe contract of new_unchecked).
        for eix in unsafe { <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(self.graph, node) } {
            for endpoint in unsafe { <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(self.graph, eix) } {
                if endpoint != node && self.visited.insert(endpoint) {
                    self.queue.push_back(endpoint);
                }
            }
        }
        Some(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BTreeGraph;
    fn diamond_btree() -> BTreeGraph<u32, &'static str> {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("1->3", [1, 3]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();
        g
    }

    #[test]
    fn bfs_diamond() {
        let g = diamond_btree();
        let order: Vec<u32> = Bfs::new(&g, 0).collect();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0);
        // Level 1: nodes 1 and 2 (either order)
        assert!(order[1..3].contains(&1));
        assert!(order[1..3].contains(&2));
        // Level 2: node 3
        assert_eq!(order[3], 3);
    }

    #[test]
    fn bfs_single_node() {
        let mut g = BTreeGraph::<u32, &str>::default();
        g.insert_node(42).unwrap();
        let order: Vec<u32> = Bfs::new(&g, 42).collect();
        assert_eq!(order, vec![42]);
    }

    #[test]
    #[should_panic]
    fn bfs_invalid_start() {
        let mut g = BTreeGraph::<u32, &str>::default();
        g.insert_node(0).ok();
        let _order: Vec<u32> = Bfs::new(&g, 99).collect();
    }

    #[test]
    fn bfs_with_cycle() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();
        let order: Vec<u32> = Bfs::new(&g, 0).collect();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn bfs_multi_source() {
        let mut g = BTreeGraph::<u32, &str>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        // No edges: 0, 1, 2 are disconnected
        let mut bfs = Bfs::new(&g, 0);
        bfs.add_start(2);
        let order: Vec<u32> = bfs.collect();
        assert_eq!(order.len(), 2);
        assert!(order.contains(&0));
        assert!(order.contains(&2));
    }
}
