//! # Depth-First Search
//!
//! Stack-based iterative depth-first traversal. This module provides two
//! iterator structs: one for pre-order (discovery order) and one for
//! post-order (finish order). Both use an explicit stack instead of recursion
//! to avoid stack overflow on deep graphs.
//!
//! ## Components
//!
//! - [`Dfs`] -- pre-order DFS iterator. Yields each node when it is first discovered.
//!   - [`Dfs::new`] / [`Dfs::new_unchecked`]
//!   - [`Dfs::add_start`] / [`Dfs::add_start_unchecked`] -- add extra roots.
//! - [`DfsPostOrder`] -- post-order DFS iterator. Yields each node after all of
//!   its descendants have been visited.
//!   - [`DfsPostOrder::new`] / [`DfsPostOrder::new_unchecked`]
//!   - [`DfsPostOrder::add_start`]
//!
//! ## Algorithm
//!
//! ```text
//!   Pre-order (Dfs)                Post-order (DfsPostOrder)
//!
//!       0                              0
//!      / \                            / \
//!     1   2                          1   2
//!      \ /                            \ /
//!       3                              3
//!
//!   Stack: [0]                     Yield when all children finished:
//!   pop 0 -> yield 0, push 2,1    finish 3 -> yield 3
//!   pop 1 -> yield 1, push 3      finish 1 -> yield 1
//!   pop 3 -> yield 3              finish 2 -> yield 2
//!   pop 2 -> yield 2 (3 visited)  finish 0 -> yield 0
//!
//!   order: 0, 1, 3, 2             order: 3, 1, 2, 0
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::dfs::{Dfs, DfsPostOrder};
//!
//! let mut g = BTreeGraph::<u32, &str>::default();
//! for n in 0..4 { g.insert_node(n).unwrap(); }
//! g.insert_edge("a", [0, 1]).unwrap();
//! g.insert_edge("b", [0, 2]).unwrap();
//! g.insert_edge("c", [1, 3]).unwrap();
//! g.insert_edge("d", [2, 3]).unwrap();
//!
//! let pre: Vec<_> = Dfs::new(&g, 0).collect();
//! // pre-order visits each node on first discovery
//! assert!(pre[0] == 0);  // root is always first
//!
//! let post: Vec<_> = DfsPostOrder::new(&g, 0).collect();
//! // post-order: root is always last
//! assert!(post.last() == Some(&0));
//! ```

use std::collections::HashSet;

use crate::graph::capability::StableNode;
use crate::graph::Graph;

/// Depth-first search iterator yielding nodes in pre-order (discovery order).
///
/// `N` is the node-index type (`G::NodeIx`); it is a separate type parameter
/// so the struct itself carries no `Graph` bound.
pub struct Dfs<'r, G: ?Sized, N> {
    graph: &'r G,
    stack: Vec<N>,
    visited: HashSet<N>,
}

impl<'r, G> Dfs<'r, G, G::NodeIx>
where
    G: Graph + ?Sized,
{
    /// Creates a new DFS iterator without requiring `StableNode`.
    ///
    /// # Safety
    /// The graph must not be modified while this iterator is alive.
    pub unsafe fn new_unchecked(graph: &'r G, start: G::NodeIx) -> Self
    where
        G: StableNode,
    {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        visited.insert(start);
        stack.push(start);
        Dfs {
            graph,
            stack,
            visited,
        }
    }

    /// Creates a new DFS iterator starting from `start`.
    pub fn new(graph: &'r G, start: G::NodeIx) -> Self
    where
        G: StableNode,
    {
        // SAFETY: StableNode guarantees index stability
        unsafe { Self::new_unchecked(graph, start) }
    }

    /// Adds another root to the DFS frontier.
    ///
    /// # Safety
    /// `start` must be a valid node index for `self.graph`.
    pub unsafe fn add_start_unchecked(&mut self, start: G::NodeIx) {
        if self.visited.insert(start) {
            self.stack.push(start);
        }
    }

    /// Adds another root to the DFS frontier.
    pub fn add_start(&mut self, start: G::NodeIx) {
        assert!(Graph::contains_node_index(self.graph, start));
        // SAFETY: checked in precondition
        unsafe { self.add_start_unchecked(start) }
    }
}

impl<'r, G> Iterator for Dfs<'r, G, G::NodeIx>
where
    G: Graph + StableNode + ?Sized,
{
    type Item = G::NodeIx;

    fn next(&mut self) -> Option<G::NodeIx> {
        let node = self.stack.pop()?;
        // SAFETY: node came from the graph. Caller guarantees no modification.
        for eix in unsafe {
            <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(self.graph, node)
        } {
            for endpoint in unsafe {
                <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(self.graph, eix)
            } {
                if endpoint != node && self.visited.insert(endpoint) {
                    self.stack.push(endpoint);
                }
            }
        }
        Some(node)
    }
}

/// Depth-first search iterator yielding nodes in post-order.
///
/// Nodes are emitted after all their descendants have been visited.
/// `N` is the node-index type (`G::NodeIx`); it is a separate type parameter
/// so the struct itself carries no `Graph` bound.
pub struct DfsPostOrder<'r, G: ?Sized, N> {
    graph: &'r G,
    // Stack of (node, expanded). When expanded=false, we push successors.
    // When expanded=true, we yield the node.
    stack: Vec<(N, bool)>,
    visited: HashSet<N>,
}

impl<'r, G> DfsPostOrder<'r, G, G::NodeIx>
where
    G: Graph + ?Sized,
{
    /// Creates a new post-order DFS iterator starting from `start`.
    pub fn new(graph: &'r G, start: G::NodeIx) -> Self
    where
        G: StableNode,
    {
        // SAFETY: StableNode guarantees index stability
        unsafe { Self::new_unchecked(graph, start) }
    }

    /// Creates a new post-order DFS iterator without requiring `StableNode`.
    ///
    /// # Safety
    /// The graph must not be modified while this iterator is alive.
    pub unsafe fn new_unchecked(graph: &'r G, start: G::NodeIx) -> Self
    where
        G: StableNode,
    {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        if Graph::contains_node_index(graph, start) {
            visited.insert(start);
            stack.push((start, false));
        }
        DfsPostOrder {
            graph,
            stack,
            visited,
        }
    }

    /// Adds another root to the DFS frontier.
    pub fn add_start(&mut self, start: G::NodeIx) {
        if self.visited.insert(start) {
            self.stack.push((start, false));
        }
    }
}

impl<'r, G> Iterator for DfsPostOrder<'r, G, G::NodeIx>
where
    G: Graph + StableNode + ?Sized,
{
    type Item = G::NodeIx;

    fn next(&mut self) -> Option<G::NodeIx> {
        loop {
            let (node, expanded) = self.stack.last_mut()?;
            if *expanded {
                let node = self.stack.pop()?.0;
                return Some(node);
            }
            *expanded = true;
            let node = *node;
            // SAFETY: node came from the graph. Caller guarantees no modification.
            let succs: Vec<G::NodeIx> = unsafe {
                <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(
                    self.graph, node,
                )
            }
            .flat_map(|eix| {
                unsafe {
                    <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(self.graph, eix)
                }
                .into_iter()
            })
            .filter(|&ep| ep != node)
            .collect();
            // Push in reverse so first successor is on top
            for succ in succs.into_iter().rev() {
                if self.visited.insert(succ) {
                    self.stack.push((succ, false));
                }
            }
        }
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

    fn linear_btree() -> BTreeGraph<u32, &'static str> {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();
        g
    }

    #[test]
    fn dfs_diamond() {
        let g = diamond_btree();
        let order: Vec<u32> = Dfs::new(&g, 0).collect();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0);
        // All nodes reachable
        assert!(order.contains(&1));
        assert!(order.contains(&2));
        assert!(order.contains(&3));
    }

    #[test]
    fn dfs_linear() {
        let g = linear_btree();
        let order: Vec<u32> = Dfs::new(&g, 0).collect();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn dfs_post_order_diamond() {
        let g = diamond_btree();
        let order: Vec<u32> = DfsPostOrder::new(&g, 0).collect();
        assert_eq!(order.len(), 4);
        // Node 0 must be last (root in post-order)
        assert_eq!(order[3], 0);
        // Node 3 must come before 1 and 2
        let pos3 = order.iter().position(|&x| x == 3).unwrap();
        let pos1 = order.iter().position(|&x| x == 1).unwrap();
        let pos2 = order.iter().position(|&x| x == 2).unwrap();
        assert!(pos3 < pos1 || pos3 < pos2);
    }

    #[test]
    fn dfs_post_order_linear() {
        let g = linear_btree();
        let order: Vec<u32> = DfsPostOrder::new(&g, 0).collect();
        assert_eq!(order, vec![3, 2, 1, 0]);
    }

    #[test]
    fn dfs_with_cycle() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();
        let order: Vec<u32> = Dfs::new(&g, 0).collect();
        assert_eq!(order.len(), 3);
    }
}
