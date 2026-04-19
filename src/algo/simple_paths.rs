//! # Simple Paths
//!
//! Enumerates all simple (loop-free) paths between two nodes using a
//! backtracking depth-first search. A *simple path* visits each node at most
//! once. The caller may constrain path length by specifying a minimum and
//! maximum number of intermediate nodes (excluding the source and target).
//!
//! ## Components
//!
//! - [`AllSimplePaths`] -- iterator yielding each path as a `Vec<G::NodeIx>`
//! - [`all_simple_paths`] -- safe constructor (requires `StableNode`)
//! - [`all_simple_paths_unchecked`] -- unsafe variant that skips index validation
//!
//! ## Algorithm
//!
//! ```text
//!   Backtracking DFS from `from` toward `to`:
//!
//!     path = [from]
//!     visited = {from}
//!
//!     procedure explore(current):
//!       for each successor s of current:
//!         if s == to and path.len >= min_length:
//!           yield path ++ [to]
//!         else if s not in visited and path.len < max_length:
//!           visited.insert(s)
//!           path.push(s)
//!           explore(s)
//!           path.pop()
//!           visited.remove(s)
//!
//!   0 ---> 1 ---> 3     path [0,1,3]
//!   |             ^
//!   +----> 2 -----+     path [0,2,3]
//! ```
//!
//! ## Example
//!
//! ```rust
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::simple_paths::all_simple_paths;
//!
//! let mut g = BTreeGraph::<u32, &str>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_node(3).unwrap();
//! g.insert_edge("a", [0, 1]).unwrap();
//! g.insert_edge("b", [0, 2]).unwrap();
//! g.insert_edge("c", [1, 3]).unwrap();
//! g.insert_edge("d", [2, 3]).unwrap();
//!
//! let paths: Vec<_> = all_simple_paths(&g, 0, 3, 0, None).collect();
//! assert_eq!(paths.len(), 2); // [0,1,3] and [0,2,3]
//! ```

use std::collections::HashSet;

use crate::graph::capability::{Directed, StableNode};
use crate::graph::Graph;

/// Iterator over all simple (non-repeating) paths between two nodes.
///
/// Each path is returned as a `Vec<G::NodeIx>`. `N` is the node-index type
/// (`G::NodeIx`); it is a separate type parameter so the struct itself carries
/// no `Graph` bound.
pub struct AllSimplePaths<'r, G: ?Sized, N> {
    graph: &'r G,
    target: N,
    min_length: usize,
    max_length: usize,
    // DFS state
    stack: Vec<N>,
    visited: HashSet<N>,
    // At each stack level, the collected successors and current index
    successors: Vec<Vec<N>>,
    indices: Vec<usize>,
}

/// Returns an iterator over all simple paths from `from` to `to`.
///
/// - `min_intermediate_nodes`: minimum number of intermediate nodes (excluding from/to)
/// - `max_intermediate_nodes`: maximum number of intermediate nodes (None for unlimited)
pub fn all_simple_paths<'r, G>(
    graph: &'r G,
    from: G::NodeIx,
    to: G::NodeIx,
    min_intermediate_nodes: usize,
    max_intermediate_nodes: Option<usize>,
) -> AllSimplePaths<'r, G, G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    assert!(Graph::contains_node_index(graph, from));
    assert!(Graph::contains_node_index(graph, to));
    // SAFETY: indices checked above; StableNode guarantees index stability.
    unsafe {
        all_simple_paths_unchecked(
            graph,
            from,
            to,
            min_intermediate_nodes,
            max_intermediate_nodes,
        )
    }
}

/// All simple paths without requiring `StableNode`.
///
/// # Safety
/// The graph must not be modified while the iterator is alive.
pub unsafe fn all_simple_paths_unchecked<'r, G>(
    graph: &'r G,
    from: G::NodeIx,
    to: G::NodeIx,
    min_intermediate_nodes: usize,
    max_intermediate_nodes: Option<usize>,
) -> AllSimplePaths<'r, G, G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    let max_length = max_intermediate_nodes
        .map(|m| m + 2) // +2 for from and to
        .unwrap_or(usize::MAX);
    let min_length = min_intermediate_nodes + 2; // +2 for from and to

    let mut visited = HashSet::new();
    visited.insert(from);

    let succs: Vec<G::NodeIx> = graph.neighbor_indices_from_unchecked(from).collect();

    AllSimplePaths {
        graph,
        target: to,
        min_length,
        max_length,
        stack: vec![from],
        visited,
        successors: vec![succs],
        indices: vec![0],
    }
}

impl<'r, G> Iterator for AllSimplePaths<'r, G, G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    type Item = Vec<G::NodeIx>;

    fn next(&mut self) -> Option<Vec<G::NodeIx>> {
        loop {
            if self.stack.is_empty() {
                return None;
            }

            let depth = self.stack.len() - 1;

            if self.indices[depth] >= self.successors[depth].len() {
                // Backtrack
                let node = self.stack.pop().unwrap();
                self.visited.remove(&node);
                self.successors.pop();
                self.indices.pop();
                continue;
            }

            let succ = self.successors[depth][self.indices[depth]];
            self.indices[depth] += 1;

            if succ == self.target {
                // Found a path
                let path_len = self.stack.len() + 1;
                if path_len >= self.min_length {
                    let mut path = self.stack.clone();
                    path.push(self.target);
                    return Some(path);
                }
                continue;
            }

            // Can we go deeper?
            if self.stack.len() + 1 >= self.max_length {
                continue;
            }

            if !self.visited.insert(succ) {
                continue; // Already on the current path
            }

            // SAFETY: succ came from the graph. Caller guarantees no modification.
            let succ_succs: Vec<G::NodeIx> =
                unsafe { self.graph.neighbor_indices_from_unchecked(succ) }.collect();

            self.stack.push(succ);
            self.successors.push(succ_succs);
            self.indices.push(0);
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

    #[test]
    fn all_paths_diamond() {
        let g = diamond_btree();
        let paths: Vec<Vec<u32>> = all_simple_paths(&g, 0, 3, 0, None).collect();
        assert_eq!(paths.len(), 2);
        // Two paths: 0->1->3 and 0->2->3
        let path_sets: HashSet<Vec<u32>> = paths.into_iter().collect();
        assert!(path_sets.contains(&vec![0, 1, 3]));
        assert!(path_sets.contains(&vec![0, 2, 3]));
    }

    #[test]
    fn all_paths_with_min() {
        let g = diamond_btree();
        // Require at least 1 intermediate node (all diamond paths have 1)
        let paths: Vec<Vec<u32>> = all_simple_paths(&g, 0, 3, 1, None).collect();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn all_paths_with_max() {
        let g = diamond_btree();
        // Max 0 intermediate nodes means direct edge only (0->3 doesn't exist)
        let paths: Vec<Vec<u32>> = all_simple_paths(&g, 0, 3, 0, Some(0)).collect();
        assert_eq!(paths.len(), 0);
    }

    #[test]
    fn all_paths_no_path() {
        let g = diamond_btree();
        // No path from 3 to 0
        let paths: Vec<Vec<u32>> = all_simple_paths(&g, 3, 0, 0, None).collect();
        assert!(paths.is_empty());
    }

    #[test]
    fn all_paths_parallel_edges() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        let paths: Vec<Vec<u32>> = all_simple_paths(&g, 0, 2, 0, None).collect();
        assert_eq!(paths.len(), 2);
        // Paths: 0->2 and 0->1->2
        let path_sets: HashSet<Vec<u32>> = paths.into_iter().collect();
        assert!(path_sets.contains(&vec![0, 2]));
        assert!(path_sets.contains(&vec![0, 1, 2]));
    }
}
