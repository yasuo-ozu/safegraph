//! # Graph Matching
//!
//! Computes matchings on undirected graphs. A **matching** is a set of edges
//! with no shared endpoints. This module provides two strategies: a fast greedy
//! heuristic that yields a *maximal* matching (no more edges can be added), and
//! an augmenting-path algorithm that finds a *maximum* matching (largest
//! possible cardinality).
//!
//! ## Components
//!
//! - [`GreedyMatching`] — lazy iterator yielding edges of a greedy maximal matching
//! - [`greedy_matching()`] — safe constructor (requires `StableEdge`)
//! - [`max_matching()`] — returns a `HashSet<EdgeIx>` of a maximum matching (requires `StableEdge`)
//!
//! ## Algorithm
//!
//! **Greedy matching** scans edges in iteration order and accepts each edge
//! whose both endpoints are still unmatched. This runs in O(E) time but may
//! miss the optimal solution.
//!
//! **Maximum matching** starts from the greedy result, then repeatedly searches
//! for augmenting paths -- paths that alternate between unmatched and matched
//! edges, starting and ending at free (unmatched) nodes. Each augmenting path
//! found increases the matching size by one (by toggling matched/unmatched
//! status along the path). The process repeats until no augmenting path exists.
//!
//! ```text
//!  Greedy phase:
//!     For each edge (u, v):
//!       +--- u and v both free? => add to matching, mark u,v matched
//!       +--- otherwise          => skip
//!
//!  Augmenting-path phase:
//!     +---> Collect free (unmatched) nodes
//!     |     |
//!     |     v
//!     |   For each free node, DFS for augmenting path:
//!     |     unmatched edge -> matched edge -> ... -> unmatched edge -> free node
//!     |     |
//!     |     +--- Path found: toggle edges along path, matching grows by 1
//!     |     +--- No path:    skip
//!     |     |
//!     +-----+  (repeat until no augmenting path found in a full round)
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::matching::{greedy_matching, max_matching};
//!
//! // Path graph: 0 -- 1 -- 2
//! let mut g = BTreeGraph::<_, _>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_edge("0-1", [0, 1]).unwrap();
//! g.insert_edge("1-2", [1, 2]).unwrap();
//!
//! // Greedy picks one edge (node 1 can only appear in one matched edge)
//! let greedy: Vec<_> = greedy_matching(&g).collect();
//! assert_eq!(greedy.len(), 1);
//!
//! // Maximum matching is also 1 for a 3-node path
//! let maximum = max_matching(&g);
//! assert_eq!(maximum.len(), 1);
//! ```

use std::collections::{HashMap, HashSet};

use crate::graph::capability::{Bigraph, StableEdge};
use crate::graph::Graph;

type Adjacency<G> = HashMap<
    <G as crate::graph::GraphProperty>::NodeIx,
    Vec<(
        <G as crate::graph::GraphProperty>::EdgeIx,
        <G as crate::graph::GraphProperty>::NodeIx,
    )>,
>;

/// Iterator that lazily yields edges forming a greedy maximal matching.
///
/// A matching is a set of edges with no shared vertices. This greedy algorithm
/// iterates edges and yields each edge if neither endpoint is already matched.
///
/// The result is a maximal matching (no more edges can be added), but not necessarily
/// a maximum matching (the largest possible).
/// `E` is the edge-index iterator type (`<G as GraphOperation<'r>>::EdgeIndices`)
/// and `N` the node-index type (`G::NodeIx`); both are separate type parameters
/// so the struct carries no `Graph` bound.
pub struct GreedyMatching<'r, G: ?Sized, E, N> {
    graph: &'r G,
    edges: E,
    matched_nodes: HashSet<N>,
}

/// Returns an iterator over edges forming a greedy maximal matching.
pub fn greedy_matching<'r, G>(graph: &'r G) -> GreedyMatching<'r, G, <G as crate::graph::GraphOperation<'r>>::EdgeIndices, G::NodeIx>
where
    G: Graph + Bigraph + StableEdge + ?Sized,
{
    GreedyMatching {
        graph,
        edges: <_ as crate::graph::GraphOperation<'_>>::edge_indices(graph),
        matched_nodes: HashSet::new(),
    }
}

impl<'r, G> Iterator for GreedyMatching<'r, G, <G as crate::graph::GraphOperation<'r>>::EdgeIndices, G::NodeIx>
where
    G: Graph + Bigraph + StableEdge + ?Sized,
{
    type Item = G::EdgeIx;

    fn next(&mut self) -> Option<G::EdgeIx> {
        loop {
            let eix = self.edges.next()?;
            let eps: Vec<G::NodeIx> = unsafe { <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(self.graph, eix) }
                .into_iter()
                .collect();
            let (a, b) = (eps[0], eps[1]);

            // Skip self-loops
            if a == b {
                continue;
            }

            if !self.matched_nodes.contains(&a) && !self.matched_nodes.contains(&b) {
                self.matched_nodes.insert(a);
                self.matched_nodes.insert(b);
                return Some(eix);
            }
        }
    }
}

/// Compute a maximum matching using augmenting paths.
///
/// Uses the Hopcroft-Karp-style augmenting path approach:
/// start with a greedy matching, then repeatedly find augmenting paths
/// via BFS/DFS to improve the matching.
///
/// Returns a set of edge indices forming the maximum matching.
pub fn max_matching<G>(graph: &G) -> HashSet<G::EdgeIx>
where
    G: Graph + Bigraph + StableEdge + ?Sized,
{
    // Build adjacency: for each node, list of (edge_ix, other_node)
    let mut adj: Adjacency<G> = HashMap::new();

    for eix in <_ as crate::graph::GraphOperation<'_>>::edge_indices(graph) {
        let eps: Vec<G::NodeIx> = unsafe { <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(graph, eix) }
            .into_iter()
            .collect();
        let (a, b) = (eps[0], eps[1]);
        if a == b {
            continue; // Skip self-loops
        }
        adj.entry(a).or_default().push((eix, b));
        adj.entry(b).or_default().push((eix, a));
    }

    // Start with greedy matching
    let mut match_of: HashMap<G::NodeIx, (G::EdgeIx, G::NodeIx)> = HashMap::new();
    let mut in_matching: HashSet<G::EdgeIx> = HashSet::new();

    for eix in <_ as crate::graph::GraphOperation<'_>>::edge_indices(graph) {
        let eps: Vec<G::NodeIx> = unsafe { <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(graph, eix) }
            .into_iter()
            .collect();
        let (a, b) = (eps[0], eps[1]);
        if a == b {
            continue;
        }
        if !match_of.contains_key(&a) && !match_of.contains_key(&b) {
            match_of.insert(a, (eix, b));
            match_of.insert(b, (eix, a));
            in_matching.insert(eix);
        }
    }

    // Augment: find augmenting paths from free (unmatched) nodes
    loop {
        let free_nodes: Vec<G::NodeIx> = <_ as crate::graph::GraphOperation<'_>>::node_indices(graph)
            .filter(|n| !match_of.contains_key(n))
            .collect();

        let mut found_augmenting = false;

        for free in &free_nodes {
            // DFS for augmenting path from this free node
            // An augmenting path alternates between unmatched and matched edges,
            // starting and ending at free nodes
            if match_of.contains_key(free) {
                continue; // May have been matched during this round
            }

            if let Some(path) = find_augmenting_path(*free, &adj, &match_of, &in_matching) {
                // Augment along the path: toggle edges in/out of matching
                for (eix, in_match) in path {
                    if in_match {
                        in_matching.remove(&eix);
                    } else {
                        in_matching.insert(eix);
                    }
                }
                // Rebuild match_of from in_matching
                match_of.clear();
                for &eix in &in_matching {
                    let eps: Vec<G::NodeIx> = unsafe { <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(graph, eix) }
                        .into_iter()
                        .collect();
                    let (a, b) = (eps[0], eps[1]);
                    match_of.insert(a, (eix, b));
                    match_of.insert(b, (eix, a));
                }
                found_augmenting = true;
            }
        }

        if !found_augmenting {
            break;
        }
    }

    in_matching
}

/// Find an augmenting path starting from a free node.
/// Returns list of (edge_ix, is_currently_in_matching) along the path.
fn find_augmenting_path<N, E>(
    start: N,
    adj: &HashMap<N, Vec<(E, N)>>,
    match_of: &HashMap<N, (E, N)>,
    in_matching: &HashSet<E>,
) -> Option<Vec<(E, bool)>>
where
    N: Copy + Eq + std::hash::Hash,
    E: Copy + Eq + std::hash::Hash,
{
    // DFS with backtracking
    // State: (current_node, must_use_unmatched_next, visited, path)
    let mut visited: HashSet<N> = HashSet::new();
    visited.insert(start);

    struct Frame<N> {
        node: N,
        use_unmatched: bool, // true = next edge must be unmatched
        neighbors_idx: usize,
    }

    let mut stack: Vec<Frame<N>> = vec![Frame {
        node: start,
        use_unmatched: true, // Start with unmatched edge
        neighbors_idx: 0,
    }];

    let mut path: Vec<(E, bool)> = Vec::new();

    loop {
        let stack_len = stack.len();
        if stack_len == 0 {
            break;
        }

        let frame = &mut stack[stack_len - 1];
        let empty_vec = Vec::new();
        let neighbors = adj.get(&frame.node).unwrap_or(&empty_vec);

        if frame.neighbors_idx >= neighbors.len() {
            // Backtrack
            stack.pop();
            path.pop();
            if let Some(top) = stack.last_mut() {
                top.neighbors_idx += 1;
            }
            continue;
        }

        let (eix, neighbor) = neighbors[frame.neighbors_idx];
        let edge_in_matching = in_matching.contains(&eix);

        // We alternate: unmatched -> matched -> unmatched -> ...
        if frame.use_unmatched == edge_in_matching {
            // Wrong type of edge, skip
            frame.neighbors_idx += 1;
            continue;
        }

        if visited.contains(&neighbor) {
            frame.neighbors_idx += 1;
            continue;
        }

        let next_use_unmatched = !frame.use_unmatched;

        // Take this edge
        path.push((eix, edge_in_matching));
        visited.insert(neighbor);

        // If we used an unmatched edge and neighbor is free, we found an augmenting path
        if !edge_in_matching && !match_of.contains_key(&neighbor) {
            return Some(path);
        }

        // Continue DFS
        stack.push(Frame {
            node: neighbor,
            use_unmatched: next_use_unmatched,
            neighbors_idx: 0,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BTreeGraph;

    #[test]
    fn greedy_matching_basic() {
        // Path: 0 -> 1 -> 2
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();

        let m: HashSet<_> = greedy_matching(&g).collect();
        // At least 1 edge in the matching
        assert!(!m.is_empty());
        // At most 1 edge (since node 1 can only be in one)
        assert!(m.len() <= 1);
    }

    #[test]
    fn max_matching_path() {
        // Path: 0 -> 1 -> 2 -> 3
        // Maximum matching = 2 (e.g., {0->1, 2->3})
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();

        let m = max_matching(&g);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn max_matching_star() {
        // Star: 0 -> 1, 0 -> 2, 0 -> 3
        // Maximum matching = 1 (center can match only one)
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("0->3", [0, 3]).unwrap();

        let m = max_matching(&g);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn max_matching_complete_bipartite() {
        // K_{2,2}: {0,1} x {2,3}
        // 0->2, 0->3, 1->2, 1->3
        // Maximum matching = 2
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("0->3", [0, 3]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("1->3", [1, 3]).unwrap();

        let m = max_matching(&g);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn matching_no_shared_vertices() {
        // Verify matching property: no two edges share a vertex
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_node(4).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();
        g.insert_edge("3->4", [3, 4]).unwrap();

        let m = max_matching(&g);
        let mut used_nodes: HashSet<i32> = HashSet::new();
        for &eix in &m {
            let tail = g.edge_tail_index(eix);
            let head = g.edge_head_index(eix);
            assert!(used_nodes.insert(tail), "Node {:?} used twice", tail);
            assert!(used_nodes.insert(head), "Node {:?} used twice", head);
        }
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn matching_empty_graph() {
        let g = BTreeGraph::<u32, &str>::default();
        let m: HashSet<_> = greedy_matching(&g).collect();
        assert!(m.is_empty());
    }
}
