//! # Bipartiteness Testing
//!
//! Determines whether a graph is bipartite (2-colorable) and, if so, produces
//! a valid 2-coloring. A graph is bipartite when its vertex set can be
//! partitioned into two disjoint sets such that every edge connects a vertex in
//! one set to a vertex in the other. Equivalently, the graph contains no
//! odd-length cycle.
//!
//! ## Components
//!
//! - [`is_bipartite()`] — returns `true` if the graph is bipartite
//! - [`bipartite_coloring()`] — safe version returning `Some(HashMap<NodeIx, bool>)` or `None` (requires `StableNode`)
//!
//! ## Algorithm
//!
//! BFS 2-coloring: starting from each unvisited node, assign it color `false`,
//! then propagate via BFS. Each neighbor receives the opposite color of its
//! parent. If a neighbor already has the same color as the current node, an
//! odd cycle exists and the graph is not bipartite. Self-loops are detected as
//! a special case (always non-bipartite).
//!
//! ```text
//!  For each unvisited node s:
//!     color[s] = false, enqueue s
//!        |
//!        v
//!  +---> Dequeue node u
//!  |     |
//!  |     v
//!  |   For each neighbor v of u:
//!  |     +--- v uncolored?  color[v] = !color[u], enqueue v
//!  |     |
//!  |     +--- color[v] == color[u]?  => NOT bipartite (odd cycle)
//!  |     |
//!  |     +--- color[v] != color[u]?  => OK, continue
//!  |     |
//!  +-----+  (queue not empty)
//!        |
//!        v  (all nodes colored without conflict)
//!     Graph IS bipartite
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::bipartite::{is_bipartite, bipartite_coloring};
//!
//! // Even cycle (4-cycle): bipartite
//! let mut g = BTreeGraph::<_, _>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_node(3).unwrap();
//! g.insert_edge("e1", [0, 1]).unwrap();
//! g.insert_edge("e2", [1, 2]).unwrap();
//! g.insert_edge("e3", [2, 3]).unwrap();
//! g.insert_edge("e4", [3, 0]).unwrap();
//! assert!(is_bipartite(&g));
//!
//! let coloring = bipartite_coloring(&g).unwrap();
//! // Adjacent nodes always have different colors
//! assert_ne!(coloring[&0], coloring[&1]);
//!
//! // Odd cycle (triangle): NOT bipartite
//! let mut h = BTreeGraph::<_, _>::default();
//! h.insert_node(0).unwrap();
//! h.insert_node(1).unwrap();
//! h.insert_node(2).unwrap();
//! h.insert_edge("e1", [0, 1]).unwrap();
//! h.insert_edge("e2", [1, 2]).unwrap();
//! h.insert_edge("e3", [2, 0]).unwrap();
//! assert!(!is_bipartite(&h));
//! ```

use std::collections::{HashMap, VecDeque};

use crate::graph::capability::{Bigraph, StableNode};
use crate::graph::Graph;

/// Check whether the graph is bipartite (2-colorable)
///
/// Returns `true` if the graph can be partitioned into two sets such that
/// every edge connects a node in one set to a node in the other.
pub fn is_bipartite<G>(graph: &G) -> bool
where
    G: Graph + Bigraph + ?Sized,
    G::Endpoints: for<'scope> crate::graph::edge::Map<
        crate::graph::context::NodeIx<'scope, G::NodeIx>,
        Mapped = [crate::graph::context::NodeIx<'scope, G::NodeIx>; 2],
    >,
{
    // Fully safe: `scope` hands `bipartite_coloring` a `Context` whose branded
    // node indices cannot escape the closure (so they can never be used after a
    // mutation); only the `bool` outcome is returned.
    graph.scope(|ctx| bipartite_coloring(ctx).is_some())
}

/// Compute a 2-coloring of the graph if it is bipartite.
///
/// Returns `Some(coloring)` where the coloring maps each node to `true` or `false`
/// (representing the two partitions), or `None` if the graph is not bipartite.
///
/// The graph is treated as undirected.
pub fn bipartite_coloring<G>(graph: &G) -> Option<HashMap<G::NodeIx, bool>>
where
    G: Graph + Bigraph + StableNode + ?Sized,
{
    let mut color: HashMap<G::NodeIx, bool> = HashMap::new();

    for start in <_ as crate::graph::GraphOperation<'_>>::node_indices(graph) {
        if color.contains_key(&start) {
            continue;
        }

        // BFS 2-coloring
        color.insert(start, false);
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            let node_color = color[&node];

            // Check all undirected neighbors (use edges_of for both directions)
            // SAFETY: edge indices are not exposed to the caller and `graph` is
            // borrowed immutably for the whole call (raw bound-free primitive).
            for eix in unsafe {
                <G as crate::graph::GraphOperation<'_>>::edge_indices_of_unchecked(graph, node)
            } {
                let mut found_other = false;
                for endpoint in unsafe {
                    <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(graph, eix)
                } {
                    if endpoint == node {
                        continue;
                    }
                    found_other = true;
                    let neighbor = endpoint;
                    if let Some(&existing_color) = color.get(&neighbor) {
                        if existing_color == node_color {
                            return None; // Same color on both sides of an edge
                        }
                    } else {
                        color.insert(neighbor, !node_color);
                        queue.push_back(neighbor);
                    }
                }
                // Self-loop: all endpoints equal node, graph is not bipartite
                if !found_other {
                    return None;
                }
            }
        }
    }

    Some(color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BTreeGraph;

    #[test]
    fn bipartite_even_cycle() {
        // 0 -> 1 -> 2 -> 3 -> 0 (4-cycle is bipartite)
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();
        g.insert_edge("3->0", [3, 0]).unwrap();

        assert!(is_bipartite(&g));
        let coloring = bipartite_coloring(&g).unwrap();
        assert_eq!(coloring.len(), 4);
        // Adjacent nodes should have different colors
        assert_ne!(coloring[&0], coloring[&1]);
        assert_ne!(coloring[&1], coloring[&2]);
        assert_ne!(coloring[&2], coloring[&3]);
    }

    #[test]
    fn not_bipartite_odd_cycle() {
        // 0 -> 1 -> 2 -> 0 (3-cycle is NOT bipartite)
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();

        assert!(!is_bipartite(&g));
        assert!(bipartite_coloring(&g).is_none());
    }

    #[test]
    fn bipartite_tree() {
        // Any tree is bipartite
        // 0 -> 1, 0 -> 2, 1 -> 3
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("1->3", [1, 3]).unwrap();

        assert!(is_bipartite(&g));
    }

    #[test]
    fn bipartite_disconnected() {
        // Two disconnected edges: 0->1, 2->3
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();

        assert!(is_bipartite(&g));
    }

    #[test]
    fn bipartite_empty() {
        let g = BTreeGraph::<u32, &str>::default();
        assert!(is_bipartite(&g));
    }

    #[test]
    fn not_bipartite_self_loop() {
        // Self-loop: 0 -> 0 is not bipartite
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_edge("0->0", [0, 0]).unwrap();

        // A self-loop creates an odd cycle of length 1
        // When checking outgoing edge 0->0, head=0 has same color as node=0
        assert!(!is_bipartite(&g));
    }
}
