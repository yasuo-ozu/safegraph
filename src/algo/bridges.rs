//! # Bridges and Articulation Points
//!
//! Identifies structural vulnerabilities in an undirected graph: **bridge edges**
//! whose removal disconnects the graph, and **articulation points** (cut vertices)
//! whose removal (along with their incident edges) increases the number of
//! connected components. Both use Tarjan's algorithm with an iterative DFS.
//!
//! ## Components
//!
//! - [`Bridges`] — lazy iterator yielding edge indices that are bridges
//! - [`bridges()`] — safe constructor (requires `StableEdge`)
//! - [`ArticulationPoints`] — lazy iterator yielding node indices that are articulation points
//! - [`articulation_points()`] — safe constructor (requires `StableNode`)
//!
//! ## Algorithm
//!
//! Tarjan's bridge-finding algorithm performs a DFS and maintains two arrays for
//! each node: `disc` (discovery time) and `low` (lowest discovery time reachable
//! through back edges). An edge `(u, v)` is a bridge when `low[v] > disc[u]`,
//! meaning there is no back edge from `v`'s subtree that reaches `u` or above.
//! Articulation points follow a similar rule: a non-root node `u` is an
//! articulation point when some child `v` has `low[v] >= disc[u]`; a root is an
//! articulation point when it has two or more DFS children.
//!
//! ```text
//!  DFS from root
//!     |
//!     v
//!  Visit node u, set disc[u] = low[u] = timer++
//!     |
//!     v
//!  For each neighbor v of u:
//!     +--- v == parent?  skip (tree edge back to parent)
//!     |
//!     +--- v already visited?  low[u] = min(low[u], disc[v])
//!     |
//!     +--- v unvisited?  recurse into v
//!              |
//!              v
//!           After returning: low[u] = min(low[u], low[v])
//!              |
//!              +--- low[v] > disc[u]  =>  edge (u,v) is a BRIDGE
//!              +--- low[v] >= disc[u] =>  u is an ARTICULATION POINT
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::bridges::{bridges, articulation_points};
//!
//! // Path graph: 0 -- 1 -- 2
//! let mut g = BTreeGraph::<_, _>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_edge("0-1", [0, 1]).unwrap();
//! g.insert_edge("1-2", [1, 2]).unwrap();
//!
//! // Both edges are bridges (removing either disconnects the graph)
//! let bridge_edges: Vec<_> = bridges(&g).collect();
//! assert_eq!(bridge_edges.len(), 2);
//!
//! // Node 1 is the only articulation point
//! let cut_vertices: Vec<_> = articulation_points(&g).collect();
//! assert_eq!(cut_vertices.len(), 1);
//! ```

use std::collections::{HashMap, HashSet};

use crate::graph::capability::{Bigraph, StableEdge, StableNode};
use crate::graph::Graph;

type EdgeAdj<G> = Vec<(
    <G as crate::graph::GraphProperty>::EdgeIx,
    <G as crate::graph::GraphProperty>::NodeIx,
)>;
type BridgeFrame<G> = (
    <G as crate::graph::GraphProperty>::NodeIx,
    Option<<G as crate::graph::GraphProperty>::EdgeIx>,
    EdgeAdj<G>,
    usize,
);
type ArticulationFrame<G> = (
    <G as crate::graph::GraphProperty>::NodeIx,
    EdgeAdj<G>,
    usize,
);

/// Iterator that lazily yields bridge edges (cut edges) in a graph.
///
/// A bridge is an edge whose removal increases the number of connected components.
/// Uses Tarjan's bridge-finding algorithm with an iterative DFS.
pub struct Bridges<'r, G: ?Sized, N, Ns, Frame> {
    graph: &'r G,
    disc: HashMap<N, usize>,
    low: HashMap<N, usize>,
    timer: usize,
    nodes: Ns,
    stack: Vec<Frame>,
}

/// Find all bridges (cut edges) in a graph.
///
/// Returns an iterator over edge indices that are bridges.
pub fn bridges<'r, G>(
    graph: &'r G,
) -> Bridges<'r, G, G::NodeIx, <G as crate::graph::GraphOperation<'r>>::NodeIndices, BridgeFrame<G>>
where
    G: Graph + Bigraph + StableEdge + ?Sized,
{
    Bridges {
        graph,
        disc: HashMap::new(),
        low: HashMap::new(),
        timer: 0,
        // SAFETY: node indices is not exposed to the caller
        nodes: <_ as crate::graph::GraphOperation<'_>>::node_indices(graph),
        stack: Vec::new(),
    }
}

impl<'r, G> Iterator
    for Bridges<'r, G, G::NodeIx, <G as crate::graph::GraphOperation<'r>>::NodeIndices, BridgeFrame<G>>
where
    G: Graph + Bigraph + StableEdge + ?Sized,
{
    type Item = G::EdgeIx;

    fn next(&mut self) -> Option<G::EdgeIx> {
        loop {
            if let Some((node, parent_edge, ref neighbors, ref mut idx)) = self.stack.last_mut() {
                let node = *node;
                let parent_edge = *parent_edge;

                if *idx < neighbors.len() {
                    let (eix, neighbor) = neighbors[*idx];
                    *idx += 1;

                    if Some(eix) == parent_edge {
                        continue;
                    }

                    if self.disc.contains_key(&neighbor) {
                        let nl = self.low[&node].min(self.disc[&neighbor]);
                        self.low.insert(node, nl);
                    } else {
                        self.disc.insert(neighbor, self.timer);
                        self.low.insert(neighbor, self.timer);
                        self.timer += 1;

                        let next_neighbors =
                            unsafe { collect_undirected_neighbors(self.graph, neighbor) };
                        self.stack.push((neighbor, Some(eix), next_neighbors, 0));
                    }
                } else {
                    let node_low = self.low[&node];
                    self.stack.pop();

                    if let Some((parent, _, _, _)) = self.stack.last() {
                        let parent = *parent;
                        let parent_low = self.low[&parent].min(node_low);
                        self.low.insert(parent, parent_low);

                        if node_low > self.disc[&parent] {
                            if let Some(pe) = parent_edge {
                                return Some(pe);
                            }
                        }
                    }
                }
            } else {
                // Find next unvisited component
                loop {
                    let start = self.nodes.next()?;
                    if !self.disc.contains_key(&start) {
                        self.disc.insert(start, self.timer);
                        self.low.insert(start, self.timer);
                        self.timer += 1;

                        let neighbors = unsafe { collect_undirected_neighbors(self.graph, start) };
                        self.stack.push((start, None, neighbors, 0));
                        break;
                    }
                }
            }
        }
    }
}

/// Iterator that lazily yields articulation points (cut vertices) in a graph.
///
/// An articulation point is a vertex whose removal (along with its edges)
/// increases the number of connected components.
/// Uses Tarjan's algorithm with an iterative DFS. Deduplicates results.
pub struct ArticulationPoints<'r, G: ?Sized, N, Ns, Frame> {
    graph: &'r G,
    disc: HashMap<N, usize>,
    low: HashMap<N, usize>,
    parent: HashMap<N, Option<N>>,
    children_count: HashMap<N, usize>,
    yielded: HashSet<N>,
    timer: usize,
    nodes: Ns,
    stack: Vec<Frame>,
}

/// Find all articulation points (cut vertices) in a graph.
///
/// Returns an iterator over node indices that are articulation points.
pub fn articulation_points<'r, G>(
    graph: &'r G,
) -> ArticulationPoints<
    'r,
    G,
    G::NodeIx,
    <G as crate::graph::GraphOperation<'r>>::NodeIndices,
    ArticulationFrame<G>,
>
where
    G: Graph + Bigraph + StableNode + ?Sized,
{
    ArticulationPoints {
        graph,
        disc: HashMap::new(),
        low: HashMap::new(),
        parent: HashMap::new(),
        children_count: HashMap::new(),
        yielded: HashSet::new(),
        timer: 0,
        nodes: <_ as crate::graph::GraphOperation<'_>>::node_indices(graph),
        stack: Vec::new(),
    }
}

impl<'r, G> Iterator
    for ArticulationPoints<
        'r,
        G,
        G::NodeIx,
        <G as crate::graph::GraphOperation<'r>>::NodeIndices,
        ArticulationFrame<G>,
    >
where
    G: Graph + Bigraph + StableNode + ?Sized,
{
    type Item = G::NodeIx;

    fn next(&mut self) -> Option<G::NodeIx> {
        loop {
            if let Some((node, ref neighbors, ref mut idx)) = self.stack.last_mut() {
                let node = *node;

                if *idx < neighbors.len() {
                    let (_eix, neighbor) = neighbors[*idx];
                    *idx += 1;

                    if self.parent[&node] == Some(neighbor) {
                        continue;
                    }

                    if self.disc.contains_key(&neighbor) {
                        let nl = self.low[&node].min(self.disc[&neighbor]);
                        self.low.insert(node, nl);
                    } else {
                        *self.children_count.entry(node).or_insert(0) += 1;
                        self.parent.insert(neighbor, Some(node));
                        self.disc.insert(neighbor, self.timer);
                        self.low.insert(neighbor, self.timer);
                        self.timer += 1;

                        let next_neighbors =
                            unsafe { collect_undirected_neighbors(self.graph, neighbor) };
                        self.stack.push((neighbor, next_neighbors, 0));
                    }
                } else {
                    let node_low = self.low[&node];
                    let node_parent = self.parent[&node];
                    self.stack.pop();

                    if let Some(par) = node_parent {
                        let par_low = self.low[&par].min(node_low);
                        self.low.insert(par, par_low);

                        let par_is_root = self.parent[&par].is_none();
                        let is_ap = if par_is_root {
                            self.children_count.get(&par).copied().unwrap_or(0) >= 2
                        } else {
                            node_low >= self.disc[&par]
                        };

                        if is_ap && self.yielded.insert(par) {
                            return Some(par);
                        }
                    }
                }
            } else {
                // Find next unvisited component
                loop {
                    let start = self.nodes.next()?;
                    if !self.disc.contains_key(&start) {
                        self.parent.insert(start, None);
                        self.disc.insert(start, self.timer);
                        self.low.insert(start, self.timer);
                        self.timer += 1;

                        let neighbors = unsafe { collect_undirected_neighbors(self.graph, start) };
                        self.stack.push((start, neighbors, 0));
                        break;
                    }
                }
            }
        }
    }
}

/// Collect all undirected neighbors of a node (both successors and predecessors).
unsafe fn collect_undirected_neighbors<G>(
    graph: &G,
    node: G::NodeIx,
) -> Vec<(G::EdgeIx, G::NodeIx)>
where
    G: Graph + Bigraph + ?Sized,
{
    let mut neighbors = Vec::new();

    for eix in <G as crate::graph::GraphOperation<'_>>::edge_indices_of_unchecked(graph, node) {
        for endpoint in <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(graph, eix) {
            if endpoint != node {
                neighbors.push((eix, endpoint));
            }
        }
    }
    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BTreeGraph;

    #[test]
    fn bridge_linear() {
        // 0 -> 1 -> 2: all edges are bridges
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();

        let b: HashSet<_> = bridges(&g).collect();
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn bridge_cycle() {
        // 0 -> 1 -> 2 -> 0: no bridges (it's a cycle)
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();

        let b: HashSet<_> = bridges(&g).collect();
        assert!(b.is_empty());
    }

    #[test]
    fn bridge_with_pendant() {
        // Cycle 0-1-2-0 with pendant edge 1->3
        // Only 1->3 is a bridge
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();
        g.insert_edge("1->3", [1, 3]).unwrap();

        let b: HashSet<_> = bridges(&g).collect();
        assert_eq!(b.len(), 1);
        assert!(b.contains(&"1->3"));
    }

    #[test]
    fn articulation_point_linear() {
        // 0 -> 1 -> 2: node 1 is an articulation point
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();

        let ap: HashSet<_> = articulation_points(&g).collect();
        assert_eq!(ap.len(), 1);
        assert!(ap.contains(&1));
    }

    #[test]
    fn articulation_point_cycle() {
        // 0 -> 1 -> 2 -> 0: no articulation points
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();

        let ap: HashSet<_> = articulation_points(&g).collect();
        assert!(ap.is_empty());
    }

    #[test]
    fn articulation_point_star() {
        // Star: center 0 connects to 1, 2, 3
        // Node 0 is an articulation point (removing it disconnects)
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("0->3", [0, 3]).unwrap();

        let ap: HashSet<_> = articulation_points(&g).collect();
        assert!(ap.contains(&0));
    }

    #[test]
    fn no_bridges_in_complete_graph() {
        // K4: complete graph on 4 nodes (all pairs connected)
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("0->3", [0, 3]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("1->3", [1, 3]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();

        let b: HashSet<_> = bridges(&g).collect();
        assert!(b.is_empty());
        let ap: HashSet<_> = articulation_points(&g).collect();
        assert!(ap.is_empty());
    }
}
