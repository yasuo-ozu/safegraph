//! # Minimum Spanning Tree
//!
//! Computes a minimum spanning tree (or forest, for disconnected graphs) using
//! Prim's algorithm. Edge weights are extracted via a user-supplied closure,
//! and the result is returned as a lazy iterator over the edge indices that
//! belong to the MST.
//!
//! ## Components
//!
//! - [`MinSpanningTree`] — lazy iterator yielding edge indices in the MST/forest
//! - [`min_spanning_tree()`] — safe constructor (requires `StableEdge`)
//! - [`min_spanning_tree_weight()`] — convenience function returning the total MST weight
//!
//! ## Algorithm
//!
//! Prim's algorithm grows the MST one edge at a time, always choosing the
//! lightest edge crossing the cut between visited and unvisited nodes. A
//! `BinaryHeap` (min-heap via reversed `Ord`) serves as the priority queue.
//! For disconnected graphs, the iterator restarts from an unvisited node once
//! the current component is exhausted, producing a minimum spanning forest.
//!
//! ```text
//! Start at any node, mark it visited
//!        |
//!        v
//!  +---> Push all incident edges to min-heap
//!  |     |
//!  |     v
//!  |   Pop lightest edge (u, v)
//!  |     |
//!  |     +---> Both endpoints visited? ---> discard, loop
//!  |     |
//!  |     +---> New endpoint found
//!  |           |
//!  |           v
//!  |         Mark new node visited, yield edge
//!  +---------+
//!        |
//!        v  (heap empty)
//!  Pick next unvisited node (new component) or stop
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::min_spanning_tree::{min_spanning_tree, min_spanning_tree_weight};
//!
//! let mut g = BTreeGraph::<_, _>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_edge(1u32, [0, 1]).unwrap();
//! g.insert_edge(2u32, [1, 2]).unwrap();
//! g.insert_edge(3u32, [0, 2]).unwrap();
//!
//! // Iterate MST edges (weights 1 and 2 are selected, weight 3 is skipped)
//! let mst_edges: Vec<_> = min_spanning_tree(&g, |&w| w).collect();
//! assert_eq!(mst_edges.len(), 2);
//!
//! // Or get the total weight directly
//! let total = min_spanning_tree_weight(&g, |&w| w);
//! assert_eq!(total, 3);
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::ops::Add;

use crate::graph::capability::{Bigraph, StableEdge};
use crate::graph::Graph;

/// A weighted edge candidate for the MST priority queue.
struct EdgeCandidate<W, E> {
    weight: W,
    edge_ix: E,
}

impl<W: Ord, E: Eq> Eq for EdgeCandidate<W, E> {}
impl<W: Ord, E: Eq> PartialEq for EdgeCandidate<W, E> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}
impl<W: Ord, E: Eq> PartialOrd for EdgeCandidate<W, E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<W: Ord, E: Eq> Ord for EdgeCandidate<W, E> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.weight.cmp(&self.weight)
    }
}

/// Iterator that lazily yields edges forming a minimum spanning tree/forest
/// using Prim's algorithm.
///
/// For disconnected graphs, yields a minimum spanning forest.
/// The graph is treated as undirected (both directions of each edge are considered).
pub struct MinSpanningTree<'r, G: ?Sized, W, F, N, E, Ns> {
    graph: &'r G,
    edge_weight: F,
    in_mst: HashSet<N>,
    heap: BinaryHeap<EdgeCandidate<W, E>>,
    nodes: Ns,
}

/// Compute a minimum spanning tree/forest using Prim's algorithm.
///
/// Returns an iterator over the edge indices forming the MST.
///
/// The `edge_weight` closure extracts a weight from the edge data.
///
/// Requires `Bigraph` to access edge endpoints.
pub fn min_spanning_tree<'r, G, W, F>(
    graph: &'r G,
    edge_weight: F,
) -> MinSpanningTree<
    'r,
    G,
    W,
    F,
    G::NodeIx,
    G::EdgeIx,
    <G as crate::graph::GraphOperation<'r>>::NodeIndices,
>
where
    G: Graph + Bigraph + StableEdge + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: Fn(&G::Edge) -> W,
{
    MinSpanningTree {
        graph,
        edge_weight,
        in_mst: HashSet::new(),
        heap: BinaryHeap::new(),
        nodes: <_ as crate::graph::GraphOperation<'_>>::node_indices(graph),
    }
}

impl<'r, G, W, F> Iterator
    for MinSpanningTree<
        'r,
        G,
        W,
        F,
        G::NodeIx,
        G::EdgeIx,
        <G as crate::graph::GraphOperation<'r>>::NodeIndices,
    >
where
    G: Graph + Bigraph + StableEdge + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: Fn(&G::Edge) -> W,
{
    type Item = G::EdgeIx;

    fn next(&mut self) -> Option<G::EdgeIx> {
        loop {
            // Try to get next MST edge from current component
            while let Some(EdgeCandidate { edge_ix, .. }) = self.heap.pop() {
                let eps: Vec<G::NodeIx> = unsafe {
                    <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(
                        self.graph, edge_ix,
                    )
                }
                .into_iter()
                .collect();
                let (a, b) = (eps[0], eps[1]);

                let new_node = if !self.in_mst.contains(&a) {
                    Some(a)
                } else if !self.in_mst.contains(&b) {
                    Some(b)
                } else {
                    None
                };

                if let Some(node) = new_node {
                    self.in_mst.insert(node);
                    unsafe {
                        add_incident_edges(
                            self.graph,
                            node,
                            &self.in_mst,
                            &self.edge_weight,
                            &mut self.heap,
                        );
                    }
                    return Some(edge_ix);
                }
            }

            // Move to next unvisited component
            loop {
                let start = self.nodes.next()?;
                if !self.in_mst.contains(&start) {
                    self.in_mst.insert(start);
                    unsafe {
                        add_incident_edges(
                            self.graph,
                            start,
                            &self.in_mst,
                            &self.edge_weight,
                            &mut self.heap,
                        );
                    }
                    break;
                }
            }
        }
    }
}

/// Add all edges from `node` to nodes not yet in MST to the priority queue.
unsafe fn add_incident_edges<G, W, F>(
    graph: &G,
    node: G::NodeIx,
    in_mst: &HashSet<G::NodeIx>,
    edge_weight: &F,
    heap: &mut BinaryHeap<EdgeCandidate<W, G::EdgeIx>>,
) where
    G: Graph + Bigraph + ?Sized,
    W: Copy + Ord,
    F: Fn(&G::Edge) -> W,
{
    for eix in <G as crate::graph::GraphOperation<'_>>::edge_indices_of_unchecked(graph, node) {
        let eps: Vec<G::NodeIx> =
            <G as crate::graph::GraphOperation<'_>>::endpoints_unchecked(graph, eix)
                .into_iter()
                .collect();
        let (a, b) = (eps[0], eps[1]);
        if (a == node && !in_mst.contains(&b)) || (b == node && !in_mst.contains(&a)) {
            let w = edge_weight(Graph::edge_unchecked(graph, eix));
            heap.push(EdgeCandidate {
                weight: w,
                edge_ix: eix,
            });
        }
    }
}

/// Compute the total weight of a minimum spanning tree/forest.
///
/// Convenience function that computes the MST and sums edge weights.
pub fn min_spanning_tree_weight<G, W, F>(graph: &G, edge_weight: F) -> W
where
    G: Graph + Bigraph + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: Fn(&G::Edge) -> W,
{
    // SAFETY: `graph` is not mutated while `mst` is consumed below, so the
    // edge indices it yields stay valid; delegate to the safe constructor.
    let mst = min_spanning_tree(unsafe { graph.unsafe_assert_stable_edge() }, &edge_weight);
    let mut total = W::default();
    for eix in mst {
        total = total + edge_weight(graph.edge(eix));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BTreeGraph;

    #[test]
    fn mst_triangle() {
        // Triangle: 0-1 (w=1), 1-2 (w=2), 0-2 (w=3)
        // MST should pick edges with weight 1 and 2
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge(1u32, [0, 1]).unwrap();
        g.insert_edge(2u32, [1, 2]).unwrap();
        g.insert_edge(3u32, [0, 2]).unwrap();

        let mst: HashSet<_> = min_spanning_tree(&g, |&w| w).collect();
        assert_eq!(mst.len(), 2);
        // MST weight should be 1 + 2 = 3
        let total: u32 = mst.iter().map(|&eix| *g.edge(eix)).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn mst_linear() {
        // Linear: 0-1 (w=5), 1-2 (w=3)
        // MST should include both edges
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge(5u32, [0, 1]).unwrap();
        g.insert_edge(3u32, [1, 2]).unwrap();

        let mst: Vec<_> = min_spanning_tree(&g, |&w| w).collect();
        assert_eq!(mst.len(), 2);
    }

    #[test]
    fn mst_disconnected() {
        // Two components: {0,1} and {2,3}
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge(1u32, [0, 1]).unwrap();
        g.insert_edge(2u32, [2, 3]).unwrap();

        let mst: Vec<_> = min_spanning_tree(&g, |&w| w).collect();
        // MST forest: one edge per component
        assert_eq!(mst.len(), 2);
    }

    #[test]
    fn mst_single_node() {
        let mut g = BTreeGraph::<u32, u32>::default();
        g.insert_node(0).unwrap();
        let mst: Vec<_> = min_spanning_tree(&g, |&w| w).collect();
        assert!(mst.is_empty());
    }

    #[test]
    fn mst_weight_function() {
        // Square: 0-1(w=1), 1-2(w=4), 2-3(w=2), 0-3(w=3)
        // MST: 0-1(1), 2-3(2), 0-3(3) = 6, skip 1-2(4)
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge(1u32, [0, 1]).unwrap();
        g.insert_edge(4u32, [1, 2]).unwrap();
        g.insert_edge(2u32, [2, 3]).unwrap();
        g.insert_edge(3u32, [0, 3]).unwrap();

        let total = min_spanning_tree_weight(&g, |&w| w);
        assert_eq!(total, 6);
    }
}
