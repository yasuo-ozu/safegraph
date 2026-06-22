//! # Maximum Flow
//!
//! Computes the maximum flow in a directed network using the Edmonds-Karp
//! algorithm, which is a BFS-based implementation of the Ford-Fulkerson method.
//! Edge capacities are extracted via a user-supplied closure. The algorithm
//! maintains a residual graph and repeatedly finds shortest augmenting paths
//! (by edge count) via BFS, guaranteeing O(VE^2) time complexity.
//!
//! ## Components
//!
//! - [`edmonds_karp()`] — returns the maximum flow value (safe, works on any graph)
//! - [`edmonds_karp_with_flows()`] — returns `(max_flow, per_edge_flow_map)` (requires `StableEdge`)
//! - [`edmonds_karp_with_flows_unchecked()`] — unsafe variant returning [`EdmondsKarpFlows`] iterator
//! - [`EdmondsKarpFlows`] — iterator yielding bottleneck values of successive augmenting paths;
//!   call `.finish()` to obtain the final `(total_flow, HashMap<EdgeIx, W>)`
//!
//! ## Algorithm
//!
//! The Edmonds-Karp algorithm works on a residual graph where each edge tracks
//! remaining forward capacity and accumulated reverse (cancellation) capacity.
//! BFS finds the shortest augmenting path from source to sink. The minimum
//! residual capacity along the path (bottleneck) is pushed as flow, updating
//! both forward and reverse residual capacities. The process repeats until no
//! augmenting path exists.
//!
//! ```text
//!  Initialize residual capacities from edge weights
//!        |
//!        v
//!  +---> BFS from source to sink in residual graph
//!  |     |
//!  |     +--- No path found? => DONE, return total flow
//!  |     |
//!  |     +--- Path found
//!  |          |
//!  |          v
//!  |        Find bottleneck = min residual capacity on path
//!  |          |
//!  |          v
//!  |        For each edge on path:
//!  |          - Forward edge: residual -= bottleneck, reverse += bottleneck
//!  |          - Reverse edge: reverse -= bottleneck, forward += bottleneck
//!  |          |
//!  |          v
//!  |        total_flow += bottleneck
//!  +---------+
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::VecGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::max_flow::edmonds_karp;
//!
//! let mut g = VecGraph::<&str, u32>::default();
//! let (s, t) = unsafe {
//!     let s = g.insert_node_unchecked("s").unwrap();
//!     let a = g.insert_node_unchecked("a").unwrap();
//!     let t = g.insert_node_unchecked("t").unwrap();
//!     g.insert_edge_unchecked(10, [s, a]).unwrap();
//!     g.insert_edge_unchecked(5, [a, t]).unwrap();
//!     (s, t)
//! };
//! // Max flow from s to t is 5 (bottleneck at a->t). `edmonds_karp` is safe
//! // and works on any graph (here a non-stable `VecGraph`).
//! let flow = edmonds_karp(&g, s, t, |&cap| cap);
//! assert_eq!(flow, 5);
//! ```

use std::collections::{HashMap, VecDeque};

use crate::graph::capability::{Bigraph, Directed, StableEdge, StableNode};
use crate::graph::Graph;

/// Compute the maximum flow from `source` to `sink` using the Edmonds-Karp algorithm
/// (BFS-based Ford-Fulkerson).
///
/// The `capacity` closure extracts a capacity value from the edge data.
/// Returns the maximum flow value.
///
/// Only considers edges in the forward direction (source -> ... -> sink).
pub fn edmonds_karp<'r, G, W, F>(graph: &'r G, source: G::NodeIx, sink: G::NodeIx, capacity: F) -> W
where
    G: Graph + Directed<'r> + Bigraph + ?Sized,
    W: Copy + Ord + Default + std::ops::Add<Output = W> + std::ops::Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    // SAFETY: any-graph entry point. `edmonds_karp_with_flows` requires both
    // `StableEdge` and `StableNode`; chain the two wrappers to supply both. Only
    // the flow value `W` escapes (no index), so the assertion is sound for the
    // duration of the call.
    edmonds_karp_with_flows(
        unsafe {
            graph
                .unsafe_assert_stable_edge()
                .unsafe_assert_stable_node()
        },
        source,
        sink,
        capacity,
    )
    .0
}

/// Compute maximum flow and return flow on each edge.
///
/// Returns `(max_flow, flow_map)` where `flow_map` maps each edge index
/// to the amount of flow passing through it.
pub fn edmonds_karp_with_flows<'r, G, W, F>(
    graph: &'r G,
    source: G::NodeIx,
    sink: G::NodeIx,
    capacity: F,
) -> (W, HashMap<G::EdgeIx, W>)
where
    G: Graph + Directed<'r> + Bigraph + StableEdge + StableNode + ?Sized,
    W: Copy + Ord + Default + std::ops::Add<Output = W> + std::ops::Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    assert!(Graph::contains_node_index(graph, source));
    assert!(Graph::contains_node_index(graph, sink));
    // SAFETY: endpoints checked above; StableEdge + StableNode keep indices valid.
    unsafe { edmonds_karp_with_flows_unchecked(graph, source, sink, capacity).finish() }
}

/// Iterator that lazily yields augmenting path bottleneck flows from Edmonds-Karp.
///
/// Each `next()` call finds one augmenting path and returns its bottleneck flow.
/// Call `finish()` to get the final `(total_flow, per_edge_flows)`.
pub struct EdmondsKarpFlows<'r, G: ?Sized, W, F, N, E> {
    graph: &'r G,
    source: N,
    sink: N,
    _capacity: std::marker::PhantomData<F>,
    residual: HashMap<E, W>,
    reverse_flow: HashMap<(N, N), W>,
    original_cap: HashMap<E, W>,
    total_flow: W,
    done: bool,
}

/// Edmonds-Karp with flow details iterator without requiring `StableNode`/`StableEdge`.
///
/// # Safety
/// The graph must not be modified while the iterator is alive.
/// `source` and `sink` must be valid node indices.
pub unsafe fn edmonds_karp_with_flows_unchecked<'r, G, W, F>(
    graph: &'r G,
    source: G::NodeIx,
    sink: G::NodeIx,
    mut capacity: F,
) -> EdmondsKarpFlows<'r, G, W, F, G::NodeIx, G::EdgeIx>
where
    G: Graph + Directed<'r> + Bigraph + StableEdge + StableNode + ?Sized,
    W: Copy + Ord + Default + std::ops::Add<Output = W> + std::ops::Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    let mut original_cap: HashMap<G::EdgeIx, W> = HashMap::new();
    let mut residual: HashMap<G::EdgeIx, W> = HashMap::new();

    for eix in <_ as crate::graph::GraphOperation<'_>>::edge_indices(graph) {
        let cap = capacity(Graph::edge_unchecked(graph, eix));
        original_cap.insert(eix, cap);
        residual.insert(eix, cap);
    }

    EdmondsKarpFlows {
        graph,
        source,
        sink,
        _capacity: std::marker::PhantomData::<F>,
        residual,
        reverse_flow: HashMap::new(),
        original_cap,
        total_flow: W::default(),
        done: false,
    }
}

impl<'r, G, W, F> EdmondsKarpFlows<'r, G, W, F, G::NodeIx, G::EdgeIx>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Default + std::ops::Add<Output = W> + std::ops::Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    /// Complete the algorithm and return `(total_flow, per_edge_flows)`.
    pub fn finish(mut self) -> (W, HashMap<G::EdgeIx, W>) {
        while self.next().is_some() {}

        let mut flows: HashMap<G::EdgeIx, W> = HashMap::new();
        for (&eix, &orig) in &self.original_cap {
            let flow = orig - self.residual[&eix];
            flows.insert(eix, flow);
        }

        (self.total_flow, flows)
    }
}

impl<'r, G, W, F> Iterator for EdmondsKarpFlows<'r, G, W, F, G::NodeIx, G::EdgeIx>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Default + std::ops::Add<Output = W> + std::ops::Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    type Item = W;

    fn next(&mut self) -> Option<W> {
        if self.done {
            return None;
        }

        let path = unsafe {
            bfs_augmenting_path(
                self.graph,
                self.source,
                self.sink,
                &self.residual,
                &self.reverse_flow,
            )
        };

        match path {
            None => {
                self.done = true;
                None
            }
            Some(augmenting_path) => {
                let mut bottleneck = None;
                for step in &augmenting_path {
                    let cap = match step {
                        AugmentStep::Forward(eix) => self.residual[eix],
                        AugmentStep::Reverse(from, to) => self
                            .reverse_flow
                            .get(&(*from, *to))
                            .copied()
                            .unwrap_or(W::default()),
                    };
                    bottleneck = Some(match bottleneck {
                        None => cap,
                        Some(b) => {
                            if cap < b {
                                cap
                            } else {
                                b
                            }
                        }
                    });
                }

                let bottleneck = bottleneck.unwrap();
                if bottleneck == W::default() {
                    self.done = true;
                    return None;
                }

                for step in &augmenting_path {
                    match step {
                        AugmentStep::Forward(eix) => {
                            let eix = *eix;
                            let tail = unsafe { self.graph.edge_tail_index_unchecked(eix) };
                            let head = unsafe { self.graph.edge_head_index_unchecked(eix) };
                            *self.residual.get_mut(&eix).unwrap() =
                                self.residual[&eix] - bottleneck;
                            let rev = self.reverse_flow.entry((head, tail)).or_default();
                            *rev = *rev + bottleneck;
                        }
                        AugmentStep::Reverse(from, to) => {
                            let rev = self.reverse_flow.get_mut(&(*from, *to)).unwrap();
                            *rev = *rev - bottleneck;
                            for eix in unsafe {
                                <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(
                                    self.graph, *to,
                                )
                            } {
                                let head = unsafe { self.graph.edge_head_index_unchecked(eix) };
                                if head == *from {
                                    *self.residual.get_mut(&eix).unwrap() =
                                        self.residual[&eix] + bottleneck;
                                    break;
                                }
                            }
                        }
                    }
                }

                self.total_flow = self.total_flow + bottleneck;
                Some(bottleneck)
            }
        }
    }
}

#[derive(Clone)]
enum AugmentStep<E, N> {
    Forward(E),
    Reverse(N, N),
}

/// BFS to find an augmenting path in the residual graph.
unsafe fn bfs_augmenting_path<'r, G, W>(
    graph: &'r G,
    source: G::NodeIx,
    sink: G::NodeIx,
    residual: &HashMap<G::EdgeIx, W>,
    reverse_flow: &HashMap<(G::NodeIx, G::NodeIx), W>,
) -> Option<Vec<AugmentStep<G::EdgeIx, G::NodeIx>>>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Default,
{
    let mut visited: HashMap<G::NodeIx, ResidualParent<G>> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut source_visited = HashMap::new();
    source_visited.insert(source, true);
    queue.push_back(source);

    while let Some(node) = queue.pop_front() {
        if node == sink {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = sink;
            while current != source {
                let (prev, step) = visited[&current].clone();
                path.push(step);
                current = prev;
            }
            path.reverse();
            return Some(path);
        }

        // Forward edges: node -> neighbor with residual > 0
        for eix in <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(graph, node)
        {
            let head = graph.edge_head_index_unchecked(eix);
            if !source_visited.contains_key(&head) && residual[&eix] > W::default() {
                source_visited.insert(head, true);
                visited.insert(head, (node, AugmentStep::Forward(eix)));
                queue.push_back(head);
            }
        }

        // Reverse edges: if there's flow from some node `pred` to `node`,
        // we can push flow back
        for eix in Directed::edge_indices_to_unchecked(graph, node) {
            let pred = graph.edge_tail_index_unchecked(eix);
            if pred == node {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = source_visited.entry(pred) {
                let rev_cap = reverse_flow
                    .get(&(node, pred))
                    .copied()
                    .unwrap_or(W::default());
                if rev_cap > W::default() {
                    e.insert(true);
                    visited.insert(pred, (node, AugmentStep::Reverse(node, pred)));
                    queue.push_back(pred);
                }
            }
        }
    }

    None
}
type ResidualParent<G> = (
    <G as crate::graph::GraphProperty>::NodeIx,
    AugmentStep<
        <G as crate::graph::GraphProperty>::EdgeIx,
        <G as crate::graph::GraphProperty>::NodeIx,
    >,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::capability::{InsertEdge, InsertNode};
    use crate::VecGraph;

    type Nix = u32;

    fn build_flow_graph() -> (VecGraph<&'static str, u32>, Nix, Nix, Nix, Nix) {
        // s -> a (cap 10), s -> b (cap 10), a -> b (cap 2), a -> t (cap 4), b -> t (cap 8)
        // Max flow s->t = 12
        let mut g = VecGraph::<&str, u32>::default();
        unsafe {
            let s = InsertNode::insert_node_unchecked(&mut g, "s").unwrap();
            let a = InsertNode::insert_node_unchecked(&mut g, "a").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "b").unwrap();
            let t = InsertNode::insert_node_unchecked(&mut g, "t").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 10, [s, a]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 10, [s, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 2, [a, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 4, [a, t]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 8, [b, t]).unwrap();
            (g, s, a, b, t)
        }
    }

    #[test]
    fn max_flow_basic() {
        let (g, s, _, _, t) = build_flow_graph();
        let flow = edmonds_karp(&g, s, t, |&cap| cap);
        assert_eq!(flow, 12);
    }

    #[test]
    fn max_flow_no_path() {
        let mut g = VecGraph::<&str, u32>::default();
        let (s, t) = unsafe {
            let s = InsertNode::insert_node_unchecked(&mut g, "s").unwrap();
            let t = InsertNode::insert_node_unchecked(&mut g, "t").unwrap();
            (s, t)
        };
        let flow = edmonds_karp(&g, s, t, |&cap| cap);
        assert_eq!(flow, 0);
    }

    #[test]
    fn max_flow_single_edge() {
        let mut g = VecGraph::<&str, u32>::default();
        let (s, t) = unsafe {
            let s = InsertNode::insert_node_unchecked(&mut g, "s").unwrap();
            let t = InsertNode::insert_node_unchecked(&mut g, "t").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 5, [s, t]).unwrap();
            (s, t)
        };
        let flow = edmonds_karp(&g, s, t, |&cap| cap);
        assert_eq!(flow, 5);
    }

    #[test]
    fn max_flow_parallel_paths() {
        // s -> a -> t (cap 3), s -> b -> t (cap 5). Max flow = 8
        let mut g = VecGraph::<&str, u32>::default();
        let (s, t) = unsafe {
            let s = InsertNode::insert_node_unchecked(&mut g, "s").unwrap();
            let a = InsertNode::insert_node_unchecked(&mut g, "a").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "b").unwrap();
            let t = InsertNode::insert_node_unchecked(&mut g, "t").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 3, [s, a]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 5, [s, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 3, [a, t]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 5, [b, t]).unwrap();
            (s, t)
        };
        let flow = edmonds_karp(&g, s, t, |&cap| cap);
        assert_eq!(flow, 8);
    }

    #[test]
    fn max_flow_with_flows() {
        let (g, s, _, _, t) = build_flow_graph();
        let (flow, flow_map) = unsafe {
            edmonds_karp_with_flows_unchecked(
                g.unsafe_assert_stable_edge().unsafe_assert_stable_node(),
                s,
                t,
                |&cap| cap,
            )
        }
        .finish();
        assert_eq!(flow, 12);
        // Verify all edges have flow entries
        assert_eq!(flow_map.len(), 5);
    }

    #[test]
    fn max_flow_bottleneck() {
        // s -> a (cap 100) -> b (cap 1) -> t (cap 100). Max flow = 1
        let mut g = VecGraph::<&str, u32>::default();
        let (s, t) = unsafe {
            let s = InsertNode::insert_node_unchecked(&mut g, "s").unwrap();
            let a = InsertNode::insert_node_unchecked(&mut g, "a").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "b").unwrap();
            let t = InsertNode::insert_node_unchecked(&mut g, "t").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 100, [s, a]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 1, [a, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 100, [b, t]).unwrap();
            (s, t)
        };
        let flow = edmonds_karp(&g, s, t, |&cap| cap);
        assert_eq!(flow, 1);
    }
}
