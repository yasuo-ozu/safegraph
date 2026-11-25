//! # Maximum Flow and Minimum Cut
//!
//! Computes the maximum flow in a directed network, and the minimum cuts that
//! certify it. Edge capacities are extracted via a user-supplied closure.
//! Two algorithms are provided:
//!
//! - **Dinic** — augments along *all* shortest paths of one length per phase.
//!   O(V²E) in general, O(E·√V) on unit networks (bipartite matching, the
//!   node-split network of [`min_vertex_cut()`]). Prefer it.
//! - **Edmonds-Karp** — one BFS over the whole residual graph per augmenting
//!   path. O(VE²); on unit capacities it needs one BFS per unit of flow, so it
//!   is quadratic where Dinic is near-linear.
//!
//! ## Components
//!
//! - [`dinic()`] — returns the maximum flow value (safe, works on any graph)
//! - [`dinic_with_flows()`] — returns `(max_flow, per_edge_flow_map)` (requires `StableEdge + StableNode`)
//! - [`dinic_min_cut()`] — returns `(max_flow, `[`MinCut`]`)`: the minimum cuts closest to
//!   the source and to the sink (requires `StableEdge + StableNode`)
//! - [`min_vertex_cut()`] — the cheapest set of nodes separating some sources from some sinks,
//!   by node capacities (requires `StableNode`)
//! - [`edmonds_karp()`] — returns the maximum flow value (safe, works on any graph)
//! - [`edmonds_karp_with_flows()`] — returns `(max_flow, per_edge_flow_map)` (requires `StableEdge`)
//! - [`edmonds_karp_with_flows_unchecked()`] — unsafe variant returning [`EdmondsKarpFlows`] iterator
//! - [`EdmondsKarpFlows`] — iterator yielding bottleneck values of successive augmenting paths;
//!   call `.finish()` to obtain the final `(total_flow, HashMap<EdgeIx, W>)`
//!
//! The Dinic functions are deterministic: they number nodes and edges in index
//! order (`Ord` on the index types), so the same graph gives the same flow on
//! every run, whatever order the backend iterates in.
//!
//! ## Minimum cuts
//!
//! After a maximum flow, the residual network has no path from the source to
//! the sink. Two minimum cuts can be read off it, and both are the same for
//! every maximum flow:
//!
//! - **closest to the source**: the source side is the nodes the source still
//!   reaches by arcs with spare capacity;
//! - **closest to the sink**: the sink side is the nodes that can still reach
//!   the sink (by forward arcs with spare capacity, or backward arcs whose flow
//!   can be cancelled).
//!
//! Every other minimum cut lies between these two. [`MinCut`] answers both.
//!
//! ## Dinic's algorithm
//!
//! ```text
//!  +---> BFS from the source: level of each node = distance by arcs
//!  |     with spare capacity
//!  |     |
//!  |     +--- Sink not reached? => DONE, return total flow
//!  |     |
//!  |     v
//!  |   Blocking flow: DFS from the source along arcs one level further,
//!  |   each node resuming at its "next arc" (arcs before it lead nowhere
//!  |   this phase). At the sink, push the path's bottleneck, then resume
//!  |   from the tail of its first saturated arc. At a dead end, step back
//!  |   and skip the arc that led there.
//!  |     |
//!  +-----+
//! ```
//!
//! ## Edmonds-Karp
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
//! ```rust
//! use safegraph::VecGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::max_flow::{dinic, edmonds_karp};
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
//! // Max flow from s to t is 5 (bottleneck at a->t). `dinic` and
//! // `edmonds_karp` are safe and work on any graph (here a non-stable `VecGraph`).
//! assert_eq!(dinic(&g, s, t, |&cap| cap), 5);
//! assert_eq!(edmonds_karp(&g, s, t, |&cap| cap), 5);
//! ```

use std::borrow::Borrow;
use std::collections::{HashMap, VecDeque};
use std::ops::{Add, Sub};

use num_traits::Bounded;

use crate::graph::capability::{Bigraph, Directed, StableEdge, StableNode};
use crate::graph::{Graph, GraphOperation};

/// Compute the maximum flow from `source` to `sink` using the Edmonds-Karp algorithm
/// (BFS-based Ford-Fulkerson).
///
/// The `capacity` closure extracts a capacity value from the edge data.
/// Returns the maximum flow value.
///
/// Only considers edges in the forward direction (source -> ... -> sink).
pub fn edmonds_karp<'r, G, W, F>(graph: &'r G, source: G::NodeIx, sink: G::NodeIx, capacity: F) -> W
where
    G: Graph + for<'x> Directed<'x> + Bigraph + ?Sized,
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
    G: Graph + for<'x> Directed<'x> + Bigraph + StableEdge + StableNode + ?Sized,
    W: Copy + Ord + Default + std::ops::Add<Output = W> + std::ops::Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    assert!(Graph::contains_node_index(graph, &source));
    assert!(Graph::contains_node_index(graph, &sink));
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
    G: Graph + for<'x> Directed<'x> + Bigraph + StableEdge + StableNode + ?Sized,
    W: Copy + Ord + Default + std::ops::Add<Output = W> + std::ops::Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    let mut original_cap: HashMap<G::EdgeIx, W> = HashMap::new();
    let mut residual: HashMap<G::EdgeIx, W> = HashMap::new();

    for eix in super::owned_edge_indices(graph) {
        let cap = capacity(Graph::edge_unchecked(graph, &eix));
        original_cap.insert(eix.clone(), cap);
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
    G: Graph + for<'x> Directed<'x> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Default + std::ops::Add<Output = W> + std::ops::Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    /// Complete the algorithm and return `(total_flow, per_edge_flows)`.
    pub fn finish(mut self) -> (W, HashMap<G::EdgeIx, W>) {
        while self.next().is_some() {}

        let mut flows: HashMap<G::EdgeIx, W> = HashMap::new();
        for (eix, &orig) in &self.original_cap {
            let flow = orig - self.residual[eix];
            flows.insert(eix.clone(), flow);
        }

        (self.total_flow, flows)
    }
}

impl<'r, G, W, F> Iterator for EdmondsKarpFlows<'r, G, W, F, G::NodeIx, G::EdgeIx>
where
    G: Graph + for<'x> Directed<'x> + Bigraph + StableNode + ?Sized,
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
                &self.source,
                &self.sink,
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
                            .get(&(from.clone(), to.clone()))
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
                            let tail: G::NodeIx = unsafe { self.graph.tail_index_unchecked(eix) }
                                .borrow()
                                .clone();
                            let head: G::NodeIx = unsafe { self.graph.head_index_unchecked(eix) }
                                .borrow()
                                .clone();
                            *self.residual.get_mut(eix).unwrap() = self.residual[eix] - bottleneck;
                            let rev = self.reverse_flow.entry((head, tail)).or_default();
                            *rev = *rev + bottleneck;
                        }
                        AugmentStep::Reverse(from, to) => {
                            let rev = self
                                .reverse_flow
                                .get_mut(&(from.clone(), to.clone()))
                                .unwrap();
                            *rev = *rev - bottleneck;
                            for eix in unsafe {
                                <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(
                                    self.graph, to,
                                )
                            } {
                                let eix: &G::EdgeIx = eix.borrow();
                                let head = unsafe { self.graph.head_index_unchecked(eix) };
                                if Borrow::<G::NodeIx>::borrow(&head) == from {
                                    *self.residual.get_mut(eix).unwrap() =
                                        self.residual[eix] + bottleneck;
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
    source: &G::NodeIx,
    sink: &G::NodeIx,
    residual: &HashMap<G::EdgeIx, W>,
    reverse_flow: &HashMap<(G::NodeIx, G::NodeIx), W>,
) -> Option<Vec<AugmentStep<G::EdgeIx, G::NodeIx>>>
where
    G: Graph + for<'x> Directed<'x> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Default,
{
    let mut visited: HashMap<G::NodeIx, ResidualParent<G>> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut source_visited = HashMap::new();
    source_visited.insert(source.clone(), true);
    queue.push_back(source.clone());

    while let Some(node) = queue.pop_front() {
        if node == *sink {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = sink.clone();
            while current != *source {
                let (prev, step) = visited[&current].clone();
                path.push(step);
                current = prev;
            }
            path.reverse();
            return Some(path);
        }

        // Forward edges: node -> neighbor with residual > 0
        for eix in
            <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(graph, &node)
        {
            let eix: &G::EdgeIx = eix.borrow();
            let head: G::NodeIx = graph.head_index_unchecked(eix).borrow().clone();
            if !source_visited.contains_key(&head) && residual[eix] > W::default() {
                source_visited.insert(head.clone(), true);
                visited.insert(
                    head.clone(),
                    (node.clone(), AugmentStep::Forward(eix.clone())),
                );
                queue.push_back(head);
            }
        }

        // Reverse edges: if there's flow from some node `pred` to `node`,
        // we can push flow back
        for eix in Directed::edge_indices_to_unchecked(graph, &node) {
            let pred: G::NodeIx = graph.tail_index_unchecked(eix).borrow().clone();
            if pred == node {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = source_visited.entry(pred.clone())
            {
                let rev_cap = reverse_flow
                    .get(&(node.clone(), pred.clone()))
                    .copied()
                    .unwrap_or(W::default());
                if rev_cap > W::default() {
                    e.insert(true);
                    visited.insert(
                        pred.clone(),
                        (
                            node.clone(),
                            AugmentStep::Reverse(node.clone(), pred.clone()),
                        ),
                    );
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

// ---------------------------------------------------------------------------
// Dinic
// ---------------------------------------------------------------------------

/// Compute the maximum flow from `source` to `sink` using Dinic's algorithm.
///
/// Same arguments and result as [`edmonds_karp()`], and likewise safe on any
/// graph, but it augments along all shortest paths of one length per phase:
/// O(V²E) in general and O(E·√V) on unit networks, where Edmonds-Karp needs
/// one BFS per unit of flow.
///
/// The `capacity` closure extracts a capacity from each edge (called once per
/// edge, in edge index order). Only edges in their own direction carry flow; a
/// capacity that is not positive (`<= W::default()`) carries none. The flow
/// value must fit in `W`.
///
/// # Panics
///
/// If `source` or `sink` is not a node of the graph, or if they are the same
/// node.
///
/// # Example
///
/// ```rust
/// use safegraph::VecGraph;
/// use safegraph::graph::Graph;
/// use safegraph::algo::max_flow::dinic;
///
/// let mut g = VecGraph::<(), u32>::default();
/// let (s, t) = unsafe {
///     let s = g.insert_node_unchecked(()).unwrap();
///     let a = g.insert_node_unchecked(()).unwrap();
///     let b = g.insert_node_unchecked(()).unwrap();
///     let t = g.insert_node_unchecked(()).unwrap();
///     g.insert_edge_unchecked(3, [s, a]).unwrap();
///     g.insert_edge_unchecked(2, [s, b]).unwrap();
///     g.insert_edge_unchecked(1, [a, b]).unwrap();
///     g.insert_edge_unchecked(2, [a, t]).unwrap();
///     g.insert_edge_unchecked(3, [b, t]).unwrap();
///     (s, t)
/// };
/// assert_eq!(dinic(&g, s, t, |&cap| cap), 5);
/// ```
pub fn dinic<G, W, F>(graph: &G, source: G::NodeIx, sink: G::NodeIx, capacity: F) -> W
where
    G: Graph + for<'x> Directed<'x> + Bigraph + ?Sized,
    W: Copy + Ord + Default + Add<Output = W> + Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    // No index escapes, so unlike `dinic_with_flows` this needs no `Stable*`
    // marker: the graph is only read, through `&G`, for the whole call.
    let mut network = edge_network(graph, capacity);
    let (s, t) = terminals(&network.nodes, &source, &sink);
    network.residual.max_flow(s, t)
}

/// Compute the maximum flow with Dinic's algorithm and return the flow on
/// each edge.
///
/// Returns `(max_flow, flow_map)` where `flow_map` maps every edge index to
/// the flow through it. See [`dinic()`] for the arguments. The flow map is
/// deterministic: the same graph gives the same map on every run.
///
/// # Panics
///
/// If `source` or `sink` is not a node of the graph, or if they are the same
/// node.
pub fn dinic_with_flows<G, W, F>(
    graph: &G,
    source: G::NodeIx,
    sink: G::NodeIx,
    capacity: F,
) -> (W, HashMap<G::EdgeIx, W>)
where
    G: Graph + for<'x> Directed<'x> + Bigraph + StableEdge + StableNode + ?Sized,
    W: Copy + Ord + Default + Add<Output = W> + Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    let mut network = edge_network(graph, capacity);
    let (s, t) = terminals(&network.nodes, &source, &sink);
    let flow = network.residual.max_flow(s, t);
    let EdgeNetwork {
        edges, residual, ..
    } = network;
    let flows = edges
        .into_iter()
        .enumerate()
        .map(|(k, edge)| (edge, residual.flow(k)))
        .collect();
    (flow, flows)
}

/// Compute the maximum flow with Dinic's algorithm and the minimum cuts
/// closest to the source and to the sink.
///
/// Returns `(max_flow, cut)`. Both cuts in [`MinCut`] are the same for every
/// maximum flow (see the [module docs](self#minimum-cuts)), and the capacity
/// of each equals `max_flow`. See [`dinic()`] for the arguments.
///
/// # Panics
///
/// If `source` or `sink` is not a node of the graph, or if they are the same
/// node.
///
/// # Example
///
/// ```rust
/// use safegraph::VecGraph;
/// use safegraph::graph::Graph;
/// use safegraph::algo::max_flow::dinic_min_cut;
///
/// // s -> a -> t, each with capacity 1: either edge is a minimum cut.
/// let mut g = VecGraph::<(), u32>::default();
/// let (s, a, t, sa, at) = unsafe {
///     let s = g.insert_node_unchecked(()).unwrap();
///     let a = g.insert_node_unchecked(()).unwrap();
///     let t = g.insert_node_unchecked(()).unwrap();
///     let sa = g.insert_edge_unchecked(1, [s, a]).unwrap();
///     let at = g.insert_edge_unchecked(1, [a, t]).unwrap();
///     (s, a, t, sa, at)
/// };
/// // `VecGraph` indices are not stable; the cut is read before `g` changes.
/// let g = unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node() };
/// let (flow, cut) = dinic_min_cut(g, s, t, |&cap| cap);
/// assert_eq!(flow, 1);
/// // The cut closest to the sink is `a -> t`: `a` cannot reach `t` any more.
/// assert!(!cut.sink_side(a) && cut.sink_side(t));
/// assert_eq!(cut.sink_cut_edges().collect::<Vec<_>>(), [&at]);
/// // The cut closest to the source is `s -> a`.
/// assert!(cut.source_side(s) && !cut.source_side(a));
/// assert_eq!(cut.source_cut_edges().collect::<Vec<_>>(), [&sa]);
/// ```
pub fn dinic_min_cut<G, W, F>(
    graph: &G,
    source: G::NodeIx,
    sink: G::NodeIx,
    capacity: F,
) -> (W, MinCut<G::NodeIx, G::EdgeIx>)
where
    G: Graph + for<'x> Directed<'x> + Bigraph + StableEdge + StableNode + ?Sized,
    W: Copy + Ord + Default + Add<Output = W> + Sub<Output = W>,
    F: FnMut(&G::Edge) -> W,
{
    let mut network = edge_network(graph, capacity);
    let (s, t) = terminals(&network.nodes, &source, &sink);
    let flow = network.residual.max_flow(s, t);
    let EdgeNetwork {
        nodes,
        edges,
        residual,
    } = network;
    let source_side = residual.reach(s, Toward::Out, |spare| spare > W::default());
    let sink_side = residual.reach(t, Toward::In, |spare| spare > W::default());
    let cut = MinCut {
        nodes,
        edges,
        ends: residual.head,
        source_side,
        sink_side,
    };
    (flow, cut)
}

/// Compute a minimum vertex cut: the cheapest set of nodes that every
/// directed path from a node of `sources` to a node of `sinks` goes through.
///
/// `capacity` gives what cutting each node costs (called once per node, in
/// node index order); `None` means the node cannot be cut. Edges cannot be
/// cut. A source or a sink is a node like any other: it is cut when that is
/// cheapest, unless `capacity` returns `None` for it (do so for the textbook
/// s–t vertex cut, which never removes `s` or `t`).
///
/// Returns `Some((cost, cut_nodes))`, with `cut_nodes` in node index order, or
/// `None` if no set of cuttable nodes separates them (some path from a source
/// to a sink has only uncuttable nodes, its ends included). Among all minimum
/// cuts it returns the one **closest to the sinks**, which is unique: the
/// nodes it leaves able to reach a sink are as few as possible.
///
/// Each node is split inside into an in-node and an out-node joined by an arc
/// of the node's capacity, and Dinic's algorithm runs on that network
/// (O(E·√V) when every capacity is 0 or 1).
///
/// A capacity that is not positive costs nothing. `W::max_value()` stands for
/// an uncuttable node, so the total capacity of all cuttable nodes must be
/// less than `W::max_value()`.
///
/// # Panics
///
/// If a node of `sources` or `sinks` is not a node of the graph.
///
/// # Example
///
/// ```rust
/// use safegraph::VecGraph;
/// use safegraph::graph::Graph;
/// use safegraph::algo::max_flow::min_vertex_cut;
///
/// // Two roots share a hub, which leads to the target. The node data is what
/// // cutting the node costs; `None` means it cannot be cut.
/// let mut g = VecGraph::<Option<u32>, ()>::default();
/// let (r1, r2, hub, target) = unsafe {
///     let r1 = g.insert_node_unchecked(Some(1)).unwrap();
///     let r2 = g.insert_node_unchecked(Some(1)).unwrap();
///     let hub = g.insert_node_unchecked(Some(1)).unwrap();
///     let target = g.insert_node_unchecked(None).unwrap();
///     g.insert_edge_unchecked((), [r1, hub]).unwrap();
///     g.insert_edge_unchecked((), [r2, hub]).unwrap();
///     g.insert_edge_unchecked((), [hub, target]).unwrap();
///     (r1, r2, hub, target)
/// };
/// // `VecGraph` indices are not stable; the cut is read before `g` changes.
/// let g = unsafe { g.unsafe_assert_stable_node() };
/// // Cutting the hub (cost 1) beats cutting both roots (cost 2).
/// assert_eq!(min_vertex_cut(g, [r1, r2], [target], |&c| c), Some((1, vec![hub])));
/// // If the target may be cut too, it ties with the hub: the cut closest to
/// // the sink wins.
/// assert_eq!(min_vertex_cut(g, [r1, r2], [target], |_| Some(1)), Some((1, vec![target])));
/// // An uncuttable source that is also a sink cannot be separated from itself.
/// assert_eq!(min_vertex_cut(g, [target], [target], |&c| c), None);
/// ```
pub fn min_vertex_cut<G, W, I, J, F>(
    graph: &G,
    sources: I,
    sinks: J,
    mut capacity: F,
) -> Option<(W, Vec<G::NodeIx>)>
where
    G: Graph + for<'x> Directed<'x> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Default + Add<Output = W> + Sub<Output = W> + Bounded,
    I: IntoIterator<Item = G::NodeIx>,
    J: IntoIterator<Item = G::NodeIx>,
    F: FnMut(&G::Node) -> Option<W>,
{
    let infinite = W::max_value();
    let nodes = sorted_node_indices(graph);
    let edges = sorted_edge_indices(graph);
    let n = nodes.len();
    // Node `i` is split into `2i`, its in-node (where its in-edges arrive),
    // and `2i + 1`, its out-node (where its out-edges leave), joined by an arc
    // of the node's capacity: cutting the node is cutting that arc. Every
    // other arc is uncuttable. `2n` is the source and `2n + 1` the sink.
    let (s, t) = (2 * n, 2 * n + 1);
    let mut residual = Residual::with_edges(n + edges.len());
    for (i, node_ix) in nodes.iter().enumerate() {
        // SAFETY: `node_ix` comes from `node_indices` of `graph`, which stays
        // borrowed for the whole call.
        let node = unsafe { <G as GraphOperation<'_>>::node_unchecked(graph, node_ix) };
        residual.add_edge(2 * i, 2 * i + 1, capacity(node).unwrap_or(infinite));
    }
    for edge_ix in &edges {
        // SAFETY: as above, for `edge_indices`.
        let (u, v) = unsafe { dense_ends(graph, &nodes, edge_ix) };
        // A self-loop never lies on a path that a cut has to break.
        if u != v {
            residual.add_edge(2 * u + 1, 2 * v, infinite);
        }
    }
    for source in sources {
        let i = dense(&nodes, &source, "a source");
        residual.add_edge(s, 2 * i, infinite);
    }
    for sink in sinks {
        let i = dense(&nodes, &sink, "a sink");
        residual.add_edge(2 * i + 1, t, infinite);
    }
    residual.link(2 * n + 2);

    // A path of uncuttable arcs cannot be cut. Ruling it out also keeps every
    // augmenting path through a finite arc, so no arithmetic exceeds the
    // capacity of a minimum cut.
    if residual.reach(s, Toward::Out, |spare| spare == infinite)[t] {
        return None;
    }
    let flow = residual.max_flow(s, t);
    // The cut closest to the sink: a node is cut when its out-node can still
    // reach the sink and its in-node cannot.
    let sink_side = residual.reach(t, Toward::In, |spare| spare > W::default());
    let cut = nodes
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| sink_side[2 * i + 1] && !sink_side[2 * i])
        .map(|(_, node_ix)| node_ix)
        .collect();
    Some((flow, cut))
}

/// The minimum cuts closest to the source and to the sink, from
/// [`dinic_min_cut()`].
///
/// - The **source side** ([`source_side`](Self::source_side)) is the nodes
///   the source still reaches in the residual network; the edges leaving it
///   are the cut closest to the source.
/// - The **sink side** ([`sink_side`](Self::sink_side)) is the nodes that can
///   still reach the sink; the edges entering it are the cut closest to the
///   sink.
///
/// The two sides never meet; a node in neither lies between the two cuts.
/// Both are the same for every maximum flow. `N` and `E` are the graph's node
/// and edge index types; node and edge lists are in index order.
#[derive(Clone, Debug)]
pub struct MinCut<N, E> {
    /// Every node index, in index order; node `i` is `nodes[i]`.
    nodes: Vec<N>,
    /// Every edge index, in index order; edge `k` is `edges[k]`.
    edges: Vec<E>,
    /// Arc heads: edge `k` goes from `ends[2k + 1]` to `ends[2k]`.
    ends: Vec<usize>,
    source_side: Vec<bool>,
    sink_side: Vec<bool>,
}

impl<N: Ord, E> MinCut<N, E> {
    /// Returns `true` if the source still reaches `node`: `node` is on the
    /// source side of the minimum cut closest to the source.
    ///
    /// # Panics
    ///
    /// If `node` is not a node of the graph.
    pub fn source_side<B: Borrow<N>>(&self, node: B) -> bool {
        self.source_side[dense(&self.nodes, Borrow::<N>::borrow(&node), "the node")]
    }

    /// Returns `true` if `node` can still reach the sink: `node` is on the
    /// sink side of the minimum cut closest to the sink.
    ///
    /// # Panics
    ///
    /// If `node` is not a node of the graph.
    pub fn sink_side<B: Borrow<N>>(&self, node: B) -> bool {
        self.sink_side[dense(&self.nodes, Borrow::<N>::borrow(&node), "the node")]
    }

    /// The nodes the source still reaches, in index order.
    pub fn source_side_nodes(&self) -> impl Iterator<Item = &N> + '_ {
        side_nodes(&self.nodes, &self.source_side)
    }

    /// The nodes that can still reach the sink, in index order.
    pub fn sink_side_nodes(&self) -> impl Iterator<Item = &N> + '_ {
        side_nodes(&self.nodes, &self.sink_side)
    }

    /// The edges of the minimum cut closest to the source: from a node the
    /// source reaches to one it does not, in index order. Each is saturated,
    /// and their capacities add up to the maximum flow.
    pub fn source_cut_edges(&self) -> impl Iterator<Item = &E> + '_ {
        let side = &self.source_side;
        self.crossing(move |v| side[v])
    }

    /// The edges of the minimum cut closest to the sink: from a node that
    /// cannot reach the sink to one that can, in index order. Each is
    /// saturated, and their capacities add up to the maximum flow.
    pub fn sink_cut_edges(&self) -> impl Iterator<Item = &E> + '_ {
        let side = &self.sink_side;
        self.crossing(move |v| !side[v])
    }

    /// The edges from a node where `on_source_part` holds to one where it
    /// does not.
    fn crossing<'a>(
        &'a self,
        on_source_part: impl Fn(usize) -> bool + 'a,
    ) -> impl Iterator<Item = &'a E> + 'a {
        let ends = &self.ends;
        self.edges
            .iter()
            .enumerate()
            .filter(move |&(k, _)| on_source_part(ends[2 * k + 1]) && !on_source_part(ends[2 * k]))
            .map(|(_, edge)| edge)
    }
}

fn side_nodes<'a, N>(nodes: &'a [N], side: &'a [bool]) -> impl Iterator<Item = &'a N> + 'a {
    nodes
        .iter()
        .zip(side)
        .filter(|&(_, &on)| on)
        .map(|(node, _)| node)
}

/// A graph as a residual network: node `i` is `nodes[i]` and edge `k` is
/// `edges[k]`, whose arcs are `2k` and `2k + 1`.
struct EdgeNetwork<N, E, W> {
    nodes: Vec<N>,
    edges: Vec<E>,
    residual: Residual<W>,
}

fn edge_network<G, W, F>(graph: &G, mut capacity: F) -> EdgeNetwork<G::NodeIx, G::EdgeIx, W>
where
    G: Graph + for<'x> Directed<'x> + Bigraph + ?Sized,
    W: Copy + Ord + Default,
    F: FnMut(&G::Edge) -> W,
{
    let nodes = sorted_node_indices(graph);
    let edges = sorted_edge_indices(graph);
    let mut residual = Residual::with_edges(edges.len());
    for edge_ix in &edges {
        // SAFETY: `edge_ix` comes from `edge_indices` of `graph`, which stays
        // borrowed for the whole call.
        let (u, v) = unsafe { dense_ends(graph, &nodes, edge_ix) };
        let edge = unsafe { <G as GraphOperation<'_>>::edge_unchecked(graph, edge_ix) };
        residual.add_edge(u, v, capacity(edge));
    }
    residual.link(nodes.len());
    EdgeNetwork {
        nodes,
        edges,
        residual,
    }
}

/// Every node index of `graph`, in index order. Numbering nodes by it makes
/// the result independent of the backend's iteration order.
fn sorted_node_indices<G>(graph: &G) -> Vec<G::NodeIx>
where
    G: Graph + ?Sized,
{
    let mut nodes: Vec<G::NodeIx> = super::owned_node_indices(graph).collect();
    nodes.sort_unstable();
    nodes
}

/// Every edge index of `graph`, in index order.
fn sorted_edge_indices<G>(graph: &G) -> Vec<G::EdgeIx>
where
    G: Graph + ?Sized,
{
    let mut edges: Vec<G::EdgeIx> = super::owned_edge_indices(graph).collect();
    edges.sort_unstable();
    edges
}

/// The dense numbers of the tail and head of `edge_ix`.
///
/// # Safety
/// `edge_ix` must be a valid edge index of `graph`, and `nodes` its sorted
/// node indices.
unsafe fn dense_ends<G>(graph: &G, nodes: &[G::NodeIx], edge_ix: &G::EdgeIx) -> (usize, usize)
where
    G: Graph + for<'x> Directed<'x> + ?Sized,
{
    let tail = <G as Directed<'_>>::edge_tail_index_unchecked(graph, edge_ix);
    let head = <G as Directed<'_>>::edge_head_index_unchecked(graph, edge_ix);
    (
        dense(
            nodes,
            Borrow::<G::NodeIx>::borrow(&tail),
            "an edge endpoint",
        ),
        dense(
            nodes,
            Borrow::<G::NodeIx>::borrow(&head),
            "an edge endpoint",
        ),
    )
}

/// The dense number of `node` in the sorted `nodes`.
fn dense<N: Ord>(nodes: &[N], node: &N, what: &str) -> usize {
    match nodes.binary_search(node) {
        Ok(i) => i,
        Err(_) => panic!("{} is not a node of the graph", what),
    }
}

fn terminals<N: Ord>(nodes: &[N], source: &N, sink: &N) -> (usize, usize) {
    let s = dense(nodes, source, "the source");
    let t = dense(nodes, sink, "the sink");
    assert!(s != t, "the source and the sink must be different nodes");
    (s, t)
}

/// Which way [`Residual::reach`] follows arcs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Toward {
    /// Along arcs: the nodes reached from the start.
    Out,
    /// Against arcs: the nodes that reach the start.
    In,
}

/// The residual network of a flow over dense node numbers.
///
/// Arc `2k` is edge `k`, with its spare capacity; arc `2k + 1` is its twin,
/// with the flow on edge `k` that can be cancelled. So `a ^ 1` is the twin of
/// arc `a`, and the tail of `a` is the head of its twin.
struct Residual<W> {
    head: Vec<usize>,
    spare: Vec<W>,
    /// The arcs out of node `v` are `out[first[v]..first[v + 1]]`, in arc
    /// order. Every arc into `v` is the twin of one of them.
    first: Vec<usize>,
    out: Vec<usize>,
}

impl<W> Residual<W>
where
    W: Copy + Ord + Default,
{
    fn with_edges(edges: usize) -> Self {
        Residual {
            head: Vec::with_capacity(2 * edges),
            spare: Vec::with_capacity(2 * edges),
            first: Vec::new(),
            out: Vec::new(),
        }
    }

    /// Adds an edge from `u` to `v`. A capacity that is not positive carries
    /// no flow, and is stored as zero so that the flow on the edge is too.
    fn add_edge(&mut self, u: usize, v: usize, capacity: W) {
        let zero = W::default();
        self.head.push(v);
        self.spare
            .push(if capacity > zero { capacity } else { zero });
        self.head.push(u);
        self.spare.push(zero);
    }

    /// Builds the arc lists of `nodes` nodes, once every edge is added.
    fn link(&mut self, nodes: usize) {
        let tail = |a: usize| self.head[a ^ 1];
        let mut first = vec![0; nodes + 1];
        for a in 0..self.head.len() {
            first[tail(a) + 1] += 1;
        }
        for v in 0..nodes {
            first[v + 1] += first[v];
        }
        let mut fill = first.clone();
        let mut out = vec![0; self.head.len()];
        for a in 0..self.head.len() {
            let u = tail(a);
            out[fill[u]] = a;
            fill[u] += 1;
        }
        self.first = first;
        self.out = out;
    }

    /// The flow on edge `k`.
    fn flow(&self, k: usize) -> W {
        self.spare[2 * k + 1]
    }

    fn arcs(&self, v: usize) -> &[usize] {
        &self.out[self.first[v]..self.first[v + 1]]
    }

    /// The nodes reached from `start` (or that reach `start`) by `usable`
    /// arcs, judged by their spare capacity.
    fn reach(&self, start: usize, toward: Toward, usable: impl Fn(W) -> bool) -> Vec<bool> {
        let mut seen = vec![false; self.first.len() - 1];
        seen[start] = true;
        let mut stack = vec![start];
        while let Some(v) = stack.pop() {
            for &a in self.arcs(v) {
                let w = self.head[a];
                // Out: `a` goes `v -> w`. In: its twin goes `w -> v`.
                let arc = if toward == Toward::Out { a } else { a ^ 1 };
                if !seen[w] && usable(self.spare[arc]) {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        seen
    }
}

impl<W> Residual<W>
where
    W: Copy + Ord + Default + Add<Output = W> + Sub<Output = W>,
{
    /// Pushes a maximum flow from `s` to `t` (Dinic's algorithm) and returns
    /// its value. Allocates the level, next-arc, queue and path buffers once.
    fn max_flow(&mut self, s: usize, t: usize) -> W {
        let nodes = self.first.len() - 1;
        let mut level = vec![usize::MAX; nodes];
        let mut queue = Vec::with_capacity(nodes);
        let mut next = vec![0; nodes];
        let mut path = Vec::new();
        let mut total = W::default();
        while self.levels(s, t, &mut level, &mut queue) {
            next.copy_from_slice(&self.first[..nodes]);
            total = total + self.blocking_flow(s, t, &level, &mut next, &mut path);
        }
        total
    }

    /// Sets the distance of each node from `s` by arcs with spare capacity,
    /// and returns whether `t` is reached. Stops once `t` has its level: every
    /// node closer than `t` has one by then, and no farther node is used.
    fn levels(&self, s: usize, t: usize, level: &mut [usize], queue: &mut Vec<usize>) -> bool {
        let zero = W::default();
        for l in level.iter_mut() {
            *l = usize::MAX;
        }
        level[s] = 0;
        queue.clear();
        queue.push(s);
        let mut i = 0;
        while i < queue.len() {
            let v = queue[i];
            i += 1;
            for &a in self.arcs(v) {
                let w = self.head[a];
                if level[w] == usize::MAX && self.spare[a] > zero {
                    level[w] = level[v] + 1;
                    if w == t {
                        return true;
                    }
                    queue.push(w);
                }
            }
        }
        false
    }

    /// Saturates every shortest path from `s` to `t` in the level graph and
    /// returns the flow pushed. `next[v]` is the next arc to try out of `v`:
    /// the arcs before it lead nowhere in this phase.
    fn blocking_flow(
        &mut self,
        s: usize,
        t: usize,
        level: &[usize],
        next: &mut [usize],
        path: &mut Vec<usize>,
    ) -> W {
        let zero = W::default();
        let mut total = zero;
        path.clear();
        let mut v = s;
        loop {
            if v == t {
                let mut pushed = self.spare[path[0]];
                for &a in path.iter() {
                    if self.spare[a] < pushed {
                        pushed = self.spare[a];
                    }
                }
                let mut saturated = path.len();
                for (i, &a) in path.iter().enumerate() {
                    self.spare[a] = self.spare[a] - pushed;
                    self.spare[a ^ 1] = self.spare[a ^ 1] + pushed;
                    if saturated == path.len() && self.spare[a] <= zero {
                        saturated = i;
                    }
                }
                total = total + pushed;
                // Resume from the tail of the first saturated arc: the path up
                // to it still has spare capacity.
                v = self.head[path[saturated] ^ 1];
                path.truncate(saturated);
                continue;
            }
            let end = self.first[v + 1];
            let mut cursor = next[v];
            while cursor < end {
                let a = self.out[cursor];
                let w = self.head[a];
                // A node at the sink's level other than the sink leads nowhere.
                if self.spare[a] > zero
                    && level[w] == level[v] + 1
                    && (w == t || level[w] < level[t])
                {
                    break;
                }
                cursor += 1;
            }
            next[v] = cursor;
            if cursor < end {
                let a = self.out[cursor];
                path.push(a);
                v = self.head[a];
            } else {
                // A dead end: back to the node before it, past the arc that
                // led here. At the source, the phase is over.
                match path.pop() {
                    Some(a) => {
                        v = self.head[a ^ 1];
                        next[v] += 1;
                    }
                    None => return total,
                }
            }
        }
    }
}

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

    // ---- Dinic ----

    #[test]
    fn dinic_basic() {
        let (g, s, _, _, t) = build_flow_graph();
        assert_eq!(dinic(&g, s, t, |&cap| cap), 12);
    }

    #[test]
    fn dinic_no_path() {
        let mut g = VecGraph::<&str, u32>::default();
        let (s, t) = unsafe {
            let s = InsertNode::insert_node_unchecked(&mut g, "s").unwrap();
            let t = InsertNode::insert_node_unchecked(&mut g, "t").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 5, [t, s]).unwrap();
            (s, t)
        };
        assert_eq!(dinic(&g, s, t, |&cap| cap), 0);
    }

    #[test]
    fn dinic_bottleneck_and_zero_capacity() {
        // s -> a (100) -> b (1) -> t (100), plus a useless s -> t of capacity 0.
        let mut g = VecGraph::<&str, u32>::default();
        let (s, t) = unsafe {
            let s = InsertNode::insert_node_unchecked(&mut g, "s").unwrap();
            let a = InsertNode::insert_node_unchecked(&mut g, "a").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "b").unwrap();
            let t = InsertNode::insert_node_unchecked(&mut g, "t").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 100, [s, a]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 1, [a, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 100, [b, t]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 0, [s, t]).unwrap();
            (s, t)
        };
        assert_eq!(dinic(&g, s, t, |&cap| cap), 1);
    }

    #[test]
    fn dinic_negative_capacity_carries_nothing() {
        let mut g = VecGraph::<(), i32>::default();
        let (s, t) = unsafe {
            let s = InsertNode::insert_node_unchecked(&mut g, ()).unwrap();
            let t = InsertNode::insert_node_unchecked(&mut g, ()).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, -4, [s, t]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 3, [s, t]).unwrap();
            (s, t)
        };
        let g = unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node() };
        let (flow, flows) = dinic_with_flows(g, s, t, |&cap| cap);
        assert_eq!(flow, 3);
        assert_eq!(flows[&0], 0);
        assert_eq!(flows[&1], 3);
    }

    #[test]
    fn dinic_cancels_flow_on_a_crossing_edge() {
        // The only shortest path s -> u -> v -> t takes the crossing u -> v.
        // Max flow 2 is s -> u -> y1 -> y2 -> t plus s -> x1 -> x2 -> v -> t,
        // which the second phase finds only by cancelling the flow on u -> v.
        let mut g = VecGraph::<(), u32>::default();
        let (s, t, uv) = unsafe {
            let mut node = || InsertNode::insert_node_unchecked(&mut g, ()).unwrap();
            let (s, u, v, t) = (node(), node(), node(), node());
            let (x1, x2, y1, y2) = (node(), node(), node(), node());
            let mut uv = None;
            for &(a, b) in &[
                (s, u),
                (u, v),
                (v, t),
                (s, x1),
                (x1, x2),
                (x2, v),
                (u, y1),
                (y1, y2),
                (y2, t),
            ] {
                let e = InsertEdge::insert_edge_unchecked(&mut g, 1, [a, b]).unwrap();
                if (a, b) == (u, v) {
                    uv = Some(e);
                }
            }
            (s, t, uv.unwrap())
        };
        assert_eq!(dinic(&g, s, t, |&cap| cap), 2);
        let g = unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node() };
        let (flow, flows) = dinic_with_flows(g, s, t, |&cap| cap);
        assert_eq!(flow, 2);
        assert_eq!(flows[&uv], 0);
    }

    #[test]
    fn dinic_with_flows_conserves_flow() {
        let (g, s, _, _, t) = build_flow_graph();
        let g = unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node() };
        let (flow, flows) = dinic_with_flows(g, s, t, |&cap| cap);
        assert_eq!(flow, 12);
        assert_eq!(flows.len(), 5);
        let mut net = [0i64; 4];
        for (eix, &f) in &flows {
            let cap = *Graph::edge(g, eix);
            assert!(f <= cap);
            let (u, v) = unsafe { crate::algo::edge_pair(g, eix) };
            net[u as usize] -= f as i64;
            net[v as usize] += f as i64;
        }
        assert_eq!(net, [-12, 0, 0, 12]);
    }

    #[test]
    #[should_panic(expected = "the source and the sink must be different nodes")]
    fn dinic_same_source_and_sink_panics() {
        let (g, s, _, _, _) = build_flow_graph();
        dinic(&g, s, s, |&cap| cap);
    }

    #[test]
    #[should_panic(expected = "the sink is not a node of the graph")]
    fn dinic_unknown_sink_panics() {
        let (g, s, _, _, _) = build_flow_graph();
        dinic(&g, s, 99, |&cap| cap);
    }

    #[test]
    fn residual_retreats_past_dead_ends() {
        // 0 -> 1 -> 2 (dead end), 1 -> 3 -> 4 = sink; 0 -> 5 -> 4.
        let mut r = Residual::<u32>::with_edges(6);
        for &(u, v) in &[(0, 1), (1, 2), (1, 3), (3, 4), (0, 5), (5, 4), (2, 6)] {
            r.add_edge(u, v, 1);
        }
        r.link(7);
        assert_eq!(r.max_flow(0, 4), 2);
        let from_source = r.reach(0, Toward::Out, |spare| spare > 0);
        assert_eq!(
            from_source,
            [true, false, false, false, false, false, false]
        );
        let to_sink = r.reach(4, Toward::In, |spare| spare > 0);
        assert_eq!(to_sink, [false, false, false, false, true, false, false]);
    }
}
