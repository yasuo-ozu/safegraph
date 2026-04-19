//! # Shortest Paths
//!
//! A collection of single-source and all-pairs shortest path algorithms for
//! weighted directed graphs. Edge weights are extracted via a closure
//! `F: FnMut(&G::Edge) -> W` rather than requiring a trait on the edge type.
//!
//! ## Components
//!
//! - [`dijkstra`] -- single-source shortest paths using a binary min-heap.
//!   Requires non-negative edge weights. O((V + E) log V).
//! - [`bellman_ford`] -- single-source shortest paths that handles negative
//!   weights. Returns [`NegativeCycleError`] if a negative cycle is reachable. O(V * E).
//! - [`astar`] -- A* search from a source to a specific goal using a heuristic
//!   function. Returns `Option<(cost, path)>`.
//! - [`floyd_warshall`] -- all-pairs shortest paths. Returns a map from
//!   `(source, target)` to distance. O(V^3).
//! - [`k_shortest_paths`] -- Yen's algorithm for the K shortest loopless paths
//!   between two nodes. Returns up to K `(cost, path)` pairs.
//! - [`NegativeCycleError`] -- error type returned when a negative-weight cycle
//!   is detected by Bellman-Ford or Floyd-Warshall.
//!
//! Each function has an `_unchecked` variant following the crate
//! convention (see [`crate::algo`] for details).
//!
//! ## Algorithm
//!
//! ```text
//!   Dijkstra's algorithm (example)
//!
//!       0 --2--> 1 --3--> 3
//!       |                 ^
//!       +---1--> 2 --1---+
//!
//!   Start: 0,  dist = {0: 0}
//!   Heap: [(0, node 0)]
//!
//!   pop (0, 0): relax 0->1 cost 2, relax 0->2 cost 1
//!     dist = {0: 0, 1: 2, 2: 1}
//!   pop (1, 2): relax 2->3 cost 1+1=2
//!     dist = {0: 0, 1: 2, 2: 1, 3: 2}
//!   pop (2, 1): no improvement
//!   pop (2, 3): done
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::shortest_path::dijkstra;
//!
//! let mut g = BTreeGraph::<u32, u32>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_edge(2, [0, 1]).unwrap();
//! g.insert_edge(1, [0, 2]).unwrap();
//! g.insert_edge(3, [1, 2]).unwrap();
//!
//! let dist = dijkstra(&g, 0, None, |e| *e);
//! // shortest path 0->2 has cost 1 (direct edge)
//! assert_eq!(dist[&2].0, 1);
//! ```

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ops::Add;

use num_traits::Bounded;

use crate::graph::capability::{Bigraph, Directed, StableNode};
use crate::graph::Graph;

/// Error returned when a negative cycle is detected.
#[derive(Debug, Clone)]
pub struct NegativeCycleError;

type DistPredMap<N, W> = HashMap<N, (W, Option<N>)>;
type PairDistMap<N, W> = HashMap<(N, N), W>;

impl std::fmt::Display for NegativeCycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "negative cycle detected")
    }
}

impl std::error::Error for NegativeCycleError {}

// Min-heap entry for Dijkstra/A*
#[derive(Debug)]
struct MinScore<W, N>(W, N);

impl<W: PartialEq, N> PartialEq for MinScore<W, N> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<W: PartialEq, N> Eq for MinScore<W, N> {}

impl<W: PartialOrd, N> PartialOrd for MinScore<W, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl<W: Ord, N> Ord for MinScore<W, N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.cmp(&self.0)
    }
}

/// Dijkstra's single-source shortest path algorithm.
///
/// Computes shortest paths from `start` to all reachable nodes (or only to `goal` if provided).
/// `edge_weight` extracts a non-negative weight from each edge.
///
/// Returns a map from node index to `(distance, optional_predecessor)`.
pub fn dijkstra<'r, G, W, F>(
    graph: &'r G,
    start: G::NodeIx,
    goal: Option<G::NodeIx>,
    edge_weight: F,
) -> HashMap<G::NodeIx, (W, Option<G::NodeIx>)>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
{
    // SAFETY: StableNode guarantee index stability
    unsafe { dijkstra_unchecked(graph, start, goal, edge_weight) }
}

/// Dijkstra without requiring `StableNode`/`StableEdge`.
///
/// # Safety
/// The graph must not be modified during the algorithm.
pub unsafe fn dijkstra_unchecked<'r, G, W, F>(
    graph: &'r G,
    start: G::NodeIx,
    goal: Option<G::NodeIx>,
    mut edge_weight: F,
) -> HashMap<G::NodeIx, (W, Option<G::NodeIx>)>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
{
    let mut dist: HashMap<G::NodeIx, (W, Option<G::NodeIx>)> = HashMap::new();
    let mut visited: HashSet<G::NodeIx> = HashSet::new();
    let mut heap = BinaryHeap::new();

    dist.insert(start, (W::default(), None));
    heap.push(MinScore(W::default(), start));

    while let Some(MinScore(cost, node)) = heap.pop() {
        if !visited.insert(node) {
            continue;
        }

        if goal == Some(node) {
            break;
        }

        for wi in <G as crate::graph::GraphOperation<'_>>::walks_from_unchecked(graph, node) {
            let (_, edge, target) = wi.get();
            if visited.contains(&target) {
                continue;
            }
            let w = edge_weight(edge);
            let new_dist = cost + w;

            let is_shorter = match dist.get(&target) {
                Some(&(d, _)) => new_dist < d,
                None => true,
            };

            if is_shorter {
                dist.insert(target, (new_dist, Some(node)));
                heap.push(MinScore(new_dist, target));
            }
        }
    }

    dist
}

/// Iterator that yields relaxation events `(node, distance, predecessor)` from the
/// Bellman-Ford single-source shortest path algorithm.
///
/// Each call to `next()` advances through the edge relaxation process and yields
/// when a node's distance is improved. After the iterator is exhausted, call
/// `finish()` to perform the negative cycle check and obtain the final result.
pub struct BellmanFord<N, W> {
    edges: Vec<(N, N, W)>,
    dist: HashMap<N, (W, Option<N>)>,
    max_rounds: usize,
    round: usize,
    edge_idx: usize,
    early_exit: bool,
    round_changed: bool,
}

/// Bellman-Ford single-source shortest path algorithm.
///
/// Supports negative edge weights. Returns `Err(NegativeCycleError)` if a negative
/// cycle is reachable from `start`.
///
/// Returns a map from node index to `(distance, optional_predecessor)`.
pub fn bellman_ford<'r, G, W, F>(
    graph: &'r G,
    start: G::NodeIx,
    edge_weight: F,
) -> Result<DistPredMap<G::NodeIx, W>, NegativeCycleError>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
{
    // SAFETY: StableNode guarantee index stability
    let iter = unsafe { bellman_ford_unchecked(graph, start, edge_weight) };
    iter.finish()
}

/// Bellman-Ford without requiring `StableNode`/`StableEdge`.
///
/// Returns an iterator that yields `(node, distance, predecessor)` each time a
/// node's distance is improved. After exhausting the iterator, call `finish()` to
/// check for negative cycles.
///
/// # Safety
/// The graph must not be modified until the returning NodeIx is accessed.
pub unsafe fn bellman_ford_unchecked<'r, G, W, F>(
    graph: &'r G,
    start: G::NodeIx,
    mut edge_weight: F,
) -> BellmanFord<G::NodeIx, W>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
{
    let node_count = <_ as crate::graph::GraphOperation<'_>>::node_indices(graph).count();

    let mut dist: HashMap<G::NodeIx, (W, Option<G::NodeIx>)> = HashMap::new();
    dist.insert(start, (W::default(), None));

    let edges: Vec<(G::NodeIx, G::NodeIx, W)> =
        <_ as crate::graph::GraphOperation<'_>>::edge_indices(graph)
        .map(|eix| {
            let tail = graph.edge_tail_index_unchecked(eix);
            let head = graph.edge_head_index_unchecked(eix);
            let w = edge_weight(Graph::edge_unchecked(graph, eix));
            (tail, head, w)
        })
        .collect();

    BellmanFord {
        edges,
        dist,
        max_rounds: node_count.saturating_sub(1),
        round: 0,
        edge_idx: 0,
        early_exit: false,
        round_changed: false,
    }
}

impl<N, W> BellmanFord<N, W>
where
    N: Copy + Eq + std::hash::Hash,
    W: Copy + Ord + Add<Output = W> + Default,
{
    /// Complete the algorithm and check for negative cycles.
    ///
    /// Drains remaining relaxation events and then verifies no negative cycle exists.
    pub fn finish(mut self) -> Result<HashMap<N, (W, Option<N>)>, NegativeCycleError> {
        // Drain the iterator
        while self.next().is_some() {}

        // Check for negative cycles
        for &(tail, head, w) in &self.edges {
            if let Some(&(d, _)) = self.dist.get(&tail) {
                let new_dist = d + w;
                let is_shorter = match self.dist.get(&head) {
                    Some(&(existing, _)) => new_dist < existing,
                    None => true,
                };
                if is_shorter {
                    return Err(NegativeCycleError);
                }
            }
        }

        Ok(self.dist)
    }
}

impl<N, W> Iterator for BellmanFord<N, W>
where
    N: Copy + Eq + std::hash::Hash,
    W: Copy + Ord + Add<Output = W> + Default,
{
    type Item = (N, W, Option<N>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.early_exit {
            return None;
        }

        loop {
            if self.round >= self.max_rounds {
                return None;
            }

            if self.edge_idx >= self.edges.len() {
                // End of round
                if !self.round_changed {
                    self.early_exit = true;
                    return None;
                }
                self.round += 1;
                self.edge_idx = 0;
                self.round_changed = false;
                if self.round >= self.max_rounds {
                    return None;
                }
            }

            let (tail, head, w) = self.edges[self.edge_idx];
            self.edge_idx += 1;

            if let Some(&(d, _)) = self.dist.get(&tail) {
                let new_dist = d + w;
                let is_shorter = match self.dist.get(&head) {
                    Some(&(existing, _)) => new_dist < existing,
                    None => true,
                };
                if is_shorter {
                    self.dist.insert(head, (new_dist, Some(tail)));
                    self.round_changed = true;
                    return Some((head, new_dist, Some(tail)));
                }
            }
        }
    }
}

/// A* shortest path algorithm with heuristic.
///
/// Returns `Some((cost, path))` if `goal` is reachable from `start`, `None` otherwise.
/// The `heuristic` function should return an admissible estimate of the remaining
/// distance from a node to `goal`.
pub fn astar<'r, G, W, F, H>(
    graph: &'r G,
    start: G::NodeIx,
    goal: G::NodeIx,
    edge_weight: F,
    heuristic: H,
) -> Option<(W, Vec<G::NodeIx>)>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
    H: FnMut(G::NodeIx) -> W,
{
    // SAFETY: StableNode guarantee index stability
    unsafe { astar_unchecked(graph, start, goal, edge_weight, heuristic) }
}

/// A* without requiring `StableNode`/`StableEdge`.
///
/// # Safety
/// The graph must not be modified until the returning NodeIx is accessed
pub unsafe fn astar_unchecked<'r, G, W, F, H>(
    graph: &'r G,
    start: G::NodeIx,
    goal: G::NodeIx,
    mut edge_weight: F,
    mut heuristic: H,
) -> Option<(W, Vec<G::NodeIx>)>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
    H: FnMut(G::NodeIx) -> W,
{
    let mut g_score: HashMap<G::NodeIx, W> = HashMap::new();
    let mut came_from: HashMap<G::NodeIx, G::NodeIx> = HashMap::new();
    let mut closed: HashSet<G::NodeIx> = HashSet::new();
    let mut heap = BinaryHeap::new();

    g_score.insert(start, W::default());
    heap.push(MinScore(W::default() + heuristic(start), start));

    while let Some(MinScore(_, node)) = heap.pop() {
        if node == goal {
            // Reconstruct path
            let cost = g_score[&goal];
            let mut path = vec![goal];
            let mut current = goal;
            while let Some(&prev) = came_from.get(&current) {
                path.push(prev);
                current = prev;
            }
            path.reverse();
            return Some((cost, path));
        }

        if !closed.insert(node) {
            continue;
        }

        let current_g = g_score[&node];

        for wi in <G as crate::graph::GraphOperation<'_>>::walks_from_unchecked(graph, node) {
            let (_, edge, target) = wi.get();
            if closed.contains(&target) {
                continue;
            }
            let w = edge_weight(edge);
            let new_g = current_g + w;

            let is_shorter = match g_score.get(&target) {
                Some(&g) => new_g < g,
                None => true,
            };

            if is_shorter {
                g_score.insert(target, new_g);
                came_from.insert(target, node);
                let f_score = new_g + heuristic(target);
                heap.push(MinScore(f_score, target));
            }
        }
    }

    None
}

/// Floyd-Warshall all-pairs shortest path algorithm.
///
/// Returns a map from `(source, target)` to the shortest distance.
/// Returns `Err(NegativeCycleError)` if a negative cycle exists.
/// Unreachable pairs are not included in the result.
pub fn floyd_warshall<'r, G, W, F>(
    graph: &'r G,
    mut edge_weight: F,
) -> Result<PairDistMap<G::NodeIx, W>, NegativeCycleError>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default + Bounded,
    F: FnMut(&G::Edge) -> W,
{
    let nodes: Vec<G::NodeIx> = <_ as crate::graph::GraphOperation<'_>>::node_indices(graph).collect();
    let inf = W::max_value();

    // dist[(i, j)] = shortest distance from i to j, using max_value() as infinity
    let mut dist: HashMap<(G::NodeIx, G::NodeIx), W> = HashMap::new();

    // Initialize: all pairs to infinity
    for &i in &nodes {
        for &j in &nodes {
            dist.insert((i, j), if i == j { W::default() } else { inf });
        }
    }

    // Initialize: direct edges
    for eix in <_ as crate::graph::GraphOperation<'_>>::edge_indices(graph) {
        let tail = graph.edge_tail_index(eix);
        let head = graph.edge_head_index(eix);
        let w = edge_weight(graph.edge(eix));
        let entry = dist.entry((tail, head)).or_insert(inf);
        if w < *entry {
            *entry = w;
        }
    }

    // Relax through intermediate nodes
    for &k in &nodes {
        for &i in &nodes {
            let d_ik = dist[&(i, k)];
            if d_ik == inf {
                continue; // skip overflow
            }
            for &j in &nodes {
                let d_kj = dist[&(k, j)];
                if d_kj == inf {
                    continue; // skip overflow
                }
                let new_dist = d_ik + d_kj;
                let entry = dist.get_mut(&(i, j)).unwrap();
                if new_dist < *entry {
                    *entry = new_dist;
                }
            }
        }
    }

    // Check for negative cycles (diagonal < 0)
    for &n in &nodes {
        if dist[&(n, n)] < W::default() {
            return Err(NegativeCycleError);
        }
    }

    // Collect results, filtering out unreachable pairs (still at infinity)
    let result = dist.into_iter().filter(|(_, d)| *d != inf).collect();
    Ok(result)
}

/// Iterator that lazily yields shortest paths using Yen's algorithm.
///
/// Each call to `next()` finds the next shortest path from `start` to `goal`.
/// The paths are yielded in ascending order of cost.
/// `N` is the node-index type (`G::NodeIx`), a separate type parameter so the
/// struct carries no `Graph` bound.
pub struct KShortestPaths<'r, G: ?Sized, W, F, N> {
    graph: &'r G,
    goal: N,
    k: usize,
    edge_weight: F,
    shortest_paths: Vec<(W, Vec<N>)>,
    candidates: BinaryHeap<std::cmp::Reverse<(W, Vec<N>)>>,
    current_k: usize,
    done: bool,
}

/// K-Shortest Paths using Yen's algorithm.
///
/// Finds up to `k` shortest paths from `start` to `goal`.
/// Returns an iterator yielding `(cost, path)` in ascending order of cost.
pub fn k_shortest_paths<'r, G, W, F>(
    graph: &'r G,
    start: G::NodeIx,
    goal: G::NodeIx,
    k: usize,
    edge_weight: F,
) -> KShortestPaths<'r, G, W, F, G::NodeIx>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
{
    // SAFETY: StableNode guarantee index stability
    unsafe { k_shortest_paths_unchecked(graph, start, goal, k, edge_weight) }
}

/// K-Shortest Paths iterator without requiring `StableNode`/`StableEdge`.
///
/// # Safety
/// The graph must not be modified while the iterator is alive.
pub unsafe fn k_shortest_paths_unchecked<'r, G, W, F>(
    graph: &'r G,
    start: G::NodeIx,
    goal: G::NodeIx,
    k: usize,
    mut edge_weight: F,
) -> KShortestPaths<'r, G, W, F, G::NodeIx>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
{
    let mut shortest_paths = Vec::new();
    let mut done = false;

    if k > 0 {
        let first = astar_unchecked(graph, start, goal, |e| edge_weight(e), |_| W::default());
        match first {
            Some(p) => shortest_paths.push(p),
            None => done = true,
        }
    } else {
        done = true;
    }

    KShortestPaths {
        graph,
        goal,
        k,
        edge_weight,
        shortest_paths,
        candidates: BinaryHeap::new(),
        current_k: 0,
        done,
    }
}

impl<'r, G, W, F> KShortestPaths<'r, G, W, F, G::NodeIx>
where
    G: Graph + Directed<'r> + Bigraph + StableNode,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
{
    /// Collect all remaining paths and return them along with already-yielded paths.
    pub fn finish(mut self) -> Vec<(W, Vec<G::NodeIx>)> {
        while self.next().is_some() {}
        self.shortest_paths
    }
}

impl<'r, G, W, F> Iterator for KShortestPaths<'r, G, W, F, G::NodeIx>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
    W: Copy + Ord + Add<Output = W> + Default,
    F: FnMut(&G::Edge) -> W,
{
    type Item = (W, Vec<G::NodeIx>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // First call: yield the first shortest path
        if self.current_k == 0 {
            if self.shortest_paths.is_empty() {
                self.done = true;
                return None;
            }
            self.current_k = 1;
            return Some(self.shortest_paths[0].clone());
        }

        if self.current_k >= self.k {
            self.done = true;
            return None;
        }

        // Find candidates from the previous shortest path
        let ki = self.current_k;
        let prev_path = self.shortest_paths[ki - 1].1.clone();

        for spur_idx in 0..prev_path.len() - 1 {
            let spur_node = prev_path[spur_idx];
            let root_path = &prev_path[..=spur_idx];
            let root_cost = if spur_idx == 0 {
                W::default()
            } else {
                let mut cost = W::default();
                for w in 0..spur_idx {
                    let from = prev_path[w];
                    let to = prev_path[w + 1];
                    for eix in unsafe { <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(self.graph, from) } {
                        if unsafe { self.graph.edge_head_index_unchecked(eix) } == to {
                            cost = cost
                                + (self.edge_weight)(unsafe { Graph::edge_unchecked(self.graph, eix) });
                            break;
                        }
                    }
                }
                cost
            };

            let mut excluded_edges: HashSet<(G::NodeIx, G::NodeIx)> = HashSet::new();
            for sp in &self.shortest_paths {
                if sp.1.len() > spur_idx && sp.1[..=spur_idx] == *root_path {
                    excluded_edges.insert((sp.1[spur_idx], sp.1[spur_idx + 1]));
                }
            }

            let root_nodes: HashSet<G::NodeIx> =
                root_path[..root_path.len() - 1].iter().copied().collect();

            let spur_path = {
                let mut dist_map: HashMap<G::NodeIx, (W, Option<G::NodeIx>)> = HashMap::new();
                let mut visited: HashSet<G::NodeIx> = HashSet::new();
                let mut heap = BinaryHeap::new();

                dist_map.insert(spur_node, (W::default(), None));
                heap.push(MinScore(W::default(), spur_node));

                while let Some(MinScore(cost, node)) = heap.pop() {
                    if !visited.insert(node) {
                        continue;
                    }
                    if node == self.goal {
                        break;
                    }
                    for eix in unsafe { <G as crate::graph::GraphOperation<'_>>::edge_indices_from_unchecked(self.graph, node) } {
                        let target = unsafe { self.graph.edge_head_index_unchecked(eix) };
                        if visited.contains(&target) || root_nodes.contains(&target) {
                            continue;
                        }
                        if excluded_edges.contains(&(node, target)) {
                            continue;
                        }
                        let w = (self.edge_weight)(unsafe { Graph::edge_unchecked(self.graph, eix) });
                        let new_dist = cost + w;
                        let is_shorter = match dist_map.get(&target) {
                            Some(&(d, _)) => new_dist < d,
                            None => true,
                        };
                        if is_shorter {
                            dist_map.insert(target, (new_dist, Some(node)));
                            heap.push(MinScore(new_dist, target));
                        }
                    }
                }

                if !dist_map.contains_key(&self.goal) || !visited.contains(&self.goal) {
                    None
                } else {
                    let spur_cost = dist_map[&self.goal].0;
                    let mut path = vec![self.goal];
                    let mut current = self.goal;
                    while let Some(&(_, Some(prev))) = dist_map.get(&current) {
                        path.push(prev);
                        current = prev;
                    }
                    path.reverse();
                    Some((spur_cost, path))
                }
            };

            if let Some((spur_cost, spur_path)) = spur_path {
                let total_cost = root_cost + spur_cost;
                let mut total_path: Vec<G::NodeIx> = root_path[..root_path.len() - 1].to_vec();
                total_path.extend(spur_path);

                let is_dup = self.shortest_paths.iter().any(|(_, p)| *p == total_path)
                    || self
                        .candidates
                        .iter()
                        .any(|std::cmp::Reverse((_, p))| *p == total_path);
                if !is_dup {
                    self.candidates
                        .push(std::cmp::Reverse((total_cost, total_path)));
                }
            }
        }

        if let Some(std::cmp::Reverse(next)) = self.candidates.pop() {
            self.shortest_paths.push(next.clone());
            self.current_k += 1;
            Some(next)
        } else {
            self.done = true;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::capability::{InsertEdge, InsertNode};
    use crate::VecGraph;

    // Use VecGraph for shortest path tests since it allows duplicate edge values
    fn weighted_vecgraph() -> (VecGraph<&'static str, u32>, [u32; 4]) {
        let mut g = VecGraph::<&str, u32>::default();
        unsafe {
            let a = InsertNode::insert_node_unchecked(&mut g, "A").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "B").unwrap();
            let c = InsertNode::insert_node_unchecked(&mut g, "C").unwrap();
            let d = InsertNode::insert_node_unchecked(&mut g, "D").unwrap();
            // A->B (w=1), A->C (w=4), B->C (w=2), B->D (w=5), C->D (w=1)
            InsertEdge::insert_edge_unchecked(&mut g, 1u32, [a, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 4, [a, c]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 2, [b, c]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 5, [b, d]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 1, [c, d]).unwrap();
            (g, [a, b, c, d])
        }
    }

    #[test]
    fn dijkstra_basic() {
        let (g, [a, b, c, d]) = weighted_vecgraph();
        let result = unsafe { dijkstra_unchecked(g.unsafe_assert_stable_node(), a, None, |&w| w) };
        assert_eq!(result[&a].0, 0);
        assert_eq!(result[&b].0, 1);
        assert_eq!(result[&c].0, 3); // A->B->C = 1+2
        assert_eq!(result[&d].0, 4); // A->B->C->D = 1+2+1
    }

    #[test]
    fn dijkstra_with_goal() {
        let (g, [a, _, c, _]) = weighted_vecgraph();
        let result = unsafe { dijkstra_unchecked(g.unsafe_assert_stable_node(), a, Some(c), |&w| w) };
        assert_eq!(result[&c].0, 3);
    }

    #[test]
    fn dijkstra_unreachable() {
        unsafe {
            let mut g = VecGraph::<&str, u32>::default();
            let a = InsertNode::insert_node_unchecked(&mut g, "A").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "B").unwrap();
            // No edges
            let result = dijkstra_unchecked(g.unsafe_assert_stable_node(), a, None, |&w| w);
            assert!(result.contains_key(&a));
            assert!(!result.contains_key(&b));
        }
    }

    #[test]
    fn bellman_ford_basic() {
        let (g, [a, b, c, d]) = weighted_vecgraph();
        let result = unsafe { bellman_ford_unchecked(g.unsafe_assert_stable_node(), a, |&w| w as i64) }
            .finish()
            .unwrap();
        assert_eq!(result[&a].0, 0);
        assert_eq!(result[&b].0, 1);
        assert_eq!(result[&c].0, 3);
        assert_eq!(result[&d].0, 4);
    }

    #[test]
    fn bellman_ford_negative_cycle() {
        unsafe {
            let mut g = VecGraph::<&str, i64>::default();
            let a = InsertNode::insert_node_unchecked(&mut g, "A").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "B").unwrap();
            let c = InsertNode::insert_node_unchecked(&mut g, "C").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 1i64, [a, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, -1, [b, c]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, -1, [c, a]).unwrap();
            let result = bellman_ford_unchecked(g.unsafe_assert_stable_node(), a, |&w| w).finish();
            assert!(result.is_err());
        }
    }

    #[test]
    fn astar_basic() {
        let (g, [a, b, c, d]) = weighted_vecgraph();
        let result = unsafe { astar_unchecked(g.unsafe_assert_stable_node(), a, d, |&w| w, |_| 0u32) };
        let (cost, path) = result.unwrap();
        assert_eq!(cost, 4);
        assert_eq!(path, vec![a, b, c, d]);
    }

    #[test]
    fn astar_unreachable() {
        unsafe {
            let mut g = VecGraph::<&str, u32>::default();
            let a = InsertNode::insert_node_unchecked(&mut g, "A").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "B").unwrap();
            let result = astar_unchecked(g.unsafe_assert_stable_node(), a, b, |&w| w, |_| 0u32);
            assert!(result.is_none());
        }
    }

    #[test]
    fn floyd_warshall_basic() {
        let (g, [a, b, c, d]) = weighted_vecgraph();
        let result = floyd_warshall(unsafe { g.unsafe_assert_stable_node() }, |&w| w).unwrap();
        assert_eq!(result[&(a, a)], 0);
        assert_eq!(result[&(a, b)], 1);
        assert_eq!(result[&(a, c)], 3);
        assert_eq!(result[&(a, d)], 4);
        // b->c = 2, b->d = min(5, 2+1) = 3
        assert_eq!(result[&(b, c)], 2);
        assert_eq!(result[&(b, d)], 3);
        // c->d = 1
        assert_eq!(result[&(c, d)], 1);
        // No path from d to others
        assert!(!result.contains_key(&(d, a)));
    }

    #[test]
    fn floyd_warshall_negative_cycle() {
        unsafe {
            let mut g = VecGraph::<&str, i64>::default();
            let a = InsertNode::insert_node_unchecked(&mut g, "A").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "B").unwrap();
            let c = InsertNode::insert_node_unchecked(&mut g, "C").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 1i64, [a, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, -1, [b, c]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, -1, [c, a]).unwrap();
            let result = floyd_warshall(g.unsafe_assert_stable_node(), |&w| w);
            assert!(result.is_err());
        }
    }

    #[test]
    fn k_shortest_paths_basic() {
        // Graph with two paths: A->B->D (cost 6) and A->C->D (cost 5)
        unsafe {
            let mut g = VecGraph::<&str, u32>::default();
            let a = InsertNode::insert_node_unchecked(&mut g, "A").unwrap();
            let b = InsertNode::insert_node_unchecked(&mut g, "B").unwrap();
            let c = InsertNode::insert_node_unchecked(&mut g, "C").unwrap();
            let d = InsertNode::insert_node_unchecked(&mut g, "D").unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 1u32, [a, b]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 5, [b, d]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 2, [a, c]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, 3, [c, d]).unwrap();

            let paths = k_shortest_paths_unchecked(g.unsafe_assert_stable_node(), a, d, 3, |&w| w).finish();
            assert_eq!(paths.len(), 2); // Only 2 paths exist
            assert_eq!(paths[0].0, 5); // A->C->D = 2+3
            assert_eq!(paths[1].0, 6); // A->B->D = 1+5
        }
    }

    #[test]
    fn k_shortest_paths_single() {
        let (g, [a, _, _, d]) = weighted_vecgraph();
        let paths = unsafe { k_shortest_paths_unchecked(g.unsafe_assert_stable_node(), a, d, 1, |&w| w) }.finish();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].0, 4); // Shortest: A->B->C->D = 1+2+1
    }
}
