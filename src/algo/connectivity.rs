//! # Connectivity
//!
//! Algorithms for analyzing how nodes in a graph are connected: strongly
//! connected components (SCCs), cycle detection, weakly connected components,
//! reachability queries, and condensation into a DAG.
//!
//! ## Components
//!
//! - [`TarjanScc`] -- lazy iterator that yields strongly connected components
//!   using Tarjan's algorithm. SCCs are emitted in reverse topological order.
//!   - [`tarjan_scc`] -- constructor.
//! - [`KosarajuScc`] -- lazy iterator that yields SCCs using Kosaraju's
//!   two-pass algorithm (forward DFS + reverse-graph DFS).
//!   - [`kosaraju_scc`] -- constructor.
//! - [`is_cyclic_directed`] -- returns `true` if the directed graph contains at
//!   least one cycle (based on topological sort failure).
//! - [`connected_components`] -- counts the number of weakly connected
//!   components (treats all edges as undirected).
//! - [`has_path_connecting`] -- returns `true` if there is a directed path from
//!   `source` to `target` (BFS-based reachability).
//! - [`condensation`] -- contracts every SCC into a single node, producing a
//!   DAG of type `VecGraph<Vec<G::NodeIx>, ()>`.
//!
//! ## Algorithm
//!
//! ```text
//!   Tarjan's SCC (example)
//!
//!       0 --> 1 --> 2       DFS from 0:
//!       ^         /         index/lowlink assignments:
//!        \       v            0: idx=0, low=0
//!         +---- 3             1: idx=1, low=0  (back-edge to 0)
//!                             2: idx=2, low=0
//!       4 (isolated)          3: idx=3, low=0
//!                           On finishing 0: lowlink == index -> pop SCC {0,1,2,3}
//!                           Node 4: singleton SCC {4}
//!
//!   Result (reverse topo order): [{0,1,2,3}, {4}]
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::connectivity::{tarjan_scc, is_cyclic_directed};
//!
//! let mut g = BTreeGraph::<u32, &str>::default();
//! for n in 0..3 { g.insert_node(n).unwrap(); }
//! g.insert_edge("a", [0, 1]).unwrap();
//! g.insert_edge("b", [1, 2]).unwrap();
//! g.insert_edge("c", [2, 0]).unwrap(); // creates cycle 0->1->2->0
//!
//! assert!(is_cyclic_directed(&g));
//!
//! let sccs: Vec<_> = tarjan_scc(&g).collect();
//! // single SCC containing all three nodes
//! assert_eq!(sccs.len(), 1);
//! assert_eq!(sccs[0].len(), 3);
//! ```

use std::collections::{HashMap, HashSet};

use super::bfs::Bfs;
use crate::graph::capability::{Bigraph, Directed, InsertEdge, InsertNode, StableNode};
use crate::graph::Graph;
use crate::VecGraph;

/// Iterator that lazily yields strongly connected components using Tarjan's algorithm.
///
/// Each call to `next()` yields one SCC as a `Vec<G::NodeIx>`.
/// SCCs are yielded in reverse topological order.
pub struct TarjanScc<'r, G: ?Sized, N, Ns> {
    graph: &'r G,
    index_counter: usize,
    scc_stack: Vec<N>,
    on_stack: HashSet<N>,
    index: HashMap<N, usize>,
    lowlink: HashMap<N, usize>,
    nodes: Ns,
    dfs_stack: Vec<(N, Vec<N>, usize)>,
}

/// Compute strongly connected components using Tarjan's algorithm.
///
/// Returns an iterator yielding SCCs in reverse topological order.
pub fn tarjan_scc<'r, G>(
    graph: &'r G,
) -> TarjanScc<'r, G, G::NodeIx, <G as crate::graph::GraphOperation<'r>>::NodeIndices>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    TarjanScc {
        graph,
        index_counter: 0,
        scc_stack: Vec::new(),
        on_stack: HashSet::new(),
        index: HashMap::new(),
        lowlink: HashMap::new(),
        nodes: <_ as crate::graph::GraphOperation<'_>>::node_indices(graph),
        dfs_stack: Vec::new(),
    }
}

impl<'r, G> Iterator
    for TarjanScc<'r, G, G::NodeIx, <G as crate::graph::GraphOperation<'r>>::NodeIndices>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    type Item = Vec<G::NodeIx>;

    fn next(&mut self) -> Option<Vec<G::NodeIx>> {
        loop {
            if let Some((current, ref succs, ref mut idx)) = self.dfs_stack.last_mut() {
                if *idx < succs.len() {
                    let succ = succs[*idx];
                    *idx += 1;
                    let current = *current;

                    if !self.index.contains_key(&succ) {
                        self.index.insert(succ, self.index_counter);
                        self.lowlink.insert(succ, self.index_counter);
                        self.index_counter += 1;
                        self.scc_stack.push(succ);
                        self.on_stack.insert(succ);

                        let succ_succs: Vec<G::NodeIx> =
                            unsafe { self.graph.neighbor_indices_from_unchecked(succ) }.collect();
                        self.dfs_stack.push((succ, succ_succs, 0));
                    } else if self.on_stack.contains(&succ) {
                        let succ_index = self.index[&succ];
                        let ll = self.lowlink.get_mut(&current).unwrap();
                        if succ_index < *ll {
                            *ll = succ_index;
                        }
                    }
                } else {
                    let current = *current;

                    if self.dfs_stack.len() > 1 {
                        let parent = self.dfs_stack[self.dfs_stack.len() - 2].0;
                        let current_ll = self.lowlink[&current];
                        let parent_ll = self.lowlink.get_mut(&parent).unwrap();
                        if current_ll < *parent_ll {
                            *parent_ll = current_ll;
                        }
                    }

                    let is_root = self.lowlink[&current] == self.index[&current];
                    self.dfs_stack.pop();

                    if is_root {
                        let mut scc = Vec::new();
                        loop {
                            let w = self.scc_stack.pop().unwrap();
                            self.on_stack.remove(&w);
                            scc.push(w);
                            if w == current {
                                break;
                            }
                        }
                        return Some(scc);
                    }
                }
            } else {
                // Find next unvisited node
                loop {
                    let node = self.nodes.next()?;
                    if !self.index.contains_key(&node) {
                        self.index.insert(node, self.index_counter);
                        self.lowlink.insert(node, self.index_counter);
                        self.index_counter += 1;
                        self.scc_stack.push(node);
                        self.on_stack.insert(node);

                        let succs: Vec<G::NodeIx> =
                            unsafe { self.graph.neighbor_indices_from_unchecked(node) }.collect();
                        self.dfs_stack.push((node, succs, 0));
                        break;
                    }
                }
            }
        }
    }
}

/// Iterator that lazily yields strongly connected components using Kosaraju's algorithm.
///
/// Phase 1 (finish order) is computed eagerly in the constructor.
/// Phase 2 (reverse DFS for SCCs) yields one SCC per `next()` call.
pub struct KosarajuScc<'r, G: ?Sized, N> {
    graph: &'r G,
    finish_order: Vec<N>,
    finish_idx: usize,
    assigned: HashSet<N>,
}

/// Compute strongly connected components using Kosaraju's algorithm.
///
/// Returns an iterator yielding SCCs.
pub fn kosaraju_scc<'r, G>(graph: &'r G) -> KosarajuScc<'r, G, G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    // Phase 1: Compute finish order via DFS (eagerly)
    let mut visited = HashSet::new();
    let mut finish_order = Vec::new();

    for node in <_ as crate::graph::GraphOperation<'_>>::node_indices(graph) {
        if visited.contains(&node) {
            continue;
        }
        let mut stack: Vec<(G::NodeIx, bool)> = vec![(node, false)];
        visited.insert(node);

        while let Some((current, expanded)) = stack.last_mut() {
            if *expanded {
                finish_order.push(*current);
                stack.pop();
            } else {
                *expanded = true;
                let current = *current;
                // SAFETY: `current` is an in-graph index and `G: StableNode`.
                let succs: Vec<G::NodeIx> =
                    unsafe { graph.neighbor_indices_from_unchecked(current) }.collect();
                for succ in succs.into_iter().rev() {
                    if visited.insert(succ) {
                        stack.push((succ, false));
                    }
                }
            }
        }
    }

    // Reverse for phase 2 iteration
    finish_order.reverse();

    KosarajuScc {
        graph,
        finish_order,
        finish_idx: 0,
        assigned: HashSet::new(),
    }
}

impl<'r, G> Iterator for KosarajuScc<'r, G, G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    type Item = Vec<G::NodeIx>;

    fn next(&mut self) -> Option<Vec<G::NodeIx>> {
        // Phase 2: find next unassigned node in reverse finish order
        while self.finish_idx < self.finish_order.len() {
            let node = self.finish_order[self.finish_idx];
            self.finish_idx += 1;

            if self.assigned.contains(&node) {
                continue;
            }

            // DFS on reverse graph using predecessors
            let mut scc = Vec::new();
            let mut stack = vec![node];
            self.assigned.insert(node);

            while let Some(current) = stack.pop() {
                scc.push(current);
                let preds: Vec<G::NodeIx> =
                    unsafe { self.graph.neighbor_indices_to_unchecked(current) }.collect();
                for pred in preds {
                    if self.assigned.insert(pred) {
                        stack.push(pred);
                    }
                }
            }

            return Some(scc);
        }

        None
    }
}

/// Returns `true` if the directed graph contains a cycle.
pub fn is_cyclic_directed<G>(graph: &G) -> bool
where
    G: Graph + for<'a> Directed<'a> + ?Sized,
    <G as crate::graph::GraphProperty>::Endpoints: for<'scope> crate::graph::edge::Map<
        crate::graph::context::NodeIx<'scope, <G as crate::graph::GraphProperty>::NodeIx>,
    >,
{
    // Fully safe: `scope` hands `toposort` a `Context` whose branded node indices
    // cannot escape the closure (so they can never be used after a mutation);
    // only the `Ok`/`Err` outcome is returned.
    graph.scope(|ctx| super::toposort::toposort(ctx).is_err())
}

/// Connected components
pub fn connected_components<G>(graph: &G) -> usize
where
    G: Graph + ?Sized,
{
    let mut visited = HashSet::new();
    let mut count = 0;

    // SAFETY: node indices are not exposed to the caller and the shared borrow
    // rules out mutation. Uses only the bound-free raw `GraphOperation`
    // primitives (`node_indices` / `walks_of_unchecked`), so it stays
    // available on any graph without a `StableNode` bound.
    for node in <_ as crate::graph::GraphOperation<'_>>::node_indices(graph) {
        if !visited.insert(node) {
            continue;
        }
        count += 1;
        // DFS through incident (undirected) neighbors
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            // SAFETY: `n` is an in-graph index (from the loop / prior neighbors).
            for wi in
                unsafe { <G as crate::graph::GraphOperation<'_>>::walks_of_unchecked(graph, n) }
            {
                // Only the neighbor index is needed; take it from the raw parts
                // (no edge deref, so no `G::Edge: '_` bound).
                let neighbor = wi.into_parts().2;
                if visited.insert(neighbor) {
                    stack.push(neighbor);
                }
            }
        }
    }

    count
}

/// Returns `true` if there is a directed path from `source` to `target`.
pub fn has_path_connecting<'r, G>(graph: &'r G, source: G::NodeIx, target: G::NodeIx) -> bool
where
    G: Graph + Directed<'r> + ?Sized,
    <G as crate::graph::GraphProperty>::Endpoints: for<'scope> crate::graph::edge::Map<
        crate::graph::context::NodeIx<'scope, <G as crate::graph::GraphProperty>::NodeIx>,
        Mapped = [crate::graph::context::NodeIx<'scope, <G as crate::graph::GraphProperty>::NodeIx>;
                     2],
    >,
{
    // `Bfs::new` validates `source`; `graph.node` validates `target`.
    assert!(Graph::contains_node_index(graph, target));
    if source == target {
        return true;
    }
    unsafe { has_path_connecting_unchecked(graph, source, target) }
}

/// Returns `true` if there is a directed path from `source` to `target`.
///
/// # Safety
/// `source` and `target` must be valid node indices for `graph`, and the graph
/// must not be modified while traversal is running.
pub unsafe fn has_path_connecting_unchecked<'r, G>(
    graph: &'r G,
    source: G::NodeIx,
    target: G::NodeIx,
) -> bool
where
    G: Graph + Directed<'r> + ?Sized,
    <G as crate::graph::GraphProperty>::Endpoints: for<'scope> crate::graph::edge::Map<
        crate::graph::context::NodeIx<'scope, <G as crate::graph::GraphProperty>::NodeIx>,
        Mapped = [crate::graph::context::NodeIx<'scope, <G as crate::graph::GraphProperty>::NodeIx>;
                     2],
    >,
{
    // Run the search inside a `scope`: the `Context` is genuinely `StableNode`
    // (no `unsafe_assert_stable_node` needed), and `wrap_node` brands the
    // caller-supplied raw indices so the BFS can start from `source` and compare
    // against `target`.
    graph.scope(|ctx| {
        let source = ctx.wrap_node(source);
        let target = ctx.wrap_node(target);
        // SAFETY: the caller guarantees `source` is a valid index, so skipping the
        // contains-check in `new_unchecked` is sound.
        unsafe { Bfs::new_unchecked(ctx, source) }.any(|n| n == target)
    })
}

/// Condenses strongly connected components into single nodes, creating a DAG.
///
/// Returns a new `VecGraph` where each node contains the members of the
/// original SCC (`Vec<G::NodeIx>`), and edges represent inter-SCC connections.
/// If `make_acyclic` is true, self-loops in the condensed graph are omitted.
pub fn condensation<'r, G>(graph: &'r G, make_acyclic: bool) -> VecGraph<Vec<G::NodeIx>, ()>
where
    G: Graph + Directed<'r> + Bigraph + StableNode + ?Sized,
{
    let sccs: Vec<Vec<G::NodeIx>> = tarjan_scc(graph).collect();

    // Map original node -> SCC index
    let mut node_to_scc: HashMap<G::NodeIx, usize> = HashMap::new();
    for (scc_idx, scc) in sccs.iter().enumerate() {
        for &node in scc {
            node_to_scc.insert(node, scc_idx);
        }
    }

    let mut condensed = VecGraph::<Vec<G::NodeIx>, ()>::default();

    // Insert SCC nodes.
    let mut scc_node_ids: Vec<u32> = Vec::new();
    for scc in &sccs {
        // SAFETY: `condensed` is a fresh append-only `VecGraph` (not `StableNode`),
        // so the returned index stays valid for the duration of this build.
        let nix =
            unsafe { InsertNode::insert_node_unchecked(&mut condensed, scc.clone()) }.unwrap();
        scc_node_ids.push(nix);
    }

    // Insert inter-SCC edges. Endpoints are read from the input graph via the
    // safe `edge_tail_index` / `edge_head_index` (panic on invalid index).
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
    // SAFETY: edge indices are not exposed to caller
    for eix in <_ as crate::graph::GraphOperation<'_>>::edge_indices(graph) {
        let tail = graph.edge_tail_index(eix);
        let head = graph.edge_head_index(eix);
        let scc_tail = node_to_scc[&tail];
        let scc_head = node_to_scc[&head];

        if make_acyclic && scc_tail == scc_head {
            continue;
        }
        if seen_edges.insert((scc_tail, scc_head)) {
            // SAFETY: `scc_node_ids` were just produced by inserting into this
            // same fresh graph and remain valid (no removals happen here).
            unsafe {
                InsertEdge::insert_edge_unchecked(
                    &mut condensed,
                    (),
                    [scc_node_ids[scc_tail], scc_node_ids[scc_head]],
                )
            }
            .ok();
        }
    }

    condensed
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

    fn cycle_graph() -> BTreeGraph<u32, &'static str> {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();
        g
    }

    fn two_scc_graph() -> BTreeGraph<u32, &'static str> {
        // SCC1: {0, 1, 2} (cycle), SCC2: {3, 4} (cycle)
        // Edge from SCC1 to SCC2: 2->3
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_node(4).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();
        g.insert_edge("3->4", [3, 4]).unwrap();
        g.insert_edge("4->3", [4, 3]).unwrap();
        g
    }

    #[test]
    fn tarjan_diamond_no_cycles() {
        let g = diamond_btree();
        let sccs: Vec<_> = tarjan_scc(&g).collect();
        // Each node is its own SCC (DAG)
        assert_eq!(sccs.len(), 4);
        for scc in &sccs {
            assert_eq!(scc.len(), 1);
        }
    }

    #[test]
    fn tarjan_cycle() {
        let g = cycle_graph();
        let sccs: Vec<_> = tarjan_scc(&g).collect();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn tarjan_two_sccs() {
        let g = two_scc_graph();
        let sccs: Vec<_> = tarjan_scc(&g).collect();
        assert_eq!(sccs.len(), 2);
        let sizes: HashSet<usize> = sccs.iter().map(|s| s.len()).collect();
        assert!(sizes.contains(&3));
        assert!(sizes.contains(&2));
    }

    #[test]
    fn kosaraju_cycle() {
        let g = cycle_graph();
        let sccs: Vec<_> = kosaraju_scc(&g).collect();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn kosaraju_two_sccs() {
        let g = two_scc_graph();
        let sccs: Vec<_> = kosaraju_scc(&g).collect();
        assert_eq!(sccs.len(), 2);
        let sizes: HashSet<usize> = sccs.iter().map(|s| s.len()).collect();
        assert!(sizes.contains(&3));
        assert!(sizes.contains(&2));
    }

    #[test]
    fn is_cyclic_yes() {
        let g = cycle_graph();
        assert!(is_cyclic_directed(&g));
    }

    #[test]
    fn is_cyclic_no() {
        let g = diamond_btree();
        assert!(!is_cyclic_directed(&g));
    }

    #[test]
    fn connected_components_connected() {
        let g = diamond_btree();
        assert_eq!(connected_components(&g), 1);
    }

    #[test]
    fn connected_components_disconnected() {
        let mut g = BTreeGraph::<u32, &str>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        // Node 2 is disconnected
        assert_eq!(connected_components(&g), 2);
    }

    #[test]
    fn has_path_yes() {
        let g = diamond_btree();
        assert!(has_path_connecting(&g, 0, 3));
    }

    #[test]
    fn has_path_no() {
        let g = diamond_btree();
        assert!(!has_path_connecting(&g, 3, 0));
    }

    #[test]
    fn has_path_self() {
        let g = diamond_btree();
        assert!(has_path_connecting(&g, 0, 0));
    }

    #[test]
    fn condensation_dag() {
        let g = diamond_btree();
        let condensed = condensation(&g, true);
        // Each node is its own SCC, so condensed has 4 nodes
        let nodes: Vec<_> = condensed.nodes().collect();
        assert_eq!(nodes.len(), 4);
    }

    #[test]
    fn condensation_two_sccs() {
        let g = two_scc_graph();
        let condensed = condensation(&g, true);
        let nodes: Vec<_> = condensed.nodes().collect();
        assert_eq!(nodes.len(), 2);
        // Should have exactly 1 edge between SCCs
        let edges: Vec<_> = condensed.edges().collect();
        assert_eq!(edges.len(), 1);
    }
}
