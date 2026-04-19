//! # Topological Sort
//!
//! Linear ordering of nodes in a directed acyclic graph (DAG) such that for
//! every directed edge u -> v, node u appears before v in the ordering. If
//! the graph contains a cycle, the sort fails with [`CycleError`].
//!
//! Two algorithms are provided: a DFS-based approach (iterative post-order)
//! and Kahn's algorithm (in-degree counting with a queue).
//!
//! ## Components
//!
//! - [`toposort`] -- DFS-based topological sort (safe, requires `StableNode` + `Directed`).
//! - [`reverse_toposort`] -- DFS post-order
//!   (reverse topological order: dependencies come *after* dependents).
//! - [`toposort_kahn`] -- Kahn's algorithm using
//!   in-degree counting and a queue.
//! - [`CycleError`] -- error type returned when a cycle is detected.
//!
//! ## Algorithm
//!
//! ```text
//!   DFS-based toposort            Kahn's algorithm
//!
//!       0                         in-degree: {0:0, 1:1, 2:1}
//!       |                         queue: [0]       (in-degree 0)
//!       v
//!       1                         dequeue 0 -> output 0
//!       |                           decrement 1's in-degree -> 0, enqueue 1
//!       v                         dequeue 1 -> output 1
//!       2                           decrement 2's in-degree -> 0, enqueue 2
//!                                 dequeue 2 -> output 2
//!   DFS post-order: [2,1,0]
//!   reverse -> [0,1,2]            result: [0, 1, 2]
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::toposort::toposort;
//!
//! let mut g = BTreeGraph::<u32, &str>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_edge("a", [0, 1]).unwrap();
//! g.insert_edge("b", [1, 2]).unwrap();
//!
//! let order = toposort(&g).unwrap();
//! assert_eq!(order, vec![0, 1, 2]);
//! ```

use std::collections::{HashMap, VecDeque};

use crate::graph::capability::{Directed, StableNode};
use crate::graph::Graph;

/// Error returned when a cycle is detected during topological sorting.
#[derive(Debug, Clone)]
pub struct CycleError<N> {
    /// A node that is part of a cycle.
    pub node: N,
}

impl<N: std::fmt::Debug> std::fmt::Display for CycleError<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cycle detected at node {:?}", self.node)
    }
}

impl<N: std::fmt::Debug> std::error::Error for CycleError<N> {}

/// Returns a topological ordering of nodes using iterative DFS.
///
/// Returns `Err(CycleError)` if the graph contains a cycle.
pub fn toposort<'r, G>(graph: &'r G) -> Result<Vec<G::NodeIx>, CycleError<G::NodeIx>>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    let mut order = reverse_toposort(graph)?;
    order.reverse();
    Ok(order)
}

/// Returns a reverse topological ordering of nodes (DFS post-order).
///
/// In this ordering, each node appears *before* all of its predecessors
/// (i.e., dependencies come *after* the nodes that depend on them).
///
/// Returns `Err(CycleError)` if the graph contains a cycle.
pub fn reverse_toposort<'r, G>(graph: &'r G) -> Result<Vec<G::NodeIx>, CycleError<G::NodeIx>>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    // States: 0 = unvisited, 1 = on stack (in progress), 2 = finished
    let mut state: HashMap<G::NodeIx, u8> = HashMap::new();
    let mut order = Vec::new();

    for node in <_ as crate::graph::GraphOperation<'_>>::node_indices(graph) {
        if state.get(&node).copied().unwrap_or(0) != 0 {
            continue;
        }
        let succs: Vec<G::NodeIx> =
            unsafe { graph.neighbor_indices_from_unchecked(node) }.collect();
        let mut stack: Vec<(G::NodeIx, Vec<G::NodeIx>, usize)> = vec![(node, succs, 0)];
        state.insert(node, 1);

        while let Some((current, ref succs, ref mut idx)) = stack.last_mut() {
            if *idx < succs.len() {
                let succ = succs[*idx];
                *idx += 1;
                match state.get(&succ).copied().unwrap_or(0) {
                    0 => {
                        state.insert(succ, 1);
                        let succ_succs: Vec<G::NodeIx> =
                            unsafe { graph.neighbor_indices_from_unchecked(succ) }.collect();
                        stack.push((succ, succ_succs, 0));
                    }
                    1 => {
                        return Err(CycleError { node: succ });
                    }
                    _ => {} // already finished
                }
            } else {
                let current = *current;
                state.insert(current, 2);
                order.push(current);
                stack.pop();
            }
        }
    }

    Ok(order)
}

/// Returns a topological ordering using Kahn's algorithm (in-degree based).
///
/// Returns `Err(CycleError)` if the graph contains a cycle.
pub fn toposort_kahn<'r, G>(graph: &'r G) -> Result<Vec<G::NodeIx>, CycleError<G::NodeIx>>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    let mut in_degree: HashMap<G::NodeIx, usize> = HashMap::new();

    for node in <_ as crate::graph::GraphOperation<'_>>::node_indices(graph) {
        in_degree.entry(node).or_insert(0);
        for succ in unsafe { graph.neighbor_indices_from_unchecked(node) } {
            *in_degree.entry(succ).or_insert(0) += 1;
        }
    }

    let mut queue: VecDeque<G::NodeIx> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&n, _)| n)
        .collect();

    let mut order = Vec::new();
    let total = in_degree.len();

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for succ in unsafe { graph.neighbor_indices_from_unchecked(node) } {
            let deg = in_degree.get_mut(&succ).unwrap();
            *deg -= 1;
            if *deg == 0 {
                queue.push_back(succ);
            }
        }
    }

    if order.len() == total {
        Ok(order)
    } else {
        let cycle_node = in_degree
            .iter()
            .find(|(_, &deg)| deg > 0)
            .map(|(&n, _)| n)
            .unwrap();
        Err(CycleError { node: cycle_node })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
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

    fn is_valid_toposort(order: &[u32], edges: &[(u32, u32)]) -> bool {
        let pos: HashMap<u32, usize> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        for &(from, to) in edges {
            if pos[&from] >= pos[&to] {
                return false;
            }
        }
        true
    }

    #[test]
    fn toposort_diamond() {
        let g = diamond_btree();
        let order = toposort(&g).unwrap();
        assert_eq!(order.len(), 4);
        let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];
        assert!(is_valid_toposort(&order, &edges));
    }

    #[test]
    fn toposort_kahn_diamond() {
        let g = diamond_btree();
        let order = toposort_kahn(&g).unwrap();
        assert_eq!(order.len(), 4);
        let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];
        assert!(is_valid_toposort(&order, &edges));
    }

    #[test]
    fn toposort_cycle() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();
        assert!(toposort(&g).is_err());
        assert!(toposort_kahn(&g).is_err());
    }

    #[test]
    fn toposort_single_node() {
        let mut g = BTreeGraph::<u32, &str>::default();
        g.insert_node(42).unwrap();
        let order = toposort(&g).unwrap();
        assert_eq!(order, vec![42]);
    }

    #[test]
    fn toposort_linear() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        let order = toposort(&g).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn reverse_toposort_diamond() {
        let g = diamond_btree();
        let order = reverse_toposort(&g).unwrap();
        assert_eq!(order.len(), 4);
        // Reverse toposort: each node appears before its predecessors
        // i.e., 3 should come before 1 and 2, and 1,2 before 0
        let pos: HashMap<u32, usize> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];
        for &(from, to) in &edges {
            assert!(
                pos[&from] > pos[&to],
                "expected {} after {} in reverse toposort",
                from,
                to
            );
        }
    }

    #[test]
    fn reverse_toposort_cycle() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->0", [1, 0]).unwrap();
        assert!(reverse_toposort(&g).is_err());
    }

    #[test]
    fn reverse_toposort_linear() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        let order = reverse_toposort(&g).unwrap();
        assert_eq!(order, vec![2, 1, 0]);
    }
}
