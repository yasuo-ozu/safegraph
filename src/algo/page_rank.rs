//! # PageRank
//!
//! Computes PageRank scores for every node using the power iteration method.
//! PageRank models a "random surfer" who follows outgoing edges with
//! probability equal to the damping factor and jumps to a random node
//! otherwise. Dangling nodes (no outgoing edges) distribute their rank
//! uniformly to all nodes.
//!
//! ## Components
//!
//! - [`page_rank`] -- safe version (requires `StableNode`), returns `HashMap<NodeIx, f64>`
//!
//! ## Algorithm
//!
//! ```text
//!   Given N nodes, damping factor d (typically 0.85):
//!
//!   1. Initialize rank(v) = 1/N for all v.
//!   2. Repeat until convergence or max iterations:
//!        dangling_sum = sum of rank(v) for all dangling nodes v
//!        For each node v:
//!          rank'(v) = (1-d)/N
//!                   + d * sum( rank(u)/out_degree(u) for u in predecessors(v) )
//!                   + d * dangling_sum / N
//!   3. Stop when max|rank'(v) - rank(v)| < tolerance.
//!
//!   Example cycle: 0 -> 1 -> 2 -> 0
//!   All nodes converge to rank = 1/3.
//! ```
//!
//! ## Example
//!
//! ```rust
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::page_rank::page_rank;
//!
//! let mut g = BTreeGraph::<u32, &str>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_edge("a", [0, 1]).unwrap();
//! g.insert_edge("b", [1, 2]).unwrap();
//! g.insert_edge("c", [2, 0]).unwrap();
//!
//! let ranks = page_rank(&g, 0.85, 100, 1e-6);
//! // All three nodes will have approximately equal rank (~0.333).
//! ```

use std::collections::HashMap;

use crate::graph::capability::{Directed, StableNode};
use crate::graph::Graph;

/// Compute PageRank scores for all nodes using the power iteration method.
///
/// - `damping_factor`: probability of following an edge (typically 0.85)
/// - `max_iterations`: maximum number of iterations
/// - `tolerance`: convergence threshold (stop when max change < tolerance)
///
/// Returns a map from node index to PageRank score.
pub fn page_rank<'r, G>(
    graph: &'r G,
    damping_factor: f64,
    max_iterations: usize,
    tolerance: f64,
) -> HashMap<G::NodeIx, f64>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    let nodes: Vec<G::NodeIx> =
        <_ as crate::graph::GraphOperation<'_>>::node_indices(graph).collect();
    let n = nodes.len();

    if n == 0 {
        return HashMap::new();
    }

    let initial_rank = 1.0 / n as f64;

    // Current ranks
    let mut rank: HashMap<G::NodeIx, f64> =
        nodes.iter().map(|&node| (node, initial_rank)).collect();

    // Precompute out-degrees
    let mut out_degree: HashMap<G::NodeIx, usize> = HashMap::new();
    for &node in &nodes {
        let deg = unsafe { graph.neighbor_indices_from_unchecked(node) }.count();
        out_degree.insert(node, deg);
    }

    // Precompute predecessors for each node
    let mut predecessors: HashMap<G::NodeIx, Vec<G::NodeIx>> = HashMap::new();
    for &node in &nodes {
        predecessors.insert(node, Vec::new());
    }
    for &node in &nodes {
        for succ in unsafe { graph.neighbor_indices_from_unchecked(node) } {
            predecessors.get_mut(&succ).unwrap().push(node);
        }
    }

    for _ in 0..max_iterations {
        let mut new_rank: HashMap<G::NodeIx, f64> = HashMap::new();

        // Collect dangling node rank sum (nodes with no outgoing edges)
        let dangling_sum: f64 = nodes
            .iter()
            .filter(|&&node| out_degree[&node] == 0)
            .map(|&node| rank[&node])
            .sum();

        for &node in &nodes {
            let mut incoming_rank = 0.0;
            for &pred in &predecessors[&node] {
                let pred_out = out_degree[&pred];
                if pred_out > 0 {
                    incoming_rank += rank[&pred] / pred_out as f64;
                }
            }

            let pr = (1.0 - damping_factor) / n as f64
                + damping_factor * (incoming_rank + dangling_sum / n as f64);
            new_rank.insert(node, pr);
        }

        // Check convergence
        let max_diff = nodes
            .iter()
            .map(|&node| (new_rank[&node] - rank[&node]).abs())
            .fold(0.0_f64, f64::max);

        rank = new_rank;

        if max_diff < tolerance {
            break;
        }
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::BTreeGraph;

    #[test]
    fn page_rank_uniform() {
        // Complete graph: all nodes should have roughly equal rank
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();
        g.insert_edge("2->0", [2, 0]).unwrap();

        let pr = page_rank(&g, 0.85, 100, 1e-6);
        assert_eq!(pr.len(), 3);
        // In a symmetric cycle, all ranks should be approximately equal
        let r0 = pr[&0];
        let r1 = pr[&1];
        let r2 = pr[&2];
        assert!((r0 - r1).abs() < 0.01);
        assert!((r1 - r2).abs() < 0.01);
        // Sum should be approximately 1.0
        assert!(((r0 + r1 + r2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn page_rank_star() {
        // Star graph: 0->1, 0->2, 0->3
        // Node 0 has no incoming, so it should have lowest rank
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("0->3", [0, 3]).unwrap();

        let pr = page_rank(&g, 0.85, 100, 1e-6);
        assert_eq!(pr.len(), 4);
        // Nodes 1, 2, 3 are dangling (no outgoing) and receive rank from node 0
        // They should have roughly equal rank, higher than node 0's rank from
        // only the base (1-d)/n contribution
        assert!(pr[&0] < pr[&1]);
    }

    #[test]
    fn page_rank_empty() {
        let g = BTreeGraph::<u32, &str>::default();
        let pr = page_rank(&g, 0.85, 100, 1e-6);
        assert!(pr.is_empty());
    }

    #[test]
    fn page_rank_single_node() {
        let mut g = BTreeGraph::<u32, &str>::default();
        g.insert_node(0).unwrap();
        let pr = page_rank(&g, 0.85, 100, 1e-6);
        assert!((pr[&0] - 1.0).abs() < 0.01);
    }
}
