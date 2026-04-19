//! # Dominators
//!
//! Computes the immediate dominator of every node reachable from a given start
//! node using the Cooper-Harvey-Kennedy iterative algorithm. A node `d`
//! *dominates* node `n` if every path from the start to `n` must pass through
//! `d`. The *immediate* dominator is the closest strict dominator.
//!
//! ## Components
//!
//! - [`Dominators`] -- iterator that lazily yields `(node, immediate_dominator)` pairs
//! - [`dominators`] -- safe constructor (requires `StableNode`)
//! - [`dominators_unchecked`] -- `unsafe` variant that skips the `start` validation (still requires `StableNode`)
//!
//! ## Algorithm
//!
//! ```text
//!   1. Compute a reverse post-order (RPO) of the graph via DFS from start.
//!   2. Initialize each node's dominator to itself (undefined).
//!   3. Iterate over nodes in RPO:
//!        idom(n) = intersect of idom(p) for every predecessor p already processed
//!      where "intersect" walks both fingers up the dominator tree by RPO number.
//!   4. Repeat until no dominator changes (fixpoint).
//!
//!         start
//!          / \
//!         v   v
//!         1   2       idom(1)=start, idom(2)=start
//!          \ /
//!           v
//!           3         idom(3)=start (intersection of 1 and 2)
//! ```
//!
//! ## Example
//!
//! ```rust
//! use safegraph::BTreeGraph;
//! use safegraph::graph::{Graph, GraphOperation};
//! use safegraph::algo::dominators::dominators;
//!
//! let mut g = BTreeGraph::<u32, &str>::default();
//! g.insert_node(0).unwrap();
//! g.insert_node(1).unwrap();
//! g.insert_node(2).unwrap();
//! g.insert_node(3).unwrap();
//! g.insert_edge("e0", [0, 1]).unwrap();
//! g.insert_edge("e1", [0, 2]).unwrap();
//! g.insert_edge("e2", [1, 3]).unwrap();
//! g.insert_edge("e3", [2, 3]).unwrap();
//!
//! for (node, idom) in dominators(&g, 0) {
//!     // e.g. (3, 0) meaning node 0 immediately dominates node 3
//! }
//! ```

use std::collections::HashMap;

use crate::graph::capability::{Directed, StableNode};
use crate::graph::Graph;

/// Iterator that lazily yields `(node, immediate_dominator)` pairs computed using the
/// Cooper-Harvey-Kennedy iterative algorithm.
///
/// The algorithm is computed eagerly in the constructor (it requires fixpoint iteration);
/// pairs are yielded lazily from the result.
pub struct Dominators<N> {
    pairs: Vec<(N, N)>,
    idx: usize,
}

/// Compute immediate dominators using the Cooper-Harvey-Kennedy iterative algorithm.
///
/// Returns an iterator yielding `(node, immediate_dominator)` pairs.
/// The `start` node dominates itself (mapped to itself).
pub fn dominators<'r, G>(graph: &'r G, start: G::NodeIx) -> Dominators<G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    // Validate the user-supplied start index (panics if invalid).
    let _ = graph.node(start);
    // SAFETY: `StableNode` guarantees the node indices stay valid for the call,
    // and `start` was validated above.
    unsafe { dominators_unchecked(graph, start) }
}

/// Immediate dominators iterator that skips the explicit `start` validation.
///
/// # Safety
/// `start` must be a valid node index for `graph`, and the graph must not be
/// modified until the resulting NodeIx is accessed.
pub unsafe fn dominators_unchecked<'r, G>(graph: &'r G, start: G::NodeIx) -> Dominators<G::NodeIx>
where
    G: Graph + Directed<'r> + StableNode + ?Sized,
{
    // Step 1: Compute reverse post-order via DFS from `start`.
    let mut visited = std::collections::HashSet::new();
    let mut rpo = Vec::new();

    let mut stack: Vec<(G::NodeIx, bool)> = vec![(start, false)];
    visited.insert(start);

    while let Some((node, expanded)) = stack.last_mut() {
        if *expanded {
            rpo.push(*node);
            stack.pop();
        } else {
            *expanded = true;
            let node = *node;
            // SAFETY: `node` is reachable from the caller-validated `start`.
            let succs: Vec<G::NodeIx> =
                unsafe { graph.neighbor_indices_from_unchecked(node) }.collect();
            for succ in succs.into_iter().rev() {
                if visited.insert(succ) {
                    stack.push((succ, false));
                }
            }
        }
    }

    rpo.reverse();

    if rpo.is_empty() {
        return Dominators {
            pairs: Vec::new(),
            idx: 0,
        };
    }

    let rpo_index: HashMap<G::NodeIx, usize> =
        rpo.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let n = rpo.len();
    let mut idom: Vec<Option<usize>> = vec![None; n];
    idom[0] = Some(0);

    // Step 2: Iterative dominator computation.
    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..n {
            let node = rpo[i];
            // SAFETY: `node` came from the RPO walk, so it is a valid index.
            let preds: Vec<G::NodeIx> =
                unsafe { graph.neighbor_indices_to_unchecked(node) }.collect();

            let mut new_idom: Option<usize> = None;

            for pred in preds {
                if let Some(&pred_rpo) = rpo_index.get(&pred) {
                    if idom[pred_rpo].is_some() {
                        new_idom = Some(match new_idom {
                            None => pred_rpo,
                            Some(current) => intersect(&idom, current, pred_rpo),
                        });
                    }
                }
            }

            if new_idom != idom[i] {
                idom[i] = new_idom;
                changed = true;
            }
        }
    }

    // Step 3: Collect pairs.
    let mut pairs = Vec::new();
    for (i, &dom) in idom.iter().enumerate() {
        if let Some(d) = dom {
            pairs.push((rpo[i], rpo[d]));
        }
    }

    Dominators { pairs, idx: 0 }
}

impl<N: Copy> Iterator for Dominators<N> {
    type Item = (N, N);

    fn next(&mut self) -> Option<(N, N)> {
        if self.idx < self.pairs.len() {
            let pair = self.pairs[self.idx];
            self.idx += 1;
            Some(pair)
        } else {
            None
        }
    }
}

/// Find the intersection (common dominator) using finger-walking.
fn intersect(idom: &[Option<usize>], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while a > b {
            a = idom[a].unwrap();
        }
        while b > a {
            b = idom[b].unwrap();
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BTreeGraph;
    use crate::graph::Graph;

    #[test]
    fn dominators_linear() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("1->2", [1, 2]).unwrap();

        let dom: HashMap<_, _> = dominators(&g, 0).collect();
        assert_eq!(dom[&0], 0); // start dominates itself
        assert_eq!(dom[&1], 0);
        assert_eq!(dom[&2], 1);
    }

    #[test]
    fn dominators_diamond() {
        let mut g = BTreeGraph::<_, _>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        g.insert_node(2).unwrap();
        g.insert_node(3).unwrap();
        g.insert_edge("0->1", [0, 1]).unwrap();
        g.insert_edge("0->2", [0, 2]).unwrap();
        g.insert_edge("1->3", [1, 3]).unwrap();
        g.insert_edge("2->3", [2, 3]).unwrap();

        let dom: HashMap<_, _> = dominators(&g, 0).collect();
        assert_eq!(dom[&0], 0);
        assert_eq!(dom[&1], 0);
        assert_eq!(dom[&2], 0);
        assert_eq!(dom[&3], 0); // 0 dominates 3 (both paths go through 0)
    }

    #[test]
    fn dominators_unreachable() {
        let mut g = BTreeGraph::<u32, &str>::default();
        g.insert_node(0).unwrap();
        g.insert_node(1).unwrap();
        // No edges, 1 is unreachable from 0
        let dom: HashMap<_, _> = dominators(&g, 0).collect();
        assert_eq!(dom.len(), 1);
        assert_eq!(dom[&0], 0);
    }
}
