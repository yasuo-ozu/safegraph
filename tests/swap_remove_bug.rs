/// Tests demonstrating the swap-remove index invalidation bug in
/// `VecGraph::take_node_unchecked` (src/vec_graph.rs, `RemoveNode` impl).
///
/// The bug: `take_node_unchecked` pre-collects outgoing (and incoming) edge
/// indices into a `Vec<EdgeIx>`, then removes them one by one via
/// `remove_edge_unchecked`, which internally calls `swap_remove` on the edges
/// vec.  `swap_remove` moves the last edge to the removed position and patches
/// all linked-list pointers (`node.next`, `edge.next`) — but the pre-collected
/// `Vec<EdgeIx>` is **not** patched.  If a later entry in that vec happens to
/// equal the old last-position index, it becomes stale/out-of-bounds after the
/// swap, causing:
///   - debug builds: panic from `debug_assert!(ix < self.edges.len())`
///   - release builds: undefined behaviour (out-of-bounds `get_unchecked`)
///
/// Triggering condition: the outgoing (or incoming) edge linked-list of the
/// removed node must contain indices in *non-decreasing* positional order.
/// Pure append-only graphs always produce decreasing order, so the bug
/// requires prior edge removals that reshuffled positions via swap_remove.
use safegraph::graph::capability::{InsertEdge, InsertNode};
use safegraph::graph::prelude::*;
use safegraph::VecGraph;

/// Minimal reproduction: 3 nodes, 4 edge insertions, 1 intermediate edge
/// removal, then `remove_node` on the hub node.
///
/// Graph construction sequence:
///
///   1. Insert e0: n1 → n2  (position 0)
///   2. Insert e1: n0 → n1  (position 1)
///   3. Insert e2: n0 → n2  (position 2)
///      n0 outgoing linked-list (reverse insertion order): [2, 1]
///
///   4. Remove e0 (n1→n2, position 0):
///      swap_remove(0) moves e2 (pos 2) → pos 0.
///      Internal pointers patched: EdgeIx(2) → EdgeIx(0).
///      n0 outgoing linked-list becomes: [0, 1]   ← INCREASING order
///
///   5. Insert e3: n0 → n2  (position 2, appended)
///      n0 outgoing linked-list: [2, 0, 1]
///
///   6. remove_node(n0):
///      Collects outgoing = [EdgeIx(2), EdgeIx(0), EdgeIx(1)]
///       - Remove EdgeIx(2): last element, just popped. edges.len() = 2. OK.
///       - Remove EdgeIx(0): swap_remove(0) moves EdgeIx(1) → pos 0.
///         edges.len() = 1.
///       - Remove EdgeIx(1): index 1 ≥ edges.len() (1). OUT OF BOUNDS.
///
/// In debug builds this panics at the `debug_assert!` inside
/// `take_edge_unchecked`.  In release builds it is undefined behaviour.
#[test]
fn remove_node_swap_remove_invalidates_precollected_edge_indices() {
    let mut g = VecGraph::<u32, u32>::default();

    // --- nodes (no removals, so NodeIx values stay valid) ---
    let n0 = unsafe { InsertNode::insert_node_unchecked(&mut g, 0).unwrap() };
    let n1 = unsafe { InsertNode::insert_node_unchecked(&mut g, 1).unwrap() };
    let n2 = unsafe { InsertNode::insert_node_unchecked(&mut g, 2).unwrap() };

    // --- edges ---
    // e0 at position 0 (NOT from n0 — used to trigger the swap later)
    let e0 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 100, [n1, n2]).unwrap() };
    // e1 at position 1 (from n0)
    let _e1 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 101, [n0, n1]).unwrap() };
    // e2 at position 2 (from n0)
    let _e2 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 102, [n0, n2]).unwrap() };
    // n0 outgoing list: [EdgeIx(2), EdgeIx(1)]  (decreasing — safe order)

    // Remove e0 (n1→n2). swap_remove(0) moves e2 from pos 2 to pos 0.
    // All linked-list pointers EdgeIx(2) are patched to EdgeIx(0).
    // n0 outgoing list becomes: [EdgeIx(0), EdgeIx(1)]  (INCREASING — triggers bug)
    g.remove_edge(e0);

    // Insert e3 from n0 (appended at position 2).
    // n0 outgoing list: [EdgeIx(2), EdgeIx(0), EdgeIx(1)]
    let _e3 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 103, [n0, n2]).unwrap() };

    // This triggers the bug: the internal loop in take_node_unchecked will
    // try to access EdgeIx(1) after the vec has shrunk to length 1.
    g.remove_node(n0);
}

/// Same bug, but via incoming edges rather than outgoing.
///
/// Construction: make n0 a *sink* node (edges point TO n0) and use the same
/// swap-remove trick to create an increasing-order incoming linked-list.
#[test]
fn remove_node_swap_remove_invalidates_precollected_incoming_edges() {
    let mut g = VecGraph::<u32, u32>::default();

    let n0 = unsafe { InsertNode::insert_node_unchecked(&mut g, 0).unwrap() };
    let n1 = unsafe { InsertNode::insert_node_unchecked(&mut g, 1).unwrap() };
    let n2 = unsafe { InsertNode::insert_node_unchecked(&mut g, 2).unwrap() };

    // e0 at position 0: unrelated edge (used to trigger swap)
    let e0 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 100, [n1, n2]).unwrap() };
    // e1 at position 1: incoming to n0
    let _e1 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 101, [n1, n0]).unwrap() };
    // e2 at position 2: incoming to n0
    let _e2 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 102, [n2, n0]).unwrap() };
    // n0 incoming list: [EdgeIx(2), EdgeIx(1)]

    // Remove e0: swap_remove(0) moves e2 (pos 2) → pos 0.
    // n0 incoming list becomes: [EdgeIx(0), EdgeIx(1)]  (increasing)
    g.remove_edge(e0);

    // Insert e3: incoming to n0 (appended at position 2).
    // n0 incoming list: [EdgeIx(2), EdgeIx(0), EdgeIx(1)]
    let _e3 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 103, [n2, n0]).unwrap() };

    // Triggers the bug in the incoming-edge removal loop.
    g.remove_node(n0);
}

/// Variant with more edges to show the bug is not limited to the minimal case.
/// Also exercises a scenario where the swapped edge is not the very last one
/// removed, but an interior element of the pre-collected vec.
#[test]
fn remove_node_swap_remove_with_extra_edges() {
    let mut g = VecGraph::<u32, u32>::default();

    let n0 = unsafe { InsertNode::insert_node_unchecked(&mut g, 0).unwrap() };
    let n1 = unsafe { InsertNode::insert_node_unchecked(&mut g, 1).unwrap() };
    let n2 = unsafe { InsertNode::insert_node_unchecked(&mut g, 2).unwrap() };
    let n3 = unsafe { InsertNode::insert_node_unchecked(&mut g, 3).unwrap() };

    // e0 (pos 0): n1 → n2  (unrelated to n0)
    let e0 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 200, [n1, n2]).unwrap() };
    // e1 (pos 1): n0 → n1
    let _e1 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 201, [n0, n1]).unwrap() };
    // e2 (pos 2): n0 → n2
    let _e2 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 202, [n0, n2]).unwrap() };
    // e3 (pos 3): n0 → n3
    let _e3 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 203, [n0, n3]).unwrap() };
    // n0 outgoing list: [3, 2, 1] (decreasing)

    // Remove e0 (pos 0): swap_remove moves e3 (pos 3) to pos 0.
    // EdgeIx(3) → EdgeIx(0) in all pointers.
    // n0 outgoing list: [0, 2, 1]  ← non-monotonic
    g.remove_edge(e0);

    // Insert e4 (pos 3): n0 → n3
    let _e4 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 204, [n0, n3]).unwrap() };
    // n0 outgoing list: [3, 0, 2, 1]

    // remove_node(n0) pre-collects [3, 0, 2, 1], then:
    //   remove(3): last element, popped. len=3. OK.
    //   remove(0): swap_remove(0), moves pos 2 to pos 0. len=2. OK.
    //              EdgeIx(2) → EdgeIx(0) in pointers (NOT in our vec).
    //   remove(2): 2 ≥ 2. OUT OF BOUNDS.
    g.remove_node(n0);
}
