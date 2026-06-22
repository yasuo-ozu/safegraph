/// Tests demonstrating that `VecGraph::take_edge_unchecked` does not patch
/// adjacency linked-list pointers after `swap_remove`, leaving stale indices
/// that corrupt subsequent operations.
///
/// When `swap_remove(i)` moves the last edge (index L) into position i, every
/// linked-list pointer that still references L (in `node.next[]` and
/// `edge.next[]`) must be rewritten to i. The current implementation skips
/// this fixup entirely.
///
/// The observable consequence depends on what happens after the stale pointer
/// is left behind:
///
///   1. **Infinite loop (hang)** — A later `insert_edge` reuses position L,
///      and `mem::replace` stores the stale L as the new edge's `next`,
///      creating a self-referential cycle in the linked list. Any traversal
///      of that list (e.g. `edge_indices_of`) loops forever.
///
///   2. **Wrong edges yielded** — Traversal follows the stale pointer into
///      an unrelated edge, producing incorrect adjacency results even when
///      no out-of-bounds access occurs.
///
///   3. **Out-of-bounds / panic** — The stale index L exceeds the current
///      vec length if no new edges have been inserted to fill the gap.
///
/// All operations in these tests use safe APIs:
///
/// - `ctx.insert_node()` / `ctx.insert_edge()` inside `scope_mut` — safe
///   because `Context` implements `StableNode`/`StableEdge` (scoped indices
///   are inherently stable within the scope).
/// - `ctx.remove_nodes_edges()` — safe removal consuming the scope context.
/// - `g.push_edge()` — safe edge insertion (discards edge index).
/// - `g.remove_node()` / `g.take_node()` / `g.take_nodes_edges()` — safe
///   `GraphOperation` methods that trigger the underlying bug.
///
/// **WARNING**: Tests that trigger the linked-list cycle will **hang** (loop
/// forever) when the bug is present. There are no timeouts.
use safegraph::graph::prelude::*;
use safegraph::VecGraph;

type G = VecGraph<u32, u32>;

// ---------------------------------------------------------------------------
// 1. Stale pointer creates a self-loop → infinite loop on traversal
// ---------------------------------------------------------------------------

/// After `remove_edge` swap-removes edge 0 (moving edge 2 to position 0),
/// the head of n0's outgoing list still says EdgeIx(2). A subsequent insert
/// reuses position 2 and `mem::replace` stores the stale EdgeIx(2) as the
/// new edge's `next[0]`, creating a cycle:
///
///   n0.next[0] = EdgeIx(2)  →  edges[2].next[0] = EdgeIx(2)  →  …
///
/// `remove_node(n0)` internally calls `edge_indices_of().collect()` on
/// this cyclic list and hangs.
#[test]
fn stale_pointer_creates_self_loop_in_outgoing_list() {
    let mut g = G::default();

    // Phase 1: build graph + remove the swap-fodder edge (safe removal).
    let (n0, n2) = g.scope_mut(|mut ctx| {
        let n0 = ctx.insert_node(0).unwrap();
        let n1 = ctx.insert_node(1).unwrap();
        let n2 = ctx.insert_node(2).unwrap();

        // edges[0]: n1→n2  (swap fodder)
        let e0 = ctx.insert_edge(100, [n1, n2]).unwrap();
        // edges[1]: n0→n1
        let _e1 = ctx.insert_edge(101, [n0, n1]).unwrap();
        // edges[2]: n0→n2  (will be swap-moved to position 0)
        let _e2 = ctx.insert_edge(102, [n0, n2]).unwrap();

        let raw = (n0.inner(), n2.inner());
        // Safe removal: swap_remove(0) moves edges[2] to position 0,
        // but linked-list pointers referencing EdgeIx(2) are NOT patched.
        ctx.remove_nodes_edges(std::iter::empty(), [e0]);
        raw
    });

    // Phase 2: insert a new edge at position 2 (= edges.len()).
    // mem::replace stores the stale EdgeIx(2) as this edge's next[0],
    // creating a self-referential cycle in n0's outgoing list.
    g.push_edge(103, [n0, n2]).unwrap();

    // Phase 3 (safe): remove_node internally traverses the cyclic list.
    // This hangs when the bug is present.
    g.remove_node(n0);
}

/// Same self-loop scenario, but on the *incoming* adjacency list.
#[test]
fn stale_pointer_creates_self_loop_in_incoming_list() {
    let mut g = G::default();

    let (n0, n2) = g.scope_mut(|mut ctx| {
        let n0 = ctx.insert_node(0).unwrap();
        let n1 = ctx.insert_node(1).unwrap();
        let n2 = ctx.insert_node(2).unwrap();

        // edges[0]: n1→n2  (swap fodder)
        let e0 = ctx.insert_edge(100, [n1, n2]).unwrap();
        // edges[1]: n1→n0  (incoming to n0)
        let _e1 = ctx.insert_edge(101, [n1, n0]).unwrap();
        // edges[2]: n2→n0  (incoming to n0, will be swap-moved)
        let _e2 = ctx.insert_edge(102, [n2, n0]).unwrap();

        let raw = (n0.inner(), n2.inner());
        ctx.remove_nodes_edges(std::iter::empty(), [e0]);
        raw
    });

    // New incoming edge to n0 at position 2 — creates the self-loop.
    g.push_edge(103, [n2, n0]).unwrap();

    // Safe: hangs due to cyclic incoming list inside remove_node.
    g.remove_node(n0);
}

// ---------------------------------------------------------------------------
// 2. Stale pointer yields wrong edges (no OOB, no cycle, just wrong data)
// ---------------------------------------------------------------------------

/// After swap_remove moves edge L to position i, the outgoing linked list of
/// the moved edge's source node still contains EdgeIx(L). If a *different*
/// edge is later inserted at position L, traversal follows the stale pointer
/// into an edge that does not belong to the node, producing incorrect results.
#[test]
fn stale_pointer_yields_wrong_edge_data() {
    let mut g = G::default();

    let (n0, n2, n3) = g.scope_mut(|mut ctx| {
        let n0 = ctx.insert_node(0).unwrap();
        let n1 = ctx.insert_node(1).unwrap();
        let n2 = ctx.insert_node(2).unwrap();
        let n3 = ctx.insert_node(3).unwrap();

        // edges[0]: n1→n2  (swap fodder)
        let e0 = ctx.insert_edge(100, [n1, n2]).unwrap();
        // edges[1]: n0→n1  (from n0)
        let _e1 = ctx.insert_edge(101, [n0, n1]).unwrap();
        // edges[2]: n0→n2  (from n0, will be swap-moved to position 0)
        let _e2 = ctx.insert_edge(102, [n0, n2]).unwrap();

        let raw = (n0.inner(), n2.inner(), n3.inner());
        ctx.remove_nodes_edges(std::iter::empty(), [e0]);
        raw
    });

    // Insert an edge NOT from n0 at position 2.
    // n0's outgoing list follows the stale EdgeIx(2) into this unrelated edge.
    g.push_edge(999, [n3, n2]).unwrap();

    // Safe: remove_node traverses n0's outgoing list. Due to the stale
    // pointer it will encounter the n3→n2 edge (value 999) which doesn't
    // belong to n0, causing incorrect unlink operations.
    // This may panic on debug assertions inside the iterator/unlink,
    // or silently corrupt the graph in release mode.
    //
    // We use take_node to observe the returned data — if the bug is fixed
    // the returned value should be 0.
    let taken = g.take_node(n0);
    assert_eq!(taken, 0, "take_node should return n0's data");
}

// ---------------------------------------------------------------------------
// 3. Stale pointer causes out-of-bounds → wrong data after backfill
// ---------------------------------------------------------------------------

/// If no new edge is inserted after `remove_edge`, the stale pointer
/// EdgeIx(L) references a position past the end of the edges vec.
/// We backfill with a dummy edge to avoid abort, then observe that
/// `take_node` encounters the wrong edge and corrupts the graph.
#[test]
fn stale_pointer_out_of_bounds_detected_via_take_node() {
    let mut g = G::default();

    let (n0, n1, n2) = g.scope_mut(|mut ctx| {
        let n0 = ctx.insert_node(0).unwrap();
        let n1 = ctx.insert_node(1).unwrap();
        let n2 = ctx.insert_node(2).unwrap();

        // edges[0]: n1→n2  (swap fodder)
        let e0 = ctx.insert_edge(100, [n1, n2]).unwrap();
        // edges[1]: n0→n1
        let _e1 = ctx.insert_edge(101, [n0, n1]).unwrap();
        // edges[2]: n0→n2  (will be swap-moved)
        let _e2 = ctx.insert_edge(102, [n0, n2]).unwrap();

        let raw = (n0.inner(), n1.inner(), n2.inner());
        ctx.remove_nodes_edges(std::iter::empty(), [e0]);
        raw
    });

    // Backfill position 2 with an unrelated edge so the stale pointer
    // doesn't hit an OOB abort.
    g.push_edge(999, [n1, n2]).unwrap();

    // Safe: remove_node follows the stale EdgeIx(2) into the dummy edge
    // and tries to unlink it from n0's adjacency lists — but the edge
    // doesn't belong to n0. Debug assertions fire; release mode corrupts.
    let taken = g.take_node(n0);
    assert_eq!(taken, 0);
}

// ---------------------------------------------------------------------------
// 4. remove_node hangs because it internally collects edge_indices_of
// ---------------------------------------------------------------------------

/// `take_node_unchecked` first collects all incident edge indices via
/// `edge_indices_of_unchecked().collect()`. If a previous `remove_edge`
/// left a cyclic linked list (scenario 1), that collect never finishes
/// and `remove_node` hangs.
#[test]
fn remove_node_hangs_due_to_cyclic_edge_list() {
    let mut g = G::default();

    let (n0, n2) = g.scope_mut(|mut ctx| {
        let n0 = ctx.insert_node(0).unwrap();
        let n1 = ctx.insert_node(1).unwrap();
        let n2 = ctx.insert_node(2).unwrap();

        let e0 = ctx.insert_edge(100, [n1, n2]).unwrap();
        let _e1 = ctx.insert_edge(101, [n0, n1]).unwrap();
        let _e2 = ctx.insert_edge(102, [n0, n2]).unwrap();

        let raw = (n0.inner(), n2.inner());
        ctx.remove_nodes_edges(std::iter::empty(), [e0]);
        raw
    });

    g.push_edge(103, [n0, n2]).unwrap();

    // Safe call — hangs forever due to cyclic linked list.
    g.remove_node(n0);
}

// ---------------------------------------------------------------------------
// 5. The swap-remove-then-remove loop inside take_node_unchecked itself
// ---------------------------------------------------------------------------

/// Even if the linked-list traversal for *collecting* edges succeeds, the
/// removal loop inside `take_node_unchecked` performs its own `swap_remove`
/// calls on the pre-collected edge indices. Each swap_remove invalidates
/// the position of the moved edge, but the pre-collected `Vec<EdgeIx>`
/// is not updated. If a later entry in that vec equals the old last-position
/// index, it becomes stale.
#[test]
fn take_node_internal_swap_invalidates_precollected_indices() {
    let mut g = G::default();

    let (n0, n3) = g.scope_mut(|mut ctx| {
        let n0 = ctx.insert_node(0).unwrap();
        let n1 = ctx.insert_node(1).unwrap();
        let n2 = ctx.insert_node(2).unwrap();
        let n3 = ctx.insert_node(3).unwrap();

        // e0 (pos 0): n1→n2  (not from n0)
        let e0 = ctx.insert_edge(200, [n1, n2]).unwrap();
        // e1 (pos 1): n0→n1
        let _e1 = ctx.insert_edge(201, [n0, n1]).unwrap();
        // e2 (pos 2): n0→n2
        let _e2 = ctx.insert_edge(202, [n0, n2]).unwrap();
        // e3 (pos 3): n0→n3
        let _e3 = ctx.insert_edge(203, [n0, n3]).unwrap();

        let raw = (n0.inner(), n3.inner());
        // swap_remove(0) moves e3 (pos 3)→pos 0. Linked-list NOT patched.
        ctx.remove_nodes_edges(std::iter::empty(), [e0]);
        raw
    });

    // Insert e4 at pos 3: n0→n3
    g.push_edge(204, [n0, n3]).unwrap();

    // Safe call — hangs or panics due to stale pre-collected indices
    // inside take_node_unchecked's swap_remove loop.
    g.remove_node(n0);
}

// ---------------------------------------------------------------------------
// 6. take_nodes_edges has the same swap-no-patch bug
// ---------------------------------------------------------------------------

/// `take_nodes_edges_unchecked` also uses swap_remove in its edge-removal
/// loop without patching. Verify it exhibits the same corruption.
#[test]
fn take_nodes_edges_swap_no_patch() {
    let mut g = G::default();

    let (n0, n2): (u32, u32) = g.scope_mut(|mut ctx| {
        let n0 = ctx.insert_node(0).unwrap();
        let n1 = ctx.insert_node(1).unwrap();
        let n2 = ctx.insert_node(2).unwrap();

        let e0 = ctx.insert_edge(100, [n1, n2]).unwrap();
        let _e1 = ctx.insert_edge(101, [n0, n1]).unwrap();
        let _e2 = ctx.insert_edge(102, [n0, n2]).unwrap();

        let raw = (n0.inner(), n2.inner());
        ctx.remove_nodes_edges(std::iter::empty(), [e0]);
        raw
    });

    g.push_edge(103, [n0, n2]).unwrap();

    // Safe call — hangs due to cyclic linked list in the bulk removal path.
    let (_nodes, _edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([n0], std::iter::empty());
}
