//! Regression tests for `Stabilized` insertion.
//!
//! The insert path skips its tombstone-reuse scan when appending is cheap:
//! either the backend reports spare capacity (`capacity_node`/`capacity_edge`
//! return `Some(cap > len)`), or — when it reports no capacity — there are no
//! tombstones to reuse. Without this, a fresh build did an O(n) scan per
//! insert (O(n²) overall); see CLAUDE.md.

use safegraph::graph::Graph;
use safegraph::{BTreeGraph, VecGraph};

#[test]
fn vec_backend_reports_capacity_btree_does_not() {
    // The Vec backend must report capacity so the append-cheap gate can fire;
    // map backends report `None` and fall back to the tombstone-existence gate.
    let g = VecGraph::<u32, u32>::default();
    assert!(
        Graph::capacity_node(&g).is_some(),
        "Vec backend reports node capacity"
    );
    assert!(
        Graph::capacity_edge(&g).is_some(),
        "Vec backend reports edge capacity"
    );

    let bg = BTreeGraph::<u32, u32>::default();
    assert_eq!(Graph::capacity_node(&bg), None);
    assert_eq!(Graph::capacity_edge(&bg), None);
}
