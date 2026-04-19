//! Zero-cost verification for the [`LinkedAdjEdgeGraph`] / `VecGraph` backend.
//!
//! These tests use the [`ir-assert`] crate to inspect the LLVM IR produced
//! for hot iterator paths and confirm the abstraction collapses away under
//! inlining on x86_64-unknown-linux-gnu (`calls.len() == 0` means every
//! accessor was inlined, no virtual / generic dispatch left).
//!
//! All test targets are passed as inline closures; all graph operations
//! go through the top-level [`Graph`] trait only.
//!
//! [`ir-assert`]: https://crates.io/crates/ir-assert
//! [`LinkedAdjEdgeGraph`]: safegraph::raw_graph::linked_adj_edge::LinkedAdjEdgeGraph

use ir_assert::assert_ir;
use safegraph::VecGraph;
use safegraph::graph::Graph;

// VecGraph's NodeIx and EdgeIx are both `u32`.

// --- Iterator paths reachable via Graph ------------------------------------

#[test]
fn vec_edges_from_count_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> usize {
            unsafe { g.edges_from_unchecked(n) }.count()
        },
    );
}

#[test]
fn vec_edges_of_count_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> usize {
            unsafe { g.edges_of_unchecked(n) }.count()
        },
    );
}

#[test]
fn vec_edges_to_count_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> usize {
            unsafe { g.unsafe_assert_stable_edge().edges_to_unchecked(n) }.count()
        },
    );
}

#[test]
fn vec_edge_indices_to_count_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> usize {
            unsafe { g.unsafe_assert_stable_edge().edge_indices_to_unchecked(n) }.count()
        },
    );
}

#[test]
fn vec_edges_from_sum_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> u32 {
            unsafe { g.edges_from_unchecked(n) }.copied().sum::<u32>()
        },
    );
}

#[test]
fn vec_edges_of_sum_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> u32 {
            unsafe { g.edges_of_unchecked(n) }.copied().sum::<u32>()
        },
    );
}

#[test]
fn vec_walks_to_sum_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> u32 {
            unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node().walks_to_unchecked(n) }
                .map(|w| w.get())
                .map(|(_, _, e)| *e)
                .sum::<u32>()
        },
    );
}

// --- Endpoint and incident lookups (Graph wrappers) ------------------------

#[test]
fn vec_endpoint_nodes_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, e: u32| -> Option<u32> {
            unsafe { g.endpoint_nodes_unchecked(e) }.next().copied()
        },
    );
}

#[test]
fn vec_edge_tail_index_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, e: u32| -> u32 {
            unsafe { g.unsafe_assert_stable_node().edge_tail_index_unchecked(e) }
        },
    );
}

#[test]
fn vec_edge_head_index_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, e: u32| -> u32 {
            unsafe { g.unsafe_assert_stable_node().edge_head_index_unchecked(e) }
        },
    );
}

#[test]
fn vec_edge_tail_load_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, e: u32| -> u32 { *unsafe { g.unsafe_assert_stable_node().edge_tail_unchecked(e) } },
    );
}

#[test]
fn vec_edge_head_load_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, e: u32| -> u32 { *unsafe { g.unsafe_assert_stable_node().edge_head_unchecked(e) } },
    );
}

#[test]
fn vec_edge_tail_indices_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, e: u32| -> Option<u32> {
            unsafe { g.unsafe_assert_stable_node().edge_tail_indices_unchecked(e) }.next()
        },
    );
}

#[test]
fn vec_edge_head_indices_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, e: u32| -> Option<u32> {
            unsafe { g.unsafe_assert_stable_node().edge_head_indices_unchecked(e) }.next()
        },
    );
}

// --- Update path (Graph::node_unchecked_mut / edge_unchecked_mut) ----------

#[test]
fn vec_node_set_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &mut VecGraph<u32, u32>, n: u32, v: u32| {
            unsafe { *g.node_unchecked_mut(n) = v }
        },
    );
}

#[test]
fn vec_edge_set_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &mut VecGraph<u32, u32>, e: u32, v: u32| {
            unsafe { *g.edge_unchecked_mut(e) = v }
        },
    );
}

// --- `next()` of each iterator type — only the per-step inner loop ---------

#[test]
fn vec_edges_from_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> Option<u32> {
            unsafe { g.edges_from_unchecked(n) }.next().copied()
        },
    );
}

#[test]
fn vec_edges_of_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> Option<u32> {
            unsafe { g.edges_of_unchecked(n) }.next().copied()
        },
    );
}

#[test]
fn vec_edges_to_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> Option<u32> {
            unsafe { g.unsafe_assert_stable_edge().edges_to_unchecked(n) }.next().copied()
        },
    );
}

#[test]
fn vec_edge_indices_to_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> Option<u32> {
            unsafe { g.unsafe_assert_stable_edge().edge_indices_to_unchecked(n) }.next()
        },
    );
}

#[test]
fn vec_walks_to_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<u32, u32>, n: u32| -> Option<u32> {
            unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node().walks_to_unchecked(n) }
                .next()
                .map(|w| w.into_parts().1)
        },
    );
}

// ---------------------------------------------------------------------------
// Varied node/edge data shapes — closures can't be generic, spot-check Wide.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct Wide {
    x: u64,
    y: u64,
}

#[test]
fn wide_edges_from_count_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<Wide, Wide>, n: u32| -> usize {
            unsafe { g.edges_from_unchecked(n) }.count()
        },
    );
}

#[test]
fn wide_walks_to_first_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<Wide, Wide>, n: u32| -> Option<u32> {
            unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node().walks_to_unchecked(n) }
                .next()
                .map(|w| w.into_parts().1)
        },
    );
}

#[test]
fn wide_edge_tail_load_no_calls() {
    assert_ir!(
        target_x86_64_unknown_linux_gnu & calls.len().eq(0),
        |g: &VecGraph<Wide, Wide>, e: u32| -> u32 { unsafe { g.unsafe_assert_stable_node().edge_tail_index_unchecked(e) } },
    );
}
