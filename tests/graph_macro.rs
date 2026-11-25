// Exercises the `graph!` macro, which needs the `macros` feature.
#![cfg(feature = "macros")]

use safegraph::graph::prelude::*;
use safegraph::graph::GraphOperation;
use safegraph::VecGraph;

#[test]
fn graph_macro_builds_vecgraph_with_default_initial() {
    let g: VecGraph<u32, u32> = safegraph::graph!(
        {0} -- {10} --> {1},
        {1} -- {11} --> {2},
        {0} -- {12} --> {2},
    );

    let nodes: Vec<u32> = g.nodes().copied().collect();
    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(nodes, vec![0, 1, 1, 2, 0, 2]);
    assert_eq!(edges, vec![10, 11, 12]);
}

#[test]
fn graph_macro_uses_custom_initial_graph() {
    let mut g = VecGraph::<u32, u32>::default();
    safegraph::graph!(
        &mut g =>
        {1} -- {7} --> {2}
    );

    let nodes: Vec<u32> = g.nodes().copied().collect();
    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(nodes, vec![1, 2]);
    assert_eq!(edges, vec![7]);
}

#[test]
fn graph_macro_supports_empty_edge_spec_with_default_edge_value() {
    let g: VecGraph<u32, u32> = safegraph::graph!({1} --> {2});

    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(edges, vec![0]);
}

#[test]
fn graph_macro_supports_compact_empty_edge_arrow() {
    let g: VecGraph<u32, u32> = safegraph::graph!({1} --> {2});

    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(edges, vec![0]);
}

#[test]
fn graph_macro_supports_reverse_arrow_forms() {
    let g: VecGraph<u32, u32> = safegraph::graph!(
        {1} <-- {2},
        {5} <-- {9} -- {6},
    );

    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(edges, vec![0, 9]);
}

#[test]
fn graph_macro_supports_chained_edges_in_single_line() {
    let g: VecGraph<u32, u32> = safegraph::graph!({0} --> {1} -- {7} --> {2} --> {3});

    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(edges, vec![0, 7, 0]);
    assert_eq!(GraphOperation::edge_indices(&g).count(), 3);
}

#[test]
fn graph_macro_supports_multiple_forward_arrows_in_one_line() {
    let mut g = VecGraph::<u32, u32>::default().stabilize();
    safegraph::graph!(&mut g => a {1} --> b {2} --> c {3});
    let nodes: Vec<u32> = g.nodes().copied().collect();
    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(nodes, vec![1, 2, 3]);
    assert_eq!(edges, vec![0, 0]);
    let edgepoints: Vec<(u32, u32)> = GraphOperation::edge_indices(&g)
        .map(|e| {
            let (s, t) = (g.tail_index(e), g.head_index(e));
            (*g.node(s), *g.node(t))
        })
        .collect();
    assert_eq!(edgepoints, vec![(1, 2), (2, 3)]);
}

#[test]
fn graph_macro_supports_mixed_forward_and_reverse_arrows_in_one_line() {
    let mut g = VecGraph::<u32, u32>::default().stabilize();
    safegraph::graph!(&mut g => a {1} --> b {2} <-- c {3});
    let nodes: Vec<u32> = g.nodes().copied().collect();
    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(nodes, vec![1, 2, 3]);
    assert_eq!(edges, vec![0, 0]);
    let edgepoints: Vec<(u32, u32)> = GraphOperation::edge_indices(&g)
        .map(|e| {
            let (s, t) = (g.tail_index(e), g.head_index(e));
            (*g.node(s), *g.node(t))
        })
        .collect();
    assert_eq!(edgepoints, vec![(1, 2), (3, 2)]);
}

#[test]
fn graph_macro_emits_distinct_nodes_for_anon_specs() {
    let g: VecGraph<u32, u32> = safegraph::graph!(
        {0} -- {1} --> {1},
        {0} -- {2} --> {2},
        {1} -- {3} --> {2},
    );

    assert_eq!(GraphOperation::node_indices(&g).count(), 6);
    assert_eq!(GraphOperation::edge_indices(&g).count(), 3);
}

#[test]
fn graph_macro_connects_edges_from_inserted_nodes() {
    let g: VecGraph<u32, u32> = safegraph::graph!(
        {5} -- {50} --> {6},
        {6} -- {60} --> {7},
    );

    let n0 = GraphOperation::node_indices(&g).next().unwrap();
    // `VecGraph` is not `StableEdge`, so use the raw `GraphOperation` method
    // (disambiguates from the `StableEdge`-bounded `Graph::…` sibling).
    let out: Vec<_> = unsafe { GraphOperation::edge_indices_from_unchecked(&g, &n0) }.collect();
    assert_eq!(out.len(), 1);
}

#[test]
fn graph_macro_supports_named_node_with_expr() {
    let mut g = VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
    safegraph::graph!(
        &mut g =>
        a {(0, 1u32)} -- {(0, 10u32)} --> b {(0, 2u32)},
        c {(0, 3u32)} -- {(0, 11u32)} --> d {(0, 4u32)}
    );

    let nodes: Vec<(i64, u32)> = g.nodes().copied().collect();
    let edges: Vec<(i64, u32)> = g.edges().copied().collect();
    assert_eq!(nodes, vec![(0, 1u32), (0, 2u32), (0, 3u32), (0, 4u32)]);
    assert_eq!(edges, vec![(0, 10u32), (0, 11u32)]);
    assert_eq!(*g.node(a), (0, 1u32));
    assert_eq!(*g.node(b), (0, 2u32));
    assert_eq!(*g.node(c), (0, 3u32));
    assert_eq!(*g.node(d), (0, 4u32));
}

#[test]
#[allow(unused_variables)]
fn graph_macro_supports_named_edge_with_expr() {
    let mut g = VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
    safegraph::graph!(
        &mut g =>
        n0 {(0, 1u32)} -- e {(0, 99u32)} --> n1 {(0, 2u32)},
        n2 {(0, 3u32)} -- f {(0, 100u32)} --> n3 {(0, 4u32)}
    );

    let nodes: Vec<(i64, u32)> = g.nodes().copied().collect();
    let edges: Vec<(i64, u32)> = g.edges().copied().collect();
    assert_eq!(nodes, vec![(0, 1u32), (0, 2u32), (0, 3u32), (0, 4u32)]);
    assert_eq!(edges, vec![(0, 99u32), (0, 100u32)]);
    assert_eq!(*g.edge(e), (0, 99u32));
    assert_eq!(*g.edge(f), (0, 100u32));
}

#[test]
fn graph_macro_can_be_used_inside_scope() {
    let outer = VecGraph::<u32, u32>::default();
    outer.scope(|_| {
        let g: VecGraph<u32, u32> = safegraph::graph!(
            {0} -- {10} --> {1},
            {1} -- {11} --> {2},
        );
        let nodes: Vec<u32> = g.nodes().copied().collect();
        let edges: Vec<u32> = g.edges().copied().collect();
        assert_eq!(nodes, vec![0, 1, 1, 2]);
        assert_eq!(edges, vec![10, 11]);
    });
}

#[test]
fn graph_macro_can_be_used_inside_scope_mut_with_bound_idents() {
    let mut g = VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
    g.scope_mut(|mut ctx| {
        safegraph::graph!(
            &mut *ctx =>
            n0 {(0, 1u32)} -- e0 {(0, 10u32)} --> n1 {(0, 2u32)}
        );
        assert_eq!(*ctx.node(n0), (0, 1u32));
        assert_eq!(*ctx.node(n1), (0, 2u32));
        assert_eq!(*ctx.edge(e0), (0, 10u32));
    });
}

/// Payloads are passed through as tokens, not parsed by the macro: any
/// expression works, including ones with commas or `=>` outside a group.
#[test]
// The closure is called in place so that its parameter list puts a comma at
// the top level of the payload.
#[allow(clippy::redundant_closure_call)]
fn graph_macro_passes_any_payload_expression_through() {
    let flag = true;
    let pair: std::collections::HashMap<u32, u32> = [(1, 2)].into_iter().collect();
    let g: VecGraph<u32, u32> = safegraph::graph!(
        // A closure (its parameter list has a comma outside any group).
        {(|a: u32, b: u32| a + b)(1, 2)} -- {if flag { 10 } else { 20 }} --> {{
            let x = 4;
            x * 2
        }},
        // A turbofish with a comma, and a `match` with `=>` inside its braces.
        {std::collections::HashMap::<u32, u32>::new().len() as u32 + pair[&1]}
            -- {match flag { true => 30, false => 40 }} --> {[5u32, 6][1]},
    );
    let nodes: Vec<u32> = g.nodes().copied().collect();
    let edges: Vec<u32> = g.edges().copied().collect();
    assert_eq!(nodes, vec![3, 8, 2, 6]);
    assert_eq!(edges, vec![10, 30]);
}

/// The graph expression before `=>` may contain `=>` inside a group.
#[test]
fn graph_macro_statement_mode_graph_expression_with_inner_arrow() {
    let mut a = VecGraph::<u32, u32>::default();
    let mut b = VecGraph::<u32, u32>::default();
    let pick_a = false;
    safegraph::graph!(
        match pick_a { true => &mut a, false => &mut b } =>
        {1} -- {7} --> {2}
    );
    assert_eq!(a.nodes().count(), 0);
    let nodes: Vec<u32> = b.nodes().copied().collect();
    assert_eq!(nodes, vec![1, 2]);
}
