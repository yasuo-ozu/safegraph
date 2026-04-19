use safegraph::graph::capability::*;
use safegraph::graph::prelude::*;
use safegraph::{BTreeGraph, VecGraph};

fn diamond_btree() -> BTreeGraph<u32, u32> {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_node(3).unwrap();
    g.insert_edge(10, [0, 1]).unwrap();
    g.insert_edge(11, [0, 2]).unwrap();
    g.insert_edge(12, [1, 3]).unwrap();
    g.insert_edge(13, [2, 3]).unwrap();
    g
}

fn diamond_vec() -> (VecGraph<u32, u32>, [u32; 4], [u32; 4]) {
    let mut g = VecGraph::<u32, u32>::default();
    let n0 = unsafe { InsertNode::insert_node_unchecked(&mut g, 0).unwrap() };
    let n1 = unsafe { InsertNode::insert_node_unchecked(&mut g, 1).unwrap() };
    let n2 = unsafe { InsertNode::insert_node_unchecked(&mut g, 2).unwrap() };
    let n3 = unsafe { InsertNode::insert_node_unchecked(&mut g, 3).unwrap() };
    let e0 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 10, [n0, n1]).unwrap() };
    let e1 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 11, [n0, n2]).unwrap() };
    let e2 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 12, [n1, n3]).unwrap() };
    let e3 = unsafe { InsertEdge::insert_edge_unchecked(&mut g, 13, [n2, n3]).unwrap() };
    (g, [n0, n1, n2, n3], [e0, e1, e2, e3])
}

#[test]
fn scoped_node_ix_inner_btree() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let nodes: Vec<_> = ctx.node_indices().collect();
        // inner() returns the underlying BTreeGraph NodeIx (&u32)
        let inner = nodes[0].inner();
        assert_eq!(inner, 0);
    });
}

#[test]
fn scoped_edge_ix_inner_btree() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let edges: Vec<_> = ctx.edge_indices().collect();
        let inner = edges[0].inner();
        assert_eq!(inner, 10);
    });
}

#[test]
fn scoped_node_ix_display_vec() {
    let (g, _, _) = diamond_vec();
    g.scope(|ctx| {
        let nodes: Vec<_> = ctx.node_indices().collect();
        let s = format!("{}", nodes[0]);
        assert_eq!(s, "0");
    });
}

#[test]
fn scoped_edge_ix_display_vec() {
    let (g, _, _) = diamond_vec();
    g.scope(|ctx| {
        let edges: Vec<_> = ctx.edge_indices().collect();
        let s = format!("{}", edges[0]);
        assert_eq!(s, "0");
    });
}

#[test]
fn scoped_node_ix_equality_btree() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let nodes: Vec<_> = ctx.node_indices().collect();
        let n0_again = ctx.node_index(0).unwrap();
        assert_eq!(nodes[0], n0_again);
    });
}

#[test]
fn scoped_node_ix_ordering_btree() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let nodes: Vec<_> = ctx.node_indices().collect();
        // BTreeGraph produces sorted node indices, scoped indices preserve order
        assert!(nodes[0] < nodes[1]);
        assert!(nodes[1] < nodes[2]);
    });
}

#[test]
fn scoped_edge_ix_inner_vec() {
    let (g, _, _) = diamond_vec();
    g.scope(|ctx| {
        let edges: Vec<_> = ctx.edge_indices().collect();
        let inner = edges[0].inner();
        // inner() returns the underlying VecGraph EdgeIx (a bare `u32`)
        let s = format!("{inner}");
        assert_eq!(s, "0");
    });
}

#[test]
fn scope_directed_edges_to_btree() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let n3 = ctx.node_index(3).unwrap();
        let incoming: Vec<_> = ctx.edges_to(n3).collect();
        assert_eq!(incoming.len(), 2);
    });
}

#[test]
fn scope_edge_indices_of_btree() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let n1 = ctx.node_index(1).unwrap();
        let all: Vec<_> = ctx.edge_indices_of(n1).collect();
        assert_eq!(all.len(), 2);
    });
}

#[test]
fn scope_incident_indices_btree() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let n1 = ctx.node_index(1).unwrap();
        let inc: Vec<_> = ctx.neighbor_indices_of(n1).collect();
        assert_eq!(inc.len(), 2);
    });
}
