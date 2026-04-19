use safegraph::graph::capability::{InsertEdge, InsertNode, StableEdge, StableNode};
use safegraph::graph::edge::Endpoints;
use safegraph::graph::prelude::*;
use safegraph::graph::stabilized::{EdgeIx, NodeIx, Stabilized};
use safegraph::raw_graph::linked_adj_edge::{EdgeRepr, LinkedAdjEdgeGraph, NodeRepr};
use safegraph::VecGraph;

// The inner graph of a stabilized `VecGraph<u32, u32>`: payloads are remapped to
// the version-tagged `NodeIx`/`EdgeIx` wrappers.
type Svg = Stabilized<
    LinkedAdjEdgeGraph<Vec<(NodeIx<u32>, NodeRepr<u32>)>, Vec<(EdgeIx<u32>, EdgeRepr<u32, u32>)>>,
    u32,
    u32,
>;

fn diamond_on<G>(ctx: &mut G) -> ([G::NodeIx; 4], [G::EdgeIx; 4])
where
    G: Graph
        + GraphProperty<Node = u32, Edge = u32>
        + InsertNode
        + InsertEdge
        + StableNode
        + StableEdge,
{
    let n0 = ctx.insert_node(0).unwrap();
    let n1 = ctx.insert_node(1).unwrap();
    let n2 = ctx.insert_node(2).unwrap();
    let n3 = ctx.insert_node(3).unwrap();
    let ep = |a, b| G::Endpoints::try_from_node_indices([a, b]).unwrap();
    let e0 = ctx.insert_edge(10, ep(n0, n1)).unwrap();
    let e1 = ctx.insert_edge(11, ep(n0, n2)).unwrap();
    let e2 = ctx.insert_edge(12, ep(n1, n3)).unwrap();
    let e3 = ctx.insert_edge(13, ep(n2, n3)).unwrap();
    ([n0, n1, n2, n3], [e0, e1, e2, e3])
}

fn new_graph() -> Svg {
    VecGraph::<u32, u32>::default().stabilize()
}

// ---- Node insertion / existence ----

#[test]
fn insert_nodes() {
    let mut g = new_graph();
    let n0 = g.insert_node(10).unwrap();
    let n1 = g.insert_node(20).unwrap();
    assert!(g.contains_node_index(n0));
    assert!(g.contains_node_index(n1));
}

#[test]
fn node_and_edge_access() {
    let mut g = VecGraph::<&str, &str>::default().stabilize();
    let n0 = g.insert_node("a").unwrap();
    let n1 = g.insert_node("b").unwrap();
    let e0 = g.insert_edge("edge", [n0, n1]).unwrap();
    assert_eq!(*g.node(n0), "a");
    assert_eq!(*g.node(n1), "b");
    assert_eq!(*g.edge(e0), "edge");
}

// ---- Edge insertion ----

#[test]
fn insert_edge_and_endpoints() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let eix = g.insert_edge(10, [n0, n1]).unwrap();
    assert!(g.contains_edge_index(eix));
    let eps = g.endpoints(eix);
    assert_eq!(eps[0], n0);
    assert_eq!(eps[1], n1);
}

// ---- Iteration ----

#[test]
fn node_and_edge_indices() {
    let mut g = new_graph();
    let (ns, es) = diamond_on(&mut g);
    let nodes: Vec<_> = g.node_indices().collect();
    assert_eq!(nodes.len(), 4);
    for n in &ns {
        assert!(nodes.contains(n));
    }
    let edges: Vec<_> = g.edge_indices().collect();
    assert_eq!(edges.len(), 4);
    for e in &es {
        assert!(edges.contains(e));
    }
}

#[test]
fn nodes_iterator() {
    let mut g = new_graph();
    diamond_on(&mut g);
    let mut vals: Vec<u32> = g.nodes().copied().collect();
    vals.sort();
    assert_eq!(vals, vec![0, 1, 2, 3]);
}

#[test]
fn edges_iterator() {
    let mut g = new_graph();
    diamond_on(&mut g);
    let mut vals: Vec<u32> = g.edges().copied().collect();
    vals.sort();
    assert_eq!(vals, vec![10, 11, 12, 13]);
}

// ---- Adjacency ----

#[test]
fn edge_indices_from() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    let out: Vec<_> = g.edge_indices_from(ns[0]).collect();
    assert_eq!(out.len(), 2);
}

#[test]
fn edges_to_incoming() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    let inc: Vec<_> = g.edge_indices_to(ns[3]).collect();
    assert_eq!(inc.len(), 2);
}

#[test]
fn edge_indices_of() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    // node 1: out=12, in=10
    let all: Vec<_> = g.edge_indices_of(ns[1]).collect();
    assert_eq!(all.len(), 2);
}

// ---- Directed: tail / head ----

#[test]
fn edge_tail_and_head() {
    let mut g = new_graph();
    let (ns, es) = diamond_on(&mut g);
    // edge 0 (10): 0->1
    assert_eq!(g.edge_tail_index(es[0]), ns[0]);
    assert_eq!(g.edge_head_index(es[0]), ns[1]);
}

#[test]
fn edge_tail_node_and_head_item() {
    let mut g = new_graph();
    let (_, es) = diamond_on(&mut g);
    // edge 2 (12): 1->3
    let tail_nodes: Vec<_> = g.edge_tail_indices(es[2]).collect();
    assert_eq!(tail_nodes.len(), 1);
    let head_nodes: Vec<_> = g.edge_head_indices(es[2]).collect();
    assert_eq!(head_nodes.len(), 1);
}

// ---- Successors / Predecessors ----

#[test]
fn successor_indices() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    let succ: Vec<_> = g.neighbor_indices_from(ns[0]).collect();
    assert_eq!(succ.len(), 2);
    assert!(succ.contains(&ns[1]));
    assert!(succ.contains(&ns[2]));
}

#[test]
fn predecessor_indices() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    let pred: Vec<_> = g.neighbor_indices_to(ns[3]).collect();
    assert_eq!(pred.len(), 2);
    assert!(pred.contains(&ns[1]));
    assert!(pred.contains(&ns[2]));
}

// ---- Incidents ----

#[test]
fn incident_indices() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    let inc: Vec<_> = g.neighbor_indices_of(ns[1]).collect();
    assert_eq!(inc.len(), 2);
    assert!(inc.contains(&ns[0]));
    assert!(inc.contains(&ns[3]));
}

// ---- Update (mutable access) ----

#[test]
fn node_mut_updates_value() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    *g.node_mut(n0) = 42;
    assert_eq!(*g.node(n0), 42);
}

#[test]
fn edge_mut_updates_value() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let e0 = g.insert_edge(10, [n0, n1]).unwrap();
    *g.edge_mut(e0) = 99;
    assert_eq!(*g.edge(e0), 99);
}

// ---- Reverse ----

#[test]
fn reverse_swaps_direction() {
    let mut g = new_graph();
    let (ns, es) = diamond_on(&mut g);
    g.reverse();
    // edge 0 (10): was 0->1, now 1->0
    assert_eq!(g.edge_tail_index(es[0]), ns[1]);
    assert_eq!(g.edge_head_index(es[0]), ns[0]);
}

#[test]
fn double_reverse_is_identity() {
    let mut g1 = new_graph();
    let (_, es) = diamond_on(&mut g1);
    let mut g2 = new_graph();
    diamond_on(&mut g2);
    g2.reverse();
    g2.reverse();
    for eix in &es {
        let eps1 = g1.endpoints(*eix);
        let eps2 = g2.endpoints(*eix);
        assert_eq!(eps1, eps2);
    }
}

// ---- Remove edge ----

#[test]
fn remove_edge_basic() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let e0 = g.insert_edge(10, [n0, n1]).unwrap();
    let _e1 = g.insert_edge(11, [n0, n1]).unwrap();
    assert_eq!(g.edge_indices().count(), 2);
    g.remove_edge(e0);
    assert_eq!(g.edge_indices().count(), 1);
    // nodes survive
    assert!(g.contains_node_index(n0));
    assert!(g.contains_node_index(n1));
}

#[test]
fn remove_node_cascades_edges() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    // remove node 1 (involved in edges 10=0->1, 12=1->3)
    g.remove_node(ns[1]);
    // Should have 3 nodes left
    assert_eq!(g.node_indices().count(), 3);
    // edges involving node 1 are gone; edges 11=0->2 and 13=2->3 remain
    assert_eq!(g.edge_indices().count(), 2);
}

// ---- Self-loop edge ----

#[test]
fn self_loop_edge() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    let e0 = g.insert_edge(10, [n0, n0]).unwrap();
    let eps = g.endpoints(e0);
    assert_eq!(eps[0], n0);
    assert_eq!(eps[1], n0);
    let out: Vec<_> = g.edge_indices_from(n0).collect();
    assert!(out.contains(&e0));
    let inc: Vec<_> = g.edge_indices_to(n0).collect();
    assert!(inc.contains(&e0));
}

// ---- Empty graph ----

#[test]
fn empty_graph() {
    let g = new_graph();
    assert_eq!(g.node_indices().count(), 0);
    assert_eq!(g.edge_indices().count(), 0);
    assert_eq!(g.nodes().count(), 0);
    assert_eq!(g.edges().count(), 0);
}

// ---- Scope ----

#[test]
fn scope_read_only() {
    let mut g = new_graph();
    diamond_on(&mut g);
    g.scope(|ctx| {
        let nodes: Vec<_> = ctx.node_indices().collect();
        assert_eq!(nodes.len(), 4);
        let edges: Vec<_> = ctx.edge_indices().collect();
        assert_eq!(edges.len(), 4);

        let [from, to] = ctx.endpoints(edges[0]);
        assert_eq!(
            *ctx.node(from) + *ctx.node(to),
            *ctx.node(from) + *ctx.node(to)
        );
    });
}

#[test]
fn scope_mut_update_node() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    let _n1 = g.insert_node(1).unwrap();
    *g.node_mut(n0) = 100;
    assert_eq!(*g.node(n0), 100);
}

#[test]
fn scope_mut_insert() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let _ = g.insert_edge(10, [n0, n1]).unwrap();
    assert_eq!(g.node_indices().count(), 2);
    assert_eq!(g.edge_indices().count(), 1);
}

// ---- Display for NodeIx / EdgeIx ----

#[test]
fn display_formatting() {
    let mut g = new_graph();
    let n = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let e = g.insert_edge(10, [n, n1]).unwrap();
    // stabilized indices display as `v{version}:{inner}`
    let n_str = format!("{n}");
    let e_str = format!("{e}");
    assert!(
        n_str.starts_with("v1:") && n_str.ends_with(":0"),
        "node display should be version-prefixed inner index, got {n_str}"
    );
    assert!(
        e_str.starts_with("v1:") && e_str.ends_with(":0"),
        "edge display should be version-prefixed inner index, got {e_str}"
    );
}

// ---- Endpoint nodes / edges_of ----

#[test]
fn endpoint_nodes_returns_references() {
    let mut g = new_graph();
    let (_, es) = diamond_on(&mut g);
    let nodes: Vec<&u32> = g.endpoint_nodes(es[0]).collect();
    assert_eq!(nodes.len(), 2);
    assert_eq!(*nodes[0], 0);
    assert_eq!(*nodes[1], 1);
}

#[test]
fn edges_of_dereferences() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    let vals: Vec<u32> = g.edges_of(ns[1]).copied().collect();
    assert_eq!(vals.len(), 2);
}

#[test]
fn incidents_returns_refs() {
    let mut g = new_graph();
    let (ns, _) = diamond_on(&mut g);
    let inc: Vec<&u32> = g.neighbors_of(ns[1]).collect();
    assert_eq!(inc.len(), 2);
}

// ---- edge_tails / edge_heads iterators ----

#[test]
fn edge_tails_and_heads_iterators() {
    let mut g = new_graph();
    let (_, es) = diamond_on(&mut g);
    let tails: Vec<_> = g.edge_tail_indices(es[1]).collect();
    assert_eq!(tails.len(), 1);
    let heads: Vec<_> = g.edge_head_indices(es[1]).collect();
    assert_eq!(heads.len(), 1);
}

#[test]
fn edge_tail_nodes_and_head_nodes() {
    let mut g = new_graph();
    let (_, es) = diamond_on(&mut g);
    let tn: Vec<_> = g.edge_tail_indices(es[2]).collect();
    assert_eq!(tn.len(), 1);
    let hn: Vec<_> = g.edge_head_indices(es[2]).collect();
    assert_eq!(hn.len(), 1);
}

// ---- Undirected self-loop deduplication ----

#[test]
fn undirected_self_loop_edge_indices_from_no_duplicate() {
    let mut g = VecGraph::<u32, u32>::default().stabilize();
    let n0 = g.insert_node(0).unwrap();
    let e0 = g.insert_edge(10, [n0, n0]).unwrap();
    let g = g.undirected();
    let from: Vec<_> = g.edge_indices_from(n0).collect();
    assert_eq!(
        from,
        vec![e0],
        "self-loop should appear exactly once in edge_indices_from"
    );
}

#[test]
fn undirected_self_loop_with_normal_edges() {
    let mut g = VecGraph::<u32, u32>::default().stabilize();
    let n0 = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let e_loop = g.insert_edge(10, [n0, n0]).unwrap();
    let e_norm = g.insert_edge(20, [n0, n1]).unwrap();
    let g = g.undirected();
    let from: Vec<_> = g.edge_indices_from(n0).collect();
    assert_eq!(
        from.len(),
        2,
        "should yield self-loop once plus normal edge"
    );
    assert!(from.contains(&e_loop));
    assert!(from.contains(&e_norm));
}

#[test]
fn undirected_multiple_self_loops_no_duplicate() {
    let mut g = VecGraph::<u32, u32>::default().stabilize();
    let n0 = g.insert_node(0).unwrap();
    let e0 = g.insert_edge(10, [n0, n0]).unwrap();
    let e1 = g.insert_edge(20, [n0, n0]).unwrap();
    let g = g.undirected();
    let from: Vec<_> = g.edge_indices_from(n0).collect();
    assert_eq!(
        from.len(),
        2,
        "two distinct self-loops should each appear once"
    );
    assert!(from.contains(&e0));
    assert!(from.contains(&e1));
}

#[test]
fn undirected_non_loop_edge_appears_from_both_endpoints() {
    let mut g = VecGraph::<u32, u32>::default().stabilize();
    let n0 = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let e0 = g.insert_edge(10, [n0, n1]).unwrap();
    let g = g.undirected();
    let from_n0: Vec<_> = g.edge_indices_from(n0).collect();
    let from_n1: Vec<_> = g.edge_indices_from(n1).collect();
    assert_eq!(from_n0, vec![e0]);
    assert_eq!(from_n1, vec![e0]);
}

// ---- take_nodes_edges ----

#[test]
fn take_nodes_edges_removes_edges_only() {
    let mut g = new_graph();
    let ([n0, n1, n2, n3], [e0, e1, e2, e3]) = diamond_on(&mut g);
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([], [e0, e3]);
    assert!(nodes.is_empty());
    assert_eq!(edges.len(), 2);
    assert!(edges.contains(&10));
    assert!(edges.contains(&13));
    assert!(!g.contains_edge_index(e0));
    assert!(!g.contains_edge_index(e3));
    assert!(g.contains_edge_index(e1));
    assert!(g.contains_edge_index(e2));
    for n in [n0, n1, n2, n3] {
        assert!(g.contains_node_index(n));
    }
}

#[test]
fn take_nodes_edges_removes_nodes_cascades() {
    let mut g = new_graph();
    let ([n0, n1, _n2, _n3], [e0, e1, e2, e3]) = diamond_on(&mut g);
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([n1], []);
    assert_eq!(nodes, vec![1]);
    assert!(edges.is_empty());
    assert!(!g.contains_node_index(n1));
    assert!(!g.contains_edge_index(e0));
    assert!(!g.contains_edge_index(e2));
    assert!(g.contains_node_index(n0));
    assert!(g.contains_edge_index(e1));
    assert!(g.contains_edge_index(e3));
}

#[test]
fn take_nodes_edges_both_nodes_and_edges() {
    let mut g = new_graph();
    let ([n0, n1, n2, n3], [e0, e1, e2, e3]) = diamond_on(&mut g);
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([n3], [e1]);
    assert_eq!(nodes, vec![3]);
    assert_eq!(edges, vec![11]);
    assert!(!g.contains_node_index(n3));
    assert!(!g.contains_edge_index(e1));
    assert!(!g.contains_edge_index(e2));
    assert!(!g.contains_edge_index(e3));
    assert!(g.contains_node_index(n0));
    assert!(g.contains_node_index(n1));
    assert!(g.contains_node_index(n2));
    assert!(g.contains_edge_index(e0));
}

#[test]
fn take_nodes_edges_empty_is_noop() {
    let mut g = new_graph();
    let (ns, es) = diamond_on(&mut g);
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([], []);
    assert!(nodes.is_empty());
    assert!(edges.is_empty());
    for n in ns {
        assert!(g.contains_node_index(n));
    }
    for e in es {
        assert!(g.contains_edge_index(e));
    }
}

#[test]
fn take_nodes_edges_all_nodes() {
    let mut g = new_graph();
    let (ns, _es) = diamond_on(&mut g);
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges(ns, []);
    assert_eq!(nodes.len(), 4);
    assert!(edges.is_empty());
    assert_eq!(g.len_node(), 0);
    assert_eq!(g.len_edge(), 0);
}

#[test]
fn take_nodes_edges_preserves_adjacency() {
    let mut g = new_graph();
    let ([n0, _n1, n2, n3], [e0, _e1, e2, _e3]) = diamond_on(&mut g);
    let _: (Vec<u32>, Vec<u32>) = g.take_nodes_edges([n2], []);
    let out_n0: Vec<_> = g.edge_indices_from(n0).collect();
    assert_eq!(out_n0.len(), 1);
    assert!(out_n0.contains(&e0));
    let inc_n3: Vec<_> = g.edge_indices_to(n3).collect();
    assert_eq!(inc_n3.len(), 1);
    assert!(inc_n3.contains(&e2));
}

#[test]
fn take_nodes_edges_self_loop() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let e_loop = g.insert_edge(10, [n0, n0]).unwrap();
    let e_norm = g.insert_edge(20, [n0, n1]).unwrap();
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([], [e_loop]);
    assert!(nodes.is_empty());
    assert_eq!(edges, vec![10]);
    assert!(!g.contains_edge_index(e_loop));
    assert!(g.contains_edge_index(e_norm));
    assert!(g.contains_node_index(n0));
}

#[test]
fn take_nodes_edges_returns_values_in_input_order() {
    let mut g = new_graph();
    let (_ns, [_e0, e1, _e2, e3]) = diamond_on(&mut g);
    let (_, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([], [e3, e1]);
    assert_eq!(edges, vec![13, 11]);
}

// ---- Multiple insertions/removals stress ----

#[test]
fn remove_then_reinsert() {
    let mut g = new_graph();
    let n0 = g.insert_node(0).unwrap();
    let n1 = g.insert_node(1).unwrap();
    let e0 = g.insert_edge(10, [n0, n1]).unwrap();
    g.remove_edge(e0);
    // reinsert — the index may differ but data should be correct
    let e1 = g.insert_edge(20, [n0, n1]).unwrap();
    assert_eq!(*g.edge(e1), 20);
    let eps = g.endpoints(e1);
    assert_eq!(eps[0], n0);
    assert_eq!(eps[1], n1);
}

#[test]
fn many_nodes_and_edges() {
    let mut g = new_graph();
    let mut ns = Vec::new();
    for i in 0..100u32 {
        ns.push(g.insert_node(i).unwrap());
    }
    // linear chain
    for i in 0..99usize {
        g.insert_edge(i as u32, [ns[i], ns[i + 1]]).unwrap();
    }
    assert_eq!(g.node_indices().count(), 100);
    assert_eq!(g.edge_indices().count(), 99);
    // middle node has 1 incoming + 1 outgoing
    let mid_edges: Vec<_> = g.edge_indices_of(ns[50]).collect();
    assert_eq!(mid_edges.len(), 2);
}
