use safegraph::BTreeGraph;
use safegraph::graph::capability::*;
use safegraph::graph::prelude::*;

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

// ---- Node insertion / existence ----

#[test]
fn insert_nodes_and_check_existence() {
    let mut g = BTreeGraph::<u32, u32>::default();
    let ix = g.insert_node(0).unwrap();
    assert_eq!(ix, 0);
    assert!(g.contains_node_index(0));
    assert!(!g.contains_node_index(99));
}

#[test]
fn duplicate_node_returns_err() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(1).unwrap();
    assert!(g.insert_node(1).is_err());
}

#[test]
fn push_discards_index() {
    let mut g = BTreeGraph::<u32, u32>::default();
    assert!(g.push(42).is_ok());
    assert!(g.contains_node_index(42));
    // duplicate via push
    assert!(g.push(42).is_err());
}

// ---- Edge insertion / existence ----

#[test]
fn insert_edge_and_check_existence() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    let ix = g.insert_edge(10, [0, 1]).unwrap();
    assert_eq!(ix, 10);
    assert!(g.contains_edge_index(10));
    assert!(!g.contains_edge_index(99));
}

#[test]
fn duplicate_edge_returns_err() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_edge(10, [0, 1]).unwrap();
    assert!(g.insert_edge(10, [0, 1]).is_err());
}

#[test]
fn push_edge_unchecked() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    assert!(unsafe { g.push_edge_unchecked(10, [0, 1]) }.is_ok());
    assert!(g.contains_edge_index(10));
}

// ---- node() / edge() safe accessors ----

#[test]
fn node_and_edge_accessors() {
    let g = diamond_btree();
    assert_eq!(*g.node(0), 0);
    assert_eq!(*g.node(3), 3);
    assert_eq!(*g.edge(10), 10);
    assert_eq!(*g.edge(13), 13);
}

#[test]
#[should_panic]
fn node_accessor_panics_on_invalid() {
    let g = diamond_btree();
    g.node(99);
}

#[test]
#[should_panic]
fn edge_accessor_panics_on_invalid() {
    let g = diamond_btree();
    g.edge(99);
}

// ---- Endpoints ----

#[test]
fn endpoints_returns_source_target() {
    let g = diamond_btree();
    let eps = g.endpoints(10);
    assert_eq!(eps[0], 0);
    assert_eq!(eps[1], 1);
}

#[test]
fn endpoint_nodes_returns_references() {
    let g = diamond_btree();
    let nodes: Vec<&u32> = g.endpoint_nodes(10).collect();
    assert_eq!(nodes.len(), 2);
    assert_eq!(*nodes[0], 0);
    assert_eq!(*nodes[1], 1);
}

// ---- Iteration: node_indices, edge_indices, nodes, edges ----

#[test]
fn node_indices_stable() {
    let g = diamond_btree();
    let indices: Vec<u32> = g.node_indices().collect();
    assert_eq!(indices, vec![0, 1, 2, 3]);
}

#[test]
fn edge_indices_stable() {
    let g = diamond_btree();
    let indices: Vec<u32> = g.edge_indices().collect();
    assert_eq!(indices, vec![10, 11, 12, 13]);
}

#[test]
fn nodes_iterator() {
    let g = diamond_btree();
    let vals: Vec<u32> = g.nodes().copied().collect();
    assert_eq!(vals, vec![0, 1, 2, 3]);
}

#[test]
fn edges_iterator() {
    let g = diamond_btree();
    let vals: Vec<u32> = g.edges().copied().collect();
    assert_eq!(vals, vec![10, 11, 12, 13]);
}

// ---- Adjacency iteration ----

#[test]
fn edge_indices_from() {
    let g = diamond_btree();
    let out: Vec<u32> = g.edge_indices_from(0).collect();
    assert_eq!(out.len(), 2);
    assert!(out.contains(&10));
    assert!(out.contains(&11));

    // leaf node 3 has no outgoing edges
    let out3: Vec<u32> = g.edge_indices_from(3).collect();
    assert!(out3.is_empty());
}

#[test]
fn edge_indices_of_includes_both_directions() {
    let g = diamond_btree();
    // node 1: outgoing 12, incoming 10
    let all: Vec<u32> = g.edge_indices_of(1).collect();
    assert_eq!(all.len(), 2);
    assert!(all.contains(&10));
    assert!(all.contains(&12));
}

#[test]
fn edges_of_dereferences() {
    let g = diamond_btree();
    let vals: Vec<u32> = g.edges_of(1).copied().collect();
    assert_eq!(vals.len(), 2);
    assert!(vals.contains(&10));
    assert!(vals.contains(&12));
}

#[test]
fn edges_to_incoming() {
    let g = diamond_btree();
    let inc: Vec<u32> = g.edge_indices_to(3).collect();
    assert_eq!(inc.len(), 2);
    assert!(inc.contains(&12));
    assert!(inc.contains(&13));

    // root node 0 has no incoming edges
    let inc0: Vec<u32> = g.edge_indices_to(0).collect();
    assert!(inc0.is_empty());
}

// ---- Directed: edge_tail / edge_head (Bigraph) ----

#[test]
fn edge_tail_and_head() {
    let g = diamond_btree();
    assert_eq!(g.edge_tail_index(10), 0);
    assert_eq!(g.edge_head_index(10), 1);
    assert_eq!(g.edge_tail_index(13), 2);
    assert_eq!(g.edge_head_index(13), 3);
}

#[test]
fn edge_tail_node_and_head_item() {
    let g = diamond_btree();
    assert_eq!(g.edge_tail_index(12), 1);
    assert_eq!(g.edge_head_index(12), 3);
}

// ---- Directed: edge_tails / edge_heads iterators ----

#[test]
fn edge_tails_and_heads_iterators() {
    let g = diamond_btree();
    let tails: Vec<u32> = g.edge_tail_indices(11).collect();
    assert_eq!(tails, vec![0]);
    let heads: Vec<u32> = g.edge_head_indices(11).collect();
    assert_eq!(heads, vec![2]);
}

#[test]
fn edge_tail_nodes_and_head_nodes() {
    let g = diamond_btree();
    let tn: Vec<u32> = g.edge_tail_indices(12).collect();
    assert_eq!(tn, vec![1]);
    let hn: Vec<u32> = g.edge_head_indices(12).collect();
    assert_eq!(hn, vec![3]);
}

// ---- Successors / Predecessors ----

#[test]
fn successor_indices() {
    let g = diamond_btree();
    let succ: Vec<u32> = g.neighbor_indices_from(0).collect();
    assert_eq!(succ.len(), 2);
    assert!(succ.contains(&1));
    assert!(succ.contains(&2));

    // node 3 is a sink
    let succ3: Vec<u32> = g.neighbor_indices_from(3).collect();
    assert!(succ3.is_empty());
}

#[test]
fn predecessors() {
    let g = diamond_btree();
    let pred: Vec<u32> = g.neighbor_indices_to(3).collect();
    assert_eq!(pred.len(), 2);
    assert!(pred.contains(&1));
    assert!(pred.contains(&2));

    // node 0 is a source
    let pred0: Vec<u32> = g.neighbor_indices_to(0).collect();
    assert!(pred0.is_empty());
}

// ---- Incidents ----

#[test]
fn incident_indices() {
    let g = diamond_btree();
    let inc: Vec<u32> = g.neighbor_indices_of(1).collect();
    assert_eq!(inc.len(), 2);
    assert!(inc.contains(&0));
    assert!(inc.contains(&3));
}

#[test]
fn incidents_returns_refs() {
    let g = diamond_btree();
    let inc: Vec<&u32> = g.neighbors_of(1).collect();
    assert_eq!(inc.len(), 2);
    assert!(inc.contains(&&0));
    assert!(inc.contains(&&3));
}

// ---- Reverse ----

#[test]
fn reverse_swaps_direction() {
    let mut g = diamond_btree();
    g.reverse();
    // edge 10 was 0->1, now 1->0
    assert_eq!(g.edge_tail_index(10), 1);
    assert_eq!(g.edge_head_index(10), 0);
    // outgoing from 3 should now have edges 12,13
    let out: Vec<u32> = g.edge_indices_from(3).collect();
    assert_eq!(out.len(), 2);
    assert!(out.contains(&12));
    assert!(out.contains(&13));
}

#[test]
fn double_reverse_is_identity() {
    let g1 = diamond_btree();
    let mut g2 = diamond_btree();
    g2.reverse();
    g2.reverse();
    for eix in g1.edge_indices() {
        let eps1 = g1.endpoints(eix);
        let eps2 = g2.endpoints(eix);
        assert_eq!(eps1, eps2);
    }
}

// ---- Remove ----

#[test]
fn remove_edge_keeps_nodes() {
    let mut g = diamond_btree();
    g.remove_edge(10);
    assert!(!g.contains_edge_index(10));
    assert!(g.contains_node_index(0));
    assert!(g.contains_node_index(1));
    // remaining outgoing from 0: only edge 11
    let out: Vec<u32> = g.edge_indices_from(0).collect();
    assert_eq!(out, vec![11]);
}

#[test]
fn remove_node_cascades_edges() {
    let mut g = diamond_btree();
    // node 1 is involved in edges 10(0->1) and 12(1->3)
    g.remove_node(1);
    assert!(!g.contains_node_index(1));
    assert!(!g.contains_edge_index(10));
    assert!(!g.contains_edge_index(12));
    // edges 11(0->2) and 13(2->3) survive
    assert!(g.contains_edge_index(11));
    assert!(g.contains_edge_index(13));
}

#[test]
fn remove_all_nodes_empties_graph() {
    let mut g = diamond_btree();
    for n in [0, 1, 2, 3] {
        if g.contains_node_index(n) {
            g.remove_node(n);
        }
    }
    let nodes: Vec<u32> = g.node_indices().collect();
    assert!(nodes.is_empty());
    let edges: Vec<u32> = g.edge_indices().collect();
    assert!(edges.is_empty());
}

// ---- UniqueNode / UniqueEdge ----

#[test]
fn node_index_lookup() {
    let g = diamond_btree();
    assert_eq!(g.node_index(2), Some(2));
    assert_eq!(g.node_index(99), None);
}

#[test]
fn edge_index_lookup() {
    let g = diamond_btree();
    assert_eq!(g.edge_index(12), Some(12));
    assert_eq!(g.edge_index(99), None);
}

// ---- Get-or-insert API ----

#[test]
fn get_or_insert_node_vacant() {
    let mut g = BTreeGraph::<u32, u32>::default();
    let node = g.get_or_insert_node(10);
    assert_eq!(node, 10);
    assert!(g.contains_node_index(10));
}

#[test]
fn get_or_insert_node_occupied() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(10).unwrap();
    let node = g.get_or_insert_node(10);
    assert_eq!(node, 10);
}

#[test]
fn get_or_insert_node_uses_node_index() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(5).unwrap();
    assert_eq!(g.node_index(5), Some(5));
    assert_eq!(g.node_index(99), None);
}

#[test]
fn get_or_insert_edge_vacant() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    let edge_ix = g.get_or_insert_edge(10, [0, 1]);
    assert_eq!(edge_ix, 10);
    assert!(g.contains_edge_index(10));
}

#[test]
fn get_or_insert_edge_occupied() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_edge(10, [0, 1]).unwrap();
    let edge_ix = g.get_or_insert_edge(10, [0, 1]);
    assert_eq!(edge_ix, 10);
}

#[test]
fn get_or_insert_edge_uses_edge_index() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_edge(10, [0, 1]).unwrap();
    assert_eq!(g.edge_index(10), Some(10));
    assert_eq!(g.edge_index(99), None);
}

// ---- Scope (immutable) ----

#[test]
fn scope_node_edge_access() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let nodes: Vec<_> = ctx.node_indices().collect();
        assert_eq!(nodes.len(), 4);

        let edges: Vec<_> = ctx.edge_indices().collect();
        assert_eq!(edges.len(), 4);

        // access node through scoped index
        assert_eq!(*ctx.node(nodes[0]), 0);

        // access edge through scoped index
        assert_eq!(*ctx.edge(edges[0]), 10);
    });
}

#[test]
fn scope_endpoints_and_adjacency() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let edges: Vec<_> = ctx.edge_indices().collect();
        let [from, to] = ctx.endpoints(edges[0]); // edge 10: 0->1
        assert_eq!(*ctx.node(from), 0);
        assert_eq!(*ctx.node(to), 1);

        // edges from node 0
        let from_0: Vec<_> = ctx.edge_indices_from(from).collect();
        assert_eq!(from_0.len(), 2);
    });
}

#[test]
fn scope_edge_tail_head() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let edges: Vec<_> = ctx.edge_indices().collect();
        let tail = ctx.edge_tail_index(edges[0]);
        let head = ctx.edge_head_index(edges[0]);
        assert_eq!(*ctx.node(tail), 0);
        assert_eq!(*ctx.node(head), 1);
    });
}

#[test]
fn scope_unique_node_lookup() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let ix = ctx.node_index(2);
        assert!(ix.is_some());
        assert_eq!(*ctx.node(ix.unwrap()), 2);
        assert!(ctx.node_index(99).is_none());
    });
}

#[test]
fn scope_unique_edge_lookup() {
    let g = diamond_btree();
    g.scope(|ctx| {
        let ix = ctx.edge_index(12);
        assert!(ix.is_some());
        assert_eq!(*ctx.edge(ix.unwrap()), 12);
        assert!(ctx.edge_index(99).is_none());
    });
}

// ---- Scope (mutable) ----

#[test]
fn scope_mut_insert_node() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.scope_mut(|mut ctx| {
        let _ = ctx.insert_node(0).unwrap();
        let _ = ctx.insert_node(1).unwrap();
        assert_eq!(ctx.node_indices().count(), 2);
    });
    assert!(g.contains_node_index(0));
    assert!(g.contains_node_index(1));
}

#[test]
fn scope_mut_get_or_insert_node() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.scope_mut(|mut ctx| {
        ctx.get_or_insert_node(42);
        assert!(ctx.node_index(42).is_some());
        assert!(ctx.node_index(99).is_none());
    });
    assert!(g.contains_node_index(42));
}

#[test]
fn scope_mut_reverse() {
    let mut g = diamond_btree();
    g.scope_mut(|mut ctx| {
        ctx.reverse();
    });
    // after reverse, edge 10 should be 1->0
    let eps = g.endpoints(10);
    assert_eq!(eps[0], 1);
    assert_eq!(eps[1], 0);
}

// ---- Self-loop edge ----

#[test]
fn self_loop_edge() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_edge(10, [0, 0]).unwrap();
    let eps = g.endpoints(10);
    assert_eq!(eps[0], 0);
    assert_eq!(eps[1], 0);
    // outgoing and incoming from node 0 both include edge 10
    let out: Vec<u32> = g.edge_indices_from(0).collect();
    assert!(out.contains(&10));
    let inc: Vec<u32> = g.edge_indices_to(0).collect();
    assert!(inc.contains(&10));
}

// ---- Empty graph ----

#[test]
fn empty_graph() {
    let g = BTreeGraph::<u32, u32>::default();
    assert!(!g.contains_node_index(0));
    assert!(!g.contains_edge_index(0));
    assert_eq!(g.node_indices().count(), 0);
    assert_eq!(g.edge_indices().count(), 0);
    assert_eq!(g.nodes().count(), 0);
    assert_eq!(g.edges().count(), 0);
}

// ---- String-keyed graph ----

#[test]
fn string_keyed_graph() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(10).unwrap();
    g.insert_node(20).unwrap();
    g.insert_edge(99, [10, 20]).unwrap();
    assert!(g.contains_node_index(10));
    assert_eq!(*g.edge(99), 99);
}

// ---- Multiple edges between same pair ----

#[test]
fn parallel_edges() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_edge(10, [0, 1]).unwrap();
    g.insert_edge(11, [0, 1]).unwrap();
    g.insert_edge(12, [1, 0]).unwrap(); // reverse direction
    let out: Vec<u32> = g.edge_indices_from(0).collect();
    assert_eq!(out.len(), 2);
    assert!(out.contains(&10));
    assert!(out.contains(&11));
}

// ---- insert_edge checks endpoints validity ----

#[test]
#[should_panic]
fn insert_edge_panics_on_invalid_endpoint() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    // node 99 doesn't exist, the safe insert_edge should panic
    g.insert_edge(10, [0, 99]).ok();
}

// ---- endpoints checks edge validity ----

#[test]
#[should_panic]
fn endpoints_panics_on_invalid() {
    let g = diamond_btree();
    g.endpoints(99);
}

// ---- edge_indices_from checks node validity ----

#[test]
#[should_panic]
fn edge_indices_from_panics_on_invalid() {
    let g = diamond_btree();
    g.edge_indices_from(99);
}

// ---- remove panics on invalid ----

#[test]
#[should_panic]
fn remove_node_panics_on_invalid() {
    let mut g = diamond_btree();
    g.remove_node(99);
}

#[test]
#[should_panic]
fn remove_edge_panics_on_invalid() {
    let mut g = diamond_btree();
    g.remove_edge(99);
}

// ---- get_or_insert_edge panics on invalid endpoint ----

#[test]
#[should_panic]
fn get_or_insert_edge_panics_on_invalid_endpoint() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    // node 99 doesn't exist
    g.get_or_insert_edge(10, [0, 99]);
}

// ---- GraphMap ----

#[test]
fn map_identity() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [0, 1]).unwrap();
    g.insert_edge(11, [1, 2]).unwrap();

    let mapped = g.map(|n| n, |e| e);

    assert!(mapped.contains_node_index(0));
    assert!(mapped.contains_node_index(1));
    assert!(mapped.contains_node_index(2));
    assert!(mapped.contains_edge_index(10));
    assert!(mapped.contains_edge_index(11));

    let eps = mapped.endpoints(10);
    assert_eq!(eps[0], 0);
    assert_eq!(eps[1], 1);
}

#[test]
fn map_transform() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_edge(4, [0, 1]).unwrap();

    // For a map-backed graph the node value IS its index, so remapping values
    // remaps indices and rewires the stored endpoints accordingly.
    let mapped = g.map(|n| n * 10, |e| e + 1);

    assert!(mapped.contains_node_index(0));
    assert!(mapped.contains_node_index(10));
    assert!(mapped.contains_edge_index(5));

    let eps = mapped.endpoints(5);
    assert_eq!(eps[0], 0);
    assert_eq!(eps[1], 10);
}

#[test]
#[should_panic(expected = "fn_node is not injective")]
fn map_non_injective_nodes() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    // Both 0 and 1 map to 0 — should panic
    let _ = g.map(|_| 0u32, |e| e);
}

#[test]
#[should_panic(expected = "fn_edge is not injective")]
fn map_non_injective_edges() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_edge(10, [0, 1]).unwrap();
    g.insert_edge(11, [0, 1]).unwrap();
    // Both edges map to 0 — should panic
    let _ = g.map(|n| n, |_| 0u32);
}

// ---- take_nodes_edges ----

#[test]
fn take_nodes_edges_removes_edges_only() {
    let mut g = diamond_btree();
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([], [10, 13]);
    assert!(nodes.is_empty());
    assert_eq!(edges.len(), 2);
    assert!(edges.contains(&10));
    assert!(edges.contains(&13));
    assert!(!g.contains_edge_index(10));
    assert!(!g.contains_edge_index(13));
    assert!(g.contains_edge_index(11));
    assert!(g.contains_edge_index(12));
    for n in [0, 1, 2, 3] {
        assert!(g.contains_node_index(n));
    }
}

#[test]
fn take_nodes_edges_removes_nodes_cascades() {
    let mut g = diamond_btree();
    // Remove node 1 (involved in edges 10=0->1, 12=1->3).
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([1], []);
    assert_eq!(nodes, vec![1]);
    assert!(edges.is_empty());
    assert!(!g.contains_node_index(1));
    assert!(!g.contains_edge_index(10));
    assert!(!g.contains_edge_index(12));
    assert!(g.contains_edge_index(11));
    assert!(g.contains_edge_index(13));
}

#[test]
fn take_nodes_edges_both_nodes_and_edges() {
    let mut g = diamond_btree();
    // Explicitly remove edge 11 (0->2) and node 3.
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([3], [11]);
    assert_eq!(nodes, vec![3]);
    assert_eq!(edges, vec![11]);
    assert!(!g.contains_node_index(3));
    assert!(!g.contains_edge_index(11));
    assert!(!g.contains_edge_index(12));
    assert!(!g.contains_edge_index(13));
    assert!(g.contains_node_index(0));
    assert!(g.contains_node_index(1));
    assert!(g.contains_node_index(2));
    assert!(g.contains_edge_index(10));
}

#[test]
fn take_nodes_edges_empty_is_noop() {
    let mut g = diamond_btree();
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([], []);
    assert!(nodes.is_empty());
    assert!(edges.is_empty());
    assert_eq!(g.node_indices().count(), 4);
    assert_eq!(g.edge_indices().count(), 4);
}

#[test]
fn take_nodes_edges_all_nodes() {
    let mut g = diamond_btree();
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([0, 1, 2, 3], []);
    assert_eq!(nodes.len(), 4);
    assert!(edges.is_empty());
    assert_eq!(g.node_indices().count(), 0);
    assert_eq!(g.edge_indices().count(), 0);
}

#[test]
fn take_nodes_edges_preserves_adjacency() {
    let mut g = diamond_btree();
    // Remove node 2 (cascades edges 11=0->2, 13=2->3).
    let _: (Vec<u32>, Vec<u32>) = g.take_nodes_edges([2], []);
    let out0: Vec<u32> = g.edge_indices_from(0).collect();
    assert_eq!(out0, vec![10]);
    let inc3: Vec<u32> = g.edge_indices_to(3).collect();
    assert_eq!(inc3, vec![12]);
}

#[test]
fn take_nodes_edges_self_loop() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_edge(10, [0, 0]).unwrap();
    g.insert_edge(20, [0, 1]).unwrap();
    let (nodes, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([], [10]);
    assert!(nodes.is_empty());
    assert_eq!(edges, vec![10]);
    assert!(!g.contains_edge_index(10));
    assert!(g.contains_edge_index(20));
    assert!(g.contains_node_index(0));
}

#[test]
fn take_nodes_edges_returns_values_in_input_order() {
    let mut g = diamond_btree();
    let (_, edges): (Vec<u32>, Vec<u32>) = g.take_nodes_edges([], [13, 11]);
    assert_eq!(edges, vec![13, 11]);
}
