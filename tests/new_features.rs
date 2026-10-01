use safegraph::graph::capability::{
    Bigraph, Directed, InsertEdge, InsertNode, RemoveNode, StableEdge, StableNode, UniqueEdge,
    UniqueNode,
};
use safegraph::graph::{prelude::*, GraphMap};
use safegraph::BTreeGraph;
use safegraph::HashGraph;
use safegraph::VecGraph;
use std::borrow::Borrow;

// ---------------------------------------------------------------------------
// endpoints_from_array round-trip
// ---------------------------------------------------------------------------

#[test]
fn endpoints_from_array_vecgraph_round_trip() {
    let mut g = VecGraph::<u32, u32>::default();
    g.scope_mut(|mut ctx| {
        let n0 = ctx.insert_node(10).unwrap();
        let n1 = ctx.insert_node(20).unwrap();
        let e0 = ctx.insert_edge(100, [n0, n1]).unwrap();

        let endpoints: Vec<_> = ctx.endpoints(e0).map(|n| n.to_ix()).collect();
        assert_eq!(endpoints, [n0, n1]);
        let built = <safegraph::graph::context::Context<'_, VecGraph<u32, u32>> as Bigraph>::endpoints_from_array([n0, n1]);
        let arr = <safegraph::graph::context::Context<'_, VecGraph<u32, u32>> as Bigraph>::endpoints_as_array(built);
        assert_eq!(arr, [n0, n1]);
    });
}

#[test]
fn endpoints_from_array_btreegraph_round_trip() {
    let mut g = BTreeGraph::<u32, u32>::new();
    let n0 = g.insert_node(10).unwrap();
    let n1 = g.insert_node(20).unwrap();
    let e0 = g.insert_edge(100, [n0, n1]).unwrap();

    assert_eq!(g.endpoints(e0), [&n0, &n1]);
    let endpoints = BTreeGraph::<u32, u32>::endpoints_from_array([n0, n1]);
    let arr = BTreeGraph::<u32, u32>::endpoints_as_array(endpoints);
    assert_eq!(arr, [n0, n1]);
}

#[test]
fn endpoints_from_array_hashgraph_round_trip() {
    let mut g = HashGraph::<u32, u32>::new();
    let n0 = g.insert_node(10).unwrap();
    let n1 = g.insert_node(20).unwrap();
    let e0 = g.insert_edge(100, [n0, n1]).unwrap();

    assert_eq!(g.endpoints(e0), [&n0, &n1]);
    let endpoints = HashGraph::<u32, u32>::endpoints_from_array([n0, n1]);
    let arr = HashGraph::<u32, u32>::endpoints_as_array(endpoints);
    assert_eq!(arr, [n0, n1]);
}

// ---------------------------------------------------------------------------
// drain
// ---------------------------------------------------------------------------

#[test]
fn drain_vecgraph_returns_all_nodes_and_edges() {
    let mut g = VecGraph::<u32, u32>::default();
    let _ = g.push(10);
    let _ = g.push(20);
    let _ = g.push(30);
    g.scope_mut(|mut ctx| {
        let indices: Vec<_> = ctx.node_indices().collect();
        ctx.insert_edge(100, [*indices[0].borrow(), *indices[1].borrow()])
            .unwrap();
        ctx.insert_edge(200, [*indices[1].borrow(), *indices[2].borrow()])
            .unwrap();
    });

    let (nodes, edges) = g.drain();
    let nodes: Vec<u32> = nodes.collect();
    let edges: Vec<u32> = edges.collect();

    assert_eq!(nodes, vec![10, 20, 30]);
    assert_eq!(edges, vec![100, 200]);
}

#[test]
fn drain_btreegraph_returns_all_nodes_and_edges() {
    let mut g = BTreeGraph::<u32, u32>::new();
    g.insert_node(10).unwrap();
    g.insert_node(20).unwrap();
    g.insert_node(30).unwrap();
    g.insert_edge(100, [10, 20]).unwrap();
    g.insert_edge(200, [20, 30]).unwrap();

    let (nodes, edges) = g.drain();
    let mut nodes: Vec<u32> = nodes.collect();
    let mut edges: Vec<u32> = edges.collect();
    nodes.sort();
    edges.sort();

    assert_eq!(nodes, vec![10, 20, 30]);
    assert_eq!(edges, vec![100, 200]);
}

#[test]
fn drain_hashgraph_returns_all_nodes_and_edges() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(10).unwrap();
    g.insert_node(20).unwrap();
    g.insert_node(30).unwrap();
    g.insert_edge(100, [10, 20]).unwrap();
    g.insert_edge(200, [20, 30]).unwrap();

    let (nodes, edges) = g.drain();
    let mut nodes: Vec<u32> = nodes.collect();
    let mut edges: Vec<u32> = edges.collect();
    nodes.sort();
    edges.sort();

    assert_eq!(nodes, vec![10, 20, 30]);
    assert_eq!(edges, vec![100, 200]);
}

#[test]
fn drain_empty_graph() {
    let g = BTreeGraph::<u32, u32>::new();
    let (nodes, edges) = g.drain();
    assert_eq!(nodes.count(), 0);
    assert_eq!(edges.count(), 0);
}

#[test]
fn drain_stabilized_skips_tombstoned() {
    let mut g = VecGraph::<u32, u32>::default().stabilize();
    let n0 = g.insert_node(10).unwrap();
    let n1 = g.insert_node(20).unwrap();
    let n2 = g.insert_node(30).unwrap();
    let _e0 = g.insert_edge(100, [n0, n1]).unwrap();
    let _e1 = g.insert_edge(200, [n1, n2]).unwrap();

    // Remove node n1 (tombstones node and its incident edges)
    g.remove_node(n1);

    let (nodes, edges) = g.drain();
    let mut nodes: Vec<u32> = nodes.collect();
    let mut edges: Vec<u32> = edges.collect();
    nodes.sort();
    edges.sort();

    assert_eq!(nodes, vec![10, 30]);
    assert!(edges.is_empty());
}

// ---------------------------------------------------------------------------
// extend_graph
// ---------------------------------------------------------------------------

#[test]
fn extend_graph_btreegraph_merges_two_graphs() {
    let mut g1 = BTreeGraph::<u32, u32>::new();
    g1.insert_node(1).unwrap();
    g1.insert_node(2).unwrap();
    g1.insert_edge(10, [1, 2]).unwrap();

    let mut g2 = BTreeGraph::<u32, u32>::new();
    g2.insert_node(3).unwrap();
    g2.insert_node(4).unwrap();
    g2.insert_edge(20, [3, 4]).unwrap();

    g1.extend_graph(g2);

    assert_eq!(g1.len_node(), 4);
    assert_eq!(g1.len_edge(), 2);
}

#[test]
fn extend_graph_hashgraph_merges_two_graphs() {
    let mut g1 = HashGraph::<u32, u32>::new();
    g1.insert_node(1).unwrap();
    g1.insert_node(2).unwrap();
    g1.insert_edge(10, [1, 2]).unwrap();

    let mut g2 = HashGraph::<u32, u32>::new();
    g2.insert_node(3).unwrap();
    g2.insert_node(4).unwrap();
    g2.insert_edge(20, [3, 4]).unwrap();

    g1.extend_graph(g2);

    assert_eq!(g1.len_node(), 4);
    assert_eq!(g1.len_edge(), 2);
}

#[test]
fn extend_graph_preserves_edge_connectivity() {
    let mut g1 = BTreeGraph::<u32, u32>::new();
    g1.insert_node(1).unwrap();
    g1.insert_node(2).unwrap();
    g1.insert_edge(10, [1, 2]).unwrap();

    let mut g2 = BTreeGraph::<u32, u32>::new();
    g2.insert_node(3).unwrap();
    g2.insert_node(4).unwrap();
    g2.insert_node(5).unwrap();
    g2.insert_edge(20, [3, 4]).unwrap();
    g2.insert_edge(30, [4, 5]).unwrap();

    g1.extend_graph(g2);

    // Total: 5 nodes, 3 edges
    assert_eq!(g1.len_node(), 5);
    assert_eq!(g1.len_edge(), 3);

    // Each edge in g1 should have valid endpoints
    for eix in g1.edge_indices() {
        let ep = g1.endpoints(*eix);
        for nix in ep {
            assert!(g1.contains_node_index(nix));
        }
    }
}

#[test]
fn extend_graph_empty_other() {
    let mut g1 = BTreeGraph::<u32, u32>::new();
    g1.insert_node(10).unwrap();

    let g2 = BTreeGraph::<u32, u32>::new();
    g1.extend_graph(g2);

    assert_eq!(g1.len_node(), 1);
    assert_eq!(g1.len_edge(), 0);
}

#[test]
fn extend_graph_into_empty() {
    let mut g1 = BTreeGraph::<u32, u32>::new();

    let mut g2 = BTreeGraph::<u32, u32>::new();
    g2.insert_node(10).unwrap();
    g2.insert_node(20).unwrap();
    g2.insert_edge(100, [10, 20]).unwrap();

    g1.extend_graph(g2);

    assert_eq!(g1.len_node(), 2);
    assert_eq!(g1.len_edge(), 1);
}

// ---------------------------------------------------------------------------
// HashGraph basic operations
// ---------------------------------------------------------------------------

#[test]
fn hashgraph_insert_nodes_and_edges() {
    let mut g = HashGraph::<u32, u32>::new();
    let n0 = g.insert_node(1).unwrap();
    let n1 = g.insert_node(2).unwrap();
    let n2 = g.insert_node(3).unwrap();

    assert_eq!(g.len_node(), 3);
    assert!(g.contains_node_index(n0));
    assert!(g.contains_node_index(n1));
    assert!(g.contains_node_index(n2));

    let e0 = g.insert_edge(10, [n0, n1]).unwrap();
    let e1 = g.insert_edge(20, [n1, n2]).unwrap();

    assert_eq!(g.len_edge(), 2);
    assert!(g.contains_edge_index(e0));
    assert!(g.contains_edge_index(e1));
}

#[test]
fn hashgraph_query_node_and_edge() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();

    assert_eq!(*g.node(1), 1);
    assert_eq!(*g.edge(10), 10);
    assert_eq!(g.endpoints(10), [&1, &2]);
}

#[test]
fn hashgraph_duplicate_node_returns_err() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    assert!(g.insert_node(1).is_err());
}

#[test]
fn hashgraph_duplicate_edge_returns_err() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();
    assert!(g.insert_edge(10, [1, 2]).is_err());
}

#[test]
fn hashgraph_remove_edge() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    let e = g.insert_edge(10, [1, 2]).unwrap();

    g.remove_edge(e);
    assert!(!g.contains_edge_index(e));
    assert_eq!(g.len_edge(), 0);
    assert_eq!(g.len_node(), 2);
}

#[test]
fn hashgraph_remove_node_cascades_edges() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_node(3).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();
    g.insert_edge(20, [2, 3]).unwrap();

    g.remove_node(2);

    assert!(!g.contains_node_index(2));
    assert_eq!(g.len_node(), 2);
    assert_eq!(g.len_edge(), 0);
}

#[test]
fn hashgraph_adjacency_walks() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_node(3).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();
    g.insert_edge(20, [1, 3]).unwrap();

    let walks: Vec<_> = g.walks_from(1).map(|w| w.get()).collect();
    assert_eq!(walks.len(), 2);

    // Verify all neighbors are reachable
    let neighbor_nodes: Vec<u32> = walks.iter().map(|&(_, _, nix)| *nix).collect();
    assert!(neighbor_nodes.contains(&2));
    assert!(neighbor_nodes.contains(&3));
}

#[test]
fn hashgraph_directed_tail_head() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();

    assert_eq!(*g.tail_index(10), 1);
    assert_eq!(*g.head_index(10), 2);
}

#[test]
fn hashgraph_unique_node_lookup() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(42).unwrap();

    assert_eq!(g.node_index(&42), Some(42));
    assert_eq!(g.node_index(&99), None);
}

#[test]
fn hashgraph_unique_edge_lookup() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();

    assert_eq!(g.edge_index(&10), Some(10));
    assert_eq!(g.edge_index(&99), None);
}

#[test]
fn hashgraph_reverse() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();

    assert_eq!(*g.tail_index(10), 1);
    assert_eq!(*g.head_index(10), 2);

    g.reverse();

    assert_eq!(*g.tail_index(10), 2);
    assert_eq!(*g.head_index(10), 1);
}

#[test]
fn hashgraph_map_transform() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();

    let g2 = g.map(|n| n * 10, |e| e * 10);
    assert!(g2.contains_node_index(10));
    assert!(g2.contains_node_index(20));
    assert!(g2.contains_edge_index(100));
}

#[test]
fn hashgraph_empty() {
    let g = HashGraph::<u32, u32>::new();
    assert_eq!(g.len_node(), 0);
    assert_eq!(g.len_edge(), 0);
}

#[test]
fn hashgraph_self_loop() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_edge(10, [1, 1]).unwrap();

    assert_eq!(g.len_edge(), 1);
    assert_eq!(g.endpoints(10), [&1, &1]);
}

#[test]
fn hashgraph_parallel_edges() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();
    g.insert_edge(20, [1, 2]).unwrap();

    assert_eq!(g.len_edge(), 2);
}

#[test]
fn hashgraph_incoming_edges() {
    let mut g = HashGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_node(3).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();
    g.insert_edge(20, [3, 2]).unwrap();

    let incoming: Vec<_> = g.edges_to(2).collect();
    assert_eq!(incoming.len(), 2);
}

/// Map-backed graphs accept non-`Copy` keys: `NodeIx`/`EdgeIx` only need `Clone`.
fn check_string_keys<G>()
where
    G: Default + Graph<Node = String, Edge = String, NodeIx = String, EdgeIx = String>,
    G: InsertNode + InsertEdge + RemoveNode + Bigraph + StableNode + StableEdge,
    G: for<'r> Directed<'r>,
    G: GraphProperty<Endpoints = [String; 2]>,
{
    let s = |v: &str| v.to_string();
    let mut g = G::default();
    for n in ["a", "b", "c", "d"] {
        g.insert_node(s(n)).unwrap();
    }
    assert!(g.insert_node(s("a")).is_err(), "node keys are unique");
    g.insert_edge(s("ab"), [s("a"), s("b")]).unwrap();
    g.insert_edge(s("bc"), [s("b"), s("c")]).unwrap();
    g.insert_edge(s("cd"), [s("c"), s("d")]).unwrap();
    g.insert_edge(s("da"), [s("d"), s("a")]).unwrap();

    assert_eq!(Borrow::<String>::borrow(&g.tail_index(s("bc"))), "b");
    assert_eq!(Borrow::<String>::borrow(&g.head_index(s("bc"))), "c");
    let from_b: Vec<String> = g
        .edge_indices_from(s("b"))
        .map(|e| e.borrow().clone())
        .collect();
    assert_eq!(from_b, ["bc"]);
    assert!(safegraph::algo::connectivity::has_path_connecting(
        &g,
        s("a"),
        s("d")
    ));

    // Batched removal: node "a" cascades "ab" and "da"; "bc" is explicit.
    g.remove_nodes_edges([s("a")], [s("bc")]);
    let mut nodes: Vec<String> = Graph::node_indices(&g)
        .map(|n| n.borrow().clone())
        .collect();
    nodes.sort();
    assert_eq!(nodes, ["b", "c", "d"]);
    let edges: Vec<String> = Graph::edge_indices(&g)
        .map(|e| e.borrow().clone())
        .collect();
    assert_eq!(edges, ["cd"]);
    assert!(!safegraph::algo::connectivity::has_path_connecting(
        &g,
        s("b"),
        s("d")
    ));
}

#[test]
fn hashgraph_string_keys() {
    check_string_keys::<HashGraph<String, String>>();
}

#[test]
fn btreegraph_string_keys() {
    check_string_keys::<BTreeGraph<String, String>>();
}

#[test]
#[should_panic(expected = "duplicate node index")]
fn remove_nodes_edges_rejects_duplicate_nodes() {
    let mut g = BTreeGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.remove_nodes_edges([1, 1], std::iter::empty());
}

#[test]
#[should_panic(expected = "duplicate edge index")]
fn remove_nodes_edges_rejects_duplicate_edges() {
    let mut g = BTreeGraph::<u32, u32>::new();
    g.insert_node(1).unwrap();
    g.insert_node(2).unwrap();
    g.insert_edge(10, [1, 2]).unwrap();
    g.remove_nodes_edges(std::iter::empty(), [10, 10]);
}
