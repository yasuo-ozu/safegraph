//! Parametric test harness for hypergraph backends.
//!
//! Hypergraph endpoints are set-shaped, not binary, so the binary-graph
//! harness in `tests/raw_graph.rs` does not apply directly. The [`backend!`]
//! macro registers each alias as a single `#[test]` delegating to
//! [`run_all`].

use std::collections::BTreeSet;

use safegraph::graph::capability::{InsertEdge, InsertNode, RemoveNode};
use safegraph::graph::context::NodeIx as ScopedNIx;
use safegraph::graph::edge::{Endpoints, Map};
use safegraph::graph::prelude::*;
use safegraph::HyperGraph;

fn run_all<G>()
where
    G: Default + 'static,
    G: Graph + InsertNode + InsertEdge + RemoveNode,
    G: GraphProperty<Node = u32, Edge = u32>,
    G::Endpoints: for<'scope> Map<ScopedNIx<'scope, <G as GraphProperty>::NodeIx>>,
    G::NodeIx: std::fmt::Debug + Ord,
    G::EdgeIx: std::fmt::Debug + Ord,
{
    // Scenario 1: read-only checks on a graph with one 3-edge and one binary
    // hyperedge.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            let a = ctx.insert_node(10).expect("insert a");
            let b = ctx.insert_node(20).expect("insert b");
            let c = ctx.insert_node(30).expect("insert c");
            let d = ctx.insert_node(40).expect("insert d");

            let ep_e0 = <_ as Endpoints>::try_from_node_indices([a, b, c]).expect("ep_e0");
            let ep_e1 = <_ as Endpoints>::try_from_node_indices([b, d]).expect("ep_e1");
            let e0 = ctx.insert_edge(100, ep_e0).expect("insert e0");
            let e1 = ctx.insert_edge(200, ep_e1).expect("insert e1");

            // Counts and individual lookups.
            assert_eq!(ctx.nodes().count(), 4);
            assert_eq!(ctx.edges().count(), 2);
            assert_eq!(*ctx.node(a), 10);
            assert_eq!(*ctx.node(d), 40);
            assert_eq!(*ctx.edge(e0), 100);
            assert_eq!(*ctx.edge(e1), 200);

            // Endpoints come back as a set: order is not meaningful.
            let ep0: BTreeSet<_> = ctx.endpoints(e0).into_iter().collect();
            let want_e0: BTreeSet<_> = [a, b, c].into_iter().collect();
            assert_eq!(ep0, want_e0);
            let ep1: BTreeSet<_> = ctx.endpoints(e1).into_iter().collect();
            let want_e1: BTreeSet<_> = [b, d].into_iter().collect();
            assert_eq!(ep1, want_e1);

            // `edge_indices_from` lists incident edges (each yielded once).
            let from_a: BTreeSet<_> = ctx.edge_indices_from(a).collect();
            assert_eq!(from_a, [e0].into_iter().collect::<BTreeSet<_>>());
            let from_b: BTreeSet<_> = ctx.edge_indices_from(b).collect();
            assert_eq!(from_b, [e0, e1].into_iter().collect::<BTreeSet<_>>());
            let from_d: BTreeSet<_> = ctx.edge_indices_from(d).collect();
            assert_eq!(from_d, [e1].into_iter().collect::<BTreeSet<_>>());

            // `walks_from` yields a hyperedge once per OTHER endpoint, so the
            // 3-edge incident on `a` shows up twice (paired with b and c).
            let walks_a: Vec<_> = ctx
                .walks_from(a)
                .map(|w| w.get())
                .map(|(eix, _, nix)| (eix, nix))
                .collect();
            assert_eq!(walks_a.len(), 2);
            for (eix, _) in &walks_a {
                assert_eq!(*eix, e0);
            }
            let neighbors: BTreeSet<_> = walks_a.iter().map(|(_, nix)| *nix).collect();
            assert_eq!(neighbors, [b, c].into_iter().collect::<BTreeSet<_>>());
        });
    }

    // Scenario 2: removing one hyperedge drops the edge count without
    // touching any node count.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            let a = ctx.insert_node(10).unwrap();
            let b = ctx.insert_node(20).unwrap();
            let c = ctx.insert_node(30).unwrap();
            let d = ctx.insert_node(40).unwrap();
            let ep_e0 = <_ as Endpoints>::try_from_node_indices([a, b, c]).unwrap();
            let ep_e1 = <_ as Endpoints>::try_from_node_indices([b, d]).unwrap();
            let e0 = ctx.insert_edge(100, ep_e0).unwrap();
            let _e1 = ctx.insert_edge(200, ep_e1).unwrap();
            ctx.remove_nodes_edges(None, Some(e0));
        });
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (4, 1), "after removing one hyperedge");
    }

    // Scenario 3: removing a node cascades to every hyperedge that included
    // it. `b` participates in both edges, so both are dropped.
    {
        let mut g = G::default();
        g.scope_mut(|mut ctx| {
            let a = ctx.insert_node(10).unwrap();
            let b = ctx.insert_node(20).unwrap();
            let c = ctx.insert_node(30).unwrap();
            let d = ctx.insert_node(40).unwrap();
            let ep_e0 = <_ as Endpoints>::try_from_node_indices([a, b, c]).unwrap();
            let ep_e1 = <_ as Endpoints>::try_from_node_indices([b, d]).unwrap();
            let _e0 = ctx.insert_edge(100, ep_e0).unwrap();
            let _e1 = ctx.insert_edge(200, ep_e1).unwrap();
            ctx.remove_nodes_edges(Some(b), None);
        });
        let (n, e) = g.scope(|ctx| (ctx.nodes().count(), ctx.edges().count()));
        assert_eq!((n, e), (3, 0), "after removing shared node");
    }
}

macro_rules! backend {
    ($name:ident, $G:ty) => {
        #[test]
        fn $name() {
            run_all::<$G>();
        }
    };
}

// Map-backed hypergraph aliases (`StableHyperGraph` / `HashHyperGraph`) don't
// currently impl `InsertEdge` (the trait demands `NC: UpdatableRandomAccess`,
// which only Vec-backed collections satisfy), so only the Vec-backed alias is
// exercised here.
backend!(seq_hash_hyper_graph, HyperGraph<u32, u32>);
