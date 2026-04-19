use safegraph::BTreeGraph;
use safegraph::graph::Graph;
use safegraph::graph::capability::{UniqueEdge, UniqueNode};

fn leak_node_ix() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    let g: &'static BTreeGraph<u32, u32> =
        unsafe { core::mem::transmute::<&mut BTreeGraph<u32, u32>, _>(&mut g) };

    let mut leaked = None;
    g.scope(|ctx| {
        leaked = Some(ctx.node_index(&0).unwrap());
    });
    let leaked = leaked.unwrap();
    assert_eq!(*g.node(leaked.inner()), 0);
}

fn leak_edge_ix() {
    let mut g = BTreeGraph::<u32, u32>::default();
    g.insert_node(0).unwrap();
    g.insert_node(1).unwrap();
    g.insert_edge(10, [0, 1]).unwrap();
    let g: &'static BTreeGraph<u32, u32> =
        unsafe { core::mem::transmute::<&mut BTreeGraph<u32, u32>, _>(&mut g) };

    let mut leaked = None;
    g.scope(|ctx| {
        leaked = Some(ctx.edge_index(&10).unwrap());
    });
    let leaked = leaked.unwrap();
    assert_eq!(*g.edge(leaked.inner()), 10);
}

fn main() {
    leak_node_ix();
    leak_edge_ix();
}
