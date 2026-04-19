use core::borrow::Borrow;

use super::capability::{
    Bigraph, InsertEdge, InsertNode, RemoveEdge, RemoveNode, StableEdge, StableNode, UniqueEdge,
    UniqueNode, UpdateEdge, UpdateNode,
};
use super::{GraphOperation, GraphProperty};

/// A wrapper that presents a graph as undirected.
///
/// Created by [`Graph::undirected`](super::Graph::undirected).
///
/// ```rust
/// use safegraph::graph::{Graph, GraphProperty, undirected::Undirected};
/// use safegraph::VecGraph;
///
/// type UndirectedVec = Undirected<VecGraph<u32, u32>>;
/// assert!(!UndirectedVec::DIRECTED);
/// ```
#[derive(Clone, Debug)]
pub struct Undirected<G> {
    pub(crate) inner: G,
}

impl<G> Undirected<G> {
    /// Unwraps and returns the inner graph.
    pub fn into_inner(self) -> G {
        self.inner
    }
}

impl<'r, G> GraphOperation<'r> for Undirected<G>
where
    G: GraphOperation<'r> + 'r,
    <G as GraphProperty>::Edge: 'r,
{
    fn contains_node_index(&self, node_ix: Self::NodeIx) -> bool {
        self.inner.contains_node_index(node_ix)
    }

    fn contains_edge_index(&self, edge_ix: Self::EdgeIx) -> bool {
        self.inner.contains_edge_index(edge_ix)
    }

    fn len_node(&self) -> usize {
        self.inner.len_node()
    }

    fn len_edge(&self) -> usize {
        self.inner.len_edge()
    }

    fn capacity_node(&self) -> Option<usize> {
        self.inner.capacity_node()
    }

    fn capacity_edge(&self) -> Option<usize> {
        self.inner.capacity_edge()
    }

    type NodeIndices = G::NodeIndices;
    type EdgeIndices = G::EdgeIndices;

    fn node_indices(&'r self) -> Self::NodeIndices {
        self.inner.node_indices()
    }

    fn edge_indices(&'r self) -> Self::EdgeIndices {
        self.inner.edge_indices()
    }

    unsafe fn node_unchecked(&self, node_ix: Self::NodeIx) -> &Self::Node {
        unsafe { self.inner.node_unchecked(node_ix) }
    }

    unsafe fn edge_unchecked(&self, edge_ix: Self::EdgeIx) -> &Self::Edge {
        unsafe { self.inner.edge_unchecked(edge_ix) }
    }

    unsafe fn endpoints_unchecked(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints {
        unsafe { self.inner.endpoints_unchecked(edge_ix) }
    }

    type EdgeIndicesFrom = G::EdgeIndicesOf;

    unsafe fn edge_indices_from_unchecked(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesFrom {
        unsafe { self.inner.edge_indices_of_unchecked(node_ix) }
    }

    type EdgeIndicesOf = G::EdgeIndicesOf;

    unsafe fn edge_indices_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesOf {
        unsafe { self.inner.edge_indices_of_unchecked(node_ix) }
    }

    type WalksFrom = G::WalksOf;
    type WalksOf = G::WalksOf;

    unsafe fn walks_from_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksFrom {
        self.inner.walks_of_unchecked(node_ix)
    }

    unsafe fn walks_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksOf {
        self.inner.walks_of_unchecked(node_ix)
    }

    type DrainNode = G::DrainNode;
    type DrainEdge = G::DrainEdge;

    fn drain(self) -> (Self::DrainNode, Self::DrainEdge) {
        self.inner.drain()
    }

    fn reverse(&mut self) {}
}

impl<G> GraphProperty for Undirected<G>
where
    G: GraphProperty,
{
    type Node = G::Node;
    type Edge = G::Edge;
    type NodeIx = G::NodeIx;
    type EdgeIx = G::EdgeIx;
    type Endpoints = G::Endpoints;
    // The whole point of this wrapper: present `G` as undirected.
    const DIRECTED: bool = false;
}

impl<'r, G> Bigraph for Undirected<G>
where
    G: GraphOperation<'r> + Bigraph,
{
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2] {
        G::endpoints_as_array(endpoints)
    }

    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints {
        G::endpoints_from_array(nodes)
    }
}
unsafe impl<'r, G> StableNode for Undirected<G> where G: GraphOperation<'r> + StableNode {}
unsafe impl<'r, G> StableEdge for Undirected<G> where G: GraphOperation<'r> + StableEdge {}

impl<G: InsertNode> InsertNode for Undirected<G> {
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node> {
        unsafe { self.inner.insert_node_unchecked(node) }
    }
}

impl<G: InsertEdge> InsertEdge for Undirected<G> {
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge> {
        unsafe { self.inner.insert_edge_unchecked(edge, endpoints) }
    }
}

impl<'r, G> UpdateNode<'r> for Undirected<G>
where
    G: UpdateNode<'r>,
    G::Edge: 'r,
{
    unsafe fn node_unchecked_mut(&mut self, node_ix: Self::NodeIx) -> &mut Self::Node {
        unsafe { <G as UpdateNode<'r>>::node_unchecked_mut(&mut self.inner, node_ix) }
    }

    type WalksFromMut = <G as UpdateNode<'r>>::WalksOfMut;
    unsafe fn walks_from_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksFromMut {
        unsafe { <G as UpdateNode<'r>>::walks_of_unchecked_mut(&mut self.inner, node_ix) }
    }

    type WalksOfMut = <G as UpdateNode<'r>>::WalksOfMut;
    unsafe fn walks_of_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksOfMut {
        unsafe { <G as UpdateNode<'r>>::walks_of_unchecked_mut(&mut self.inner, node_ix) }
    }
}

impl<G: UpdateEdge> UpdateEdge for Undirected<G> {
    unsafe fn edge_unchecked_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge {
        unsafe { <G as UpdateEdge>::edge_unchecked_mut(&mut self.inner, edge_ix) }
    }
}

impl<G> UniqueNode for Undirected<G>
where
    G: UniqueNode + for<'r> GraphOperation<'r>,
{
    fn node_index(&self, node: impl Borrow<Self::Node>) -> Option<Self::NodeIx> {
        <G as UniqueNode>::node_index(&self.inner, node)
    }
}

impl<G> UniqueEdge for Undirected<G>
where
    G: UniqueEdge + for<'r> GraphOperation<'r>,
{
    fn edge_index(&self, edge: impl Borrow<Self::Edge>) -> Option<Self::EdgeIx> {
        <G as UniqueEdge>::edge_index(&self.inner, edge)
    }
}

impl<G: RemoveEdge> RemoveEdge for Undirected<G> {
    unsafe fn take_edge_unchecked(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge {
        unsafe { <G as RemoveEdge>::take_edge_unchecked(&mut self.inner, edge_ix) }
    }
}

impl<G: RemoveNode> RemoveNode for Undirected<G> {
    unsafe fn take_node_unchecked(&mut self, node_ix: Self::NodeIx) -> Self::Node {
        unsafe { <G as RemoveNode>::take_node_unchecked(&mut self.inner, node_ix) }
    }

    unsafe fn take_nodes_edges_unchecked<IN, IE>(
        &mut self,
        node_indices: impl IntoIterator<Item = Self::NodeIx>,
        edge_indices: impl IntoIterator<Item = Self::EdgeIx>,
    ) -> (IN, IE)
    where
        IN: Default + Extend<Self::Node>,
        IE: Default + Extend<Self::Edge>,
    {
        unsafe {
            <G as RemoveNode>::take_nodes_edges_unchecked(
                &mut self.inner,
                node_indices,
                edge_indices,
            )
        }
    }
}
