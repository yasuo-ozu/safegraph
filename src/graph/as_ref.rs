//! Reference-borrowing graph wrapper.
//!
//! [`Graph::as_ref`](super::Graph::as_ref) hands out `&AsRef<G>` (an `impl
//! Graph` reachable by shared reference);
//! [`Graph::as_mut`](super::Graph::as_mut) hands out `&mut AsRef<G>`, a fully
//! mutable graph — `AsRef` forwards every mutation/lookup capability
//! (`Insert*`/`Update*`/`Remove*`/`Unique*`) to the wrapped `G`.

use core::borrow::Borrow;

use super::capability::{
    Bigraph, Directed, InsertEdge, InsertNode, RemoveEdge, RemoveNode, StableEdge, StableNode,
    UniqueEdge, UniqueNode, UpdateEdge, UpdateNode,
};
use super::{GraphOperation, GraphProperty};

/// See [`Graph::as_ref`](super::Graph::as_ref),
/// [`Graph::as_mut`](super::Graph::as_mut)
#[repr(transparent)]
pub struct AsRef<G: ?Sized>(pub(crate) G);

impl<G: ?Sized + GraphProperty> GraphProperty for AsRef<G> {
    type Node = G::Node;
    type Edge = G::Edge;
    type NodeIx = G::NodeIx;
    type EdgeIx = G::EdgeIx;
    type Endpoints = G::Endpoints;
    const DIRECTED: bool = G::DIRECTED;
}

impl<'a, G> GraphOperation<'a> for AsRef<G>
where
    G: ?Sized + for<'b> GraphOperation<'b>,
{
    fn contains_node_index(&self, node_ix: Self::NodeIx) -> bool {
        <G as GraphOperation<'_>>::contains_node_index(&self.0, node_ix)
    }

    fn contains_edge_index(&self, edge_ix: Self::EdgeIx) -> bool {
        <G as GraphOperation<'_>>::contains_edge_index(&self.0, edge_ix)
    }

    fn len_node(&self) -> usize {
        <G as GraphOperation<'_>>::len_node(&self.0)
    }

    fn len_edge(&self) -> usize {
        <G as GraphOperation<'_>>::len_edge(&self.0)
    }

    fn capacity_node(&self) -> Option<usize> {
        <G as GraphOperation<'_>>::capacity_node(&self.0)
    }

    fn capacity_edge(&self) -> Option<usize> {
        <G as GraphOperation<'_>>::capacity_edge(&self.0)
    }

    type NodeIndices = <G as GraphOperation<'a>>::NodeIndices;
    type EdgeIndices = <G as GraphOperation<'a>>::EdgeIndices;

    fn node_indices(&'a self) -> Self::NodeIndices {
        <G as GraphOperation<'a>>::node_indices(&self.0)
    }

    fn edge_indices(&'a self) -> Self::EdgeIndices {
        <G as GraphOperation<'a>>::edge_indices(&self.0)
    }

    unsafe fn node_unchecked(&self, node_ix: Self::NodeIx) -> &Self::Node {
        <G as GraphOperation<'_>>::node_unchecked(&self.0, node_ix)
    }

    unsafe fn edge_unchecked(&self, edge_ix: Self::EdgeIx) -> &Self::Edge {
        <G as GraphOperation<'_>>::edge_unchecked(&self.0, edge_ix)
    }

    unsafe fn endpoints_unchecked(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints {
        <G as GraphOperation<'_>>::endpoints_unchecked(&self.0, edge_ix)
    }

    type EdgeIndicesFrom = <G as GraphOperation<'a>>::EdgeIndicesFrom;

    unsafe fn edge_indices_from_unchecked(
        &'a self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesFrom {
        <G as GraphOperation<'a>>::edge_indices_from_unchecked(&self.0, node_ix)
    }

    type EdgeIndicesOf = <G as GraphOperation<'a>>::EdgeIndicesOf;

    unsafe fn edge_indices_of_unchecked(&'a self, node_ix: Self::NodeIx) -> Self::EdgeIndicesOf {
        <G as GraphOperation<'a>>::edge_indices_of_unchecked(&self.0, node_ix)
    }

    type WalksFrom
        = <G as GraphOperation<'a>>::WalksFrom;

    unsafe fn walks_from_unchecked(&'a self, node_ix: Self::NodeIx) -> Self::WalksFrom {
        <G as GraphOperation<'a>>::walks_from_unchecked(&self.0, node_ix)
    }

    type WalksOf
        = <G as GraphOperation<'a>>::WalksOf;

    unsafe fn walks_of_unchecked(&'a self, node_ix: Self::NodeIx) -> Self::WalksOf {
        <G as GraphOperation<'a>>::walks_of_unchecked(&self.0, node_ix)
    }

    type DrainNode = <G as GraphOperation<'a>>::DrainNode;
    type DrainEdge = <G as GraphOperation<'a>>::DrainEdge;

    fn drain(self) -> (Self::DrainNode, Self::DrainEdge)
    where
        Self: Sized,
    {
        // `AsRef` is only ever held by reference (via the `Graph::as_ref`
        // reinterpretation); an owned value cannot be constructed outside this
        // crate, so the by-value `drain` is unreachable.
        unreachable!("AsRef view is borrowed and never drained")
    }

    fn reverse(&mut self) {
        self.0.reverse()
    }
}

// SAFETY: AsRef forwards to `G`; if the underlying graph guarantees a stability
// marker then the wrapper inherits the real guarantee unchanged.
unsafe impl<G: ?Sized + GraphProperty + StableNode> StableNode for AsRef<G> {}
unsafe impl<G: ?Sized + GraphProperty + StableEdge> StableEdge for AsRef<G> {}

// SAFETY: forwards both directions to `G`'s own `Bigraph` impl unchanged.
impl<G: ?Sized + Bigraph> Bigraph for AsRef<G> {
    #[inline]
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2] {
        G::endpoints_as_array(endpoints)
    }
    #[inline]
    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints {
        G::endpoints_from_array(nodes)
    }
}

impl<'r, G> Directed<'r> for AsRef<G>
where
    G: ?Sized + Directed<'r>,
{
    type EdgeIndicesTo = <G as Directed<'r>>::EdgeIndicesTo;
    type EdgeTailIndices = <G as Directed<'r>>::EdgeTailIndices;
    type EdgeHeadIndices = <G as Directed<'r>>::EdgeHeadIndices;
    type WalksTo
        = <G as Directed<'r>>::WalksTo;

    #[inline]
    unsafe fn walks_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksTo {
        <G as Directed<'r>>::walks_to_unchecked(&self.0, node_ix)
    }
    #[inline]
    unsafe fn edge_indices_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesTo {
        <G as Directed<'r>>::edge_indices_to_unchecked(&self.0, node_ix)
    }
    #[inline]
    unsafe fn edge_head_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeHeadIndices {
        <G as Directed<'r>>::edge_head_indices_unchecked(&self.0, edge_ix)
    }
    #[inline]
    unsafe fn edge_tail_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeTailIndices {
        <G as Directed<'r>>::edge_tail_indices_unchecked(&self.0, edge_ix)
    }
}

impl<G: ?Sized + InsertNode> InsertNode for AsRef<G> {
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node> {
        <G as InsertNode>::insert_node_unchecked(&mut self.0, node)
    }
}

impl<G: ?Sized + InsertEdge> InsertEdge for AsRef<G> {
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge> {
        <G as InsertEdge>::insert_edge_unchecked(&mut self.0, edge, endpoints)
    }
}

impl<'r, G> UpdateNode<'r> for AsRef<G>
where
    G: ?Sized + UpdateNode<'r>,
    G::Edge: 'r,
{
    unsafe fn node_unchecked_mut(&mut self, node_ix: Self::NodeIx) -> &mut Self::Node {
        <G as UpdateNode<'r>>::node_unchecked_mut(&mut self.0, node_ix)
    }

    type WalksFromMut = <G as UpdateNode<'r>>::WalksFromMut;
    unsafe fn walks_from_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksFromMut {
        <G as UpdateNode<'r>>::walks_from_unchecked_mut(&mut self.0, node_ix)
    }

    type WalksOfMut = <G as UpdateNode<'r>>::WalksOfMut;
    unsafe fn walks_of_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksOfMut {
        <G as UpdateNode<'r>>::walks_of_unchecked_mut(&mut self.0, node_ix)
    }
}

impl<G: ?Sized + UpdateEdge> UpdateEdge for AsRef<G> {
    unsafe fn edge_unchecked_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge {
        <G as UpdateEdge>::edge_unchecked_mut(&mut self.0, edge_ix)
    }
}

impl<G: ?Sized + UniqueNode> UniqueNode for AsRef<G> {
    fn node_index(&self, node: impl Borrow<Self::Node>) -> Option<Self::NodeIx> {
        <G as UniqueNode>::node_index(&self.0, node)
    }
}

impl<G: ?Sized + UniqueEdge> UniqueEdge for AsRef<G> {
    fn edge_index(&self, edge: impl Borrow<Self::Edge>) -> Option<Self::EdgeIx> {
        <G as UniqueEdge>::edge_index(&self.0, edge)
    }
}

impl<G: ?Sized + RemoveEdge> RemoveEdge for AsRef<G> {
    unsafe fn take_edge_unchecked(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge {
        <G as RemoveEdge>::take_edge_unchecked(&mut self.0, edge_ix)
    }
}

impl<G: ?Sized + RemoveNode> RemoveNode for AsRef<G> {
    unsafe fn take_node_unchecked(&mut self, node_ix: Self::NodeIx) -> Self::Node {
        <G as RemoveNode>::take_node_unchecked(&mut self.0, node_ix)
    }

    // Forward the batch removal so the inner backend's fast path (if any) is kept.
    unsafe fn take_nodes_edges_unchecked<IN, IE>(
        &mut self,
        node_indices: impl IntoIterator<Item = Self::NodeIx>,
        edge_indices: impl IntoIterator<Item = Self::EdgeIx>,
    ) -> (IN, IE)
    where
        IN: Default + Extend<Self::Node>,
        IE: Default + Extend<Self::Edge>,
    {
        <G as RemoveNode>::take_nodes_edges_unchecked(&mut self.0, node_indices, edge_indices)
    }
}
