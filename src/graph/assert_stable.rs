//! Unsafe wrapper that adds a stability marker to any graph.
//!
//! These back [`Graph::unsafe_assert_stable_node`](super::Graph::unsafe_assert_stable_node)
//! / `…_edge` (and the `_mut`
//! variants): an algorithm written once, safely, against a `Stable*` bound runs
//! on a non-`Stable*` graph (e.g. a plain `VecGraph`) by reinterpreting `&graph`
//! as `&AssertStable<…, G>` — the caller takes on, via the `unsafe` contract, the
//! obligation that the relevant index handles stay valid for the duration of the
//! call.
//!
//! The wrapper forwards every operation to `G` unchanged. The two kinds
//! **compose**: chaining
//! `g.unsafe_assert_stable_node().unsafe_assert_stable_edge()` (in either order)
//! yields a view that is simultaneously `StableNode` and `StableEdge`, because
//! each kind forwards the *other* marker from its inner graph. The views are
//! constructed solely by the `Graph::unsafe_assert_stable_*` methods (a reference
//! reinterpretation) and never own a graph.

use core::marker::PhantomData;

use crate::graph::capability::{
    Bigraph, Directed, InsertEdge, InsertNode, RemoveEdge, RemoveNode, StableEdge, StableNode,
    UpdateEdge, UpdateNode,
};
use crate::graph::{GraphOperation, GraphProperty};

/// [`AssertStable`] kind selecting the [`StableNode`] assertion.
pub enum TNode {}

/// [`AssertStable`] kind selecting the [`StableEdge`] assertion.
pub enum TEdge {}

/// `#[repr(transparent)]` wrapper over `G` that unsafely asserts one stability
/// marker, selected by `Kind`: [`TNode`] → [`StableNode`], [`TEdge`] →
/// [`StableEdge`].
///
/// See the [module docs](self). Obtained via
/// [`Graph::unsafe_assert_stable_node`](crate::graph::Graph::unsafe_assert_stable_node)
/// / `…_edge` (and the `_mut` variants); never constructed directly.
#[repr(transparent)]
pub struct AssertStable<Kind, G: ?Sized> {
    _kind: PhantomData<Kind>,
    pub(crate) inner: G,
}

impl<Kind, G: GraphProperty + ?Sized> GraphProperty for AssertStable<Kind, G> {
    type Node = G::Node;
    type Edge = G::Edge;
    type NodeIx = G::NodeIx;
    type EdgeIx = G::EdgeIx;
    type Endpoints = G::Endpoints;
    const DIRECTED: bool = G::DIRECTED;
}

impl<'r, Kind, G> GraphOperation<'r> for AssertStable<Kind, G>
where
    G: GraphOperation<'r> + ?Sized,
{
    #[inline]
    fn contains_node_index(&self, node_ix: Self::NodeIx) -> bool {
        self.inner.contains_node_index(node_ix)
    }
    #[inline]
    fn contains_edge_index(&self, edge_ix: Self::EdgeIx) -> bool {
        self.inner.contains_edge_index(edge_ix)
    }
    #[inline]
    fn len_node(&self) -> usize {
        self.inner.len_node()
    }
    #[inline]
    fn len_edge(&self) -> usize {
        self.inner.len_edge()
    }
    #[inline]
    fn capacity_node(&self) -> Option<usize> {
        self.inner.capacity_node()
    }
    #[inline]
    fn capacity_edge(&self) -> Option<usize> {
        self.inner.capacity_edge()
    }

    type NodeIndices = <G as GraphOperation<'r>>::NodeIndices;
    type EdgeIndices = <G as GraphOperation<'r>>::EdgeIndices;

    #[inline]
    fn node_indices(&'r self) -> Self::NodeIndices {
        self.inner.node_indices()
    }
    #[inline]
    fn edge_indices(&'r self) -> Self::EdgeIndices {
        self.inner.edge_indices()
    }
    #[inline]
    unsafe fn node_unchecked(&self, node_ix: Self::NodeIx) -> &Self::Node {
        self.inner.node_unchecked(node_ix)
    }
    #[inline]
    unsafe fn edge_unchecked(&self, edge_ix: Self::EdgeIx) -> &Self::Edge {
        self.inner.edge_unchecked(edge_ix)
    }
    #[inline]
    unsafe fn endpoints_unchecked(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints {
        self.inner.endpoints_unchecked(edge_ix)
    }

    type EdgeIndicesFrom = <G as GraphOperation<'r>>::EdgeIndicesFrom;
    #[inline]
    unsafe fn edge_indices_from_unchecked(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesFrom {
        self.inner.edge_indices_from_unchecked(node_ix)
    }

    type EdgeIndicesOf = <G as GraphOperation<'r>>::EdgeIndicesOf;
    #[inline]
    unsafe fn edge_indices_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesOf {
        self.inner.edge_indices_of_unchecked(node_ix)
    }

    type WalksFrom
        = <G as GraphOperation<'r>>::WalksFrom;
    #[inline]
    unsafe fn walks_from_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksFrom {
        self.inner.walks_from_unchecked(node_ix)
    }

    type WalksOf
        = <G as GraphOperation<'r>>::WalksOf;
    #[inline]
    unsafe fn walks_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksOf {
        self.inner.walks_of_unchecked(node_ix)
    }

    type DrainNode = <G as GraphOperation<'r>>::DrainNode;
    type DrainEdge = <G as GraphOperation<'r>>::DrainEdge;

    fn drain(self) -> (Self::DrainNode, Self::DrainEdge)
    where
        Self: Sized,
    {
        // These views are only ever held by reference (via the
        // `Graph::unsafe_assert_stable_*` reinterpretation); an owned value
        // cannot be constructed outside this crate, so by-value `drain` is
        // unreachable.
        unreachable!("AssertStable view is borrowed and never drained")
    }

    fn reverse(&mut self) {
        self.inner.reverse()
    }
}

impl<'r, Kind, G> Directed<'r> for AssertStable<Kind, G>
where
    G: Directed<'r> + ?Sized,
{
    type EdgeIndicesTo = <G as Directed<'r>>::EdgeIndicesTo;
    type EdgeTailIndices = <G as Directed<'r>>::EdgeTailIndices;
    type EdgeHeadIndices = <G as Directed<'r>>::EdgeHeadIndices;
    type WalksTo
        = <G as Directed<'r>>::WalksTo;

    #[inline]
    unsafe fn walks_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksTo {
        self.inner.walks_to_unchecked(node_ix)
    }
    #[inline]
    unsafe fn edge_indices_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesTo {
        self.inner.edge_indices_to_unchecked(node_ix)
    }
    #[inline]
    unsafe fn edge_head_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeHeadIndices {
        self.inner.edge_head_indices_unchecked(edge_ix)
    }
    #[inline]
    unsafe fn edge_tail_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeTailIndices {
        self.inner.edge_tail_indices_unchecked(edge_ix)
    }
}

// SAFETY: forwards both directions to `G`'s own `Bigraph` impl unchanged.
impl<Kind, G> Bigraph for AssertStable<Kind, G>
where
    G: Bigraph + ?Sized,
{
    #[inline]
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2] {
        G::endpoints_as_array(endpoints)
    }
    #[inline]
    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints {
        G::endpoints_from_array(nodes)
    }
}

// SAFETY: the wrapper does not change the graph; the `unsafe_assert_stable_*`
// method that produces it is `unsafe` precisely so the caller asserts the
// index-stability contract that this `Kind` represents.
unsafe impl<G: GraphProperty + ?Sized> StableNode for AssertStable<TNode, G> {}
unsafe impl<G: GraphProperty + ?Sized> StableEdge for AssertStable<TEdge, G> {}

// SAFETY: each kind leaves the other index kind's behavior completely unchanged
// (it only forwards), so when the inner graph already guarantees that kind's
// stability the wrapper inherits the real guarantee unchanged.
unsafe impl<G: StableEdge + ?Sized> StableEdge for AssertStable<TNode, G> {}
unsafe impl<G: StableNode + ?Sized> StableNode for AssertStable<TEdge, G> {}

impl<Kind, G: ?Sized + InsertNode> InsertNode for AssertStable<Kind, G> {
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node> {
        <G as InsertNode>::insert_node_unchecked(&mut self.inner, node)
    }
}

impl<Kind, G: ?Sized + InsertEdge> InsertEdge for AssertStable<Kind, G> {
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge> {
        <G as InsertEdge>::insert_edge_unchecked(&mut self.inner, edge, endpoints)
    }
}

impl<'r, Kind, G> UpdateNode<'r> for AssertStable<Kind, G>
where
    G: ?Sized + UpdateNode<'r>,
    G::Edge: 'r,
{
    unsafe fn node_unchecked_mut(&mut self, node_ix: Self::NodeIx) -> &mut Self::Node {
        <G as UpdateNode<'r>>::node_unchecked_mut(&mut self.inner, node_ix)
    }

    type WalksFromMut = <G as UpdateNode<'r>>::WalksFromMut;
    unsafe fn walks_from_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksFromMut {
        <G as UpdateNode<'r>>::walks_from_unchecked_mut(&mut self.inner, node_ix)
    }

    type WalksOfMut = <G as UpdateNode<'r>>::WalksOfMut;
    unsafe fn walks_of_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksOfMut {
        <G as UpdateNode<'r>>::walks_of_unchecked_mut(&mut self.inner, node_ix)
    }
}

impl<Kind, G: ?Sized + UpdateEdge> UpdateEdge for AssertStable<Kind, G> {
    unsafe fn edge_unchecked_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge {
        <G as UpdateEdge>::edge_unchecked_mut(&mut self.inner, edge_ix)
    }
}

impl<Kind, G: ?Sized + RemoveEdge> RemoveEdge for AssertStable<Kind, G> {
    unsafe fn take_edge_unchecked(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge {
        <G as RemoveEdge>::take_edge_unchecked(&mut self.inner, edge_ix)
    }
}

impl<Kind, G: ?Sized + RemoveNode> RemoveNode for AssertStable<Kind, G> {
    unsafe fn take_node_unchecked(&mut self, node_ix: Self::NodeIx) -> Self::Node {
        <G as RemoveNode>::take_node_unchecked(&mut self.inner, node_ix)
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
        <G as RemoveNode>::take_nodes_edges_unchecked(&mut self.inner, node_indices, edge_indices)
    }
}
