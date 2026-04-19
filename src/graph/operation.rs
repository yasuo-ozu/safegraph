//! [`GraphOperation`] trait and its associated iterator helpers.
//!
//! This module exposes the lifetime-parameterised graph operations that every
//! storage backend must implement, along with three small iterator adapters
//! ([`NodeRefIter`], [`EdgeRefIter`], [`NeighborIndices`]) used in default
//! implementations of [`GraphOperation`] / [`Graph`].
//!
//! [`Graph`]: super::Graph

use super::walk_item::WalkItem;
use super::GraphProperty;

pub struct NeighborIndices<N> {
    pub(crate) iter: std::vec::IntoIter<N>,
}

impl<N> Iterator for NeighborIndices<N> {
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct NodeRefIter<'r, I, G>(pub(crate) &'r G, pub(crate) I)
where
    G: GraphOperation<'r> + ?Sized,
    I: Iterator<Item = G::NodeIx>;

impl<'r, I, G> Iterator for NodeRefIter<'r, I, G>
where
    G: GraphOperation<'r> + ?Sized,
    I: Iterator<Item = G::NodeIx>,
{
    type Item = &'r G::Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|ix| {
            // SAFETY: index comes from graph-derived iterators.
            unsafe { self.0.node_unchecked(ix) }
        })
    }
}

pub struct EdgeRefIter<'r, I, G>(pub(crate) &'r G, pub(crate) I)
where
    G: GraphOperation<'r> + ?Sized,
    I: Iterator<Item = G::EdgeIx>;

impl<'r, I, G> Iterator for EdgeRefIter<'r, I, G>
where
    G: GraphOperation<'r> + ?Sized,
    I: Iterator<Item = G::EdgeIx>,
{
    type Item = &'r G::Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|ix| {
            // SAFETY: index comes from graph-derived iterators.
            unsafe { self.0.edge_unchecked(ix) }
        })
    }
}

/// Lifetime-parameterized graph operations.
///
/// Provides the required per-implementation operations on a graph.
/// The lifetime `'r` ties borrowed iterators and references to the
/// graph's borrow scope.
///
/// Types that implement `GraphOperation<'r>` for all lifetimes automatically
/// get the [`Graph`](super::Graph) trait, which provides whole-graph operations
/// and convenience methods.
///
/// Do not call this trait directly. Use methods exposed from
/// [`Graph`](super::Graph) instead.
pub trait GraphOperation<'r>: GraphProperty {
    /// Returns `true` if `node_ix` refers to a live node in this graph.
    fn contains_node_index(&self, node_ix: Self::NodeIx) -> bool;
    /// Returns `true` if `edge_ix` refers to a live edge in this graph.
    fn contains_edge_index(&self, edge_ix: Self::EdgeIx) -> bool;

    /// Returns the number of live nodes in the graph.
    fn len_node(&self) -> usize;
    /// Returns the number of live edges in the graph.
    fn len_edge(&self) -> usize;
    /// Returns the total node capacity, if applicable (e.g., `Vec`-backed graphs).
    /// `None` means the concept of fixed capacity does not apply.
    fn capacity_node(&self) -> Option<usize> {
        None
    }
    /// Returns the total edge capacity, if applicable.
    fn capacity_edge(&self) -> Option<usize> {
        None
    }

    /// Iterator type returned by [`node_indices`](Self::node_indices).
    type NodeIndices: Iterator<Item = Self::NodeIx>;
    /// Iterator type returned by [`edge_indices`](Self::edge_indices).
    type EdgeIndices: Iterator<Item = Self::EdgeIx>;

    /// Returns an iterator over every node index currently in the graph.
    fn node_indices(&'r self) -> Self::NodeIndices;

    /// Returns an iterator over every edge index currently in the graph.
    fn edge_indices(&'r self) -> Self::EdgeIndices;

    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn node_unchecked(&self, node_ix: Self::NodeIx) -> &Self::Node;
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_unchecked(&self, edge_ix: Self::EdgeIx) -> &Self::Edge;

    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn endpoints_unchecked(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints;

    /// Iterator type returned by [`edge_indices_from_unchecked`](Self::edge_indices_from_unchecked).
    type EdgeIndicesFrom: Iterator<Item = Self::EdgeIx>;

    /// Returns edges starting from `node_ix`.
    /// For directed graphs this returns outgoing edges only.
    /// For undirected graphs this returns all connecting edges.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edge_indices_from_unchecked(&'r self, node_ix: Self::NodeIx)
        -> Self::EdgeIndicesFrom;

    /// Iterator type returned by [`edge_indices_of_unchecked`](Self::edge_indices_of_unchecked).
    type EdgeIndicesOf: Iterator<Item = Self::EdgeIx>;

    /// Returns all edges connected with `node_ix` (both directions for directed graphs).
    ///
    /// Each incident edge is yielded exactly once, including a self-loop
    /// (`[node_ix, node_ix]`) — it appears once, not twice.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edge_indices_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesOf;

    /// Type returned by [`walks_from_unchecked`](Self::walks_from_unchecked).
    ///
    /// Each item is a [`WalkItem`] wrapping `(EdgeIx, &Edge, NodeIx)` — an
    /// edge, its data, and the neighbor node at the other end. For hypergraphs
    /// the same edge may appear multiple times (once per endpoint that is not
    /// the source node). The item is lifetime-erased (see [`WalkItem`]); deref
    /// it to obtain the borrowed tuple.
    type WalksFrom: Iterator<Item = WalkItem<'r, Self::EdgeIx, Self::Edge, Self::NodeIx>>;

    /// Returns (EdgeIx, &Edge, NodeIx) triples for outgoing edges.
    ///
    /// For directed graphs, follows outgoing edges. For undirected, all edges.
    /// Self-loops will also be visited once.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_from_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksFrom;

    /// Type returned by [`walks_of_unchecked`](Self::walks_of_unchecked).
    ///
    /// Each item is a [`WalkItem`] wrapping `(EdgeIx, &Edge, NodeIx)`; see
    /// [`WalksFrom`](Self::WalksFrom).
    type WalksOf: Iterator<Item = WalkItem<'r, Self::EdgeIx, Self::Edge, Self::NodeIx>>;

    /// Returns (EdgeIx, &Edge, NodeIx) triples for all incident edges.
    ///
    /// Includes both incoming and outgoing edges for directed graphs. Each
    /// incident edge is yielded exactly once, including a self-loop (yielded
    /// once, with the node itself as the neighbor) — never twice.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksOf;

    /// Iterator type for drained nodes returned by [`drain`](Self::drain).
    type DrainNode: Iterator<Item = Self::Node>;
    /// Iterator type for drained edges returned by [`drain`](Self::drain).
    type DrainEdge: Iterator<Item = Self::Edge>;

    /// Consume this graph, returning iterators over all node and edge data.
    ///
    /// The node iterator yields items in the same order as
    /// [`node_indices`](Self::node_indices) and the edge
    /// iterator yields items in the same order as
    /// [`edge_indices`](Self::edge_indices).
    fn drain(self) -> (Self::DrainNode, Self::DrainEdge)
    where
        Self: Sized;

    /// Reverse all edge directions in the graph.
    fn reverse(&mut self);
}
