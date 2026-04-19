//! Capability traits that describe what operations a graph backend supports.
//!
//! These traits are **not intended for direct use by library users.** The
//! [`Graph`](super::Graph) facade re-exposes every operation with
//! index-checking assertions and ergonomic signatures. Prefer calling
//! methods on `Graph` (e.g. [`Graph::insert_node`](super::Graph::insert_node),
//! [`Graph::take_node`](super::Graph::take_node)) rather than invoking
//! these traits directly.
//!
//! Graph backends implement the capability traits that match their storage
//! guarantees; the `Graph` blanket impl then wires the safe convenience
//! layer on top.

use core::borrow::Borrow;

use super::walk_item::{WalkItemMut, WalkItemTo};
use super::GraphProperty;

/// Mutable access to node data and incident-edge traversal with mutable edge
/// references.
///
/// Use [`Graph::walks_from_mut`](super::Graph::walks_from_mut) /
/// [`Graph::walks_of_mut`](super::Graph::walks_of_mut) instead of calling
/// these methods directly.
pub trait UpdateNode<'r>: GraphProperty {
    /// Returns a mutable reference to the node data at `node_ix`.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn node_unchecked_mut(&mut self, node_ix: Self::NodeIx) -> &mut Self::Node;

    /// see [`UpdateNode::walks_of_unchecked_mut()`]
    type WalksFromMut: Iterator<Item = WalkItemMut<'r, Self::EdgeIx, Self::Edge, Self::NodeIx>>;

    /// Mutable counterpart of
    /// [`walks_from_unchecked`](super::GraphOperation::walks_from_unchecked).
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_from_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksFromMut;

    /// see [`UpdateNode::walks_of_unchecked_mut()`]
    type WalksOfMut: Iterator<Item = WalkItemMut<'r, Self::EdgeIx, Self::Edge, Self::NodeIx>>;

    /// Mutable counterpart of
    /// [`walks_of_unchecked`](super::GraphOperation::walks_of_unchecked).
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_of_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksOfMut;
}

/// Mutable access to edge data.
///
/// Use [`Graph::edge_mut`](super::Graph::edge_mut) (index-checked) or
/// [`Graph::edge_unchecked_mut`](super::Graph::edge_unchecked_mut) instead
/// of calling this trait directly.
pub trait UpdateEdge: GraphProperty {
    /// Returns a mutable reference to the edge data at `edge_ix`.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_unchecked_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge;
}

/// Marker: node indices are stable across mutations.
///
/// A `NodeIx` returned by this graph always refers to the same node for
/// the lifetime of the graph. Removing a node invalidates only that index
/// (`contains_node_index` returns `false`); all other node indices remain
/// valid.
///
/// Map-backed graphs (e.g. `BTreeGraph`, `HashGraph`) and
/// [`Stabilized`](super::stabilized::Stabilized) implement this
/// automatically. Vec-backed graphs do not, because swap-remove can
/// relocate the last element. Use
/// [`Graph::scope`](super::Graph::scope) or
/// [`Graph::unsafe_assert_stable_node`](super::Graph::unsafe_assert_stable_node)
/// to satisfy `StableNode` bounds on non-stable graphs.
///
/// # Safety
/// Implementor must guarantee that a `NodeIx` always refers to the same
/// node for the lifetime of the graph. If the node has been removed,
/// `contains_node_index()` must return `false`.
pub unsafe trait StableNode: GraphProperty {}

/// Marker: edge indices are stable across mutations.
///
/// An `EdgeIx` returned by this graph always refers to the same edge for
/// the lifetime of the graph. Removing an edge invalidates only that index
/// (`contains_edge_index` returns `false`); all other edge indices remain
/// valid.
///
/// See [`StableNode`] for the analogous node guarantee and the list of
/// graphs that implement it. Use
/// [`Graph::unsafe_assert_stable_edge`](super::Graph::unsafe_assert_stable_edge)
/// to satisfy this bound on non-stable graphs.
///
/// # Safety
/// Implementor must guarantee that an `EdgeIx` always refers to the same
/// edge for the lifetime of the graph. If the edge has been removed,
/// `contains_edge_index()` must return `false`.
pub unsafe trait StableEdge: GraphProperty {}

/// Ability to insert new nodes into the graph.
///
/// Use [`Graph::insert_node`](super::Graph::insert_node) (index-checked)
/// or [`Graph::push`](super::Graph::push) (discards the returned index)
/// instead of calling this trait directly.
///
/// Insertion returns `Err(node)` when the graph rejects the value (e.g.
/// [`UniqueNode`] graphs that already contain an equal node).
pub trait InsertNode: GraphProperty {
    /// Inserts `node` into the graph and returns its index.
    ///
    /// Returns `Err(node)` if the graph rejects the insertion (e.g. a
    /// duplicate in a [`UniqueNode`] graph).
    ///
    /// # Safety
    /// The returned `NodeIx` must not be used after the graph is modified,
    /// unless the graph also implements [`StableNode`].
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node>;
}

/// Ability to insert new edges into the graph.
///
/// Use [`Graph::insert_edge`](super::Graph::insert_edge) (index-checked)
/// or [`Graph::push_edge`](super::Graph::push_edge) (discards the returned
/// index) instead of calling this trait directly.
///
/// Insertion returns `Err(edge)` when the graph rejects the value (e.g.
/// [`UniqueEdge`] graphs that already contain an equal edge).
pub trait InsertEdge: GraphProperty {
    /// Inserts `edge` connecting `endpoints` and returns its index.
    ///
    /// Returns `Err(edge)` if the graph rejects the insertion (e.g. a
    /// duplicate in a [`UniqueEdge`] graph).
    ///
    /// # Safety
    /// `endpoints` must contain valid node indices currently held by this
    /// graph. The returned `EdgeIx` must not be used after the graph is
    /// modified, unless the graph also implements [`StableEdge`].
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge>;
}

/// Node values are unique: at most one node holds any given value.
///
/// This enables value-to-index lookup. Use
/// [`Graph::get_or_insert_node`](super::Graph::get_or_insert_node) instead
/// of calling [`node_index`](UniqueNode::node_index) directly.
///
/// Requires [`StableNode`] because the returned index must remain valid
/// for the caller to use.
///
/// Implemented automatically by map-backed graphs (`BTreeGraph`,
/// `HashGraph`) where node values serve as map keys.
pub trait UniqueNode: StableNode {
    /// Returns the index of the node whose value equals `node`, or `None`
    /// if no such node exists.
    fn node_index(&self, node: impl Borrow<Self::Node>) -> Option<Self::NodeIx>;
}

/// Edge values are unique: at most one edge holds any given value.
///
/// This enables value-to-index lookup. Use
/// [`Graph::get_or_insert_edge`](super::Graph::get_or_insert_edge) instead
/// of calling [`edge_index`](UniqueEdge::edge_index) directly.
///
/// Requires [`StableEdge`] because the returned index must remain valid
/// for the caller to use.
///
/// Implemented automatically by map-backed graphs (`BTreeGraph`,
/// `HashGraph`) where edge values serve as map keys.
pub trait UniqueEdge: StableEdge {
    /// Returns the index of the edge whose value equals `edge`, or `None`
    /// if no such edge exists.
    fn edge_index(&self, edge: impl Borrow<Self::Edge>) -> Option<Self::EdgeIx>;
}

// ---- Directed / Bigraph ----

/// A directed graph: edges have a source (tail) and target (head)
/// distinction.
///
/// Use the [`Graph`](super::Graph) convenience methods instead of calling
/// this trait directly:
/// [`walks_to`](super::Graph::walks_to),
/// [`edge_indices_to`](super::Graph::edge_indices_to),
/// [`edge_tail_indices`](super::Graph::edge_tail_indices),
/// [`edge_head_indices`](super::Graph::edge_head_indices).
pub trait Directed<'r>: GraphProperty {
    /// Iterator over edge indices incoming to a node.
    type EdgeIndicesTo: Iterator<Item = Self::EdgeIx>;
    /// Iterator over the tail (source) node indices of an edge.
    type EdgeTailIndices: Iterator<Item = Self::NodeIx>;
    /// Iterator over the head (target) node indices of an edge.
    type EdgeHeadIndices: Iterator<Item = Self::NodeIx>;

    /// Iterator over [`WalkItemTo`] wrapping `(source_node, edge, &edge_data)`
    /// triples for incoming edges; deref to obtain the borrowed tuple.
    type WalksTo: Iterator<Item = WalkItemTo<'r, Self::NodeIx, Self::EdgeIx, Self::Edge>>;

    /// Returns incoming walks to `node_ix`.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksTo;

    /// Returns edge indices incoming to `node_ix`.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edge_indices_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesTo;

    /// Returns the head (target) node indices of `edge_ix`.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_head_indices_unchecked(&'r self, edge_ix: Self::EdgeIx)
        -> Self::EdgeHeadIndices;

    /// Returns the tail (source) node indices of `edge_ix`.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_tail_indices_unchecked(&'r self, edge_ix: Self::EdgeIx)
        -> Self::EdgeTailIndices;
}

/// A bigraph (binary graph): each edge connects exactly two nodes.
///
/// Combined with [`Directed`], enables single-node
/// [`Graph::edge_head`](super::Graph::edge_head) /
/// [`Graph::edge_tail`](super::Graph::edge_tail) accessors on the
/// [`Graph`](super::Graph) facade.
pub trait Bigraph: GraphProperty {
    /// Convert an [`Endpoints`](super::GraphProperty::Endpoints) value into a
    /// `[tail, head]` array.
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2];

    /// Construct an [`Endpoints`](super::GraphProperty::Endpoints) value from a
    /// `[tail, head]` array.
    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints;
}

/// Ability to remove nodes (and their incident edges) from the graph.
///
/// Use [`Graph::take_node`](super::Graph::take_node) (index-checked,
/// returns data) or
/// [`Graph::remove_node`](super::Graph::remove_node) (index-checked,
/// discards data) instead of calling this trait directly.
///
/// Removing a node always removes all edges incident to it, so this trait
/// requires [`RemoveEdge`].
pub trait RemoveNode: RemoveEdge {
    /// Removes the node at `node_ix` and returns its data.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn take_node_unchecked(&mut self, node_ix: Self::NodeIx) -> Self::Node;

    /// Removes the node at `node_ix`, discarding its data.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn remove_node_unchecked(&mut self, node_ix: Self::NodeIx)
    where
        Self: RemoveNode,
    {
        let _ = <Self as RemoveNode>::take_node_unchecked(self, node_ix);
    }

    /// Batch-removes selected nodes and edges, returning removed payloads.
    ///
    /// Edges are removed first via
    /// [`take_edge_unchecked`](RemoveEdge::take_edge_unchecked), then nodes
    /// via [`take_node_unchecked`](Self::take_node_unchecked). Backends
    /// with cheaper batch removal may override.
    ///
    /// # Safety
    /// All `node_indices` and `edge_indices` must be valid indices currently
    /// held by this graph, and neither sequence may contain duplicates (each
    /// payload can be moved out only once). Implementations must ensure that
    /// removing one index does not invalidate any remaining index in the
    /// batch.
    unsafe fn take_nodes_edges_unchecked<IN, IE>(
        &mut self,
        node_indices: impl IntoIterator<Item = Self::NodeIx>,
        edge_indices: impl IntoIterator<Item = Self::EdgeIx>,
    ) -> (IN, IE)
    where
        IN: Default + Extend<Self::Node>,
        IE: Default + Extend<Self::Edge>,
    {
        let mut nodes_out = IN::default();
        let mut edges_out = IE::default();
        for eix in edge_indices {
            let e = unsafe { <Self as RemoveEdge>::take_edge_unchecked(self, eix) };
            edges_out.extend(core::iter::once(e));
        }
        for nix in node_indices {
            let v = unsafe { <Self as RemoveNode>::take_node_unchecked(self, nix) };
            nodes_out.extend(core::iter::once(v));
        }
        (nodes_out, edges_out)
    }
}

/// Ability to remove edges from the graph.
///
/// Use [`Graph::take_edge`](super::Graph::take_edge) (index-checked,
/// returns data) or
/// [`Graph::remove_edge`](super::Graph::remove_edge) (index-checked,
/// discards data) instead of calling this trait directly.
pub trait RemoveEdge: GraphProperty {
    /// Removes the edge at `edge_ix` and returns its data.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn take_edge_unchecked(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge;

    /// Removes the edge at `edge_ix`, discarding its data.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn remove_edge_unchecked(&mut self, edge_ix: Self::EdgeIx)
    where
        Self: RemoveEdge,
    {
        let _ = <Self as RemoveEdge>::take_edge_unchecked(self, edge_ix);
    }
}
