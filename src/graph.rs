use std::fmt::Display;
use std::hash::Hash;

pub mod as_ref;
pub mod assert_stable;
pub mod capability;
pub mod context;
pub mod edge;
pub mod operation;
pub mod prelude;
pub mod stabilized;
pub mod undirected;
pub mod walk_item;

use assert_stable::{AssertStable, TEdge, TNode};
use capability::*;
use edge::Endpoints;
pub use operation::{EdgeRefIter, GraphOperation, NeighborIndices, NodeRefIter};

type StabilizedGraph<'r, G, N, E> = stabilized::Stabilized<
    <G as GraphMap<'r, stabilized::NodeIx<N>, stabilized::EdgeIx<E>>>::Mapped,
    N,
    E,
>;

/// Property for [`Graph`].
///
/// This trait is inherited by [`Graph`], so use `T: Graph` boundary instead
/// of `T: GraphProperty`.
pub trait GraphProperty {
    /// The data stored in each node.
    type Node;
    /// The data stored in each edge.
    type Edge;
    /// A lightweight handle that identifies a node within the graph.
    type NodeIx: Copy + Eq + Ord + Hash + Display + std::fmt::Debug;
    /// A lightweight handle that identifies an edge within the graph.
    type EdgeIx: Copy + Eq + Ord + Hash + Display + std::fmt::Debug;
    /// The collection of node indices that form an edge's endpoints.
    type Endpoints: Endpoints<NodeIx = Self::NodeIx>;
    /// Whether this graph is directed (`true`) or undirected (`false`).
    ///
    /// A [`Directed`] graph sets this to `true`; the
    /// [`Undirected`](undirected::Undirected) view sets it to `false`.
    const DIRECTED: bool;
}

/// Trait for graph types that support transforming all node and edge data
/// while preserving the graph topology.
pub trait GraphMap<'r, NewNode, NewEdge>: Sized + GraphProperty {
    /// The type of graph produced after mapping.
    type Mapped;

    /// Transform all node and edge data, preserving topology.
    fn map<FN, FE>(self, fn_node: FN, fn_edge: FE) -> Self::Mapped
    where
        FN: FnMut(Self::Node) -> NewNode,
        FE: FnMut(Self::Edge) -> NewEdge;
}

/// Main API exposed to operate with graphs.
///
///
/// # Safety
///
/// This trait is auto-infered from [`GraphOperation`] and [`GraphProperty`]. Do not implement
/// [`Graph`] directly for graphs.
pub unsafe trait Graph: for<'r> GraphOperation<'r> {
    /// Extend this graph with all nodes and edges from `other`.
    ///
    /// Nodes and edges from `other` are treated as entirely new; no
    /// deduplication is performed. The `other` graph is consumed via
    /// [`drain`](GraphOperation::drain).
    ///
    /// Node indices from `other` are mapped to fresh indices in `self`, and
    /// edge endpoints are adjusted accordingly.
    fn extend_graph<G>(&mut self, other: G)
    where
        Self: InsertNode + InsertEdge,
        G: Graph + GraphProperty<Node = Self::Node, Edge = Self::Edge>,
    {
        // Snapshot indices/endpoints before draining; `drain` consumes
        // `other` and invalidates everything reachable through it.
        let node_indices: Vec<G::NodeIx> =
            <G as GraphOperation<'_>>::node_indices(&other).collect();
        let edge_endpoints: Vec<G::Endpoints> = <G as GraphOperation<'_>>::edge_indices(&other)
            .map(|eix| unsafe { <G as GraphOperation<'_>>::endpoints_unchecked(&other, eix) })
            .collect();
        // UFCS: `other: Graph`, and both `Graph` and `GraphOperation` are in
        // scope here, so plain `.drain()` would be ambiguous.
        let (drain_nodes, drain_edges) = <G as GraphOperation<'_>>::drain(other);

        let mut node_map = std::collections::HashMap::new();
        for (old_nix, node) in node_indices.into_iter().zip(drain_nodes) {
            let new_nix = unsafe {
                crate::unwrap_unchecked(
                    <Self as InsertNode>::insert_node_unchecked(self, node).ok(),
                )
            };
            node_map.insert(old_nix, new_nix);
        }

        for (endpoints, edge) in edge_endpoints.into_iter().zip(drain_edges) {
            let mapped = Self::Endpoints::try_from_node_indices(
                endpoints
                    .into_iter()
                    .map(|nix| *node_map.get(&nix).expect("endpoint mapping failed")),
            )
            .expect("endpoint construction failed");
            unsafe {
                crate::unwrap_unchecked(
                    <Self as InsertEdge>::insert_edge_unchecked(self, edge, mapped).ok(),
                );
            }
        }
    }

    // ---- Scope ----

    /// Borrow this graph in a scoped context with lifetime-tagged indices.
    ///
    /// This is useful when you want indices that cannot escape the closure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use safegraph::graph::Graph;
    /// use safegraph::VecGraph;
    ///
    /// let g = VecGraph::<u32, u32>::default();
    /// g.scope(|ctx| {
    ///     let _count = ctx.nodes().count();
    /// });
    /// ```
    fn scope<R, F>(&self, f: F) -> R
    where
        F: for<'scope> FnOnce(&context::Context<'scope, Self>) -> R,
    {
        // SAFETY: `context::Context` is #[repr(transparent)]
        f(unsafe { core::mem::transmute::<&Self, &context::Context<'_, Self>>(self) })
    }

    /// Mutably borrow this graph in a scoped context with lifetime-tagged indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use safegraph::graph::Graph;
    /// use safegraph::VecGraph;
    ///
    /// let mut g = VecGraph::<u32, u32>::default();
    /// g.scope_mut(|mut ctx| {
    ///     let _ = ctx.push(1);
    /// });
    /// ```
    fn scope_mut<R, F>(&mut self, f: F) -> R
    where
        F: for<'scope> FnOnce(context::RemovableContext<'_, 'scope, Self>) -> R,
    {
        // SAFETY: `context::Context` is #[repr(transparent)]
        let ctx =
            unsafe { core::mem::transmute::<&mut Self, &mut context::Context<'_, Self>>(self) };
        f(context::RemovableContext::new(ctx))
    }

    /// Reinterpret `&self` as a graph that unsafely claims [`StableNode`].
    ///
    /// Prefer a safe alternative — [`Graph::scope()`] (a zero-cost wrapper) or
    /// [`Graph::stabilize()`] (a safe wrapper that adds some runtime cost) — over
    /// this, unless you fully understand how this library's [`StableNode`]
    /// mechanism works.
    ///
    /// # Safety
    /// The view claims [`StableNode`] whether or not `Self` actually implements
    /// it, so the caller must guarantee that — for as long as the view, and every
    /// `NodeIx` obtained through it, are in use — the graph is not mutated in any
    /// way that invalidates or repurposes a live node index. The hazard is a
    /// removal on a non-stable backend (e.g. `VecGraph`'s swap-remove, which moves
    /// another node into the freed slot, leaving a previously valid `NodeIx`
    /// pointing at a different node). Read-only use, and mutations that preserve
    /// every existing node index, are sound; index-invalidating removals are
    /// undefined behavior.
    #[inline]
    unsafe fn unsafe_assert_stable_node(&self) -> &AssertStable<TNode, Self> {
        // SAFETY: `AssertStable` is #[repr(transparent)] over `Self`.
        core::mem::transmute::<&Self, &AssertStable<TNode, Self>>(self)
    }

    /// Reinterpret `&self` as a graph that unsafely claims [`StableEdge`].
    ///
    /// Edge-stability counterpart of
    /// [`unsafe_assert_stable_node`](Self::unsafe_assert_stable_node). Prefer a
    /// safe alternative — [`Graph::scope()`] or [`Graph::stabilize()`] — over
    /// this, unless you fully understand how this library's [`StableEdge`]
    /// mechanism works.
    ///
    /// # Safety
    /// The view claims [`StableEdge`] whether or not `Self` actually implements
    /// it, so the caller must guarantee that no mutation invalidates or
    /// repurposes a live edge index (e.g. a swap-remove that moves another edge
    /// into the freed slot) while the view — or any `EdgeIx` obtained through it —
    /// is in use. Index-invalidating edge removals are undefined behavior.
    #[inline]
    unsafe fn unsafe_assert_stable_edge(&self) -> &AssertStable<TEdge, Self> {
        // SAFETY: `AssertStable` is #[repr(transparent)] over `Self`.
        core::mem::transmute::<&Self, &AssertStable<TEdge, Self>>(self)
    }

    /// Reinterpret `&mut self` as a graph that unsafely claims [`StableNode`],
    /// granting mutable access.
    ///
    /// Mutable counterpart of
    /// [`unsafe_assert_stable_node`](Self::unsafe_assert_stable_node), for a
    /// [`StableNode`]-bounded algorithm that needs to mutate a graph which is not
    /// itself `StableNode`. Prefer [`Graph::scope_mut()`] over this, unless you
    /// fully understand how this library's [`StableNode`] mechanism works.
    ///
    /// # Safety
    /// Carries the same contract as
    /// [`unsafe_assert_stable_node`](Self::unsafe_assert_stable_node): the view
    /// claims [`StableNode`] whether or not `Self` actually implements it. Because
    /// it grants `&mut` access the obligation is sharper — the caller must not
    /// perform, through this view or any other handle, a mutation that
    /// invalidates or repurposes a live node index (notably a swap-remove on a
    /// non-stable backend) while the view, or a `NodeIx` derived from it, is in
    /// use.
    #[inline]
    unsafe fn unsafe_assert_stable_node_mut(&mut self) -> &mut AssertStable<TNode, Self> {
        // SAFETY: `AssertStable` is #[repr(transparent)] over `Self`.
        core::mem::transmute::<&mut Self, &mut AssertStable<TNode, Self>>(self)
    }

    /// Reinterpret `&mut self` as a graph that unsafely claims [`StableEdge`],
    /// granting mutable access.
    ///
    /// Mutable counterpart of
    /// [`unsafe_assert_stable_edge`](Self::unsafe_assert_stable_edge). Prefer
    /// [`Graph::scope_mut()`] over this, unless you fully understand how this
    /// library's [`StableEdge`] mechanism works.
    ///
    /// # Safety
    /// Carries the same contract as
    /// [`unsafe_assert_stable_edge`](Self::unsafe_assert_stable_edge), but with
    /// `&mut` access: the caller must not perform, through this view or any other
    /// handle, a mutation that invalidates or repurposes a live edge index while
    /// the view, or an `EdgeIx` derived from it, is in use.
    #[inline]
    unsafe fn unsafe_assert_stable_edge_mut(&mut self) -> &mut AssertStable<TEdge, Self> {
        // SAFETY: `AssertStable` is #[repr(transparent)] over `Self`.
        core::mem::transmute::<&mut Self, &mut AssertStable<TEdge, Self>>(self)
    }

    /// Removes all nodes and edges from the graph.
    ///
    /// This default implementation removes edges first, then nodes.
    fn clear(&mut self)
    where
        Self: RemoveNode + RemoveEdge,
    {
        #[derive(Default)]
        struct Sink;
        impl<T> Extend<T> for Sink {
            fn extend<I: IntoIterator<Item = T>>(&mut self, _iter: I) {}
        }
        // SAFETY: indices are collected into owned `Vec`s (no live borrows)
        // before the mutable removal call.
        let edges: Vec<_> = <Self as GraphOperation<'_>>::edge_indices(self).collect();
        let nodes: Vec<_> = <Self as GraphOperation<'_>>::node_indices(self).collect();
        let _: (Sink, Sink) =
            unsafe { <Self as RemoveNode>::take_nodes_edges_unchecked(self, nodes, edges) };
    }

    /// Convert this graph into a tombstone-versioned stable graph wrapper.
    ///
    /// The wrapper keeps index stability across removals by soft-deleting
    /// entries and tracking generation numbers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use safegraph::graph::Graph;
    /// use safegraph::VecGraph;
    ///
    /// let mut g = VecGraph::<u32, u32>::default().stabilize();
    /// let n = g.insert_node(1).unwrap();
    /// assert!(g.contains_node_index(n));
    /// ```
    fn stabilize<'r, N, E>(self) -> StabilizedGraph<'r, Self, N, E>
    where
        Self: Sized
            + GraphProperty<Node = N, Edge = E>
            + GraphMap<'r, stabilized::NodeIx<N>, stabilized::EdgeIx<E>>,
    {
        let live_nodes = <Self as GraphOperation<'r>>::len_node(&self);
        let live_edges = <Self as GraphOperation<'r>>::len_edge(&self);
        let mapped = GraphMap::map(
            self,
            |n| stabilized::NodeIx {
                version: 1,
                inner: n,
            },
            |e| stabilized::EdgeIx {
                version: 1,
                inner: e,
            },
        );
        stabilized::Stabilized::from_mapped(mapped, live_nodes, live_edges)
    }

    /// Wrap this graph in an [`Undirected`](undirected::Undirected) view.
    ///
    /// In the wrapped view `edge_indices_from(nix)` returns every incident
    /// edge (both outgoing and incoming on the underlying directed graph),
    /// and `walks_from` skips self-loops. The wrapper takes ownership; use
    /// `into_inner` to recover the original graph.
    fn undirected(self) -> undirected::Undirected<Self>
    where
        Self: Sized,
    {
        undirected::Undirected { inner: self }
    }

    /// Returns `true` if `node_ix` refers to a live node in this graph.
    fn contains_node_index(&self, node_ix: Self::NodeIx) -> bool {
        <Self as GraphOperation<'_>>::contains_node_index(self, node_ix)
    }

    /// Returns `true` if `edge_ix` refers to a live edge in this graph.
    fn contains_edge_index(&self, edge_ix: Self::EdgeIx) -> bool {
        <Self as GraphOperation<'_>>::contains_edge_index(self, edge_ix)
    }

    /// Returns the number of live nodes in the graph.
    fn len_node(&self) -> usize {
        <Self as GraphOperation<'_>>::len_node(self)
    }

    /// Returns the number of live edges in the graph.
    fn len_edge(&self) -> usize {
        <Self as GraphOperation<'_>>::len_edge(self)
    }

    /// Returns the total node capacity, if applicable (e.g. `Vec`-backed
    /// graphs). `None` means fixed capacity does not apply.
    fn capacity_node(&self) -> Option<usize> {
        <Self as GraphOperation<'_>>::capacity_node(self)
    }

    /// Returns the total edge capacity, if applicable.
    fn capacity_edge(&self) -> Option<usize> {
        <Self as GraphOperation<'_>>::capacity_edge(self)
    }

    /// Reverses the orientation of every edge in place (no-op for undirected
    /// graphs).
    fn reverse(&mut self) {
        <Self as GraphOperation<'_>>::reverse(self)
    }

    /// Consumes this graph, returning iterators over all node and edge data.
    ///
    /// The node iterator yields items in the same order as
    /// [`node_indices`](Self::node_indices) and the edge iterator in the same
    /// order as [`edge_indices`](Self::edge_indices).
    ///
    /// The drained iterators own their data, so the lifetime is irrelevant; we
    /// pin it to `'static` (`Graph: for<'r> GraphOperation<'r>` includes it) to
    /// keep the signature free of a parameter.
    fn drain(
        self,
    ) -> (
        <Self as GraphOperation<'static>>::DrainNode,
        <Self as GraphOperation<'static>>::DrainEdge,
    )
    where
        Self: Sized,
    {
        <Self as GraphOperation<'static>>::drain(self)
    }

    /// Returns an iterator over all node indices. Requires [`StableNode`].
    fn node_indices(&self) -> <Self as GraphOperation<'_>>::NodeIndices
    where
        Self: StableNode,
    {
        // SAFETY: Self impls StableNode
        <Self as GraphOperation<'_>>::node_indices(self)
    }

    /// Returns an iterator over all edge indices. Requires [`StableEdge`].
    fn edge_indices(&self) -> <Self as GraphOperation<'_>>::EdgeIndices
    where
        Self: StableEdge,
    {
        // SAFETY: Self impls StableEdge
        <Self as GraphOperation<'_>>::edge_indices(self)
    }

    /// Returns a reference to the node at `node_ix`. Panics if the index is invalid.
    fn node(&self, node_ix: Self::NodeIx) -> &Self::Node {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        unsafe { <Self as GraphOperation<'_>>::node_unchecked(self, node_ix) }
    }

    /// Returns a reference to the node at `node_ix`, without checking validity.
    ///
    /// Unchecked sibling of [`node`](Self::node).
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn node_unchecked(&self, node_ix: Self::NodeIx) -> &Self::Node {
        <Self as GraphOperation<'_>>::node_unchecked(self, node_ix)
    }

    /// Returns a reference to the edge at `edge_ix`. Panics if the index is invalid.
    fn edge(&self, edge_ix: Self::EdgeIx) -> &Self::Edge {
        assert!(<Self as GraphOperation<'_>>::contains_edge_index(
            self, edge_ix
        ));
        unsafe { <Self as GraphOperation<'_>>::edge_unchecked(self, edge_ix) }
    }

    /// Returns a reference to the edge at `edge_ix`, without checking validity.
    ///
    /// Unchecked sibling of [`edge`](Self::edge).
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_unchecked(&self, edge_ix: Self::EdgeIx) -> &Self::Edge {
        <Self as GraphOperation<'_>>::edge_unchecked(self, edge_ix)
    }

    /// Returns an iterator over references to all node data.
    fn nodes(&self) -> NodeRefIter<'_, <Self as GraphOperation<'_>>::NodeIndices, Self> {
        NodeRefIter(self, <Self as GraphOperation<'_>>::node_indices(self))
    }

    /// Returns an iterator over references to all edge data.
    fn edges(&self) -> EdgeRefIter<'_, <Self as GraphOperation<'_>>::EdgeIndices, Self> {
        EdgeRefIter(self, <Self as GraphOperation<'_>>::edge_indices(self))
    }

    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn node_unchecked_mut<'a>(&'a mut self, node_ix: Self::NodeIx) -> &'a mut Self::Node
    where
        Self: UpdateNode<'a>,
    {
        <Self as UpdateNode<'a>>::node_unchecked_mut(self, node_ix)
    }

    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_unchecked_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge
    where
        Self: UpdateEdge,
    {
        <Self as UpdateEdge>::edge_unchecked_mut(self, edge_ix)
    }

    /// Returns a mutable reference to the node at `node_ix`. Panics if the index is invalid.
    fn node_mut<'r>(&'r mut self, node_ix: Self::NodeIx) -> &'r mut Self::Node
    where
        Self: UpdateNode<'r>,
    {
        assert!(<Self as GraphOperation<'r>>::contains_node_index(
            self, node_ix
        ));
        unsafe { <Self as UpdateNode<'r>>::node_unchecked_mut(self, node_ix) }
    }

    /// Returns a mutable reference to the edge at `edge_ix`. Panics if the index is invalid.
    fn edge_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge
    where
        Self: UpdateEdge,
    {
        assert!(<Self as GraphOperation<'_>>::contains_edge_index(
            self, edge_ix
        ));
        unsafe { <Self as UpdateEdge>::edge_unchecked_mut(self, edge_ix) }
    }

    /// Returns the endpoints of `edge_ix`. Panics if the index is invalid. Requires [`StableNode`].
    fn endpoints(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints
    where
        Self: StableNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: edge index checked above; `StableNode` keeps the result valid.
        unsafe { <Self as GraphOperation<'_>>::endpoints_unchecked(self, edge_ix) }
    }

    /// Returns the endpoints of `edge_ix`, without checking index validity.
    ///
    /// Unchecked sibling of [`endpoints`](Self::endpoints): it keeps the same
    /// [`StableNode`] bound but skips the index-validity check.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn endpoints_unchecked(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints
    where
        Self: StableNode,
    {
        <Self as GraphOperation<'_>>::endpoints_unchecked(self, edge_ix)
    }

    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn endpoint_nodes_unchecked(
        &self,
        edge_ix: Self::EdgeIx,
    ) -> NodeRefIter<'_, <Self::Endpoints as IntoIterator>::IntoIter, Self> {
        NodeRefIter(
            self,
            <Self as GraphOperation<'_>>::endpoints_unchecked(self, edge_ix).into_iter(),
        )
    }

    /// Returns an iterator over references to the endpoint nodes of `edge_ix`.
    /// Panics if the index is invalid.
    fn endpoint_nodes(
        &self,
        edge_ix: Self::EdgeIx,
    ) -> NodeRefIter<'_, <Self::Endpoints as IntoIterator>::IntoIter, Self> {
        assert!(<Self as GraphOperation<'_>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: edge index validity is checked above.
        unsafe { self.endpoint_nodes_unchecked(edge_ix) }
    }

    /// Returns edge indices from `node_ix`.
    ///
    /// For directed graphs, this yields outgoing edges.
    /// For undirected graphs, this yields all connected edges.
    fn edge_indices_from(
        &self,
        node_ix: Self::NodeIx,
    ) -> <Self as GraphOperation<'_>>::EdgeIndicesFrom
    where
        Self: StableEdge,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; `StableEdge` keeps the result valid.
        unsafe { <Self as GraphOperation<'_>>::edge_indices_from_unchecked(self, node_ix) }
    }

    /// Returns edge indices from `node_ix`, without checking index validity.
    ///
    /// Unchecked sibling of [`edge_indices_from`](Self::edge_indices_from): it
    /// keeps the same [`StableEdge`] bound but skips the index-validity check.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edge_indices_from_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> <Self as GraphOperation<'_>>::EdgeIndicesFrom
    where
        Self: StableEdge,
    {
        <Self as GraphOperation<'_>>::edge_indices_from_unchecked(self, node_ix)
    }

    /// Returns an iterator over references to edges from `node_ix`,
    /// without checking index validity.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edges_from_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> EdgeRefIter<'_, <Self as GraphOperation<'_>>::EdgeIndicesFrom, Self> {
        EdgeRefIter(
            self,
            <Self as GraphOperation<'_>>::edge_indices_from_unchecked(self, node_ix),
        )
    }

    /// Returns an iterator over references to edges starting from `node_ix`.
    /// Panics if the index is invalid.
    fn edges_from(
        &self,
        node_ix: Self::NodeIx,
    ) -> EdgeRefIter<'_, <Self as GraphOperation<'_>>::EdgeIndicesFrom, Self> {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index validity is checked above.
        unsafe { self.edges_from_unchecked(node_ix) }
    }

    /// Returns all edge indices incident to `node_ix`.
    ///
    /// For directed graphs this includes both incoming and outgoing edges.
    fn edge_indices_of(&self, node_ix: Self::NodeIx) -> <Self as GraphOperation<'_>>::EdgeIndicesOf
    where
        Self: StableEdge,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; `StableEdge` keeps the result valid.
        unsafe { <Self as GraphOperation<'_>>::edge_indices_of_unchecked(self, node_ix) }
    }

    /// Returns all edge indices incident to `node_ix`, without checking validity.
    ///
    /// Unchecked sibling of [`edge_indices_of`](Self::edge_indices_of): it keeps
    /// the same [`StableEdge`] bound but skips the index-validity check.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edge_indices_of_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> <Self as GraphOperation<'_>>::EdgeIndicesOf
    where
        Self: StableEdge,
    {
        <Self as GraphOperation<'_>>::edge_indices_of_unchecked(self, node_ix)
    }

    /// Returns an iterator over references to all incident edges of `node_ix`,
    /// without checking index validity.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edges_of_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> EdgeRefIter<'_, <Self as GraphOperation<'_>>::EdgeIndicesOf, Self> {
        EdgeRefIter(
            self,
            <Self as GraphOperation<'_>>::edge_indices_of_unchecked(self, node_ix),
        )
    }

    /// Returns an iterator over references to all edges incident to `node_ix`.
    /// Panics if the index is invalid.
    fn edges_of(
        &self,
        node_ix: Self::NodeIx,
    ) -> EdgeRefIter<'_, <Self as GraphOperation<'_>>::EdgeIndicesOf, Self> {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index validity is checked above.
        unsafe { self.edges_of_unchecked(node_ix) }
    }

    /// Returns neighbor indices reachable via outgoing edges (directed) or all edges (undirected).
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn neighbor_indices_from_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> NeighborIndices<Self::NodeIx>
    where
        Self: StableNode,
    {
        NeighborIndices {
            iter: <Self as GraphOperation<'_>>::walks_from_unchecked(self, node_ix)
                .map(|wi| wi.into_parts().2)
                .collect::<Vec<_>>()
                .into_iter(),
        }
    }

    /// Returns node indices directly reachable from `node_ix`.
    ///
    /// For directed graphs this follows outgoing edges.
    /// For undirected graphs this returns adjacent nodes.
    fn neighbor_indices_from(&self, node_ix: Self::NodeIx) -> NeighborIndices<Self::NodeIx>
    where
        Self: StableNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; `StableNode` keeps the result valid.
        unsafe { self.neighbor_indices_from_unchecked(node_ix) }
    }

    /// Returns node references directly reachable from `node_ix`.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn neighbors_from_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> NodeRefIter<'_, NeighborIndices<Self::NodeIx>, Self>
    where
        Self: StableNode,
    {
        NodeRefIter(self, self.neighbor_indices_from_unchecked(node_ix))
    }

    /// Returns an iterator over references to neighbor nodes reachable from `node_ix`.
    /// Panics if the index is invalid. Requires [`StableNode`].
    fn neighbors_from(
        &self,
        node_ix: Self::NodeIx,
    ) -> NodeRefIter<'_, NeighborIndices<Self::NodeIx>, Self>
    where
        Self: StableNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: Self impls StableNode, node_ix checked above.
        unsafe { self.neighbors_from_unchecked(node_ix) }
    }

    /// Returns neighbor indices connected to `node_ix` via any edge direction.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn neighbor_indices_of_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> NeighborIndices<Self::NodeIx>
    where
        Self: StableNode,
    {
        NeighborIndices {
            iter: <Self as GraphOperation<'_>>::walks_of_unchecked(self, node_ix)
                .map(|wi| wi.into_parts().2)
                .collect::<Vec<_>>()
                .into_iter(),
        }
    }

    /// Returns node indices directly connected with `node_ix`.
    ///
    /// This includes all adjacent nodes regardless of edge direction.
    fn neighbor_indices_of(&self, node_ix: Self::NodeIx) -> NeighborIndices<Self::NodeIx>
    where
        Self: StableNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; `StableNode` keeps the result valid.
        unsafe { self.neighbor_indices_of_unchecked(node_ix) }
    }

    /// Returns an iterator over references to all neighbor nodes of `node_ix`,
    /// without checking index validity.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn neighbors_of_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> NodeRefIter<'_, NeighborIndices<Self::NodeIx>, Self>
    where
        Self: StableNode,
    {
        // SAFETY: the NodeIx is only used until the Graph is not modified.
        NodeRefIter(self, self.neighbor_indices_of_unchecked(node_ix))
    }

    /// Returns an iterator over references to all neighbor nodes of `node_ix`.
    /// Panics if the index is invalid. Requires [`StableNode`].
    fn neighbors_of(
        &self,
        node_ix: Self::NodeIx,
    ) -> NodeRefIter<'_, NeighborIndices<Self::NodeIx>, Self>
    where
        Self: StableNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: checked in precondition
        unsafe { self.neighbors_of_unchecked(node_ix) }
    }

    /// Returns (EdgeIx, &Edge, NodeIx) triples for outgoing walks. Requires [`StableEdge`] + [`StableNode`].
    fn walks_from(&self, node_ix: Self::NodeIx) -> <Self as GraphOperation<'_>>::WalksFrom
    where
        Self: StableEdge + StableNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; StableEdge + StableNode keep it valid.
        unsafe { <Self as GraphOperation<'_>>::walks_from_unchecked(self, node_ix) }
    }

    /// Returns (EdgeIx, &Edge, NodeIx) triples for all incident walks. Requires [`StableEdge`] + [`StableNode`].
    fn walks_of(&self, node_ix: Self::NodeIx) -> <Self as GraphOperation<'_>>::WalksOf
    where
        Self: StableEdge + StableNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; StableEdge + StableNode keep it valid.
        unsafe { <Self as GraphOperation<'_>>::walks_of_unchecked(self, node_ix) }
    }

    /// Unchecked sibling of [`walks_from`](Self::walks_from): it keeps the same
    /// [`StableEdge`] + [`StableNode`] bound but skips the index-validity check.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_from_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> <Self as GraphOperation<'_>>::WalksFrom
    where
        Self: StableEdge + StableNode,
    {
        <Self as GraphOperation<'_>>::walks_from_unchecked(self, node_ix)
    }

    /// Unchecked sibling of [`walks_of`](Self::walks_of): it keeps the same
    /// [`StableEdge`] + [`StableNode`] bound but skips the index-validity check.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_of_unchecked(
        &self,
        node_ix: Self::NodeIx,
    ) -> <Self as GraphOperation<'_>>::WalksOf
    where
        Self: StableEdge + StableNode,
    {
        <Self as GraphOperation<'_>>::walks_of_unchecked(self, node_ix)
    }

    /// Returns (NodeIx, EdgeIx, &Edge) triples for incoming edges.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_to_unchecked<'r>(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> <Self as Directed<'r>>::WalksTo
    where
        Self: Directed<'r> + StableEdge + StableNode,
    {
        <Self as Directed<'r>>::walks_to_unchecked(self, node_ix)
    }

    /// Returns (NodeIx, EdgeIx, &Edge) triples for incoming walks. Requires [`Directed`] + [`StableEdge`] + [`StableNode`].
    fn walks_to<'r>(&'r self, node_ix: Self::NodeIx) -> <Self as Directed<'r>>::WalksTo
    where
        Self: Directed<'r> + StableEdge + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; StableEdge + StableNode keep it valid.
        unsafe { <Self as Directed<'r>>::walks_to_unchecked(self, node_ix) }
    }

    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_from_unchecked_mut<'r>(
        &'r mut self,
        node_ix: Self::NodeIx,
    ) -> <Self as capability::UpdateNode<'r>>::WalksFromMut
    where
        Self: capability::UpdateNode<'r> + StableEdge + StableNode,
    {
        <Self as capability::UpdateNode<'r>>::walks_from_unchecked_mut(self, node_ix)
    }

    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn walks_of_unchecked_mut<'r>(
        &'r mut self,
        node_ix: Self::NodeIx,
    ) -> <Self as capability::UpdateNode<'r>>::WalksOfMut
    where
        Self: capability::UpdateNode<'r> + StableEdge + StableNode,
    {
        <Self as capability::UpdateNode<'r>>::walks_of_unchecked_mut(self, node_ix)
    }

    /// Returns (EdgeIx, &mut Edge, NodeIx) triples for outgoing walks.
    /// Requires [`capability::UpdateNode`] + [`StableEdge`] + [`StableNode`].
    fn walks_from_mut<'r>(
        &'r mut self,
        node_ix: Self::NodeIx,
    ) -> <Self as capability::UpdateNode<'r>>::WalksFromMut
    where
        Self: capability::UpdateNode<'r> + StableEdge + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; StableEdge + StableNode keep it valid.
        unsafe { <Self as capability::UpdateNode<'r>>::walks_from_unchecked_mut(self, node_ix) }
    }

    /// Returns (EdgeIx, &mut Edge, NodeIx) triples for all incident walks.
    /// Requires [`capability::UpdateNode`] + [`StableEdge`] + [`StableNode`].
    fn walks_of_mut<'r>(
        &'r mut self,
        node_ix: Self::NodeIx,
    ) -> <Self as capability::UpdateNode<'r>>::WalksOfMut
    where
        Self: capability::UpdateNode<'r> + StableEdge + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; StableEdge + StableNode keep it valid.
        unsafe { <Self as capability::UpdateNode<'r>>::walks_of_unchecked_mut(self, node_ix) }
    }

    /// Returns edges incoming to `node_ix`.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edge_indices_to_unchecked<'r>(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> <Self as Directed<'r>>::EdgeIndicesTo
    where
        Self: Directed<'r> + StableEdge,
    {
        <Self as Directed<'r>>::edge_indices_to_unchecked(self, node_ix)
    }

    /// Returns edge indices incoming to `node_ix`.
    ///
    /// This is available only for directed graphs.
    fn edge_indices_to<'r>(&'r self, node_ix: Self::NodeIx) -> <Self as Directed<'r>>::EdgeIndicesTo
    where
        Self: Directed<'r> + StableEdge,
    {
        assert!(<Self as GraphOperation<'r>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; `StableEdge` keeps the result valid.
        unsafe { <Self as Directed<'r>>::edge_indices_to_unchecked(self, node_ix) }
    }

    /// Returns an iterator over references to edges incoming to `node_ix`,
    /// without checking index validity.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn edges_to_unchecked<'r>(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> EdgeRefIter<'r, <Self as Directed<'r>>::EdgeIndicesTo, Self>
    where
        Self: Directed<'r> + StableEdge,
    {
        EdgeRefIter(
            self,
            <Self as Directed<'r>>::edge_indices_to_unchecked(self, node_ix),
        )
    }

    /// Returns an iterator over references to edges incoming to `node_ix`.
    /// Panics if the index is invalid.
    fn edges_to<'r>(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> EdgeRefIter<'r, <Self as Directed<'r>>::EdgeIndicesTo, Self>
    where
        Self: Directed<'r> + StableEdge,
    {
        assert!(<Self as GraphOperation<'r>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; `StableEdge` keeps the result valid.
        unsafe { self.edges_to_unchecked(node_ix) }
    }

    /// Returns neighbor indices reachable via incoming edges (directed graphs only).
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn neighbor_indices_to_unchecked<'r>(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> NeighborIndices<Self::NodeIx>
    where
        Self: Directed<'r> + StableNode,
    {
        NeighborIndices {
            iter: <Self as Directed<'r>>::walks_to_unchecked(self, node_ix)
                .map(|wi| wi.into_parts().0)
                .collect::<Vec<_>>()
                .into_iter(),
        }
    }

    /// Returns node indices that have an edge into `node_ix`.
    ///
    /// This is available only for directed graphs.
    fn neighbor_indices_to<'r>(&'r self, node_ix: Self::NodeIx) -> NeighborIndices<Self::NodeIx>
    where
        Self: Directed<'r> + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: node index checked above; `StableNode` keeps the result valid.
        unsafe { self.neighbor_indices_to_unchecked(node_ix) }
    }

    /// Returns an iterator over references to nodes with an edge into `node_ix`.
    ///
    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn neighbors_to_unchecked<'r>(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> NodeRefIter<'r, NeighborIndices<Self::NodeIx>, Self>
    where
        Self: Directed<'r> + StableNode,
    {
        NodeRefIter(self, self.neighbor_indices_to_unchecked(node_ix))
    }

    /// Returns an iterator over references to nodes with an edge into `node_ix`.
    /// Panics if the index is invalid.
    fn neighbors_to<'r>(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> NodeRefIter<'r, NeighborIndices<Self::NodeIx>, Self>
    where
        Self: Directed<'r> + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_node_index(
            self, node_ix
        ));
        unsafe { self.neighbors_to_unchecked(node_ix) }
    }

    /// Returns the source (tail) nodes of `edge_ix`.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_tail_indices_unchecked<'r>(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> <Self as Directed<'r>>::EdgeTailIndices
    where
        Self: Directed<'r> + StableNode,
    {
        <Self as Directed<'r>>::edge_tail_indices_unchecked(self, edge_ix)
    }

    /// Returns the source (tail) node indices of `edge_ix`. Requires [`Directed`] + [`StableNode`].
    fn edge_tail_indices<'r>(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> <Self as Directed<'r>>::EdgeTailIndices
    where
        Self: Directed<'r> + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: edge index checked above; `StableNode` keeps the result valid.
        unsafe { <Self as Directed<'r>>::edge_tail_indices_unchecked(self, edge_ix) }
    }

    /// Returns an iterator over references to the source (tail) nodes of `edge_ix`.
    /// Panics if the index is invalid.
    fn edge_tails<'r>(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> NodeRefIter<'r, <Self as Directed<'r>>::EdgeTailIndices, Self>
    where
        Self: Directed<'r>,
    {
        assert!(<Self as GraphOperation<'r>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: node indices are consumed immediately while graph is immutably borrowed.
        unsafe {
            NodeRefIter(
                self,
                <Self as Directed<'r>>::edge_tail_indices_unchecked(self, edge_ix),
            )
        }
    }

    /// Returns the target (head) nodes of `edge_ix`.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_head_indices_unchecked<'r>(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> <Self as Directed<'r>>::EdgeHeadIndices
    where
        Self: Directed<'r> + StableNode,
    {
        <Self as Directed<'r>>::edge_head_indices_unchecked(self, edge_ix)
    }

    /// Returns the target (head) node indices of `edge_ix`. Requires [`Directed`] + [`StableNode`].
    fn edge_head_indices<'r>(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> <Self as Directed<'r>>::EdgeHeadIndices
    where
        Self: Directed<'r> + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: edge index checked above; `StableNode` keeps the result valid.
        unsafe { <Self as Directed<'r>>::edge_head_indices_unchecked(self, edge_ix) }
    }

    /// Returns an iterator over references to the target (head) nodes of `edge_ix`.
    /// Panics if the index is invalid.
    fn edge_heads<'r>(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> NodeRefIter<'r, <Self as Directed<'r>>::EdgeHeadIndices, Self>
    where
        Self: Directed<'r>,
    {
        assert!(<Self as GraphOperation<'r>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: node indices are consumed immediately while graph is immutably borrowed.
        unsafe {
            NodeRefIter(
                self,
                <Self as Directed<'r>>::edge_head_indices_unchecked(self, edge_ix),
            )
        }
    }

    /// Returns an iterator over references to the source (tail) nodes of `edge_ix`,
    /// without checking index validity.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_tails_unchecked<'r>(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> NodeRefIter<'r, <Self as Directed<'r>>::EdgeTailIndices, Self>
    where
        Self: Directed<'r>,
    {
        NodeRefIter(
            self,
            <Self as Directed<'r>>::edge_tail_indices_unchecked(self, edge_ix),
        )
    }

    /// Returns an iterator over references to the target (head) nodes of `edge_ix`,
    /// without checking index validity.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_heads_unchecked<'r>(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> NodeRefIter<'r, <Self as Directed<'r>>::EdgeHeadIndices, Self>
    where
        Self: Directed<'r>,
    {
        NodeRefIter(
            self,
            <Self as Directed<'r>>::edge_head_indices_unchecked(self, edge_ix),
        )
    }

    /// Returns the single source (tail / index 0) node of `edge_ix`.
    ///
    /// Requires [`Directed`] + [`Bigraph`].
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_tail_index_unchecked<'r>(&'r self, edge_ix: Self::EdgeIx) -> Self::NodeIx
    where
        Self: Directed<'r> + Bigraph + StableNode,
    {
        let endpoints = <Self as GraphOperation<'r>>::endpoints_unchecked(self, edge_ix);
        <Self as Bigraph>::endpoints_as_array(endpoints)[0]
    }

    /// Returns the single source (tail) node index of `edge_ix`.
    /// Panics if the index is invalid. Requires [`Directed`] + [`Bigraph`] + [`StableNode`].
    fn edge_tail_index<'r>(&'r self, edge_ix: Self::EdgeIx) -> Self::NodeIx
    where
        Self: Directed<'r> + Bigraph + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: edge index validity is checked above.
        unsafe { self.edge_tail_index_unchecked(edge_ix) }
    }

    /// Returns a reference to the single source (tail) node of `edge_ix`.
    /// Panics if the index is invalid. Requires [`Directed`] + [`Bigraph`] + [`StableNode`].
    fn edge_tail<'r>(&'r self, edge_ix: Self::EdgeIx) -> &'r Self::Node
    where
        Self: Directed<'r> + Bigraph + StableNode,
    {
        // SAFETY: `edge_tail_index` is safe and returns a valid node index.
        unsafe { <Self as GraphOperation<'r>>::node_unchecked(self, self.edge_tail_index(edge_ix)) }
    }

    /// Returns the single target (head / index 1) node of `edge_ix`.
    ///
    /// Requires [`Directed`] + [`Bigraph`].
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_head_index_unchecked<'r>(&'r self, edge_ix: Self::EdgeIx) -> Self::NodeIx
    where
        Self: Directed<'r> + Bigraph + StableNode,
    {
        let endpoints = <Self as GraphOperation<'r>>::endpoints_unchecked(self, edge_ix);
        <Self as Bigraph>::endpoints_as_array(endpoints)[1]
    }

    /// Returns the single target (head) node index of `edge_ix`.
    /// Panics if the index is invalid. Requires [`Directed`] + [`Bigraph`] + [`StableNode`].
    fn edge_head_index<'r>(&'r self, edge_ix: Self::EdgeIx) -> Self::NodeIx
    where
        Self: Directed<'r> + Bigraph + StableNode,
    {
        assert!(<Self as GraphOperation<'r>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: edge index validity is checked above.
        unsafe { self.edge_head_index_unchecked(edge_ix) }
    }

    /// Returns a reference to the single target (head) node of `edge_ix`.
    /// Panics if the index is invalid. Requires [`Directed`] + [`Bigraph`] + [`StableNode`].
    fn edge_head<'r>(&'r self, edge_ix: Self::EdgeIx) -> &'r Self::Node
    where
        Self: Directed<'r> + Bigraph + StableNode,
    {
        // SAFETY: `edge_head_index` is safe and returns a valid node index.
        unsafe { <Self as GraphOperation<'r>>::node_unchecked(self, self.edge_head_index(edge_ix)) }
    }

    /// Returns a reference to the single source (tail) node of `edge_ix`,
    /// without checking index validity.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_tail_unchecked<'r>(&'r self, edge_ix: Self::EdgeIx) -> &'r Self::Node
    where
        Self: Directed<'r> + Bigraph + StableNode,
    {
        <Self as GraphOperation<'r>>::node_unchecked(self, self.edge_tail_index_unchecked(edge_ix))
    }

    /// Returns a reference to the single target (head) node of `edge_ix`,
    /// without checking index validity.
    ///
    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn edge_head_unchecked<'r>(&'r self, edge_ix: Self::EdgeIx) -> &'r Self::Node
    where
        Self: Directed<'r> + Bigraph + StableNode,
    {
        <Self as GraphOperation<'r>>::node_unchecked(self, self.edge_head_index_unchecked(edge_ix))
    }

    /// Inserts a node and returns its index, skipping the [`StableNode`] bound
    /// that the safe [`insert_node`](Self::insert_node) requires.
    #[allow(clippy::missing_safety_doc)] // shares the crate-wide `*_unchecked` contract
    unsafe fn insert_node_unchecked(&mut self, node: Self::Node) -> Result<Self::NodeIx, Self::Node>
    where
        Self: InsertNode,
    {
        <Self as InsertNode>::insert_node_unchecked(self, node)
    }

    /// Inserts a node and returns its stable index. Requires [`InsertNode`] + [`StableNode`].
    fn insert_node(&mut self, node: Self::Node) -> Result<Self::NodeIx, Self::Node>
    where
        Self: InsertNode + StableNode,
    {
        unsafe { <Self as InsertNode>::insert_node_unchecked(self, node) }
    }

    /// Inserts a node, discarding the index. Requires [`InsertNode`].
    fn push(&mut self, node: Self::Node) -> Result<(), Self::Node>
    where
        Self: InsertNode,
    {
        unsafe { <Self as InsertNode>::insert_node_unchecked(self, node).map(|_| ()) }
    }

    /// # Safety
    /// `endpoints` must contain valid node indices currently held by this graph.
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge>
    where
        Self: InsertEdge,
    {
        <Self as InsertEdge>::insert_edge_unchecked(self, edge, endpoints)
    }

    /// Inserts an edge and returns its stable index. Panics if endpoint nodes are invalid.
    /// Requires [`InsertEdge`] + [`StableEdge`].
    fn insert_edge(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge>
    where
        Self: InsertEdge + StableEdge,
    {
        assert!(endpoints
            .iter()
            .all(|ix| <Self as GraphOperation<'_>>::contains_node_index(self, ix)));
        // SAFETY: endpoints checked above; `StableEdge` keeps the returned index valid.
        unsafe { <Self as InsertEdge>::insert_edge_unchecked(self, edge, endpoints) }
    }

    /// # Safety
    /// `endpoints` must contain valid node indices currently held by this graph.
    unsafe fn push_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<(), Self::Edge>
    where
        Self: InsertEdge,
    {
        <Self as InsertEdge>::insert_edge_unchecked(self, edge, endpoints).map(|_| ())
    }

    /// Inserts an edge, discarding the index. Panics if endpoint nodes are invalid.
    fn push_edge(&mut self, edge: Self::Edge, endpoints: Self::Endpoints) -> Result<(), Self::Edge>
    where
        Self: InsertEdge,
    {
        assert!(endpoints
            .iter()
            .all(|ix| <Self as GraphOperation<'_>>::contains_node_index(self, ix)));
        // SAFETY: endpoints checked above; the returned index is discarded.
        unsafe { self.push_edge_unchecked(edge, endpoints) }
    }

    /// Returns the index of the node if it already exists, or inserts it and
    /// returns the new index.
    #[allow(clippy::missing_safety_doc)] // shares the crate-wide `*_unchecked` contract
    unsafe fn get_or_insert_node_unchecked(&mut self, node: Self::Node) -> Self::NodeIx
    where
        Self: InsertNode + UniqueNode,
    {
        match self.node_index(&node) {
            Some(ix) => ix,
            None => crate::unwrap_unchecked(
                <Self as InsertNode>::insert_node_unchecked(self, node).ok(),
            ),
        }
    }

    /// Returns the index of the node if it already exists, or inserts it and
    /// returns the new index.
    fn get_or_insert_node(&mut self, node: Self::Node) -> Self::NodeIx
    where
        Self: InsertNode + UniqueNode + StableNode,
    {
        unsafe { self.get_or_insert_node_unchecked(node) }
    }

    /// Returns the index of the edge if it already exists, or inserts it with
    /// the given endpoints and returns the new index.
    ///
    /// # Safety
    /// Endpoint nodes must be valid node indices currently held by this graph.
    unsafe fn get_or_insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Self::EdgeIx
    where
        Self: InsertEdge + UniqueEdge,
    {
        match self.edge_index(&edge) {
            Some(ix) => ix,
            None => crate::unwrap_unchecked(
                <Self as InsertEdge>::insert_edge_unchecked(self, edge, endpoints).ok(),
            ),
        }
    }

    /// Returns the index of the edge if it already exists, or inserts it with
    /// the given endpoints and returns the new index.
    fn get_or_insert_edge(&mut self, edge: Self::Edge, endpoints: Self::Endpoints) -> Self::EdgeIx
    where
        Self: InsertEdge + UniqueEdge + StableEdge,
    {
        assert!(endpoints
            .iter()
            .all(|ix| <Self as GraphOperation<'_>>::contains_node_index(self, ix)));
        // SAFETY: endpoints checked above; `StableEdge` keeps the returned index valid.
        unsafe { self.get_or_insert_edge_unchecked(edge, endpoints) }
    }

    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn remove_node_unchecked(&mut self, node_ix: Self::NodeIx)
    where
        Self: RemoveNode,
    {
        <Self as RemoveNode>::remove_node_unchecked(self, node_ix)
    }

    /// Removes the node at `node_ix` and all its incident edges.
    /// Panics if the index is invalid.
    fn remove_node(&mut self, node_ix: Self::NodeIx)
    where
        Self: RemoveNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: checked in precondition
        unsafe { <Self as RemoveNode>::remove_node_unchecked(self, node_ix) }
    }

    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn remove_edge_unchecked(&mut self, edge_ix: Self::EdgeIx)
    where
        Self: RemoveEdge,
    {
        <Self as RemoveEdge>::remove_edge_unchecked(self, edge_ix);
    }

    /// Removes the edge at `edge_ix`. Panics if the index is invalid.
    fn remove_edge(&mut self, edge_ix: Self::EdgeIx)
    where
        Self: RemoveEdge,
    {
        assert!(<Self as GraphOperation<'_>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: checked in precondition
        unsafe { <Self as RemoveEdge>::remove_edge_unchecked(self, edge_ix) }
    }

    /// # Safety
    /// `node_ix` must be a valid node index currently held by this graph.
    unsafe fn take_node_unchecked(&mut self, node_ix: Self::NodeIx) -> Self::Node
    where
        Self: RemoveNode,
    {
        <Self as RemoveNode>::take_node_unchecked(self, node_ix)
    }

    /// Removes the node at `node_ix` and returns its data.
    /// Panics if the index is invalid.
    fn take_node(&mut self, node_ix: Self::NodeIx) -> Self::Node
    where
        Self: RemoveNode,
    {
        assert!(<Self as GraphOperation<'_>>::contains_node_index(
            self, node_ix
        ));
        // SAFETY: checked above.
        unsafe { <Self as RemoveNode>::take_node_unchecked(self, node_ix) }
    }

    /// # Safety
    /// `edge_ix` must be a valid edge index currently held by this graph.
    unsafe fn take_edge_unchecked(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge
    where
        Self: RemoveEdge,
    {
        <Self as RemoveEdge>::take_edge_unchecked(self, edge_ix)
    }

    /// Removes the edge at `edge_ix` and returns its data.
    /// Panics if the index is invalid.
    fn take_edge(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge
    where
        Self: RemoveEdge,
    {
        assert!(<Self as GraphOperation<'_>>::contains_edge_index(
            self, edge_ix
        ));
        // SAFETY: checked above.
        unsafe { <Self as RemoveEdge>::take_edge_unchecked(self, edge_ix) }
    }

    /// Removes selected nodes and edges, returning removed payloads.
    ///
    /// Edges are removed first, then nodes.
    ///
    /// Panics if any index is invalid or appears more than once.
    fn take_nodes_edges<IN, IE>(
        &mut self,
        node_indices: impl IntoIterator<Item = Self::NodeIx>,
        edge_indices: impl IntoIterator<Item = Self::EdgeIx>,
    ) -> (IN, IE)
    where
        Self: RemoveNode,
        IN: Default + Extend<Self::Node>,
        IE: Default + Extend<Self::Edge>,
    {
        let edge_indices: Vec<_> = edge_indices.into_iter().collect();
        let node_indices: Vec<_> = node_indices.into_iter().collect();
        let mut seen_edges = std::collections::HashSet::with_capacity(edge_indices.len());
        for &eix in &edge_indices {
            assert!(<Self as GraphOperation<'_>>::contains_edge_index(self, eix));
            assert!(seen_edges.insert(eix), "duplicate edge index {}", eix);
        }
        let mut seen_nodes = std::collections::HashSet::with_capacity(node_indices.len());
        for &nix in &node_indices {
            assert!(<Self as GraphOperation<'_>>::contains_node_index(self, nix));
            assert!(seen_nodes.insert(nix), "duplicate node index {}", nix);
        }
        // SAFETY: All indices validated above, duplicates rejected.
        unsafe {
            <Self as RemoveNode>::take_nodes_edges_unchecked(self, node_indices, edge_indices)
        }
    }

    /// Removes selected nodes and edges.
    ///
    /// Edges are removed first, then nodes.
    ///
    /// # Safety
    /// All `node_indices` and `edge_indices` must be valid indices currently
    /// held by this graph, and neither sequence may contain duplicates.
    unsafe fn remove_nodes_edges_unchecked(
        &mut self,
        node_indices: impl IntoIterator<Item = Self::NodeIx>,
        edge_indices: impl IntoIterator<Item = Self::EdgeIx>,
    ) where
        Self: RemoveNode,
    {
        #[derive(Default)]
        struct Sync;
        impl<T> Extend<T> for Sync {
            fn extend<I: IntoIterator<Item = T>>(&mut self, _iter: I) {}
        }
        // SAFETY: caller guarantees index validity.
        let _: (Sync, Sync) =
            <Self as RemoveNode>::take_nodes_edges_unchecked(self, node_indices, edge_indices);
    }

    /// Removes selected nodes and edges.
    ///
    /// Edges are removed first, then nodes.
    fn remove_nodes_edges(
        &mut self,
        node_indices: impl IntoIterator<Item = Self::NodeIx>,
        edge_indices: impl IntoIterator<Item = Self::EdgeIx>,
    ) where
        Self: RemoveNode,
    {
        let edge_indices: Vec<_> = edge_indices.into_iter().collect();
        let node_indices: Vec<_> = node_indices.into_iter().collect();
        for &eix in &edge_indices {
            assert!(<Self as GraphOperation<'_>>::contains_edge_index(self, eix));
        }
        for &nix in &node_indices {
            assert!(<Self as GraphOperation<'_>>::contains_node_index(self, nix));
        }
        // SAFETY: validated above.
        unsafe {
            self.remove_nodes_edges_unchecked(node_indices, edge_indices);
        }
    }

    // ---- as_ref / as_mut ----

    /// Reinterpret `&self` as a `#[repr(transparent)]`
    /// [`AsRef`](as_ref::AsRef) view — an `impl Graph` forwarding every read to
    /// `self` (same `Node`/`Edge`/index types as `Self`).
    ///
    /// See [`as_ref::AsRef`] for details.
    fn as_ref(&self) -> &as_ref::AsRef<Self> {
        // SAFETY: `#[repr(transparent)]` keeps the layout (incl. `?Sized` metadata).
        unsafe { core::mem::transmute::<&Self, &as_ref::AsRef<Self>>(self) }
    }

    /// Reinterpret `&mut self` as a `#[repr(transparent)]`
    /// [`AsRef`](as_ref::AsRef) view. The resulting `&mut AsRef<Self>` is a fully
    /// mutable graph: `AsRef` forwards every mutation/lookup capability to `self`.
    ///
    /// See [`as_ref::AsRef`] for details.
    fn as_mut(&mut self) -> &mut as_ref::AsRef<Self> {
        // SAFETY: `#[repr(transparent)]` keeps the layout (incl. `?Sized` metadata).
        unsafe { core::mem::transmute::<&mut Self, &mut as_ref::AsRef<Self>>(self) }
    }
}

// SAFETY: auto impl
unsafe impl<T: for<'r> GraphOperation<'r> + ?Sized> Graph for T {}
