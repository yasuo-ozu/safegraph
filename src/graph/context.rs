use core::borrow::Borrow;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use std::fmt::{self, Display};

use super::capability::*;
use super::edge::Map;
use super::walk_item::{WalkItem, WalkItemMut, WalkItemTo};
use super::{GraphOperation, GraphProperty};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeIx<'scope, I>(crate::Invariant<'scope>, I);

impl<'scope, I: Display> Display for NodeIx<'scope, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.1.fmt(f)
    }
}

impl<'scope, I: fmt::Debug> fmt::Debug for NodeIx<'scope, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.1.fmt(f)
    }
}

impl<'scope, I> NodeIx<'scope, I> {
    pub fn inner(self) -> I {
        self.1
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EdgeIx<'scope, I>(crate::Invariant<'scope>, I);

impl<'scope, I: Display> Display for EdgeIx<'scope, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.1.fmt(f)
    }
}

impl<'scope, I: fmt::Debug> fmt::Debug for EdgeIx<'scope, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.1.fmt(f)
    }
}

impl<'scope, I> EdgeIx<'scope, I> {
    pub fn inner(self) -> I {
        self.1
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct Context<'scope, G: ?Sized> {
    _scope: crate::Invariant<'scope>,
    graph: G,
}

impl<'scope, G: ?Sized + GraphProperty> Context<'scope, G> {
    /// Brands a raw node index from the wrapped graph as a scoped index — the
    /// inverse of [`NodeIx::inner`].
    ///
    /// This is the controlled entry point for bringing an externally-held index
    /// into a scope (e.g. to seed a traversal from a caller-supplied node). The
    /// brand only governs escape (the result still cannot outlive the scope); the
    /// caller is responsible for the index actually belonging to this graph, the
    /// same as for any index passed to the `*_unchecked` accessors.
    pub fn wrap_node(&self, node_ix: G::NodeIx) -> NodeIx<'scope, G::NodeIx> {
        NodeIx(PhantomData, node_ix)
    }

    /// Brands a raw edge index as a scoped index — the inverse of
    /// [`EdgeIx::inner`]. See [`wrap_node`](Self::wrap_node).
    pub fn wrap_edge(&self, edge_ix: G::EdgeIx) -> EdgeIx<'scope, G::EdgeIx> {
        EdgeIx(PhantomData, edge_ix)
    }
}

pub struct Walks<'scope, I, Eix, Nix> {
    inner: I,
    _scope: crate::Invariant<'scope>,
    _marker: PhantomData<(Eix, Nix)>,
}

impl<'scope, I, Eix, Nix> Walks<'scope, I, Eix, Nix> {
    fn new(inner: I) -> Self {
        Self {
            inner,
            _scope: PhantomData,
            _marker: PhantomData,
        }
    }
}

// Brand the indices of each `WalkItem` with the scope, leaving the (erased)
// edge pointer untouched — so no deref and no `Edge: 'r` bound. The two impls
// are disjoint (an iterator has a single `Item`).
impl<'scope, 'r, I, Eix, E: ?Sized, Nix> Iterator for Walks<'scope, I, Eix, Nix>
where
    I: Iterator<Item = WalkItem<'r, Eix, E, Nix>>,
{
    type Item = WalkItem<'r, EdgeIx<'scope, Eix>, E, NodeIx<'scope, Nix>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|wi| {
            let (eix, edge_ptr, nix) = wi.into_parts();
            // SAFETY: `edge_ptr` is valid for `'r` (from the inner `WalkItem`).
            unsafe {
                WalkItem::from_parts(EdgeIx(PhantomData, eix), edge_ptr, NodeIx(PhantomData, nix))
            }
        })
    }
}

/// Mutable counterpart of [`Walks`] (separate struct because coherence cannot
/// see that an iterator's `Item` is either a `WalkItem` or a `WalkItemMut`,
/// never both).
pub struct WalksMut<'scope, I, Eix, Nix> {
    inner: I,
    _scope: crate::Invariant<'scope>,
    _marker: PhantomData<(Eix, Nix)>,
}

impl<'scope, I, Eix, Nix> WalksMut<'scope, I, Eix, Nix> {
    fn new(inner: I) -> Self {
        Self {
            inner,
            _scope: PhantomData,
            _marker: PhantomData,
        }
    }
}

impl<'scope, 'r, I, Eix, E: ?Sized, Nix> Iterator for WalksMut<'scope, I, Eix, Nix>
where
    I: Iterator<Item = WalkItemMut<'r, Eix, E, Nix>>,
{
    type Item = WalkItemMut<'r, EdgeIx<'scope, Eix>, E, NodeIx<'scope, Nix>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|wi| {
            let (eix, edge_ptr, nix) = wi.into_parts();
            // SAFETY: `edge_ptr` is uniquely valid for `'r` (from the inner item).
            unsafe {
                WalkItemMut::from_parts(
                    EdgeIx(PhantomData, eix),
                    edge_ptr,
                    NodeIx(PhantomData, nix),
                )
            }
        })
    }
}

/// A mutable, scope-branded graph context that supports node/edge removal.
///
/// Obtained from [`Graph::scope_mut`](super::Graph::scope_mut). Derefs to
/// [`Context`], so all read and insert operations are available through it.
/// Removal methods consume `self` so that branded indices from the same
/// scope cannot be used after the topology changes.
///
/// ```rust
/// use safegraph::graph::Graph;
/// use safegraph::VecGraph;
///
/// let mut g = VecGraph::<u32, u32>::default();
/// g.push(10).unwrap();
/// g.push(20).unwrap();
/// g.scope_mut(|ctx| {
///     let n: Vec<_> = ctx.node_indices().collect();
///     ctx.remove_nodes_edges(n, []);
/// });
/// assert_eq!(g.len_node(), 0);
/// ```
pub struct RemovableContext<'r, 'scope, G: ?Sized> {
    _scope: crate::Invariant<'scope>,
    context: &'r mut Context<'scope, G>,
}

fn dedup_checked<I>(iter: impl Iterator<Item = I>) -> Vec<I>
where
    I: Copy + Eq + core::hash::Hash + Display,
{
    let mut seen = std::collections::HashSet::with_capacity(iter.size_hint().0);
    iter.map(|ix| {
        assert!(seen.insert(ix), "duplicate index {} in batch removal", ix);
        ix
    })
    .collect()
}

#[derive(Default)]
struct Sink;
impl<T> Extend<T> for Sink {
    fn extend<I: IntoIterator<Item = T>>(&mut self, _iter: I) {}
}

impl<'r, 'scope, G: ?Sized> RemovableContext<'r, 'scope, G> {
    pub(crate) fn new(context: &'r mut Context<'scope, G>) -> Self {
        Self {
            context,
            _scope: core::marker::PhantomData,
        }
    }

    /// Removes the given nodes and edges, discarding their data.
    ///
    /// Edges are removed before nodes so that incident-edge cleanup
    /// works correctly. Consumes `self` to prevent use of stale branded
    /// indices.
    ///
    /// # Panics
    /// Panics if any index appears more than once.
    pub fn remove_nodes_edges(
        self,
        node_indices: impl IntoIterator<Item = NodeIx<'scope, G::NodeIx>>,
        edge_indices: impl IntoIterator<Item = EdgeIx<'scope, G::EdgeIx>>,
    ) where
        G: GraphOperation<'r> + RemoveNode,
        G::Endpoints: Map<NodeIx<'scope, G::NodeIx>>,
    {
        // Share the single-element fast path and the general path with
        // `take_nodes_edges`; discard the returned payloads.
        let _: (Sink, Sink) = self.take_nodes_edges(node_indices, edge_indices);
    }

    /// Removes the given nodes and edges, returning their data.
    ///
    /// Edges are removed before nodes so that incident-edge cleanup
    /// works correctly. Consumes `self` to prevent use of stale branded
    /// indices.
    ///
    /// The return type parameters `IN` and `IE` are the collections that
    /// receive the removed node and edge payloads (e.g. `Vec<N>`,
    /// `Vec<E>`).
    ///
    /// # Panics
    /// Panics if any index appears more than once.
    pub fn take_nodes_edges<IN, IE>(
        mut self,
        node_indices: impl IntoIterator<Item = NodeIx<'scope, G::NodeIx>>,
        edge_indices: impl IntoIterator<Item = EdgeIx<'scope, G::EdgeIx>>,
    ) -> (IN, IE)
    where
        G: GraphOperation<'r> + RemoveNode,
        G::Endpoints: Map<NodeIx<'scope, G::NodeIx>>,
        IN: Default + Extend<G::Node>,
        IE: Default + Extend<G::Edge>,
    {
        fn internal<'r, 'scope, IN, IE, G>(
            graph: &mut RemovableContext<'r, 'scope, G>,
            node_indices: impl Iterator<Item = NodeIx<'scope, G::NodeIx>>,
            edge_indices: impl Iterator<Item = EdgeIx<'scope, G::EdgeIx>>,
        ) -> (IN, IE)
        where
            G: GraphOperation<'r> + RemoveNode + ?Sized,
            G::Endpoints: Map<NodeIx<'scope, G::NodeIx>>,
            IN: Default + Extend<G::Node>,
            IE: Default + Extend<G::Edge>,
        {
            let node_indices = dedup_checked(node_indices.map(|NodeIx(_, nix)| nix));
            let edge_indices = dedup_checked(edge_indices.map(|EdgeIx(_, eix)| eix));
            // SAFETY: scoped indices are produced from this context; duplicates
            // rejected above.
            unsafe {
                <G as RemoveNode>::take_nodes_edges_unchecked(
                    &mut graph.context.graph,
                    node_indices,
                    edge_indices,
                )
            }
        }

        let mut node_indices = node_indices.into_iter();
        let mut edge_indices = edge_indices.into_iter();
        if matches!(
            (node_indices.size_hint().1, edge_indices.size_hint().1),
            (Some(0), Some(0)) | (Some(0), Some(1)) | (Some(1), Some(0)) | (Some(1), Some(1))
        ) {
            let mut nodes_out = IN::default();
            let mut edges_out = IE::default();
            let first_edge_index = edge_indices.next();
            let first_node_index = node_indices.next();
            if let Some(EdgeIx(_, eix)) = first_edge_index {
                // SAFETY: scoped indices are produced from this context and
                // stay valid for the scope.
                let e =
                    unsafe { <G as RemoveEdge>::take_edge_unchecked(&mut self.context.graph, eix) };
                edges_out.extend(core::iter::once(e));
            }
            if let Some(NodeIx(_, nix)) = first_node_index {
                // SAFETY: scoped indices are produced from this context and
                // stay valid for the scope.
                let n =
                    unsafe { <G as RemoveNode>::take_node_unchecked(&mut self.context.graph, nix) };
                nodes_out.extend(core::iter::once(n));
            }

            // Most cases, collect() operation and memory allocation for zero-sized iterator is
            // thrown away during optimization, so the deligated iterator and Vec is not costly.
            let (nodes_out_internal, edges_out_internal): (Vec<_>, Vec<_>) = internal(
                &mut self,
                node_indices.filter(|n| first_node_index.map(|ix| &ix == n) != Some(true)),
                edge_indices.filter(|n| first_edge_index.map(|ix| &ix == n) != Some(true)),
            );
            nodes_out.extend(nodes_out_internal);
            edges_out.extend(edges_out_internal);
            (nodes_out, edges_out)
        } else {
            internal(&mut self, node_indices, edge_indices)
        }
    }

    /// Removes every node and edge from the graph.
    ///
    /// Consumes `self` to prevent use of stale branded indices.
    pub fn clear(self)
    where
        G: for<'rr> GraphOperation<'rr> + RemoveNode,
        G::Endpoints: Map<NodeIx<'scope, G::NodeIx>>,
    {
        // SAFETY: indices are collected into owned `Vec`s (no live borrows)
        // before the mutable removal call.
        let edges: Vec<_> = <G as GraphOperation<'_>>::edge_indices(&self.context.graph).collect();
        let nodes: Vec<_> = <G as GraphOperation<'_>>::node_indices(&self.context.graph).collect();
        let _: (Sink, Sink) = unsafe {
            <G as RemoveNode>::take_nodes_edges_unchecked(&mut self.context.graph, nodes, edges)
        };
    }
}

impl<'r, 'scope, G: ?Sized> Deref for RemovableContext<'r, 'scope, G> {
    type Target = Context<'scope, G>;

    fn deref(&self) -> &Self::Target {
        self.context
    }
}

impl<'r, 'scope, G: ?Sized> DerefMut for RemovableContext<'r, 'scope, G> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.context
    }
}

pub struct NodeIxIter<'scope, I> {
    iter: I,
    _scope: crate::Invariant<'scope>,
}

impl<'scope, I> Iterator for NodeIxIter<'scope, I>
where
    I: Iterator,
{
    type Item = NodeIx<'scope, I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ix| NodeIx(PhantomData, ix))
    }
}

pub struct EdgeIxIter<'scope, I> {
    iter: I,
    _scope: crate::Invariant<'scope>,
}

impl<'scope, I> Iterator for EdgeIxIter<'scope, I>
where
    I: Iterator,
{
    type Item = EdgeIx<'scope, I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ix| EdgeIx(PhantomData, ix))
    }
}

impl<'scope, 'r, G> GraphOperation<'r> for Context<'scope, G>
where
    G: for<'rr> GraphOperation<'rr> + ?Sized,
    <G as GraphProperty>::Endpoints: Map<NodeIx<'scope, <G as GraphProperty>::NodeIx>>,
{
    #[inline]
    fn contains_node_index(&self, NodeIx(_, _node_ix): Self::NodeIx) -> bool {
        true
    }

    #[inline]
    fn contains_edge_index(&self, EdgeIx(_, _edge_ix): Self::EdgeIx) -> bool {
        true
    }

    type NodeIndices = NodeIxIter<'scope, <G as GraphOperation<'r>>::NodeIndices>;
    type EdgeIndices = EdgeIxIter<'scope, <G as GraphOperation<'r>>::EdgeIndices>;

    #[inline]
    fn node_indices(&'r self) -> Self::NodeIndices {
        NodeIxIter {
            iter: <G as GraphOperation<'r>>::node_indices(&self.graph),
            _scope: PhantomData,
        }
    }

    #[inline]
    fn edge_indices(&'r self) -> Self::EdgeIndices {
        EdgeIxIter {
            iter: <G as GraphOperation<'r>>::edge_indices(&self.graph),
            _scope: PhantomData,
        }
    }

    #[inline]
    unsafe fn node_unchecked(&self, NodeIx(_, node_ix): Self::NodeIx) -> &Self::Node {
        <G as GraphOperation<'_>>::node_unchecked(&self.graph, node_ix)
    }

    #[inline]
    unsafe fn edge_unchecked(&self, EdgeIx(_, edge_ix): Self::EdgeIx) -> &Self::Edge {
        <G as GraphOperation<'_>>::edge_unchecked(&self.graph, edge_ix)
    }

    #[inline]
    unsafe fn endpoints_unchecked(&self, EdgeIx(_, edge_ix): Self::EdgeIx) -> Self::Endpoints {
        <G as GraphOperation<'_>>::endpoints_unchecked(&self.graph, edge_ix)
            .map_forward(|nix| NodeIx(PhantomData, nix))
    }

    type EdgeIndicesFrom = EdgeIxIter<'scope, <G as GraphOperation<'r>>::EdgeIndicesFrom>;

    #[inline]
    unsafe fn edge_indices_from_unchecked(
        &'r self,
        NodeIx(_, node_ix): Self::NodeIx,
    ) -> Self::EdgeIndicesFrom {
        EdgeIxIter {
            iter: unsafe {
                <G as GraphOperation<'r>>::edge_indices_from_unchecked(&self.graph, node_ix)
            },
            _scope: PhantomData,
        }
    }

    type EdgeIndicesOf = EdgeIxIter<'scope, <G as GraphOperation<'r>>::EdgeIndicesOf>;

    #[inline]
    unsafe fn edge_indices_of_unchecked(
        &'r self,
        NodeIx(_, node_ix): Self::NodeIx,
    ) -> Self::EdgeIndicesOf {
        EdgeIxIter {
            iter: unsafe {
                <G as GraphOperation<'r>>::edge_indices_of_unchecked(&self.graph, node_ix)
            },
            _scope: PhantomData,
        }
    }

    type WalksFrom = Walks<
        'scope,
        <G as GraphOperation<'r>>::WalksFrom,
        <G as GraphProperty>::EdgeIx,
        <G as GraphProperty>::NodeIx,
    >;
    type WalksOf = Walks<
        'scope,
        <G as GraphOperation<'r>>::WalksOf,
        <G as GraphProperty>::EdgeIx,
        <G as GraphProperty>::NodeIx,
    >;

    #[inline]
    unsafe fn walks_from_unchecked(&'r self, NodeIx(_, node_ix): Self::NodeIx) -> Self::WalksFrom {
        Walks::new(unsafe { <G as GraphOperation<'r>>::walks_from_unchecked(&self.graph, node_ix) })
    }

    #[inline]
    unsafe fn walks_of_unchecked(&'r self, NodeIx(_, node_ix): Self::NodeIx) -> Self::WalksOf {
        Walks::new(unsafe { <G as GraphOperation<'r>>::walks_of_unchecked(&self.graph, node_ix) })
    }

    fn len_node(&self) -> usize {
        <G as GraphOperation<'_>>::len_node(&self.graph)
    }

    fn len_edge(&self) -> usize {
        <G as GraphOperation<'_>>::len_edge(&self.graph)
    }

    fn capacity_node(&self) -> Option<usize> {
        <G as GraphOperation<'_>>::capacity_node(&self.graph)
    }

    fn capacity_edge(&self) -> Option<usize> {
        <G as GraphOperation<'_>>::capacity_edge(&self.graph)
    }

    type DrainNode = std::vec::IntoIter<Self::Node>;
    type DrainEdge = std::vec::IntoIter<Self::Edge>;

    fn drain(self) -> (Self::DrainNode, Self::DrainEdge)
    where
        Self: Sized,
    {
        // Context is only constructed by transmuting a reference; it is never
        // owned directly, so `drain` should never be called on it.
        unreachable!("Context does not own graph data and cannot be drained")
    }

    fn reverse(&mut self) {
        <G as GraphOperation<'_>>::reverse(&mut self.graph)
    }
}

impl<'scope, G> GraphProperty for Context<'scope, G>
where
    G: GraphProperty + ?Sized,
    G::Endpoints: Map<NodeIx<'scope, G::NodeIx>>,
{
    type Node = G::Node;
    type Edge = G::Edge;
    type NodeIx = NodeIx<'scope, G::NodeIx>;
    type EdgeIx = EdgeIx<'scope, G::EdgeIx>;
    type Endpoints = <G::Endpoints as Map<NodeIx<'scope, G::NodeIx>>>::Mapped;
    const DIRECTED: bool = G::DIRECTED;
}

impl<'scope, 'r, G: 'r + ?Sized> Directed<'r> for Context<'scope, G>
where
    G: Directed<'r>,
    Self: super::Graph<
        NodeIx = NodeIx<'scope, G::NodeIx>,
        EdgeIx = EdgeIx<'scope, G::EdgeIx>,
        Node = G::Node,
        Edge = G::Edge,
    >,
{
    type EdgeIndicesTo = EdgeIxIter<'scope, G::EdgeIndicesTo>;
    type EdgeTailIndices = NodeIxIter<'scope, G::EdgeTailIndices>;
    type EdgeHeadIndices = NodeIxIter<'scope, G::EdgeHeadIndices>;
    type WalksTo = std::iter::Map<
        G::WalksTo,
        fn(
            WalkItemTo<
                'r,
                <G as GraphProperty>::NodeIx,
                <G as GraphProperty>::EdgeIx,
                <G as GraphProperty>::Edge,
            >,
        ) -> WalkItemTo<
            'r,
            NodeIx<'scope, <G as GraphProperty>::NodeIx>,
            EdgeIx<'scope, <G as GraphProperty>::EdgeIx>,
            <G as GraphProperty>::Edge,
        >,
    >;

    unsafe fn walks_to_unchecked(&'r self, NodeIx(_, node_ix): Self::NodeIx) -> Self::WalksTo {
        <G as Directed<'r>>::walks_to_unchecked(&self.graph, node_ix).map(
            (|wi: WalkItemTo<'r, G::NodeIx, G::EdgeIx, G::Edge>| {
                let (nix, eix, edge_ptr) = wi.into_parts();
                // SAFETY: `edge_ptr` is valid for `'r` (from the inner item).
                unsafe {
                    WalkItemTo::from_parts(
                        NodeIx(PhantomData, nix),
                        EdgeIx(PhantomData, eix),
                        edge_ptr,
                    )
                }
            }) as fn(_) -> _,
        )
    }

    unsafe fn edge_indices_to_unchecked(
        &'r self,
        NodeIx(_, node_ix): Self::NodeIx,
    ) -> Self::EdgeIndicesTo {
        EdgeIxIter {
            iter: <G as Directed<'r>>::edge_indices_to_unchecked(&self.graph, node_ix),
            _scope: PhantomData,
        }
    }

    unsafe fn edge_tail_indices_unchecked(
        &'r self,
        EdgeIx(_, edge_ix): Self::EdgeIx,
    ) -> Self::EdgeTailIndices {
        NodeIxIter {
            iter: <G as Directed<'r>>::edge_tail_indices_unchecked(&self.graph, edge_ix),
            _scope: PhantomData,
        }
    }

    unsafe fn edge_head_indices_unchecked(
        &'r self,
        EdgeIx(_, edge_ix): Self::EdgeIx,
    ) -> Self::EdgeHeadIndices {
        NodeIxIter {
            iter: <G as Directed<'r>>::edge_head_indices_unchecked(&self.graph, edge_ix),
            _scope: PhantomData,
        }
    }
}

impl<'scope, G> Bigraph for Context<'scope, G>
where
    G: for<'rr> GraphOperation<'rr> + Bigraph + ?Sized,
    <G as GraphProperty>::Endpoints: Map<NodeIx<'scope, <G as GraphProperty>::NodeIx>>,
{
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2] {
        let raw = <<G as GraphProperty>::Endpoints as Map<NodeIx<'scope, G::NodeIx>>>::map_backward(
            endpoints,
            |NodeIx(_, nix)| nix,
        );
        <[G::NodeIx; 2] as Map<NodeIx<'scope, G::NodeIx>>>::map_forward(
            <G as Bigraph>::endpoints_as_array(raw),
            |inner| NodeIx(PhantomData, inner),
        )
    }

    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints {
        let raw = <[G::NodeIx; 2] as Map<NodeIx<'scope, G::NodeIx>>>::map_backward(
            nodes,
            |NodeIx(_, nix)| nix,
        );
        <<G as GraphProperty>::Endpoints as Map<NodeIx<'scope, G::NodeIx>>>::map_forward(
            <G as Bigraph>::endpoints_from_array(raw),
            |inner| NodeIx(PhantomData, inner),
        )
    }
}

impl<'r, 'scope, G: ?Sized> UpdateNode<'r> for Context<'scope, G>
where
    Self: super::Graph<
        Edge = G::Edge,
        EdgeIx = EdgeIx<'scope, G::EdgeIx>,
        Node = G::Node,
        NodeIx = NodeIx<'scope, G::NodeIx>,
    >,
    G: UpdateNode<'r>,
    G::Edge: 'r,
    G::Endpoints: Map<NodeIx<'scope, G::NodeIx>, Mapped = Self::Endpoints>,
{
    unsafe fn node_unchecked_mut(&mut self, NodeIx(_, node_ix): Self::NodeIx) -> &mut Self::Node {
        <G as UpdateNode<'_>>::node_unchecked_mut(&mut self.graph, node_ix)
    }

    type WalksFromMut = WalksMut<
        'scope,
        <G as UpdateNode<'r>>::WalksFromMut,
        <G as GraphProperty>::EdgeIx,
        <G as GraphProperty>::NodeIx,
    >;
    unsafe fn walks_from_unchecked_mut(
        &'r mut self,
        NodeIx(_, node_ix): Self::NodeIx,
    ) -> Self::WalksFromMut {
        WalksMut::new(<G as UpdateNode<'r>>::walks_from_unchecked_mut(
            &mut self.graph,
            node_ix,
        ))
    }

    type WalksOfMut = WalksMut<
        'scope,
        <G as UpdateNode<'r>>::WalksOfMut,
        <G as GraphProperty>::EdgeIx,
        <G as GraphProperty>::NodeIx,
    >;
    unsafe fn walks_of_unchecked_mut(
        &'r mut self,
        NodeIx(_, node_ix): Self::NodeIx,
    ) -> Self::WalksOfMut {
        WalksMut::new(<G as UpdateNode<'r>>::walks_of_unchecked_mut(
            &mut self.graph,
            node_ix,
        ))
    }
}

impl<'scope, G: ?Sized> UpdateEdge for Context<'scope, G>
where
    Self: super::Graph<Edge = G::Edge, EdgeIx = EdgeIx<'scope, G::EdgeIx>>,
    G: UpdateEdge,
{
    unsafe fn edge_unchecked_mut(&mut self, EdgeIx(_, edge_ix): Self::EdgeIx) -> &mut Self::Edge {
        <G as UpdateEdge>::edge_unchecked_mut(&mut self.graph, edge_ix)
    }
}

// SAFETY: Within a scope, node indices are inherently stable:
// - `InsertNode` guarantees insertion does not invalidate existing indices.
// - Removal via `RemovableContext` consumes the context, preventing use of
//   invalidated indices.
// - `contains_node_index` returns `true` unconditionally for scoped indices.
unsafe impl<'scope, G: ?Sized> StableNode for Context<'scope, G> where Self: super::Graph {}

// SAFETY: Same reasoning as StableNode — edge indices are stable within a scope.
unsafe impl<'scope, G: ?Sized> StableEdge for Context<'scope, G> where Self: super::Graph {}

impl<'scope, G: ?Sized> InsertNode for Context<'scope, G>
where
    G: InsertNode,
    Self: super::Graph<NodeIx = NodeIx<'scope, G::NodeIx>, Node = G::Node>,
{
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node> {
        <G as InsertNode>::insert_node_unchecked(&mut self.graph, node)
            .map(|ix| NodeIx(PhantomData, ix))
    }
}

impl<'scope, G: ?Sized> InsertEdge for Context<'scope, G>
where
    G: InsertEdge,
    Self: super::Graph<EdgeIx = EdgeIx<'scope, G::EdgeIx>, Edge = G::Edge>,
    G::Endpoints: Map<NodeIx<'scope, G::NodeIx>, Mapped = Self::Endpoints>,
{
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge> {
        <G as InsertEdge>::insert_edge_unchecked(
            &mut self.graph,
            edge,
            <<G as super::GraphProperty>::Endpoints>::map_backward(endpoints, |NodeIx(_, ix)| ix),
        )
        .map(|ix| EdgeIx(PhantomData, ix))
    }
}

impl<'scope, G: ?Sized> UniqueNode for Context<'scope, G>
where
    G: super::Graph + UniqueNode,
    Self: super::Graph<NodeIx = NodeIx<'scope, G::NodeIx>, Node = G::Node>,
{
    fn node_index(&self, node: impl Borrow<Self::Node>) -> Option<Self::NodeIx> {
        <G as UniqueNode>::node_index(&self.graph, node).map(|ix| NodeIx(PhantomData, ix))
    }
}
impl<'scope, G: ?Sized> UniqueEdge for Context<'scope, G>
where
    G: super::Graph + UniqueEdge,
    Self: super::Graph<EdgeIx = EdgeIx<'scope, G::EdgeIx>, Edge = G::Edge>,
{
    fn edge_index(&self, edge: impl Borrow<Self::Edge>) -> Option<Self::EdgeIx> {
        <G as UniqueEdge>::edge_index(&self.graph, edge).map(|ix| EdgeIx(PhantomData, ix))
    }
}
