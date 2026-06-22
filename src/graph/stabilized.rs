//! Tombstone-versioned graph wrapper providing stable node and edge indices.
//!
//! Each node and edge in the inner graph is stored as `(version: i64, data)`.
//! A positive version means the entry is live; a negative version means it has
//! been tombstoned (soft-deleted). On insert, tombstoned slots may be reused
//! for nodes; edges are always appended.

use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;

use super::capability::*;
use super::edge::Map;
use super::walk_item::{WalkItem, WalkItemMut, WalkItemTo};
use super::{Graph, GraphOperation, GraphProperty};

pub struct Walks<'r, G, N, E, I>
where
    G: super::GraphProperty<Node = NodeIx<N>, Edge = EdgeIx<E>> + ?Sized,
{
    pub(crate) graph: &'r G,
    pub(crate) inner: I,
    pub(crate) _marker: PhantomData<(N, E)>,
}

impl<'r, G, N, E, I> Iterator for Walks<'r, G, N, E, I>
where
    G: super::GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>> + ?Sized,
    I: Iterator<Item = WalkItem<'r, G::EdgeIx, EdgeIx<E>, G::NodeIx>>,
{
    type Item = WalkItem<'r, EdgeIx<G::EdgeIx>, E, NodeIx<G::NodeIx>>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (eix, edge_ptr, nix) = self.inner.next()?.into_parts();
            // `edge_ptr: *const EdgeIx<E>` is valid for `'r`. Read the version
            // and reproject to `&E` via raw pointers, so neither `E: 'r` nor
            // `N: 'r` is required (which would force `'static` under `for<'r>`).
            // SAFETY: the pointer is valid for `'r` (from the inner `WalkItem`).
            let ver = unsafe { (*edge_ptr).version };
            if ver <= 0 {
                continue;
            }
            // The neighbor node must carry its live version so the yielded index
            // is valid for lookup/comparison; `0` would be rejected by
            // `contains_node_index`. SAFETY: `nix` came from the inner graph's
            // own walk iterator, so it is a valid node index.
            let nver = version_of(unsafe { self.graph.node_unchecked(nix) });
            // SAFETY: `edge_ptr` points to a live `EdgeIx<E>` for `'r`, so its
            // `inner: E` field address is valid for `'r`.
            let inner_ptr: *const E = unsafe { core::ptr::addr_of!((*edge_ptr).inner) };
            return Some(unsafe {
                WalkItem::from_parts(
                    EdgeIx {
                        version: ver,
                        inner: eix,
                    },
                    inner_ptr,
                    NodeIx {
                        version: nver,
                        inner: nix,
                    },
                )
            });
        }
    }
}

pub struct WalksMut<'r, G, N, E, I>
where
    G: super::GraphProperty<Node = NodeIx<N>, Edge = EdgeIx<E>> + ?Sized,
{
    pub(crate) inner: I,
    pub(crate) _marker: PhantomData<(&'r mut G, N, E)>,
}

impl<'r, G, N, E, I> Iterator for WalksMut<'r, G, N, E, I>
where
    G: super::GraphProperty<Node = NodeIx<N>, Edge = EdgeIx<E>> + ?Sized,
    I: Iterator<Item = WalkItemMut<'r, G::EdgeIx, EdgeIx<E>, G::NodeIx>>,
{
    type Item = WalkItemMut<'r, EdgeIx<G::EdgeIx>, E, NodeIx<G::NodeIx>>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (eix, edge_ptr, nix) = self.inner.next()?.into_parts();
            // SAFETY: `edge_ptr: *mut EdgeIx<E>` is uniquely valid for `'r`.
            let ver = unsafe { (*edge_ptr).version };
            if ver <= 0 {
                continue;
            }
            // SAFETY: project to the `inner: E` field; valid and unique for `'r`.
            let inner_ptr: *mut E = unsafe { core::ptr::addr_of_mut!((*edge_ptr).inner) };
            return Some(unsafe {
                WalkItemMut::from_parts(
                    EdgeIx {
                        version: ver,
                        inner: eix,
                    },
                    inner_ptr,
                    NodeIx {
                        version: 0,
                        inner: nix,
                    },
                )
            });
        }
    }
}

pub struct WalksTo<'r, G, N, E, I>
where
    G: super::GraphProperty<Node = NodeIx<N>, Edge = EdgeIx<E>> + ?Sized,
{
    pub(crate) graph: &'r G,
    pub(crate) inner: I,
    pub(crate) _marker: PhantomData<(N, E)>,
}

impl<'r, G, N, E, I> Iterator for WalksTo<'r, G, N, E, I>
where
    G: super::GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>> + ?Sized,
    I: Iterator<Item = WalkItemTo<'r, G::NodeIx, G::EdgeIx, EdgeIx<E>>>,
{
    type Item = WalkItemTo<'r, NodeIx<G::NodeIx>, EdgeIx<G::EdgeIx>, E>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (nix, eix, edge_ptr) = self.inner.next()?.into_parts();
            // SAFETY: `edge_ptr: *const EdgeIx<E>` is valid for `'r`.
            let ver = unsafe { (*edge_ptr).version };
            if ver <= 0 {
                continue;
            }
            // SAFETY: `nix` came from the inner graph's own walk iterator; look up
            // its live version so the yielded index is valid (see `Walks`).
            let nver = version_of(unsafe { self.graph.node_unchecked(nix) });
            // SAFETY: project to the `inner: E` field; valid for `'r`.
            let inner_ptr: *const E = unsafe { core::ptr::addr_of!((*edge_ptr).inner) };
            return Some(unsafe {
                WalkItemTo::from_parts(
                    NodeIx {
                        version: nver,
                        inner: nix,
                    },
                    EdgeIx {
                        version: ver,
                        inner: eix,
                    },
                    inner_ptr,
                )
            });
        }
    }
}

pub struct NodeIndices<'r, N, E, G, I> {
    graph: &'r G,
    inner: I,
    _marker: PhantomData<(N, E)>,
}

impl<'r, N, E, G, I> Iterator for NodeIndices<'r, N, E, G, I>
where
    G: 'r + GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>>,
    N: 'r,
    E: 'r,
    I: Iterator<Item = G::NodeIx>,
{
    type Item = NodeIx<G::NodeIx>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let inner_ix = self.inner.next()?;
            // SAFETY: index comes from a graph-derived iterator.
            let ver = unsafe { self.graph.node_unchecked(inner_ix) }.version;
            if ver > 0 {
                return Some(NodeIx {
                    version: ver,
                    inner: inner_ix,
                });
            }
        }
    }
}

pub struct EdgeIndices<'r, N, E, G, I> {
    graph: &'r G,
    inner: I,
    _marker: PhantomData<(N, E)>,
}

impl<'r, N, E, G, I> Iterator for EdgeIndices<'r, N, E, G, I>
where
    G: 'r + GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>>,
    N: 'r,
    E: 'r,
    I: Iterator<Item = G::EdgeIx>,
{
    type Item = EdgeIx<G::EdgeIx>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let inner_eix = self.inner.next()?;
            // SAFETY: index comes from a graph-derived iterator.
            let ver = unsafe { version_of(self.graph.edge_unchecked(inner_eix)) };
            if ver > 0 {
                return Some(EdgeIx {
                    version: ver,
                    inner: inner_eix,
                });
            }
        }
    }
}

/// A graph wrapper that provides stable indices via tombstone versioning.
///
/// Created by [`crate::graph::Graph::stabilize()`]. The inner graph stores versioned data
/// `NodeIx<N>` / `EdgeIx<E>` where the sign of the version indicates liveness.
#[derive(Clone, Debug)]
pub struct Stabilized<G, N, E> {
    inner: G,
    live_nodes: usize,
    live_edges: usize,
    _marker: PhantomData<(N, E)>,
}

impl<G, N, E> Stabilized<G, N, E> {
    /// Construct a `Stabilized` wrapper from a pre-mapped graph whose node data
    /// is `NodeIx<N>` and edge data is `EdgeIx<E>`, with all versions initially 1.
    pub(crate) fn from_mapped(inner: G, live_nodes: usize, live_edges: usize) -> Self {
        Self {
            inner,
            live_nodes,
            live_edges,
            _marker: PhantomData,
        }
    }

    /// Number of live (non-tombstoned) nodes.
    pub fn live_nodes(&self) -> usize {
        self.live_nodes
    }

    /// Number of live (non-tombstoned) edges.
    pub fn live_edges(&self) -> usize {
        self.live_edges
    }
}

impl<G, N, E> Stabilized<G, N, E>
where
    G: Graph<Node = NodeIx<N>, Edge = EdgeIx<E>> + RemoveNode,
{
    /// Consume the wrapper and return the inner graph with every tombstoned
    /// (soft-deleted) node and edge physically removed.
    ///
    /// `Stabilized` soft-deletes by flipping an entry's version sign, so the
    /// inner graph still physically holds removed entries. This purges them via
    /// `G`'s [`RemoveNode`] / [`RemoveEdge`], leaving only the live data — so it
    /// requires `G: Remove{Node,Edge}`. The returned graph keeps the
    /// version-tagged `NodeIx<N>` / `EdgeIx<E>` payloads (all now positive).
    pub fn into_inner(mut self) -> G {
        // Tombstoned (version <= 0) inner indices. Collect before mutating so the
        // read borrows end first; node indices are unaffected by edge removal, so
        // both lists stay valid for their respective pass.
        let dead_edges: Vec<G::EdgeIx> = <G as GraphOperation<'_>>::edge_indices(&self.inner)
            .filter(|&ix| {
                version_of(unsafe { <G as GraphOperation<'_>>::edge_unchecked(&self.inner, ix) })
                    <= 0
            })
            .collect();
        let dead_nodes: Vec<G::NodeIx> = <G as GraphOperation<'_>>::node_indices(&self.inner)
            .filter(|&ix| {
                version_of(unsafe { <G as GraphOperation<'_>>::node_unchecked(&self.inner, ix) })
                    <= 0
            })
            .collect();

        // Edges first (no cascade); afterwards every tombstoned node is
        // edge-free, so the node pass cascades nothing.
        //
        // SAFETY: every collected index is currently live in `inner`; the
        // backend's batch removal handles descending order / swap-remove
        // relocation within each pass.
        let _: (Vec<G::Node>, Vec<G::Edge>) = unsafe {
            <G as RemoveNode>::take_nodes_edges_unchecked(
                &mut self.inner,
                core::iter::empty::<G::NodeIx>(),
                dead_edges,
            )
        };
        let _: (Vec<G::Node>, Vec<G::Edge>) = unsafe {
            <G as RemoveNode>::take_nodes_edges_unchecked(
                &mut self.inner,
                dead_nodes,
                core::iter::empty::<G::EdgeIx>(),
            )
        };
        self.inner
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeIx<Ix> {
    pub(crate) version: i64,
    pub(crate) inner: Ix,
}

impl<Ix: Display> Display for NodeIx<Ix> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}:{}", self.version, self.inner)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EdgeIx<Ix> {
    pub(crate) version: i64,
    pub(crate) inner: Ix,
}

impl<Ix: Display> Display for EdgeIx<Ix> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}:{}", self.version, self.inner)
    }
}

trait Versioned {
    fn ver(&self) -> i64;
}
impl<T> Versioned for NodeIx<T> {
    #[inline]
    fn ver(&self) -> i64 {
        self.version
    }
}
impl<T> Versioned for EdgeIx<T> {
    #[inline]
    fn ver(&self) -> i64 {
        self.version
    }
}

#[inline]
fn version_of<T: Versioned>(pair: &T) -> i64 {
    pair.ver()
}

impl<'r, G: 'r, N: 'r, E: 'r> GraphOperation<'r> for Stabilized<G, N, E>
where
    G: GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>>,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    fn contains_node_index(&self, ix: Self::NodeIx) -> bool {
        if ix.version <= 0 || !self.inner.contains_node_index(ix.inner) {
            return false;
        }
        // SAFETY: we checked contains_node_index above.
        let ver = unsafe { version_of(self.inner.node_unchecked(ix.inner)) };
        ver == ix.version
    }

    fn contains_edge_index(&self, ix: Self::EdgeIx) -> bool {
        if ix.version <= 0 || !self.inner.contains_edge_index(ix.inner) {
            return false;
        }
        let ver = unsafe { version_of(self.inner.edge_unchecked(ix.inner)) };
        ver == ix.version
    }

    fn len_node(&self) -> usize {
        self.live_nodes
    }

    fn len_edge(&self) -> usize {
        self.live_edges
    }

    fn capacity_node(&self) -> Option<usize> {
        self.inner.capacity_node()
    }

    fn capacity_edge(&self) -> Option<usize> {
        self.inner.capacity_edge()
    }

    type NodeIndices = NodeIndices<'r, N, E, G, G::NodeIndices>;
    type EdgeIndices = EdgeIndices<'r, N, E, G, G::EdgeIndices>;

    fn node_indices(&'r self) -> Self::NodeIndices {
        NodeIndices {
            graph: &self.inner,
            inner: self.inner.node_indices(),
            _marker: PhantomData,
        }
    }

    fn edge_indices(&'r self) -> Self::EdgeIndices {
        EdgeIndices {
            graph: &self.inner,
            inner: self.inner.edge_indices(),
            _marker: PhantomData,
        }
    }

    unsafe fn node_unchecked(&self, ix: Self::NodeIx) -> &Self::Node {
        &self.inner.node_unchecked(ix.inner).inner
    }

    unsafe fn edge_unchecked(&self, ix: Self::EdgeIx) -> &Self::Edge {
        &self.inner.edge_unchecked(ix.inner).inner
    }

    unsafe fn endpoints_unchecked(&self, ix: Self::EdgeIx) -> Self::Endpoints {
        <G as GraphOperation<'_>>::endpoints_unchecked(&self.inner, ix.inner).map_forward(|nix| {
            NodeIx {
                version: version_of(unsafe { self.inner.node_unchecked(nix) }),
                inner: nix,
            }
        })
    }

    type EdgeIndicesFrom = EdgeIndices<'r, N, E, G, G::EdgeIndicesFrom>;

    unsafe fn edge_indices_from_unchecked(&'r self, ix: Self::NodeIx) -> Self::EdgeIndicesFrom {
        EdgeIndices {
            graph: &self.inner,
            inner: self.inner.edge_indices_from_unchecked(ix.inner),
            _marker: PhantomData,
        }
    }

    type EdgeIndicesOf = EdgeIndices<'r, N, E, G, G::EdgeIndicesOf>;

    unsafe fn edge_indices_of_unchecked(&'r self, ix: Self::NodeIx) -> Self::EdgeIndicesOf {
        EdgeIndices {
            graph: &self.inner,
            inner: self.inner.edge_indices_of_unchecked(ix.inner),
            _marker: PhantomData,
        }
    }

    type WalksFrom = Walks<'r, G, N, E, <G as GraphOperation<'r>>::WalksFrom>;
    type WalksOf = Walks<'r, G, N, E, <G as GraphOperation<'r>>::WalksOf>;

    unsafe fn walks_from_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksFrom {
        Walks {
            graph: &self.inner,
            inner: unsafe {
                <G as GraphOperation<'r>>::walks_from_unchecked(&self.inner, node_ix.inner)
            },
            _marker: PhantomData,
        }
    }

    unsafe fn walks_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksOf {
        Walks {
            graph: &self.inner,
            inner: unsafe {
                <G as GraphOperation<'r>>::walks_of_unchecked(&self.inner, node_ix.inner)
            },
            _marker: PhantomData,
        }
    }

    type DrainNode = std::iter::FilterMap<G::DrainNode, fn(NodeIx<N>) -> Option<N>>;
    type DrainEdge = std::iter::FilterMap<G::DrainEdge, fn(EdgeIx<E>) -> Option<E>>;

    fn drain(self) -> (Self::DrainNode, Self::DrainEdge) {
        let (nodes, edges) = self.inner.drain();
        (
            nodes.filter_map(
                (|NodeIx { version, inner }| if version > 0 { Some(inner) } else { None })
                    as fn(NodeIx<N>) -> Option<N>,
            ),
            edges.filter_map(
                (|EdgeIx { version, inner }| if version > 0 { Some(inner) } else { None })
                    as fn(EdgeIx<E>) -> Option<E>,
            ),
        )
    }

    fn reverse(&mut self) {
        self.inner.reverse();
    }
}

impl<G, N, E> GraphProperty for Stabilized<G, N, E>
where
    G: GraphProperty<Node = NodeIx<N>, Edge = EdgeIx<E>>,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    type Node = N;
    type Edge = E;
    type NodeIx = NodeIx<G::NodeIx>;
    type EdgeIx = EdgeIx<G::EdgeIx>;
    type Endpoints = <G::Endpoints as Map<NodeIx<G::NodeIx>>>::Mapped;
    const DIRECTED: bool = G::DIRECTED;
}

impl<'r, G: 'r, N: 'r, E: 'r> Directed<'r> for Stabilized<G, N, E>
where
    G: GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>> + Directed<'r>,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    type EdgeIndicesTo = EdgeIndices<'r, N, E, G, <G as Directed<'r>>::EdgeIndicesTo>;
    type EdgeTailIndices = NodeIndices<'r, N, E, G, <G as Directed<'r>>::EdgeTailIndices>;
    type EdgeHeadIndices = NodeIndices<'r, N, E, G, <G as Directed<'r>>::EdgeHeadIndices>;
    type WalksTo = WalksTo<'r, G, N, E, <G as Directed<'r>>::WalksTo>;

    unsafe fn walks_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksTo {
        WalksTo {
            graph: &self.inner,
            inner: unsafe { <G as Directed<'r>>::walks_to_unchecked(&self.inner, node_ix.inner) },
            _marker: PhantomData,
        }
    }

    unsafe fn edge_indices_to_unchecked(&'r self, ix: Self::NodeIx) -> Self::EdgeIndicesTo {
        EdgeIndices {
            graph: &self.inner,
            inner: <G as Directed<'r>>::edge_indices_to_unchecked(&self.inner, ix.inner),
            _marker: PhantomData,
        }
    }

    unsafe fn edge_tail_indices_unchecked(&'r self, ix: Self::EdgeIx) -> Self::EdgeTailIndices {
        NodeIndices {
            graph: &self.inner,
            inner: unsafe {
                <G as Directed<'r>>::edge_tail_indices_unchecked(&self.inner, ix.inner)
            },
            _marker: PhantomData,
        }
    }

    unsafe fn edge_head_indices_unchecked(&'r self, ix: Self::EdgeIx) -> Self::EdgeHeadIndices {
        NodeIndices {
            graph: &self.inner,
            inner: unsafe {
                <G as Directed<'r>>::edge_head_indices_unchecked(&self.inner, ix.inner)
            },
            _marker: PhantomData,
        }
    }
}

impl<'r, G: 'r, N: 'r, E: 'r> Bigraph for Stabilized<G, N, E>
where
    G: GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>> + Bigraph,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2] {
        let mut versions = std::collections::HashMap::new();
        let raw = <G::Endpoints as Map<NodeIx<G::NodeIx>>>::map_backward(endpoints, |nix| {
            versions.insert(nix.inner, nix.version);
            nix.inner
        });
        <[G::NodeIx; 2] as Map<NodeIx<G::NodeIx>>>::map_forward(
            <G as Bigraph>::endpoints_as_array(raw),
            |inner| NodeIx {
                version: versions[&inner],
                inner,
            },
        )
    }

    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints {
        let mut versions = std::collections::HashMap::new();
        let raw = <[G::NodeIx; 2] as Map<NodeIx<G::NodeIx>>>::map_backward(nodes, |nix| {
            versions.insert(nix.inner, nix.version);
            nix.inner
        });
        <G::Endpoints as Map<NodeIx<G::NodeIx>>>::map_forward(
            <G as Bigraph>::endpoints_from_array(raw),
            |inner| NodeIx {
                version: versions[&inner],
                inner,
            },
        )
    }
}

// SAFETY: Tombstoned indices return false from contains_*_index; live indices
// always refer to the same node/edge (versions prevent ABA).
unsafe impl<'r, G: 'r, N: 'r, E: 'r> StableNode for Stabilized<G, N, E>
where
    G: GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>>,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
}

unsafe impl<'r, G: 'r, N: 'r, E: 'r> StableEdge for Stabilized<G, N, E>
where
    G: GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>>,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
}

impl<'r, G: 'r, N: 'r, E: 'r> UpdateNode<'r> for Stabilized<G, N, E>
where
    G: GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>> + UpdateNode<'r>,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    unsafe fn node_unchecked_mut(&mut self, ix: Self::NodeIx) -> &mut Self::Node {
        &mut <G as UpdateNode<'r>>::node_unchecked_mut(&mut self.inner, ix.inner).inner
    }

    type WalksFromMut = WalksMut<'r, G, N, E, <G as UpdateNode<'r>>::WalksFromMut>;
    unsafe fn walks_from_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksFromMut {
        WalksMut {
            inner: unsafe {
                <G as UpdateNode<'r>>::walks_from_unchecked_mut(&mut self.inner, node_ix.inner)
            },
            _marker: PhantomData,
        }
    }

    type WalksOfMut = WalksMut<'r, G, N, E, <G as UpdateNode<'r>>::WalksOfMut>;
    unsafe fn walks_of_unchecked_mut(&'r mut self, node_ix: Self::NodeIx) -> Self::WalksOfMut {
        WalksMut {
            inner: unsafe {
                <G as UpdateNode<'r>>::walks_of_unchecked_mut(&mut self.inner, node_ix.inner)
            },
            _marker: PhantomData,
        }
    }
}

impl<'r, G: 'r, N: 'r, E: 'r> UpdateEdge for Stabilized<G, N, E>
where
    G: GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>> + UpdateEdge,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    unsafe fn edge_unchecked_mut(&mut self, ix: Self::EdgeIx) -> &mut Self::Edge {
        &mut <G as UpdateEdge>::edge_unchecked_mut(&mut self.inner, ix.inner).inner
    }
}

impl<G, N, E> InsertNode for Stabilized<G, N, E>
where
    G: Graph<Node = NodeIx<N>, Edge = EdgeIx<E>> + InsertNode + for<'a> UpdateNode<'a>,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node> {
        // Try to reuse a tombstoned node slot rather than grow the inner graph.
        // Skip the scan when appending is cheap: either the backend reports
        // spare capacity (no reallocation), or — when it reports no capacity
        // info — there are no tombstones to reuse anyway (dead == 0, so the
        // scan would always fail). The latter keeps fresh builds O(n) instead
        // of O(n²).
        let can_append_cheaply = match Graph::capacity_node(&self.inner) {
            Some(cap) => cap > Graph::len_node(&self.inner),
            None => Graph::len_node(&self.inner) <= self.live_nodes,
        };
        if !can_append_cheaply {
            // Scan for a tombstoned slot to reuse.
            let tombstone = {
                <G as GraphOperation<'_>>::node_indices(&self.inner)
                    .find(|&ix| Graph::node_unchecked(&self.inner, ix).version < 0)
            };
            if let Some(inner_ix) = tombstone {
                let entry = <G as UpdateNode<'_>>::node_unchecked_mut(&mut self.inner, inner_ix);
                let new_version = (-entry.version) + 1;
                *entry = NodeIx {
                    version: new_version,
                    inner: node,
                };
                self.live_nodes += 1;
                return Ok(NodeIx {
                    version: new_version,
                    inner: inner_ix,
                });
            }
        }

        // Fall back to inner append.
        match <G as InsertNode>::insert_node_unchecked(
            &mut self.inner,
            NodeIx {
                version: 1,
                inner: node,
            },
        ) {
            Ok(inner_ix) => {
                self.live_nodes += 1;
                Ok(NodeIx {
                    version: 1,
                    inner: inner_ix,
                })
            }
            Err(NodeIx { inner: node, .. }) => Err(node),
        }
    }
}

impl<G, N, E> InsertEdge for Stabilized<G, N, E>
where
    G: Graph<Node = NodeIx<N>, Edge = EdgeIx<E>> + InsertEdge + UpdateEdge,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge> {
        // Strip the version tags back off to recover the inner graph's endpoints.
        // Works for any endpoint shape (binary array or hyperedge set) via `Map`.
        let inner_endpoints = <G::Endpoints as Map<NodeIx<G::NodeIx>>>::map_backward(
            endpoints,
            |NodeIx { inner, .. }| inner,
        );
        // See `insert_node_unchecked`: append-cheap when the backend reports
        // spare capacity, or (no capacity info) when there are no tombstones.
        let can_append_cheaply = match Graph::capacity_edge(&self.inner) {
            Some(cap) => cap > Graph::len_edge(&self.inner),
            None => Graph::len_edge(&self.inner) <= self.live_edges,
        };
        if !can_append_cheaply {
            // Reuse only tombstones whose stored endpoints match the requested
            // endpoints (`Endpoints: Eq`). Generic `Graph` API does not provide
            // endpoint-rewire mutation.
            let tombstone = {
                <G as GraphOperation<'_>>::edge_indices(&self.inner).find(|&ix| {
                    Graph::edge_unchecked(&self.inner, ix).version < 0
                        && <G as GraphOperation<'_>>::endpoints_unchecked(&self.inner, ix)
                            == inner_endpoints
                })
            };
            if let Some(inner_ix) = tombstone {
                let entry = <G as UpdateEdge>::edge_unchecked_mut(&mut self.inner, inner_ix);
                let new_version = (-entry.version) + 1;
                *entry = EdgeIx {
                    version: new_version,
                    inner: edge,
                };
                self.live_edges += 1;
                return Ok(EdgeIx {
                    version: new_version,
                    inner: inner_ix,
                });
            }
        }
        match <G as InsertEdge>::insert_edge_unchecked(
            &mut self.inner,
            EdgeIx {
                version: 1,
                inner: edge,
            },
            inner_endpoints,
        ) {
            Ok(inner_ix) => {
                self.live_edges += 1;
                Ok(EdgeIx {
                    version: 1,
                    inner: inner_ix,
                })
            }
            Err(EdgeIx { inner: edge, .. }) => Err(edge),
        }
    }
}

impl<'r, G: 'r, N: 'r, E: 'r + Clone> RemoveEdge for Stabilized<G, N, E>
where
    G: GraphOperation<'r, Node = NodeIx<N>, Edge = EdgeIx<E>> + UpdateEdge,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    unsafe fn take_edge_unchecked(&mut self, ix: Self::EdgeIx) -> Self::Edge {
        let edge = unsafe { self.inner.edge_unchecked(ix.inner).inner.clone() };
        <Self as RemoveEdge>::remove_edge_unchecked(self, ix);
        edge
    }

    unsafe fn remove_edge_unchecked(&mut self, ix: Self::EdgeIx) {
        let entry = <G as UpdateEdge>::edge_unchecked_mut(&mut self.inner, ix.inner);
        debug_assert!(entry.version > 0);
        entry.version = -entry.version; // tombstone
        self.live_edges -= 1;
    }
}

impl<G, N: Clone, E: Clone> RemoveNode for Stabilized<G, N, E>
where
    G: Graph<Node = NodeIx<N>, Edge = EdgeIx<E>> + for<'a> UpdateNode<'a> + UpdateEdge,
    G::Endpoints: Map<NodeIx<G::NodeIx>>,
{
    unsafe fn take_node_unchecked(&mut self, ix: Self::NodeIx) -> Self::Node {
        let node = unsafe { Graph::node_unchecked(&self.inner, ix.inner).inner.clone() };
        <Self as RemoveNode>::remove_node_unchecked(self, ix);
        node
    }

    unsafe fn remove_node_unchecked(&mut self, ix: Self::NodeIx) {
        let incident_edges: Vec<G::EdgeIx> = {
            <G as GraphOperation<'_>>::edge_indices_of_unchecked(&self.inner, ix.inner).collect()
        };
        for inner_eix in incident_edges {
            let edge_entry = <G as UpdateEdge>::edge_unchecked_mut(&mut self.inner, inner_eix);
            if edge_entry.version > 0 {
                edge_entry.version = -edge_entry.version;
                self.live_edges -= 1;
            }
        }
        let node_entry = <G as UpdateNode<'_>>::node_unchecked_mut(&mut self.inner, ix.inner);
        debug_assert!(node_entry.version > 0);
        node_entry.version = -node_entry.version;
        self.live_nodes -= 1;
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
        let mut nodes = IN::default();
        let mut edges = IE::default();

        for eix in edge_indices {
            let edge = unsafe { <Self as RemoveEdge>::take_edge_unchecked(self, eix) };
            edges.extend(core::iter::once(edge));
        }
        for nix in node_indices {
            let node = unsafe { <Self as RemoveNode>::take_node_unchecked(self, nix) };
            nodes.extend(core::iter::once(node));
        }

        (nodes, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VecGraph;

    #[test]
    fn removed_indices_stay_invalid_after_many_insertions() {
        let mut g = VecGraph::<u32, u32>::default().stabilize();

        let mut nodes = Vec::new();
        for i in 0..12u32 {
            nodes.push(g.insert_node(i).unwrap());
        }

        let mut edges = Vec::new();
        for i in 0..11u32 {
            edges.push(
                g.insert_edge(100 + i, [nodes[i as usize], nodes[i as usize + 1]])
                    .unwrap(),
            );
        }

        let removed_nodes = vec![nodes[2], nodes[5], nodes[8]];
        for n in &removed_nodes {
            g.remove_node(*n);
        }
        let removed_edge_explicit = edges[10];
        g.remove_edge(removed_edge_explicit);

        let removed_edges = vec![edges[1], edges[2], removed_edge_explicit];
        for n in &removed_nodes {
            assert!(!Graph::contains_node_index(&g, *n));
        }
        for e in &removed_edges {
            assert!(!Graph::contains_edge_index(&g, *e));
        }

        let mut new_nodes = Vec::new();
        for i in 0..30u32 {
            new_nodes.push((1000 + i, g.insert_node(1000 + i).unwrap()));
        }
        let mut new_edges = Vec::new();
        for i in 0..20u32 {
            let a = new_nodes[i as usize].1;
            let b = new_nodes[i as usize + 1].1;
            new_edges.push((5000 + i, g.insert_edge(5000 + i, [a, b]).unwrap()));
        }

        for n in &removed_nodes {
            assert!(!Graph::contains_node_index(&g, *n));
        }
        for e in &removed_edges {
            assert!(!Graph::contains_edge_index(&g, *e));
        }

        for (value, n) in &new_nodes {
            assert!(Graph::contains_node_index(&g, *n));
            assert_eq!(*g.node(*n), *value);
        }
        for (value, e) in &new_edges {
            assert!(Graph::contains_edge_index(&g, *e));
            assert_eq!(*g.edge(*e), *value);
        }
    }

    #[test]
    fn stabilized_fresh_build_appends_without_growing_tombstones() {
        let mut g = VecGraph::<u32, u32>::default().stabilize();
        let n = 200u32;
        let ixs: Vec<_> = (0..n).map(|i| g.insert_node(i).unwrap()).collect();
        for i in 0..n {
            g.insert_edge(1000 + i, [ixs[i as usize], ixs[((i + 1) % n) as usize]])
                .unwrap();
        }

        assert_eq!(Graph::node_indices(&g).count(), n as usize);
        assert_eq!(Graph::edge_indices(&g).count(), n as usize);
        assert_eq!(*g.node(ixs[5]), 5);

        assert_eq!(Graph::len_node(&g.inner), n as usize);
        assert_eq!(Graph::len_edge(&g.inner), n as usize);
    }

    #[test]
    fn stabilized_reuses_tombstoned_node_slot_when_capacity_full() {
        let mut g = VecGraph::<u32, u32>::default().stabilize();
        let n0 = g.insert_node(1).unwrap();
        let _n1 = g.insert_node(2).unwrap();

        while Graph::capacity_node(&g).unwrap() > Graph::len_node(&g.inner) {
            g.insert_node(999).unwrap();
        }
        g.remove_node(n0);
        let inner_len_before = Graph::len_node(&g.inner);

        let _n2 = g.insert_node(3).unwrap();
        let inner_len_after = Graph::len_node(&g.inner);

        assert_eq!(
            inner_len_after, inner_len_before,
            "reused the tombstoned slot"
        );
    }

    #[test]
    fn stabilized_reuses_tombstoned_edge_slot_for_same_endpoints_when_capacity_full() {
        let mut g = VecGraph::<u32, u32>::default().stabilize();
        let n0 = g.insert_node(1).unwrap();
        let n1 = g.insert_node(2).unwrap();
        let e0 = g.insert_edge(10, [n0, n1]).unwrap();

        while Graph::capacity_edge(&g).unwrap() > Graph::len_edge(&g.inner) {
            g.insert_edge(777, [n0, n1]).unwrap();
        }
        g.remove_edge(e0);
        let inner_len_before = Graph::len_edge(&g.inner);

        let _e1 = g.insert_edge(11, [n0, n1]).unwrap();
        let inner_len_after = Graph::len_edge(&g.inner);

        assert_eq!(
            inner_len_after, inner_len_before,
            "reused the tombstoned slot"
        );
    }

    #[test]
    fn into_inner_purges_tombstones() {
        let mut g = VecGraph::<u32, u32>::default().stabilize();
        let n0 = g.insert_node(0).unwrap();
        let n1 = g.insert_node(1).unwrap();
        let n2 = g.insert_node(2).unwrap();
        let n3 = g.insert_node(3).unwrap();
        let e01 = g.insert_edge(100, [n0, n1]).unwrap();
        let _e12 = g.insert_edge(101, [n1, n2]).unwrap();
        let _e23 = g.insert_edge(102, [n2, n3]).unwrap();

        g.remove_node(n2);
        g.remove_edge(e01);
        let (live_nodes, live_edges) = (g.live_nodes(), g.live_edges());
        assert_eq!((live_nodes, live_edges), (3, 0));

        let inner = g.into_inner();

        assert_eq!(Graph::len_node(&inner), live_nodes);
        assert_eq!(Graph::len_edge(&inner), live_edges);
        for ix in <_ as GraphOperation<'_>>::node_indices(&inner) {
            assert!(
                version_of(unsafe { <_ as GraphOperation<'_>>::node_unchecked(&inner, ix) }) > 0
            );
        }
        for ix in <_ as GraphOperation<'_>>::edge_indices(&inner) {
            assert!(
                version_of(unsafe { <_ as GraphOperation<'_>>::edge_unchecked(&inner, ix) }) > 0
            );
        }
    }

    #[allow(dead_code)]
    fn _stabilized_supports_hyperedge_endpoints(
        g: &Stabilized<crate::HyperGraph<NodeIx<u32>, EdgeIx<u32>>, u32, u32>,
    ) -> usize {
        Graph::len_node(g) + Graph::len_edge(g)
    }
}
