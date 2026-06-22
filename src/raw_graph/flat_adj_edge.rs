//! Nested-collection directed graph backend.
//!
//! A single outer [`RandomAccess`] collection holds the nodes; each node's
//! storage is a [`NodeRepr`] bundling that node's outgoing-edge collection
//! `EC` with an optional incoming-edge reverse index `IS`. Each edge is
//! identified by [`EdgeIx`] — the pair `(head VIx, EIx)` of the head node and
//! the inner-collection index within that head's outgoing edges.
//!
//! Data layout:
//!
//! ```text
//! nodes: NC                                            // head-node collection
//!   - Index = VIx
//!   - Value = V
//!   - Storage = NodeRepr<EC, IS>
//!       - outgoing: EC                                 // outgoing adjacency
//!           - Index = EIx, Value = E, Storage = VIx
//!       - incoming: IS                                 // reverse index (or TNone)
//! ```
//!
//! The incoming-store parameter `IS` selects the performance profile:
//!
//! | `IS`                              | incoming queries | write overhead |
//! |-----------------------------------|------------------|----------------|
//! | [`TNone`]                         | O(V + E) scan    | none           |
//! | a set (`Vec`/`HashSet`/`BTreeSet`) | O(in-degree)     | per insert/remove |
//!
//! "incoming queries" covers `walks_to`, `edge_indices_to`, the incoming
//! pass of `walks_of` / `edge_indices_of`, and `take_node`'s incident-edge
//! sweep. Outgoing queries (`walks_from`, `edge_indices_from`) are always
//! O(out-degree). With `IS = TNone` there is no reverse index at all.

use core::borrow::Borrow;
use core::marker::PhantomData;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::collection::{
    Collection, CollectionBiject, DrainEntries, InsertableCollection, RandomAccess,
    RandomAccessRef, RemovableRandomAccess, StableCollection, UpdatableRandomAccess,
};
use crate::graph::capability::{
    Bigraph, Directed, InsertEdge, InsertNode, RemoveEdge, RemoveNode, StableEdge, StableNode,
    UniqueNode, UpdateEdge, UpdateNode,
};
use crate::graph::operation::GraphOperation;
use crate::graph::walk_item::{WalkItem, WalkItemMut, WalkItemTo};
use crate::graph::{GraphMap, GraphProperty};

pub use super::hyper_edge::{IncidenceSet, IncidenceSetRef};

/// Edge index for [`FlatAdjEdgeGraph`]. Composed of `(head VIx, EIx)` where the
/// head node owns this edge in its outgoing collection.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EdgeIx<VIx, EIx>(pub VIx, pub EIx);

impl<VIx: Display, EIx: Display> Display for EdgeIx<VIx, EIx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "e({},{})", self.0, self.1)
    }
}

/// Sentinel "no incoming index" marker.
///
/// Use as the `IS` parameter of [`NodeRepr`] to skip the per-node
/// incoming-edge store entirely. Writes that would touch the incoming
/// set become no-ops; incoming queries (`walks_to`,
/// `edge_indices_to`, the incoming pass of `walks_of` /
/// `edge_indices_of`, `take_node_unchecked`'s incident-edge sweep) fall
/// back to scanning every node's `outgoing` collection — O(V + E).
#[derive(Copy, Clone, Debug, Default)]
pub struct TNone;

/// Operations on a node collection (`NC`) regarding incoming edges of
/// [`FlatAdjEdgeGraph`].
///
/// Parameterised over the in-store type `IS` so the two impl heads
/// (`IS: IncidenceSet<...>` vs `IS = TNone`) are non-overlapping at the
/// trait-signature level:
///
/// - When `IS: IncidenceSet<EdgeIx<VIx, EIx>>`: uses the maintained
///   reverse index (`MAINTAINED = true`).
/// - When `IS = TNone`: writes are no-ops; queries scan every node's
///   `outgoing` collection (`MAINTAINED = false`).
pub trait IncomingOps<EC, IS, VIx, EIx>
where
    EC: RandomAccess<Index = EIx, Storage = VIx>,
{
    /// `true` when this node collection maintains a real reverse index.
    const MAINTAINED: bool;

    /// Record that `eix` is now incoming to `nodes[tail]`. No-op when
    /// `MAINTAINED == false`.
    ///
    /// # Safety
    /// `tail` must be a valid index currently held by this collection.
    unsafe fn record_insert(&mut self, tail: VIx, eix: EdgeIx<VIx, EIx>);

    /// Drop `eix` from the incoming set of `nodes[at]`. No-op when
    /// `MAINTAINED == false`.
    ///
    /// # Safety
    /// `at` must be a valid index currently held by this collection.
    unsafe fn record_remove(&mut self, at: VIx, eix: &EdgeIx<VIx, EIx>);

    /// Is `eix` in the maintained incoming set of `nodes[at]`?
    /// Always `false` when `MAINTAINED == false`.
    ///
    /// # Safety
    /// `at` must be a valid index currently held by this collection.
    unsafe fn record_contains(&self, at: VIx, eix: &EdgeIx<VIx, EIx>) -> bool;

    /// Collect every EdgeIx currently incoming to `node_ix`.
    ///
    /// - `MAINTAINED == true`: copies from `nodes[node_ix].incoming`.
    /// - `MAINTAINED == false`: scans every node's `outgoing` collection
    ///   for edges whose target is `node_ix`.
    ///
    /// # Safety
    /// `node_ix` must be a valid index currently held by this collection.
    unsafe fn collect_incoming(&self, node_ix: VIx) -> Vec<EdgeIx<VIx, EIx>>;
}

impl<NC, EC, V, E, VIx, EIx, IS> IncomingOps<EC, IS, VIx, EIx> for NC
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>> + UpdatableRandomAccess,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx>,
    IS: IncidenceSet<EdgeIx<VIx, EIx>> + for<'a> IncidenceSetRef<'a, EdgeIx<VIx, EIx>>,
    VIx: Copy + Eq + 'static,
    EIx: Copy + Eq + 'static,
{
    const MAINTAINED: bool = true;

    #[inline]
    unsafe fn record_insert(&mut self, tail: VIx, eix: EdgeIx<VIx, EIx>) {
        let inc = &mut unsafe { self.get_storage_unchecked_mut(&tail) }.incoming;
        IncidenceSet::insert(inc, eix);
    }
    #[inline]
    unsafe fn record_remove(&mut self, at: VIx, eix: &EdgeIx<VIx, EIx>) {
        let inc = &mut unsafe { self.get_storage_unchecked_mut(&at) }.incoming;
        IncidenceSet::remove(inc, eix);
    }
    #[inline]
    unsafe fn record_contains(&self, at: VIx, eix: &EdgeIx<VIx, EIx>) -> bool {
        let inc = &unsafe { self.get_storage_unchecked(&at) }.incoming;
        IncidenceSet::contains(inc, eix)
    }
    unsafe fn collect_incoming(&self, node_ix: VIx) -> Vec<EdgeIx<VIx, EIx>> {
        let inc = &unsafe { self.get_storage_unchecked(&node_ix) }.incoming;
        IncidenceSetRef::iter(inc).collect()
    }
}

impl<NC, EC, V, E, VIx, EIx> IncomingOps<EC, TNone, VIx, EIx> for NC
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, TNone>>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx>,
    for<'a> NC: RandomAccessRef<'a>,
    for<'a> EC: RandomAccessRef<'a>,
    VIx: Copy + Eq + 'static,
    EIx: Copy + Eq + 'static,
{
    const MAINTAINED: bool = false;

    #[inline]
    unsafe fn record_insert(&mut self, _tail: VIx, _eix: EdgeIx<VIx, EIx>) {}
    #[inline]
    unsafe fn record_remove(&mut self, _at: VIx, _eix: &EdgeIx<VIx, EIx>) {}
    #[inline]
    unsafe fn record_contains(&self, _at: VIx, _eix: &EdgeIx<VIx, EIx>) -> bool {
        false
    }
    unsafe fn collect_incoming(&self, node_ix: VIx) -> Vec<EdgeIx<VIx, EIx>> {
        let mut out = Vec::new();
        for head in self.indices() {
            let inner = &unsafe { self.get_storage_unchecked(&head) }.outgoing;
            for eix in inner.indices() {
                let target = unsafe { inner.get_storage_unchecked(&eix) };
                if *target == node_ix {
                    out.push(EdgeIx(head, eix));
                }
            }
        }
        out
    }
}

/// Per-node storage held by the outer collection of [`FlatAdjEdgeGraph`].
///
/// Bundles the outgoing-adjacency collection with a reverse-index store
/// of edges whose tail is this node. Use [`TNone`] for `IS` to skip the
/// reverse index entirely.
#[derive(Clone, Debug)]
pub struct NodeRepr<EC, IS> {
    /// EC: `RandomAccess<Index = EIx, Value = E, Storage = VIx>` —
    /// per-node outgoing-edge collection. Each entry's `Value` is the
    /// edge data; its `Storage` is the target node id.
    pub(crate) outgoing: EC,
    /// IS: `IncidenceSet<EdgeIx<VIx, EIx>>` — reverse index of full edge
    /// ids whose tail is THIS node. Use any `Vec`, `HashSet`, or
    /// `BTreeSet` of `EdgeIx<VIx, EIx>` for a maintained index, or
    /// [`TNone`] to disable it.
    pub(crate) incoming: IS,
}

/// Nested-collection directed graph: a single outer collection of nodes,
/// each owning its outgoing-edge collection plus an optional incoming-edge
/// reverse index ([`NodeRepr`]).
///
/// With a maintained reverse index (`IS` a set type) `walks_to` and
/// `edge_indices_to` are O(in-degree), at the cost of doubled write work —
/// every edge insertion / removal also touches the tail node's `incoming`
/// list. With `IS = TNone` those queries fall back to an O(V + E) scan and
/// writes carry no reverse-index overhead.
#[derive(Clone, Debug)]
pub struct FlatAdjEdgeGraph<NC> {
    /// NC:
    /// `RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>`
    /// — the outer node collection. Each entry pairs node data (`V`)
    /// with a [`NodeRepr`] holding both that node's outgoing-edge
    /// collection and its incoming-edge reverse index.
    pub(crate) nodes: NC,
}

impl<NC: Default> Default for FlatAdjEdgeGraph<NC> {
    fn default() -> Self {
        Self {
            nodes: NC::default(),
        }
    }
}

impl<NC: Default> FlatAdjEdgeGraph<NC> {
    /// Equivalent to [`Default::default`].
    pub fn new() -> Self {
        Self::default()
    }
}

/// Iterator over outgoing edge indices of a head node.
pub struct EdgeIndicesFromIter<VIx, II> {
    head: VIx,
    inner: II,
}

impl<VIx: Copy, EIx, II> Iterator for EdgeIndicesFromIter<VIx, II>
where
    II: Iterator<Item = EIx>,
{
    type Item = EdgeIx<VIx, EIx>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|eix| EdgeIx(self.head, eix))
    }
}

/// Iterator over outgoing walks (`(EdgeIx, &Edge, target VIx)`).
pub struct WalksFromIter<'r, VIx, EC, II> {
    head: VIx,
    inner_collection: &'r EC,
    inner_indices: II,
}

impl<'r, VIx, EC, II> Iterator for WalksFromIter<'r, VIx, EC, II>
where
    VIx: Copy + 'r,
    EC: RandomAccess<Storage = VIx>,
    EC::Value: 'r,
    II: Iterator<Item = EC::Index>,
{
    type Item = WalkItem<'r, EdgeIx<VIx, EC::Index>, EC::Value, VIx>;
    fn next(&mut self) -> Option<Self::Item> {
        let eix = self.inner_indices.next()?;
        // SAFETY: `eix` came from this collection's own `indices()`.
        let (edge_val, target) = unsafe { self.inner_collection.get_both_unchecked(&eix) };
        Some(WalkItem::new(EdgeIx(self.head, eix), edge_val, *target))
    }
}

/// Iterator over incoming-edge walks (`(source VIx, EdgeIx, &Edge)`).
///
/// Walks the tail node's pre-built `incoming` set (O(in-degree)) and for
/// each `EdgeIx(head, eix_in_head)` resolves the edge data through the
/// outer graph. Generic over `II = IS::Iter` so the choice of incidence
/// set type leaks through cleanly.
pub struct WalksToIter<'r, NC, E, II> {
    nodes: &'r NC,
    incoming_iter: II,
    _marker: PhantomData<&'r E>,
}

impl<'r, NC, EC, V, E, VIx, EIx, IS, II> Iterator for WalksToIter<'r, NC, E, II>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>> + 'r,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx> + 'r,
    IS: 'r,
    VIx: Copy + Eq + 'r,
    EIx: Copy + Eq + 'r,
    E: 'r,
    II: Iterator<Item = EdgeIx<VIx, EIx>>,
{
    type Item = WalkItemTo<'r, VIx, EdgeIx<VIx, EIx>, E>;
    fn next(&mut self) -> Option<Self::Item> {
        let edge_ix = self.incoming_iter.next()?;
        // SAFETY: incoming set only holds valid EdgeIx values; the dual
        // index is maintained on every insert / remove.
        let inner = &unsafe { self.nodes.get_storage_unchecked(&edge_ix.0) }.outgoing;
        let edge = unsafe { inner.get_value_unchecked(&edge_ix.1) };
        Some(WalkItemTo::new(edge_ix.0, edge_ix, edge))
    }
}

/// Iterator over incoming-edge indices — thin newtype around an
/// `IncidenceSet::Iter` so the backend's concrete iter type stays
/// internal.
pub struct EdgeIndicesToIter<II> {
    incoming_iter: II,
}

impl<VIx: Copy, EIx: Copy, II> Iterator for EdgeIndicesToIter<II>
where
    II: Iterator<Item = EdgeIx<VIx, EIx>>,
{
    type Item = EdgeIx<VIx, EIx>;
    fn next(&mut self) -> Option<Self::Item> {
        self.incoming_iter.next()
    }
}

impl<NC, EC, V, E, VIx, EIx, IS> GraphProperty for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    type Node = V;
    type Edge = E;
    type NodeIx = VIx;
    type EdgeIx = EdgeIx<VIx, EIx>;
    type Endpoints = [VIx; 2];
    const DIRECTED: bool = true;
}

impl<NC, EC, V, E, VIx, EIx, IS> Bigraph for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2] {
        endpoints
    }
    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints {
        nodes
    }
}

impl<'r, NC, EC, V, E, VIx, EIx, IS> GraphOperation<'r> for FlatAdjEdgeGraph<NC>
where
    // `reverse` rebuilds the head-indexed adjacency, so the collections must be
    // writable (insertable + default + drainable); every real flat instance
    // (Vec/BTreeMap-backed) satisfies this, and slice-backed nodes are already
    // excluded by the `DrainEntries` bound that `drain` needs.
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>
        + IncomingOps<EC, IS, VIx, EIx>
        + InsertableCollection<InsertedIndex = VIx>
        + DrainEntries
        + Default
        + 'r,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx>
        + InsertableCollection<InsertedIndex = EIx>
        + DrainEntries
        + Default
        + 'r,
    for<'a> NC: RandomAccessRef<'a>,
    for<'a> EC: RandomAccessRef<'a>,
    IS: Default + 'static,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    V: 'r,
    E: 'r,
{
    fn contains_node_index(&self, node_ix: Self::NodeIx) -> bool {
        self.nodes.contains_index(&node_ix)
    }

    fn contains_edge_index(&self, edge_ix: Self::EdgeIx) -> bool {
        if !self.nodes.contains_index(&edge_ix.0) {
            return false;
        }
        let inner = &unsafe { self.nodes.get_storage_unchecked(&edge_ix.0) }.outgoing;
        inner.contains_index(&edge_ix.1)
    }

    fn len_node(&self) -> usize {
        Collection::len(&self.nodes)
    }

    fn len_edge(&self) -> usize {
        let mut total = 0;
        for nix in self.nodes.indices() {
            let inner = &unsafe { self.nodes.get_storage_unchecked(&nix) }.outgoing;
            total += Collection::len(inner);
        }
        total
    }

    type NodeIndices = <NC as RandomAccessRef<'r>>::Indices;
    fn node_indices(&'r self) -> Self::NodeIndices {
        self.nodes.indices()
    }

    type EdgeIndices = std::vec::IntoIter<EdgeIx<VIx, EIx>>;
    fn edge_indices(&'r self) -> Self::EdgeIndices {
        let mut out = Vec::new();
        for head in self.nodes.indices() {
            let inner = &unsafe { self.nodes.get_storage_unchecked(&head) }.outgoing;
            for eix in inner.indices() {
                out.push(EdgeIx(head, eix));
            }
        }
        out.into_iter()
    }

    unsafe fn node_unchecked(&self, node_ix: Self::NodeIx) -> &Self::Node {
        unsafe { self.nodes.get_value_unchecked(&node_ix) }
    }

    unsafe fn edge_unchecked(&self, edge_ix: Self::EdgeIx) -> &Self::Edge {
        let inner = &unsafe { self.nodes.get_storage_unchecked(&edge_ix.0) }.outgoing;
        // The raw round-trip sidesteps the pre-1.63 borrow checker, which could
        // not prove `EC` outlives the `&self` borrow through the nested
        // `nodes -> outgoing -> value` access (E0311). SAFETY: the value lives
        // as long as `&self`.
        let val: *const E = unsafe { inner.get_value_unchecked(&edge_ix.1) };
        unsafe { &*val }
    }

    unsafe fn endpoints_unchecked(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints {
        let inner = &unsafe { self.nodes.get_storage_unchecked(&edge_ix.0) }.outgoing;
        let target = unsafe { inner.get_storage_unchecked(&edge_ix.1) };
        [edge_ix.0, *target]
    }

    type EdgeIndicesFrom = EdgeIndicesFromIter<VIx, <EC as RandomAccessRef<'r>>::Indices>;
    unsafe fn edge_indices_from_unchecked(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesFrom {
        let inner = &unsafe { self.nodes.get_storage_unchecked(&node_ix) }.outgoing;
        EdgeIndicesFromIter {
            head: node_ix,
            inner: inner.indices(),
        }
    }

    type EdgeIndicesOf = std::vec::IntoIter<EdgeIx<VIx, EIx>>;
    unsafe fn edge_indices_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesOf {
        let mut out = Vec::new();
        let inner = &unsafe { self.nodes.get_storage_unchecked(&node_ix) }.outgoing;
        for eix in inner.indices() {
            out.push(EdgeIx(node_ix, eix));
        }
        // Incoming via IncomingOps — skip self-loops (already in outgoing).
        for eix in unsafe { IncomingOps::collect_incoming(&self.nodes, node_ix) } {
            if eix.0 != node_ix {
                out.push(eix);
            }
        }
        out.into_iter()
    }

    type WalksFrom = WalksFromIter<'r, VIx, EC, <EC as RandomAccessRef<'r>>::Indices>;
    unsafe fn walks_from_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksFrom {
        let inner = &unsafe { self.nodes.get_storage_unchecked(&node_ix) }.outgoing;
        WalksFromIter {
            head: node_ix,
            inner_collection: inner,
            inner_indices: inner.indices(),
        }
    }

    type WalksOf = std::vec::IntoIter<WalkItem<'r, EdgeIx<VIx, EIx>, E, VIx>>;
    unsafe fn walks_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksOf {
        let mut out = Vec::new();
        let inner = &unsafe { self.nodes.get_storage_unchecked(&node_ix) }.outgoing;
        for eix in inner.indices() {
            let (edge_val, target) = unsafe { inner.get_both_unchecked(&eix) };
            out.push(WalkItem::new(EdgeIx(node_ix, eix), edge_val, *target));
        }
        // Incoming via IncomingOps — skip self-loops.
        for eix in unsafe { IncomingOps::collect_incoming(&self.nodes, node_ix) } {
            if eix.0 == node_ix {
                continue;
            }
            let other = &unsafe { self.nodes.get_storage_unchecked(&eix.0) }.outgoing;
            let edge_val = unsafe { other.get_value_unchecked(&eix.1) };
            out.push(WalkItem::new(eix, edge_val, eix.0));
        }
        out.into_iter()
    }

    type DrainNode = std::vec::IntoIter<V>;
    type DrainEdge = std::vec::IntoIter<E>;
    fn drain(self) -> (Self::DrainNode, Self::DrainEdge) {
        // Edge payloads live inside each node's storage (`NodeRepr::outgoing`),
        // so consume the node collection entries-first (`DrainEntries`) to keep
        // both the node value and its storage, then drain each node's outgoing
        // edge collection.
        let mut node_vals: Vec<V> = Vec::with_capacity(self.nodes.len());
        let mut edge_vals: Vec<E> = Vec::new();
        for (value, repr) in self.nodes.into_entries() {
            node_vals.push(value);
            edge_vals.extend(repr.outgoing.into_values());
        }
        (node_vals.into_iter(), edge_vals.into_iter())
    }

    fn reverse(&mut self) {
        // An edge `h -> t` is stored in `h`'s outgoing list; reversing it means
        // moving it into `t`'s. Take the whole graph, re-insert every node value
        // in its original order (which reproduces the same node indices —
        // positional for Vec, key=value for maps, so the index mapping is the
        // identity), then re-insert each edge with its endpoints swapped.
        let old = std::mem::take(self);
        let mut reversed: Vec<(VIx, VIx, E)> = Vec::new();
        for (value, repr) in old.nodes.into_entries() {
            // SAFETY: inserting into the freshly-emptied graph; the returned index
            // equals the original (identity mapping), keeping the targets snapshotted
            // below valid in the rebuilt graph.
            let head = unsafe {
                crate::unwrap_unchecked(
                    <Self as InsertNode>::insert_node_unchecked(self, value).ok(),
                )
            };
            for (edge, target) in repr.outgoing.into_entries() {
                // original `head -> target`  ⇒  reversed `target -> head`
                reversed.push((target, head, edge));
            }
        }
        for (new_head, new_tail, edge) in reversed {
            // SAFETY: both endpoints are valid node indices in the rebuilt graph.
            unsafe {
                let _ =
                    <Self as InsertEdge>::insert_edge_unchecked(self, edge, [new_head, new_tail]);
            }
        }
    }
}

impl<'r, NC, EC, V, E, VIx, EIx, IS> Directed<'r> for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>
        + IncomingOps<EC, IS, VIx, EIx>
        + 'r,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx> + 'r,
    for<'a> NC: RandomAccessRef<'a>,
    for<'a> EC: RandomAccessRef<'a>,
    IS: 'static,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    V: 'r,
    E: 'r,
{
    type EdgeIndicesTo = std::vec::IntoIter<EdgeIx<VIx, EIx>>;
    unsafe fn edge_indices_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesTo {
        unsafe { IncomingOps::collect_incoming(&self.nodes, node_ix) }.into_iter()
    }

    type WalksTo = std::vec::IntoIter<WalkItemTo<'r, VIx, EdgeIx<VIx, EIx>, E>>;
    unsafe fn walks_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksTo {
        let eixs = unsafe { IncomingOps::collect_incoming(&self.nodes, node_ix) };
        let mut out = Vec::with_capacity(eixs.len());
        for eix in eixs {
            let inner = &unsafe { self.nodes.get_storage_unchecked(&eix.0) }.outgoing;
            let edge = unsafe { inner.get_value_unchecked(&eix.1) };
            out.push(WalkItemTo::new(eix.0, eix, edge));
        }
        out.into_iter()
    }

    type EdgeTailIndices = core::iter::Once<VIx>;
    unsafe fn edge_tail_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeTailIndices {
        core::iter::once(edge_ix.0)
    }

    type EdgeHeadIndices = core::iter::Once<VIx>;
    unsafe fn edge_head_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeHeadIndices {
        let inner = &unsafe { self.nodes.get_storage_unchecked(&edge_ix.0) }.outgoing;
        let target = unsafe { inner.get_storage_unchecked(&edge_ix.1) };
        core::iter::once(*target)
    }
}

impl<NC, EC, V, E, VIx, EIx, IS> InsertNode for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>
        + InsertableCollection<InsertedIndex = VIx>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx> + Default,
    IS: Default,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node> {
        let storage = NodeRepr {
            outgoing: EC::default(),
            incoming: IS::default(),
        };
        self.nodes.insert(node, storage).map_err(|(v, _)| v)
    }
}

impl<NC, EC, V, E, VIx, EIx, IS> InsertEdge for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>
        + IncomingOps<EC, IS, VIx, EIx>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx>
        + InsertableCollection<InsertedIndex = EIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge> {
        let [head, tail] = endpoints;
        let inner = &mut unsafe { self.nodes.get_storage_unchecked_mut(&head) }.outgoing;
        let eix = match inner.insert(edge, tail) {
            Ok(eix) => eix,
            Err((v, _)) => return Err(v),
        };
        let new_edge_ix = EdgeIx(head, eix);
        unsafe { IncomingOps::record_insert(&mut self.nodes, tail, new_edge_ix) };
        Ok(new_edge_ix)
    }
}

impl<'r, NC, EC, V, E, VIx, EIx, IS> UpdateNode<'r> for FlatAdjEdgeGraph<NC>
where
    NC: UpdatableRandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>> + 'r,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx> + 'r,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    V: 'r,
    E: 'r,
{
    unsafe fn node_unchecked_mut(&mut self, node_ix: Self::NodeIx) -> &mut Self::Node {
        unsafe { self.nodes.get_value_unchecked_mut(&node_ix) }
    }

    type WalksFromMut = std::iter::Empty<WalkItemMut<'r, EdgeIx<VIx, EIx>, E, VIx>>;
    unsafe fn walks_from_unchecked_mut(&'r mut self, _node_ix: Self::NodeIx) -> Self::WalksFromMut {
        std::iter::empty()
    }

    type WalksOfMut = std::iter::Empty<WalkItemMut<'r, EdgeIx<VIx, EIx>, E, VIx>>;
    unsafe fn walks_of_unchecked_mut(&'r mut self, _node_ix: Self::NodeIx) -> Self::WalksOfMut {
        std::iter::empty()
    }
}

impl<NC, EC, V, E, VIx, EIx, IS> UpdateEdge for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>,
    EC: UpdatableRandomAccess<Index = EIx, Value = E, Storage = VIx> + 'static,
    IS: 'static,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn edge_unchecked_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge {
        let inner = &mut unsafe { self.nodes.get_storage_unchecked_mut(&edge_ix.0) }.outgoing;
        unsafe { inner.get_value_unchecked_mut(&edge_ix.1) }
    }
}

impl<NC, EC, V, E, VIx, EIx, IS> RemoveEdge for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>
        + IncomingOps<EC, IS, VIx, EIx>,
    EC: RemovableRandomAccess<Index = EIx, Value = E, Storage = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn take_edge_unchecked(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge {
        let head = edge_ix.0;
        let node_storage = unsafe { self.nodes.get_storage_unchecked_mut(&head) };
        let (edge_value, target, swapped) =
            unsafe { node_storage.outgoing.take_unchecked(&edge_ix.1) };

        unsafe { IncomingOps::record_remove(&mut self.nodes, target, &edge_ix) };

        // If the inner collection swap-removed, an edge previously at
        // (head, old_last) now sits at (head, edge_ix.1). Rewrite the
        // relocated edge's reference in its tail's incoming set.
        if let Some(old_last) = swapped {
            let old_edge_ix = EdgeIx(head, old_last);
            let new_edge_ix = EdgeIx(head, edge_ix.1);
            let relocated_target = *unsafe {
                self.nodes
                    .get_storage_unchecked(&head)
                    .outgoing
                    .get_storage_unchecked(&edge_ix.1)
            };
            if unsafe { IncomingOps::record_contains(&self.nodes, relocated_target, &old_edge_ix) }
            {
                unsafe {
                    IncomingOps::record_remove(&mut self.nodes, relocated_target, &old_edge_ix)
                };
                unsafe {
                    IncomingOps::record_insert(&mut self.nodes, relocated_target, new_edge_ix)
                };
            }
        }

        edge_value
    }
}

impl<NC, EC, V, E, VIx, EIx, IS> RemoveNode for FlatAdjEdgeGraph<NC>
where
    NC: RemovableRandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>
        + IncomingOps<EC, IS, VIx, EIx>,
    EC: RemovableRandomAccess<Index = EIx, Value = E, Storage = VIx>,
    for<'a> NC: RandomAccessRef<'a>,
    for<'a> EC: RandomAccessRef<'a>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn take_node_unchecked(&mut self, node_ix: Self::NodeIx) -> Self::Node {
        let mut incoming_snapshot: Vec<EdgeIx<VIx, EIx>> =
            unsafe { IncomingOps::collect_incoming(&self.nodes, node_ix) };
        // The snapshot is taken once, but each `take_edge_unchecked` may
        // swap-relocate a same-head edge into a slot a later snapshot entry
        // still points at (stale index → wrong removal or panic). `EdgeIx`
        // orders lexicographically on `(head, eix)` and cross-head removals
        // never relocate each other's inner slots, so a single global
        // descending sort gives the per-head descending order that makes
        // each swap-remove disturb only already-processed (higher) slots.
        incoming_snapshot.sort_unstable_by(|a, b| b.cmp(a));
        // Self-loops are reached again via the outgoing pass below.
        for eix in incoming_snapshot.into_iter().filter(|e| e.0 != node_ix) {
            let _ = unsafe { self.take_edge_unchecked(eix) };
        }
        // Descending order so inner swap-remove relocations don't
        // invalidate later eixs.
        let mut outgoing_eixs: Vec<EIx> = {
            let storage = unsafe { self.nodes.get_storage_unchecked(&node_ix) };
            storage.outgoing.indices().collect()
        };
        outgoing_eixs.sort_unstable_by(|a, b| b.cmp(a));
        for eix in outgoing_eixs {
            let _ = unsafe { self.take_edge_unchecked(EdgeIx(node_ix, eix)) };
        }

        let (data, _storage, swapped) = unsafe { self.nodes.take_unchecked(&node_ix) };

        // If NC swap-removed, the relocated node's old VIx is now `node_ix`.
        // Rewrite outgoing-edge targets and (when maintained) incoming-set
        // EdgeIx heads.
        if let Some(old_last) = swapped {
            let heads: Vec<VIx> = self.nodes.indices().collect();
            for head in heads {
                let storage = unsafe { self.nodes.get_storage_unchecked_mut(&head) };
                let eixs: Vec<EIx> = storage.outgoing.indices().collect();
                for eix in eixs {
                    let target = unsafe { storage.outgoing.get_storage_unchecked_mut(&eix) };
                    if *target == old_last {
                        *target = node_ix;
                    }
                }
                if NC::MAINTAINED {
                    // Set-typed IS can't be mutated in place — collect via
                    // the trait, drop matching entries, re-insert rewritten.
                    let entries: Vec<EdgeIx<VIx, EIx>> =
                        unsafe { IncomingOps::collect_incoming(&self.nodes, head) };
                    for entry in &entries {
                        if entry.0 == old_last {
                            unsafe { IncomingOps::record_remove(&mut self.nodes, head, entry) };
                        }
                    }
                    for entry in entries {
                        if entry.0 == old_last {
                            unsafe {
                                IncomingOps::record_insert(
                                    &mut self.nodes,
                                    head,
                                    EdgeIx(node_ix, entry.1),
                                )
                            };
                        }
                    }
                }
            }
        }

        data
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
        // The default impl removes in the given order, which is unsound for a
        // `Vec`-backed inner collection: `take_edge_unchecked` swap-relocates a
        // same-head edge into the just-vacated slot, invalidating later batch
        // indices. Removing in descending `(head, eix)` order makes every inner
        // swap-remove disturb only an already-processed (higher) slot. The
        // per-edge incoming-set maintenance is inherited from
        // `take_edge_unchecked` / `take_node_unchecked`.
        let mut edges_to_remove: Vec<Self::EdgeIx> = edge_indices.into_iter().collect();
        edges_to_remove.sort_unstable_by(|a, b| b.cmp(a));
        edges_to_remove.dedup();
        let mut edges_out = IE::default();
        for eix in edges_to_remove {
            let data = unsafe { <Self as RemoveEdge>::take_edge_unchecked(self, eix) };
            edges_out.extend(core::iter::once(data));
        }

        let mut nodes_to_remove: Vec<Self::NodeIx> = node_indices.into_iter().collect();
        nodes_to_remove.sort_unstable_by(|a, b| b.cmp(a));
        let mut nodes_out = IN::default();
        for nix in nodes_to_remove {
            let data = unsafe { <Self as RemoveNode>::take_node_unchecked(self, nix) };
            nodes_out.extend(core::iter::once(data));
        }
        (nodes_out, edges_out)
    }
}

unsafe impl<NC, EC, V, E, VIx, EIx, IS> StableNode for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>> + StableCollection,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
}

unsafe impl<NC, EC, V, E, VIx, EIx, IS> StableEdge for FlatAdjEdgeGraph<NC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>> + StableCollection,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx> + StableCollection,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
}

impl<NC, EC, V, E, VIx, EIx, IS> UniqueNode for FlatAdjEdgeGraph<NC>
where
    NC: StableCollection
        + CollectionBiject
        + RandomAccess<Index = VIx, Value = V, Storage = NodeRepr<EC, IS>>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    V: PartialEq,
{
    fn node_index(&self, node: impl Borrow<Self::Node>) -> Option<Self::NodeIx> {
        unsafe { self.nodes.value_to_key_unchecked(node.borrow()) }.copied()
    }
}

// ---------------------------------------------------------------------------
// GraphMap — Vec-backed FlatAdjEdgeGraph
//
// Indices are u32 positions and don't change under payload mapping; rebuild
// each (value, storage) tuple at both layers with the value transformed. The
// per-node incoming set `IS` holds position-based `EdgeIx` values, which the
// payload mapping leaves untouched, so it threads through unchanged.
// ---------------------------------------------------------------------------

impl<'r, N, E, NewN, NewE, IS> GraphMap<'r, NewN, NewE>
    for FlatAdjEdgeGraph<Vec<(N, NodeRepr<Vec<(E, u32)>, IS>)>>
{
    type Mapped = FlatAdjEdgeGraph<Vec<(NewN, NodeRepr<Vec<(NewE, u32)>, IS>)>>;

    fn map<FN, FE>(self, mut fn_node: FN, mut fn_edge: FE) -> Self::Mapped
    where
        FN: FnMut(Self::Node) -> NewN,
        FE: FnMut(Self::Edge) -> NewE,
    {
        // Element type (`(NewN, NodeRepr<Vec<(NewE, u32)>, IS>)`) is
        // inferred from the `Self::Mapped` return type.
        let nodes: Vec<_> = self
            .nodes
            .into_iter()
            .map(|(v, NodeRepr { outgoing, incoming })| {
                let new_outgoing: Vec<(NewE, u32)> =
                    outgoing.into_iter().map(|(e, t)| (fn_edge(e), t)).collect();
                (
                    fn_node(v),
                    NodeRepr {
                        outgoing: new_outgoing,
                        incoming,
                    },
                )
            })
            .collect();
        FlatAdjEdgeGraph { nodes }
    }
}
