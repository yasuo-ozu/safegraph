//! Undirected hypergraph backend.
//!
//! A hypergraph generalizes ordinary graphs by allowing each edge ("hyper-
//! edge") to connect an arbitrary number of nodes, not just two. This
//! backend maintains a dual index so both directions of lookup are
//! O(degree) rather than O(V + E):
//!
//! ```text
//! nodes: NC                              // outer node collection
//!   - Index   = VIx
//!   - Value   = V
//!   - Storage = IS                       // set of incident EIx
//!
//! edges: EC                              // outer edge collection
//!   - Index   = EIx
//!   - Value   = E
//!   - Storage = ES                       // set of endpoint VIx (impl `Endpoints`)
//! ```
//!
//! The invariant `eix ∈ nodes[vix].Storage  ⇔  vix ∈ edges[eix].Storage`
//! is maintained by every insert / remove path.

use core::marker::PhantomData;
use std::collections::{btree_set, hash_set, BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::collection::{
    Collection, InsertableCollection, RandomAccess, RandomAccessRef, RemovableRandomAccess,
    StableCollection, UpdatableRandomAccess,
};
use crate::graph::capability::{
    InsertEdge, InsertNode, RemoveEdge, RemoveNode, StableEdge, StableNode, UniqueEdge, UniqueNode,
    UpdateEdge, UpdateNode,
};
use crate::graph::edge::Endpoints;
use crate::graph::operation::GraphOperation;
use crate::graph::walk_item::{WalkItem, WalkItemMut};
use crate::graph::GraphProperty;

/// Per-node container of incident edge indices. Used as the `Storage` type
/// on the outer node collection.
pub trait IncidenceSet<I>: Default {
    fn insert(&mut self, item: I);
    fn remove(&mut self, item: &I);
    fn contains(&self, item: &I) -> bool;
    fn len(&self) -> usize;
    /// Returns `true` if the incidence set is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Iteration access for an [`IncidenceSet`], parameterized by borrow lifetime
/// to sidestep GATs (matches the HRTB pattern used elsewhere in this crate).
pub trait IncidenceSetRef<'a, I>: IncidenceSet<I>
where
    Self: 'a,
{
    type Iter: Iterator<Item = I>;
    fn iter(&'a self) -> Self::Iter;
}

// --- HashSet<I> ---

impl<I: Copy + Eq + Hash> IncidenceSet<I> for HashSet<I> {
    #[inline]
    fn insert(&mut self, item: I) {
        HashSet::insert(self, item);
    }
    #[inline]
    fn remove(&mut self, item: &I) {
        HashSet::remove(self, item);
    }
    #[inline]
    fn contains(&self, item: &I) -> bool {
        HashSet::contains(self, item)
    }
    #[inline]
    fn len(&self) -> usize {
        HashSet::len(self)
    }
}

impl<'a, I: Copy + Eq + Hash + 'a> IncidenceSetRef<'a, I> for HashSet<I> {
    type Iter = std::iter::Copied<hash_set::Iter<'a, I>>;
    #[inline]
    fn iter(&'a self) -> Self::Iter {
        HashSet::iter(self).copied()
    }
}

// --- BTreeSet<I> ---

impl<I: Copy + Eq + Ord> IncidenceSet<I> for BTreeSet<I> {
    #[inline]
    fn insert(&mut self, item: I) {
        BTreeSet::insert(self, item);
    }
    #[inline]
    fn remove(&mut self, item: &I) {
        BTreeSet::remove(self, item);
    }
    #[inline]
    fn contains(&self, item: &I) -> bool {
        BTreeSet::contains(self, item)
    }
    #[inline]
    fn len(&self) -> usize {
        BTreeSet::len(self)
    }
}

impl<'a, I: Copy + Eq + Ord + 'a> IncidenceSetRef<'a, I> for BTreeSet<I> {
    type Iter = std::iter::Copied<btree_set::Iter<'a, I>>;
    #[inline]
    fn iter(&'a self) -> Self::Iter {
        BTreeSet::iter(self).copied()
    }
}

// --- Vec<I> (multiset semantics; `remove` deletes one occurrence) ---

impl<I: Copy + Eq> IncidenceSet<I> for Vec<I> {
    #[inline]
    fn insert(&mut self, item: I) {
        Vec::push(self, item);
    }
    fn remove(&mut self, item: &I) {
        if let Some(pos) = <[I]>::iter(self).position(|x| x == item) {
            // swap_remove keeps O(1); order of iteration doesn't matter
            // for incidence semantics.
            self.swap_remove(pos);
        }
    }
    fn contains(&self, item: &I) -> bool {
        <[I]>::iter(self).any(|x| x == item)
    }
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<'a, I: Copy + Eq + 'a> IncidenceSetRef<'a, I> for Vec<I> {
    type Iter = std::iter::Copied<std::slice::Iter<'a, I>>;
    #[inline]
    fn iter(&'a self) -> Self::Iter {
        <[I]>::iter(self).copied()
    }
}

/// Undirected hypergraph backed by two cross-referenced [`RandomAccess`]
/// collections.
///
/// See the module-level docs for the storage layout.
#[derive(Clone, Debug)]
pub struct HyperGraph<NC, EC> {
    pub(crate) nodes: NC,
    pub(crate) edges: EC,
}

impl<NC: Default, EC: Default> Default for HyperGraph<NC, EC> {
    fn default() -> Self {
        Self {
            nodes: NC::default(),
            edges: EC::default(),
        }
    }
}

impl<NC: Default, EC: Default> HyperGraph<NC, EC> {
    /// Equivalent to [`Default::default`].
    pub fn new() -> Self {
        Self::default()
    }
}

/// `BTreeMap`-backed hypergraph (Key=Value pattern). Indices stable;
/// implements [`StableNode`] + [`StableEdge`].
pub type StableHyperGraph<N, E> = HyperGraph<
    std::collections::BTreeMap<N, BTreeSet<E>>,
    std::collections::BTreeMap<E, BTreeSet<N>>,
>;

/// `HashMap`-backed hypergraph (Key=Value pattern). Same stability as
/// [`StableHyperGraph`], hash-based iteration order.
pub type HashHyperGraph<N, E> = HyperGraph<HashMap<N, HashSet<E>>, HashMap<E, HashSet<N>>>;

impl<NC, EC, V, E, VIx, EIx, IS, ES> GraphProperty for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES>,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    type Node = V;
    type Edge = E;
    type NodeIx = VIx;
    type EdgeIx = EIx;
    type Endpoints = ES;
    // Hyperedges are undirected (no source/target distinction).
    const DIRECTED: bool = false;
}

/// `edge_indices_from_unchecked` result: thin wrapper around the
/// incidence-set iterator.
pub struct EdgeIndicesFromIter<I> {
    inner: I,
}

impl<EIx, I> Iterator for EdgeIndicesFromIter<I>
where
    I: Iterator<Item = EIx>,
{
    type Item = EIx;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// `walks_from_unchecked` / `walks_of_unchecked` iterator.
///
/// Flat-maps over incident hyperedges and yields `(EIx, &Edge, neighbor VIx)`
/// for every endpoint of each edge except the origin node. Same edge
/// appears multiple times when its arity is > 2.
pub struct Walks<'r, NC, EC, VIx, EIx, IS, ES>
where
    NC: RandomAccess<Index = VIx, Storage = IS>,
    EC: RandomAccess<Index = EIx, Storage = ES>,
    IS: IncidenceSetRef<'r, EIx>,
    ES: IntoIterator,
{
    origin: VIx,
    edges: &'r EC,
    incidence_iter: IS::Iter,
    cur_edge: Option<(EIx, ES::IntoIter)>,
    _marker: PhantomData<&'r NC>,
}

impl<'r, NC, EC, V, E, VIx, EIx, IS, ES> Iterator for Walks<'r, NC, EC, VIx, EIx, IS, ES>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES>,
    IS: IncidenceSet<EIx> + IncidenceSetRef<'r, EIx>,
    ES: Endpoints<NodeIx = VIx> + 'r,
    VIx: Copy + Eq + 'r,
    EIx: Copy + 'r,
    V: 'r,
    E: 'r,
{
    type Item = WalkItem<'r, EIx, E, VIx>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((eix, ep_iter)) = self.cur_edge.as_mut() {
                let eix = *eix;
                for vix in ep_iter.by_ref() {
                    if vix == self.origin {
                        continue;
                    }
                    // SAFETY: eix is from a maintained incidence set.
                    let edge_value = unsafe { self.edges.get_value_unchecked(&eix) };
                    return Some(WalkItem::new(eix, edge_value, vix));
                }
                self.cur_edge = None;
            }
            let eix = self.incidence_iter.next()?;
            let endpoints = unsafe { self.edges.get_storage_unchecked(&eix) }.clone();
            self.cur_edge = Some((eix, endpoints.into_iter()));
        }
    }
}

impl<'r, NC, EC, V, E, VIx, EIx, IS, ES> GraphOperation<'r> for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS> + 'r,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES> + 'r,
    IS: IncidenceSet<EIx> + for<'a> IncidenceSetRef<'a, EIx> + 'static,
    ES: Endpoints<NodeIx = VIx> + 'r,
    for<'a> NC: RandomAccessRef<'a>,
    for<'a> EC: RandomAccessRef<'a>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    V: 'r,
    E: 'r,
{
    fn contains_node_index(&self, node_ix: Self::NodeIx) -> bool {
        self.nodes.contains_index(&node_ix)
    }

    fn contains_edge_index(&self, edge_ix: Self::EdgeIx) -> bool {
        self.edges.contains_index(&edge_ix)
    }

    fn len_node(&self) -> usize {
        Collection::len(&self.nodes)
    }

    fn len_edge(&self) -> usize {
        Collection::len(&self.edges)
    }

    type NodeIndices = <NC as RandomAccessRef<'r>>::Indices;
    fn node_indices(&'r self) -> Self::NodeIndices {
        self.nodes.indices()
    }

    type EdgeIndices = <EC as RandomAccessRef<'r>>::Indices;
    fn edge_indices(&'r self) -> Self::EdgeIndices {
        self.edges.indices()
    }

    unsafe fn node_unchecked(&self, node_ix: Self::NodeIx) -> &Self::Node {
        unsafe { self.nodes.get_value_unchecked(&node_ix) }
    }

    unsafe fn edge_unchecked(&self, edge_ix: Self::EdgeIx) -> &Self::Edge {
        unsafe { self.edges.get_value_unchecked(&edge_ix) }
    }

    unsafe fn endpoints_unchecked(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints {
        unsafe { self.edges.get_storage_unchecked(&edge_ix) }.clone()
    }

    type EdgeIndicesFrom = EdgeIndicesFromIter<<IS as IncidenceSetRef<'r, EIx>>::Iter>;
    unsafe fn edge_indices_from_unchecked(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesFrom {
        let storage = unsafe { self.nodes.get_storage_unchecked(&node_ix) };
        EdgeIndicesFromIter {
            inner: IncidenceSetRef::iter(storage),
        }
    }

    type EdgeIndicesOf = EdgeIndicesFromIter<<IS as IncidenceSetRef<'r, EIx>>::Iter>;
    unsafe fn edge_indices_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesOf {
        // Undirected: same as edge_indices_from.
        unsafe { self.edge_indices_from_unchecked(node_ix) }
    }

    type WalksFrom = Walks<'r, NC, EC, VIx, EIx, IS, ES>;
    unsafe fn walks_from_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksFrom {
        let storage = unsafe { self.nodes.get_storage_unchecked(&node_ix) };
        Walks {
            origin: node_ix,
            edges: &self.edges,
            incidence_iter: IncidenceSetRef::iter(storage),
            cur_edge: None,
            _marker: PhantomData,
        }
    }

    type WalksOf = Walks<'r, NC, EC, VIx, EIx, IS, ES>;
    unsafe fn walks_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksOf {
        // Undirected: identical to walks_from.
        unsafe { self.walks_from_unchecked(node_ix) }
    }

    type DrainNode = NC::IntoValues;
    type DrainEdge = EC::IntoValues;
    fn drain(self) -> (Self::DrainNode, Self::DrainEdge) {
        (self.nodes.into_values(), self.edges.into_values())
    }

    fn reverse(&mut self) {
        // Undirected: no-op.
    }
}

impl<NC, EC, V, E, VIx, EIx, IS, ES> InsertNode for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS>
        + InsertableCollection<InsertedIndex = VIx>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES>,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node> {
        self.nodes.insert(node, IS::default()).map_err(|(v, _)| v)
    }
}

impl<NC, EC, V, E, VIx, EIx, IS, ES> InsertEdge for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS> + UpdatableRandomAccess,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES>
        + InsertableCollection<InsertedIndex = EIx>,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge> {
        let endpoints_clone = endpoints.clone();
        let eix = match self.edges.insert(edge, endpoints) {
            Ok(eix) => eix,
            Err((v, _)) => return Err(v),
        };
        for vix in endpoints_clone.into_iter() {
            let storage = unsafe { self.nodes.get_storage_unchecked_mut(&vix) };
            storage.insert(eix);
        }
        Ok(eix)
    }
}

impl<'r, NC, EC, V, E, VIx, EIx, IS, ES> UpdateNode<'r> for HyperGraph<NC, EC>
where
    NC: UpdatableRandomAccess<Index = VIx, Value = V, Storage = IS> + 'r,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES> + 'r,
    IS: IncidenceSet<EIx> + 'static,
    ES: Endpoints<NodeIx = VIx> + 'r,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    V: 'r,
    E: 'r,
{
    unsafe fn node_unchecked_mut(&mut self, node_ix: Self::NodeIx) -> &mut Self::Node {
        unsafe { self.nodes.get_value_unchecked_mut(&node_ix) }
    }

    type WalksFromMut = std::iter::Empty<WalkItemMut<'r, EIx, E, VIx>>;
    unsafe fn walks_from_unchecked_mut(&'r mut self, _node_ix: Self::NodeIx) -> Self::WalksFromMut {
        std::iter::empty()
    }

    type WalksOfMut = std::iter::Empty<WalkItemMut<'r, EIx, E, VIx>>;
    unsafe fn walks_of_unchecked_mut(&'r mut self, _node_ix: Self::NodeIx) -> Self::WalksOfMut {
        std::iter::empty()
    }
}

impl<NC, EC, V, E, VIx, EIx, IS, ES> UpdateEdge for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS>,
    EC: UpdatableRandomAccess<Index = EIx, Value = E, Storage = ES>,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn edge_unchecked_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge {
        unsafe { self.edges.get_value_unchecked_mut(&edge_ix) }
    }
}

impl<NC, EC, V, E, VIx, EIx, IS, ES> RemoveEdge for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS> + UpdatableRandomAccess,
    EC: RemovableRandomAccess<Index = EIx, Value = E, Storage = ES>,
    for<'a> EC: RandomAccessRef<'a>,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn take_edge_unchecked(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge {
        let (e, endpoints, swapped) = unsafe { self.edges.take_unchecked(&edge_ix) };
        for vix in endpoints.into_iter() {
            let storage = unsafe { self.nodes.get_storage_unchecked_mut(&vix) };
            storage.remove(&edge_ix);
        }
        // If the outer EC swap-removed, rewrite references to old_last.
        if let Some(old_last) = swapped {
            // The relocated edge now sits at `edge_ix`. Its endpoints'
            // incidence sets currently hold `old_last`; they need to point
            // at `edge_ix`.
            let relocated_endpoints = unsafe { self.edges.get_storage_unchecked(&edge_ix) }.clone();
            for vix in relocated_endpoints.into_iter() {
                let storage = unsafe { self.nodes.get_storage_unchecked_mut(&vix) };
                storage.remove(&old_last);
                storage.insert(edge_ix);
            }
        }
        e
    }
}

// ---------------------------------------------------------------------------
// RemoveNode — cascade: drop every incident edge, then take the node.
// Handles outer NC swap-remove by rewriting references to old_last.
// ---------------------------------------------------------------------------

impl<NC, EC, V, E, VIx, EIx, IS, ES> RemoveNode for HyperGraph<NC, EC>
where
    NC: RemovableRandomAccess<Index = VIx, Value = V, Storage = IS> + UpdatableRandomAccess,
    EC: RemovableRandomAccess<Index = EIx, Value = E, Storage = ES> + UpdatableRandomAccess,
    for<'a> NC: RandomAccessRef<'a>,
    for<'a> EC: RandomAccessRef<'a>,
    IS: IncidenceSet<EIx> + for<'a> IncidenceSetRef<'a, EIx> + 'static,
    ES: Endpoints<NodeIx = VIx> + 'static,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
    unsafe fn take_node_unchecked(&mut self, node_ix: Self::NodeIx) -> Self::Node {
        // Re-reading the incidence set after each removal picks up EIxs
        // that were rewritten by `take_edge_unchecked`'s swap-remove fixup
        // (Vec-backed EC).
        loop {
            let next_eix = {
                let storage = unsafe { self.nodes.get_storage_unchecked(&node_ix) };
                IncidenceSetRef::iter(storage).next()
            };
            let eix = match next_eix {
                Some(eix) => eix,
                None => break,
            };
            let _ = unsafe { <Self as RemoveEdge>::take_edge_unchecked(self, eix) };
        }

        let (data, _storage, swapped) = unsafe { self.nodes.take_unchecked(&node_ix) };

        // On NC swap-remove, rewrite every edge whose endpoints reference
        // `old_last` to reference `node_ix` instead.
        if let Some(old_last) = swapped {
            let all_eixs: Vec<EIx> = self.edges.indices().collect();
            for eix in all_eixs {
                let endpoints_clone = unsafe { self.edges.get_storage_unchecked(&eix) }.clone();
                let mut needs_rewrite = false;
                for vix in endpoints_clone.iter() {
                    if vix == old_last {
                        needs_rewrite = true;
                        break;
                    }
                }
                if !needs_rewrite {
                    continue;
                }
                let rewritten: Vec<VIx> = endpoints_clone
                    .into_iter()
                    .map(|v| if v == old_last { node_ix } else { v })
                    .collect();
                let new_endpoints = match <ES as Endpoints>::try_from_node_indices(rewritten) {
                    Some(ep) => ep,
                    None => continue, // shouldn't happen
                };
                let entry = unsafe { self.edges.get_storage_unchecked_mut(&eix) };
                *entry = new_endpoints;
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
        let mut nodes_out = IN::default();
        let mut edges_out = IE::default();
        for eix in edge_indices {
            if !self.edges.contains_index(&eix) {
                continue;
            }
            let e = unsafe { <Self as RemoveEdge>::take_edge_unchecked(self, eix) };
            edges_out.extend(core::iter::once(e));
        }
        for nix in node_indices {
            if !self.nodes.contains_index(&nix) {
                continue;
            }
            let v = unsafe { self.take_node_unchecked(nix) };
            nodes_out.extend(core::iter::once(v));
        }
        (nodes_out, edges_out)
    }
}

unsafe impl<NC, EC, V, E, VIx, EIx, IS, ES> StableNode for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS> + StableCollection,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES>,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
}

unsafe impl<NC, EC, V, E, VIx, EIx, IS, ES> StableEdge for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES> + StableCollection,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
{
}

impl<NC, EC, V, E, VIx, EIx, IS, ES> UniqueNode for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS>
        + crate::collection::CollectionBiject
        + StableCollection,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES>,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    V: PartialEq,
{
    fn node_index(&self, node: impl core::borrow::Borrow<Self::Node>) -> Option<Self::NodeIx> {
        // SAFETY: CollectionBiject contract guarantees an inverse lookup
        // when the node value is present.
        unsafe { self.nodes.value_to_key_unchecked(node.borrow()) }.copied()
    }
}

impl<NC, EC, V, E, VIx, EIx, IS, ES> UniqueEdge for HyperGraph<NC, EC>
where
    NC: RandomAccess<Index = VIx, Value = V, Storage = IS>,
    EC: RandomAccess<Index = EIx, Value = E, Storage = ES>
        + crate::collection::CollectionBiject
        + StableCollection,
    IS: IncidenceSet<EIx>,
    ES: Endpoints<NodeIx = VIx>,
    VIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    EIx: Copy + Eq + Ord + Hash + Display + Debug + 'static,
    E: PartialEq,
{
    fn edge_index(&self, edge: impl core::borrow::Borrow<Self::Edge>) -> Option<Self::EdgeIx> {
        unsafe { self.edges.value_to_key_unchecked(edge.borrow()) }.copied()
    }
}
