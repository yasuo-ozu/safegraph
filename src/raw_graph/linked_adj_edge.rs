//! Graph backend built directly on the [`RandomAccess`] trait family.
//!
//! Storage is split into two independent `RandomAccess` collections — one for
//! nodes, one for edges — so users can freely combine `Vec`, `BTreeMap`, and
//! `HashMap` (or any other future backend) on each side.
//!
//! The graph itself is direction-agnostic: edges store `[NIx; 2]` endpoints
//! with positional meaning (slot 0 / slot 1) but the type system does not
//! track "directed vs undirected". `walks_from_unchecked` follows OUTGOING
//! only; `walks_of_unchecked` traverses both adjacency lists with self-loop
//! deduplication.
//!
//! [`RandomAccess`]: crate::collection::RandomAccess
//!
//! # Layout
//!
//! Each entry in the node collection carries (`Value = N`, `Storage =
//! NodeRepr<ESlot>`), and each edge entry carries (`Value = E`, `Storage =
//! EdgeRepr<NIx, ESlot>`). For Vec-backed sequences the entries are tuples
//! `(N, NodeRepr<u32>)`; for Map-backed maps the entries are `(N,
//! NodeRepr<Option<I>>)` with `N` as the key.

use crate::unwrap_unchecked;
use core::marker::PhantomData;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use core::borrow::Borrow;

use crate::collection::{
    Collection, CollectionBiject, InsertableCollection, RandomAccess, RandomAccessRef,
    RemovableRandomAccess, StableCollection, UpdatableRandomAccess,
};
use crate::graph::capability::{
    Bigraph, Directed, InsertEdge, InsertNode, RemoveEdge, RemoveNode, StableEdge, StableNode,
    UniqueEdge, UniqueNode, UpdateEdge, UpdateNode,
};
use crate::graph::operation::GraphOperation;
use crate::graph::walk_item::{WalkItem, WalkItemMut, WalkItemTo};
use crate::graph::{GraphMap, GraphProperty};

const OUTGOING: usize = 0;
const INCOMING: usize = 1;

/// Per-node adjacency-list head pointers.
///
/// `next[OUTGOING]` and `next[INCOMING]` are slot values produced by
/// `EC::to_slot` / `EC::sentinel`; convert via `EC::from_slot` at the
/// boundary.
#[derive(Clone, Debug)]
pub struct NodeRepr<ESlot> {
    pub(crate) next: [ESlot; 2],
}

/// Per-edge endpoints and next-edge pointers for the two adjacency lists.
#[derive(Clone, Debug)]
pub struct EdgeRepr<NIx, ESlot> {
    pub(crate) next: [ESlot; 2],
    pub(crate) node: [NIx; 2],
}

/// Graph backed by two independent [`RandomAccess`] collections.
///
/// No trait bounds on the type definition itself — constraints live on the
/// `impl` blocks. See module docs for the expected `Storage` shape on each
/// collection.
#[derive(Clone, Debug)]
pub struct LinkedAdjEdgeGraph<NC, EC> {
    pub(crate) nodes: NC,
    pub(crate) edges: EC,
}

impl<NC: Default, EC: Default> Default for LinkedAdjEdgeGraph<NC, EC> {
    fn default() -> Self {
        Self {
            nodes: NC::default(),
            edges: EC::default(),
        }
    }
}

impl<NC: Default, EC: Default> LinkedAdjEdgeGraph<NC, EC> {
    /// Equivalent to [`Default::default`]. Provided for ergonomic parity with
    /// the old `BTreeGraph::new()` / `HashGraph::new()` constructors.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Walks one direction's adjacency chain from a node.
///
/// `IS_INCOMING` selects the chain: `false` for outgoing, `true` for incoming.
/// `next` holds the raw slot value; convert via `EC::from_slot` to test
/// whether we've reached the sentinel.
pub struct EdgeIndicesDirected<'a, NIx, EC, ESlot, const IS_INCOMING: bool> {
    edges: &'a EC,
    next: ESlot,
    #[cfg(debug_assertions)]
    node_key: NIx,
    _marker: PhantomData<NIx>,
}

impl<'a, NIx, EC, ESlot, const IS_INCOMING: bool> Iterator
    for EdgeIndicesDirected<'a, NIx, EC, ESlot, IS_INCOMING>
where
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NIx, ESlot>>,
    ESlot: Copy + Eq + Hash + 'a,
    NIx: Copy + Eq + 'a,
{
    type Item = EC::Index;

    fn next(&mut self) -> Option<EC::Index> {
        let eix = EC::from_slot(self.next)?;
        // SAFETY: `eix` came from an in-graph adjacency list.
        let storage = unsafe { self.edges.get_storage_unchecked(&eix) };
        #[cfg(debug_assertions)]
        debug_assert!(storage.node[IS_INCOMING as usize] == self.node_key);
        self.next = storage.next[IS_INCOMING as usize];
        Some(eix)
    }
}

/// All incident edges of a node: outgoing then incoming, with self-loops
/// yielded only once.
pub struct EdgeIndicesOf<'a, NIx, EC, ESlot> {
    edges: &'a EC,
    outgoing_next: ESlot,
    incoming_next: ESlot,
    node_key: NIx,
}

impl<'a, NIx, EC, ESlot> Iterator for EdgeIndicesOf<'a, NIx, EC, ESlot>
where
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NIx, ESlot>>,
    ESlot: Copy + Eq + Hash + 'a,
    NIx: Copy + Eq + 'a,
{
    type Item = EC::Index;

    fn next(&mut self) -> Option<EC::Index> {
        if let Some(eix) = EC::from_slot(self.outgoing_next) {
            // SAFETY: from in-graph adjacency list.
            let storage = unsafe { self.edges.get_storage_unchecked(&eix) };
            self.outgoing_next = storage.next[OUTGOING];
            #[cfg(debug_assertions)]
            debug_assert!(storage.node[OUTGOING] == self.node_key);
            return Some(eix);
        }
        loop {
            let eix = EC::from_slot(self.incoming_next)?;
            // SAFETY: from in-graph adjacency list.
            let storage = unsafe { self.edges.get_storage_unchecked(&eix) };
            self.incoming_next = storage.next[INCOMING];
            debug_assert!(storage.node[INCOMING] == self.node_key);
            // Skip self-loops — already yielded by outgoing pass.
            if storage.node[OUTGOING] != self.node_key {
                return Some(eix);
            }
        }
    }
}

/// Walk triples for one direction. Parameterised over `ER` (the edge-
/// collection reference type): `&'r EC` for shared walks, `&'r mut EC` for
/// mutable walks. Two `Iterator` impls (below) handle each case.
pub struct WalksDirected<NIx, ER, ESlot, const IS_INCOMING: bool> {
    edges: ER,
    next: ESlot,
    #[cfg(debug_assertions)]
    node_key: NIx,
    _marker: PhantomData<NIx>,
}

// --- Shared walks: ER = &'r EC ---

impl<'r, NIx, EC, ESlot, const IS_INCOMING: bool> Iterator
    for WalksDirected<NIx, &'r EC, ESlot, IS_INCOMING>
where
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NIx, ESlot>>,
    EC::Value: 'r,
    ESlot: Copy + Eq + Hash + 'r,
    NIx: Copy + Eq + 'r,
{
    type Item = WalkItem<'r, EC::Index, EC::Value, NIx>;

    fn next(&mut self) -> Option<Self::Item> {
        let eix = EC::from_slot(self.next)?;
        // self.edges: &'r EC (Copy). Passing it preserves 'r in the &self
        // parameter of the trait method, so the returned references are 'r.
        // SAFETY: `eix` came from an in-graph adjacency list.
        let (edge, storage) = unsafe { self.edges.get_both_unchecked(&eix) };
        self.next = storage.next[IS_INCOMING as usize];
        #[cfg(debug_assertions)]
        debug_assert!(storage.node[IS_INCOMING as usize] == self.node_key);
        Some(WalkItem::new(
            eix,
            edge,
            storage.node[(!IS_INCOMING) as usize],
        ))
    }
}

// --- Mutable walks: ER = &'r mut EC ---

impl<'r, NIx, EC, ESlot, const IS_INCOMING: bool> Iterator
    for WalksDirected<NIx, &'r mut EC, ESlot, IS_INCOMING>
where
    EC: UpdatableRandomAccess<Slot = ESlot, Storage = EdgeRepr<NIx, ESlot>>,
    EC::Value: 'r,
    ESlot: Copy + Eq + Hash + 'r,
    NIx: Copy + Eq + 'r,
{
    type Item = WalkItemMut<'r, EC::Index, EC::Value, NIx>;

    fn next(&mut self) -> Option<Self::Item> {
        let eix = EC::from_slot(self.next)?;
        // SAFETY: each next() yields a distinct EIx (graph walks visit each
        // edge at most once), so the &'r mut references we hand out never
        // alias. We use raw pointers to escape the per-call &mut self borrow
        // and re-extend to 'r — the underlying *self.edges borrow is held
        // by the iterator for the full 'r.
        unsafe {
            let next_slot;
            let opp;
            {
                let storage = self.edges.get_storage_unchecked(&eix);
                next_slot = storage.next[IS_INCOMING as usize];
                opp = storage.node[(!IS_INCOMING) as usize];
                #[cfg(debug_assertions)]
                debug_assert!(storage.node[IS_INCOMING as usize] == self.node_key);
            }
            self.next = next_slot;
            // Now take mutable access to the value.
            let edge: &mut EC::Value = self.edges.get_value_unchecked_mut(&eix);
            let edge: &'r mut EC::Value = core::mem::transmute(edge);
            Some(WalkItemMut::new(eix, edge, opp))
        }
    }
}

/// All-incidents walk: outgoing then incoming with self-loop dedup.
/// Parameterised over `ER` the same way as `WalksDirected`.
pub struct WalksOf<NIx, ER, ESlot> {
    edges: ER,
    node_key: NIx,
    outgoing_next: ESlot,
    incoming_next: ESlot,
}

// --- Shared walks_of: ER = &'r EC ---

impl<'r, NIx, EC, ESlot> Iterator for WalksOf<NIx, &'r EC, ESlot>
where
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NIx, ESlot>>,
    EC::Value: 'r,
    ESlot: Copy + Eq + Hash + 'r,
    NIx: Copy + Eq + 'r,
{
    type Item = WalkItem<'r, EC::Index, EC::Value, NIx>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(eix) = EC::from_slot(self.outgoing_next) {
            // SAFETY: from in-graph adjacency list.
            let (edge, storage) = unsafe { self.edges.get_both_unchecked(&eix) };
            self.outgoing_next = storage.next[OUTGOING];
            debug_assert!(storage.node[OUTGOING] == self.node_key);
            return Some(WalkItem::new(eix, edge, storage.node[INCOMING]));
        }
        loop {
            let eix = EC::from_slot(self.incoming_next)?;
            // SAFETY: from in-graph adjacency list.
            let (edge, storage) = unsafe { self.edges.get_both_unchecked(&eix) };
            self.incoming_next = storage.next[INCOMING];
            debug_assert!(storage.node[INCOMING] == self.node_key);
            if storage.node[OUTGOING] != self.node_key {
                return Some(WalkItem::new(eix, edge, storage.node[OUTGOING]));
            }
        }
    }
}

// --- Mutable walks_of: ER = &'r mut EC ---

impl<'r, NIx, EC, ESlot> Iterator for WalksOf<NIx, &'r mut EC, ESlot>
where
    EC: UpdatableRandomAccess<Slot = ESlot, Storage = EdgeRepr<NIx, ESlot>>,
    EC::Value: 'r,
    ESlot: Copy + Eq + Hash + 'r,
    NIx: Copy + Eq + 'r,
{
    type Item = WalkItemMut<'r, EC::Index, EC::Value, NIx>;

    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: as in WalksDirected mut impl — distinct EIx per call, raw
        // pointer used to extend lifetime to 'r.
        unsafe {
            // Outgoing first.
            if let Some(eix) = EC::from_slot(self.outgoing_next) {
                let (next_slot, opp);
                {
                    let storage = self.edges.get_storage_unchecked(&eix);
                    next_slot = storage.next[OUTGOING];
                    opp = storage.node[INCOMING];
                    debug_assert!(storage.node[OUTGOING] == self.node_key);
                }
                self.outgoing_next = next_slot;
                let edge: &mut EC::Value = self.edges.get_value_unchecked_mut(&eix);
                let edge: &'r mut EC::Value = core::mem::transmute(edge);
                return Some(WalkItemMut::new(eix, edge, opp));
            }
            loop {
                let eix = EC::from_slot(self.incoming_next)?;
                let (next_slot, target_out, target_in);
                {
                    let storage = self.edges.get_storage_unchecked(&eix);
                    next_slot = storage.next[INCOMING];
                    target_out = storage.node[OUTGOING];
                    target_in = storage.node[INCOMING];
                    debug_assert!(target_in == self.node_key);
                }
                self.incoming_next = next_slot;
                if target_out != self.node_key {
                    let edge: &mut EC::Value = self.edges.get_value_unchecked_mut(&eix);
                    let edge: &'r mut EC::Value = core::mem::transmute(edge);
                    return Some(WalkItemMut::new(eix, edge, target_out));
                }
                // self-loop already yielded by outgoing pass — skip.
            }
        }
    }
}

impl<NC, EC, ESlot> GraphProperty for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
    type Node = NC::Value;
    type Edge = EC::Value;
    type NodeIx = NC::Index;
    type EdgeIx = EC::Index;
    type Endpoints = [NC::Index; 2];
    const DIRECTED: bool = true;
}

impl<'r, NC, EC, ESlot> GraphOperation<'r> for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>> + 'r,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>> + 'r,
    for<'a> NC: RandomAccessRef<'a>,
    for<'a> EC: RandomAccessRef<'a>,
    ESlot: Copy + Eq + Hash + 'r,
    NC::Index: Display + Debug + 'r,
    EC::Index: Display + Debug,
    NC::Value: 'r,
    EC::Value: 'r,
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

    fn capacity_node(&self) -> Option<usize> {
        Collection::capacity(&self.nodes)
    }
    fn capacity_edge(&self) -> Option<usize> {
        Collection::capacity(&self.edges)
    }

    type NodeIndices = <NC as RandomAccessRef<'r>>::Indices;
    type EdgeIndices = <EC as RandomAccessRef<'r>>::Indices;

    fn node_indices(&'r self) -> Self::NodeIndices {
        self.nodes.indices()
    }
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
        let storage = unsafe { self.edges.get_storage_unchecked(&edge_ix) };
        storage.node
    }

    type EdgeIndicesFrom = EdgeIndicesDirected<'r, NC::Index, EC, ESlot, false>;

    unsafe fn edge_indices_from_unchecked(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesFrom {
        let head_slot = unsafe { self.nodes.get_storage_unchecked(&node_ix) }.next[OUTGOING];
        EdgeIndicesDirected {
            edges: &self.edges,
            next: head_slot,
            #[cfg(debug_assertions)]
            node_key: node_ix,
            _marker: PhantomData,
        }
    }

    type EdgeIndicesOf = EdgeIndicesOf<'r, NC::Index, EC, ESlot>;

    unsafe fn edge_indices_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesOf {
        let node_storage = unsafe { self.nodes.get_storage_unchecked(&node_ix) };
        EdgeIndicesOf {
            edges: &self.edges,
            outgoing_next: node_storage.next[OUTGOING],
            incoming_next: node_storage.next[INCOMING],
            node_key: node_ix,
        }
    }

    type WalksFrom = WalksDirected<NC::Index, &'r EC, ESlot, false>;

    unsafe fn walks_from_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksFrom {
        let head_slot = unsafe { self.nodes.get_storage_unchecked(&node_ix) }.next[OUTGOING];
        WalksDirected {
            edges: &self.edges,
            next: head_slot,
            #[cfg(debug_assertions)]
            node_key: node_ix,
            _marker: PhantomData,
        }
    }

    type WalksOf = WalksOf<NC::Index, &'r EC, ESlot>;

    unsafe fn walks_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksOf {
        let node_storage = unsafe { self.nodes.get_storage_unchecked(&node_ix) };
        WalksOf {
            edges: &self.edges,
            node_key: node_ix,
            outgoing_next: node_storage.next[OUTGOING],
            incoming_next: node_storage.next[INCOMING],
        }
    }

    type DrainNode = NC::IntoValues;
    type DrainEdge = EC::IntoValues;

    fn drain(self) -> (Self::DrainNode, Self::DrainEdge) {
        (
            Collection::into_values(self.nodes),
            Collection::into_values(self.edges),
        )
    }

    fn reverse(&mut self) {
        let node_keys: Vec<NC::Index> = self.nodes.indices().collect();
        for nix in node_keys {
            let s = unsafe { self.nodes.get_storage_unchecked_mut(&nix) };
            s.next.swap(0, 1);
        }
        let edge_keys: Vec<EC::Index> = self.edges.indices().collect();
        for eix in edge_keys {
            let s = unsafe { self.edges.get_storage_unchecked_mut(&eix) };
            s.next.swap(0, 1);
            s.node.swap(0, 1);
        }
    }
}

impl<NC, EC, ESlot> Bigraph for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2] {
        endpoints
    }
    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints {
        nodes
    }
}

impl<NC, EC, ESlot> InsertNode for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>
        + InsertableCollection<InsertedIndex = <NC as RandomAccess>::Index>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
    unsafe fn insert_node_unchecked(
        &mut self,
        node: Self::Node,
    ) -> Result<Self::NodeIx, Self::Node> {
        let sentinel = EC::sentinel();
        let storage = NodeRepr {
            next: [sentinel, sentinel],
        };
        self.nodes.insert(node, storage).map_err(|(v, _)| v)
    }
}

impl<NC, EC, ESlot> InsertEdge for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>
        + InsertableCollection<InsertedIndex = <EC as RandomAccess>::Index>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
    unsafe fn insert_edge_unchecked(
        &mut self,
        edge: Self::Edge,
        endpoints: Self::Endpoints,
    ) -> Result<Self::EdgeIx, Self::Edge> {
        let [from, to] = endpoints;
        let old_out = unsafe { self.nodes.get_storage_unchecked(&from) }.next[OUTGOING];
        let old_in = unsafe { self.nodes.get_storage_unchecked(&to) }.next[INCOMING];
        let edge_storage = EdgeRepr {
            next: [old_out, old_in],
            node: [from, to],
        };
        let eix = match self.edges.insert(edge, edge_storage) {
            Ok(eix) => eix,
            Err((e, _)) => return Err(e),
        };
        let new_slot = EC::to_slot(eix);
        unsafe { self.nodes.get_storage_unchecked_mut(&from) }.next[OUTGOING] = new_slot;
        unsafe { self.nodes.get_storage_unchecked_mut(&to) }.next[INCOMING] = new_slot;
        Ok(eix)
    }
}

impl<'r, NC, EC, ESlot> UpdateNode<'r> for LinkedAdjEdgeGraph<NC, EC>
where
    NC: UpdatableRandomAccess<Storage = NodeRepr<ESlot>> + 'r,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>> + 'r,
    ESlot: Copy + Eq + Hash + 'r,
    NC::Index: Display + Debug + 'r,
    EC::Index: Display + Debug + 'r,
    EC::Value: 'r,
{
    unsafe fn node_unchecked_mut(&mut self, node_ix: Self::NodeIx) -> &mut Self::Node {
        unsafe { self.nodes.get_value_unchecked_mut(&node_ix) }
    }

    type WalksFromMut = std::iter::Empty<WalkItemMut<'r, EC::Index, EC::Value, NC::Index>>;
    unsafe fn walks_from_unchecked_mut(&'r mut self, _node_ix: Self::NodeIx) -> Self::WalksFromMut {
        std::iter::empty()
    }

    type WalksOfMut = std::iter::Empty<WalkItemMut<'r, EC::Index, EC::Value, NC::Index>>;
    unsafe fn walks_of_unchecked_mut(&'r mut self, _node_ix: Self::NodeIx) -> Self::WalksOfMut {
        std::iter::empty()
    }
}

impl<NC, EC, ESlot> UpdateEdge for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: UpdatableRandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
    unsafe fn edge_unchecked_mut(&mut self, edge_ix: Self::EdgeIx) -> &mut Self::Edge {
        unsafe { self.edges.get_value_unchecked_mut(&edge_ix) }
    }
}

impl<'r, NC, EC, ESlot> Directed<'r> for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>> + 'r,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>> + 'r,
    for<'a> NC: RandomAccessRef<'a>,
    for<'a> EC: RandomAccessRef<'a>,
    ESlot: Copy + Eq + Hash + 'r,
    NC::Index: Display + Debug + 'r,
    EC::Index: Display + Debug,
    NC::Value: 'r,
    EC::Value: 'r,
{
    type EdgeIndicesTo = EdgeIndicesDirected<'r, NC::Index, EC, ESlot, true>;
    type EdgeTailIndices = core::iter::Once<NC::Index>;
    type EdgeHeadIndices = core::iter::Once<NC::Index>;
    type WalksTo = core::iter::Map<
        WalksDirected<NC::Index, &'r EC, ESlot, true>,
        fn(
            WalkItem<'r, EC::Index, EC::Value, NC::Index>,
        ) -> WalkItemTo<'r, NC::Index, EC::Index, EC::Value>,
    >;

    unsafe fn walks_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksTo {
        let head_slot = unsafe { self.nodes.get_storage_unchecked(&node_ix) }.next[INCOMING];
        let walks: WalksDirected<NC::Index, &'r EC, ESlot, true> = WalksDirected {
            edges: &self.edges,
            next: head_slot,
            #[cfg(debug_assertions)]
            node_key: node_ix,
            _marker: PhantomData,
        };
        // Reorder `(edge_ix, &edge, node_ix)` to `(node_ix, edge_ix, &edge)`
        // without a deref (so no `EC::Value: 'r`): shuffle the raw parts.
        walks.map(
            (|wi: WalkItem<'r, EC::Index, EC::Value, NC::Index>| {
                let (eix, edge_ptr, nix) = wi.into_parts();
                // SAFETY: the pointer is valid for `'r` (from `WalkItem::new`).
                unsafe { WalkItemTo::from_parts(nix, eix, edge_ptr) }
            })
                as fn(
                    WalkItem<'r, EC::Index, EC::Value, NC::Index>,
                ) -> WalkItemTo<'r, NC::Index, EC::Index, EC::Value>,
        )
    }

    unsafe fn edge_indices_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::EdgeIndicesTo {
        let head_slot = unsafe { self.nodes.get_storage_unchecked(&node_ix) }.next[INCOMING];
        EdgeIndicesDirected {
            edges: &self.edges,
            next: head_slot,
            #[cfg(debug_assertions)]
            node_key: node_ix,
            _marker: PhantomData,
        }
    }

    unsafe fn edge_tail_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeTailIndices {
        let storage = unsafe { self.edges.get_storage_unchecked(&edge_ix) };
        core::iter::once(storage.node[OUTGOING])
    }

    unsafe fn edge_head_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeHeadIndices {
        let storage = unsafe { self.edges.get_storage_unchecked(&edge_ix) };
        core::iter::once(storage.node[INCOMING])
    }
}

/// Walk the linked list at `(node_ix, dir)` and replace any pointer to `target`
/// with `replacement`.
///
/// # Safety
/// `node_ix` must be valid and `nodes`/`edges` must be the same collections.
unsafe fn replace_in_list<NC, EC, ESlot>(
    nodes: &mut NC,
    edges: &mut EC,
    node_ix: NC::Index,
    target: EC::Index,
    replacement: ESlot,
    dir: usize,
) where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
{
    let mut cur = &mut unsafe { nodes.get_storage_unchecked_mut(&node_ix) }.next[dir];
    loop {
        debug_assert!(EC::from_slot(*cur).is_some());
        let unwrapped = unwrap_unchecked(EC::from_slot(*cur));
        if unwrapped == target {
            break;
        }
        cur = &mut unsafe { edges.get_storage_unchecked_mut(&unwrapped) }.next[dir];
    }
    *cur = replacement;
}

/// Remove `node_ix` after all incident edges have been removed. If the backend
/// relocates the last node into the freed slot, patches every incident edge to
/// point at the new key.
///
/// # Safety
/// `node_ix` must be valid and have no incident edges remaining.
unsafe fn remove_node_inner<NC, EC, ESlot>(
    nodes: &mut NC,
    edges: &mut EC,
    node_ix: NC::Index,
) -> NC::Value
where
    NC: RemovableRandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
{
    let (data, _storage, swapped) = unsafe { nodes.take_unchecked(&node_ix) };

    if let Some(_old_last) = swapped {
        // The entry at `old_last` was moved to `node_ix`. Re-write every
        // incident edge's endpoint to the new key.
        for dir in [OUTGOING, INCOMING] {
            let head_slot = unsafe { nodes.get_storage_unchecked(&node_ix) }.next[dir];
            let mut cur = EC::from_slot(head_slot);
            while let Some(eix) = cur {
                let s = unsafe { edges.get_storage_unchecked_mut(&eix) };
                s.node[dir] = node_ix;
                let next_slot = s.next[dir];
                cur = EC::from_slot(next_slot);
            }
        }
    }

    data
}

/// Walk the `dir` adjacency chain of `node_ix` once, splicing out every edge
/// contained in `dying`. `expected` is the number of dying edges linked into
/// this chain; the walk stops as soon as the last one is unlinked, so the
/// cost is bounded by the position of the last dying link, never more than
/// the chain length.
///
/// Used by [`LinkedAdjEdgeGraph::take_nodes_edges_unchecked`]: with `k` dying
/// edges in a chain this is one bounded walk instead of `k` head-to-target
/// walks through [`replace_in_list`].
///
/// # Safety
/// `node_ix` must be a valid node index whose `dir` chain is intact and
/// contains exactly `expected` edges from `dying`.
unsafe fn unlink_dying_edges<NC, EC, ESlot>(
    nodes: &mut NC,
    edges: &mut EC,
    node_ix: NC::Index,
    dying: &std::collections::HashSet<EC::Index>,
    mut expected: usize,
    dir: usize,
) where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
{
    // `pred` identifies the pointer that may need rewriting: `None` is the
    // node's head pointer, `Some(e)` is edge `e`'s `next[dir]`.
    let mut pred: Option<EC::Index> = None;
    while expected > 0 {
        let cur_slot = match pred {
            None => unsafe { nodes.get_storage_unchecked(&node_ix) }.next[dir],
            Some(p) => unsafe { edges.get_storage_unchecked(&p) }.next[dir],
        };
        debug_assert!(EC::from_slot(cur_slot).is_some());
        let eix = unwrap_unchecked(EC::from_slot(cur_slot));
        if dying.contains(&eix) {
            let next = unsafe { edges.get_storage_unchecked(&eix) }.next[dir];
            match pred {
                None => unsafe { nodes.get_storage_unchecked_mut(&node_ix) }.next[dir] = next,
                Some(p) => unsafe { edges.get_storage_unchecked_mut(&p) }.next[dir] = next,
            }
            expected -= 1;
        } else {
            pred = Some(eix);
        }
    }
}

/// Walk the `dir` adjacency chain of `node_ix` once, rewriting every pointer
/// whose slot value appears in `moved` (stale position → current position).
/// `expected` is the number of stale pointers in this chain; the walk stops
/// once all of them are patched.
///
/// Used by [`LinkedAdjEdgeGraph::take_nodes_edges_unchecked`] to repair the
/// adjacency lists after a whole batch of `swap_remove` relocations in one
/// pass per affected chain.
///
/// # Safety
/// `node_ix` must be a valid node index. Every pointer in its `dir` chain
/// must either be valid or have its target's current position recorded in
/// `moved`, and exactly `expected` of them must be stale.
unsafe fn patch_moved_edges<NC, EC, ESlot>(
    nodes: &mut NC,
    edges: &mut EC,
    node_ix: NC::Index,
    moved: &std::collections::HashMap<ESlot, ESlot>,
    mut expected: usize,
    dir: usize,
) where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
{
    let mut pred: Option<EC::Index> = None;
    while expected > 0 {
        let mut cur_slot = match pred {
            None => unsafe { nodes.get_storage_unchecked(&node_ix) }.next[dir],
            Some(p) => unsafe { edges.get_storage_unchecked(&p) }.next[dir],
        };
        if let Some(&current) = moved.get(&cur_slot) {
            match pred {
                None => unsafe { nodes.get_storage_unchecked_mut(&node_ix) }.next[dir] = current,
                Some(p) => unsafe { edges.get_storage_unchecked_mut(&p) }.next[dir] = current,
            }
            cur_slot = current;
            expected -= 1;
        }
        debug_assert!(EC::from_slot(cur_slot).is_some());
        let eix = unwrap_unchecked(EC::from_slot(cur_slot));
        pred = Some(eix);
    }
}

impl<NC, EC, ESlot> RemoveEdge for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RemovableRandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
    unsafe fn take_edge_unchecked(&mut self, edge_ix: Self::EdgeIx) -> Self::Edge {
        let storage = unsafe { self.edges.get_storage_unchecked(&edge_ix) };
        let [from_node, to_node] = storage.node;
        let next_out_slot = storage.next[OUTGOING];
        let next_in_slot = storage.next[INCOMING];

        unsafe {
            replace_in_list(
                &mut self.nodes,
                &mut self.edges,
                from_node,
                edge_ix,
                next_out_slot,
                OUTGOING,
            );
            replace_in_list(
                &mut self.nodes,
                &mut self.edges,
                to_node,
                edge_ix,
                next_in_slot,
                INCOMING,
            );
        }

        let (data, _storage, swapped) = unsafe { self.edges.take_unchecked(&edge_ix) };

        if let Some(old_last) = swapped {
            // The entry at `old_last` was moved to `edge_ix`. Re-link adjacency
            // pointers that referenced `old_last`.
            let moved_storage = unsafe { self.edges.get_storage_unchecked(&edge_ix) };
            let moved_endpoints = moved_storage.node;
            let new_slot = EC::to_slot(edge_ix);
            unsafe {
                replace_in_list(
                    &mut self.nodes,
                    &mut self.edges,
                    moved_endpoints[OUTGOING],
                    old_last,
                    new_slot,
                    OUTGOING,
                );
                replace_in_list(
                    &mut self.nodes,
                    &mut self.edges,
                    moved_endpoints[INCOMING],
                    old_last,
                    new_slot,
                    INCOMING,
                );
            }
        }

        data
    }
}

impl<NC, EC, ESlot> RemoveNode for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RemovableRandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RemovableRandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
    unsafe fn take_node_unchecked(&mut self, node_ix: Self::NodeIx) -> Self::Node {
        // Drain both adjacency chains. eix is always the head of node_ix's
        // `dir` chain, so we advance it with a direct pointer write rather than
        // walking via replace_in_list. The outgoing pass removes self-loops
        // entirely (also unlinked from node_ix's incoming chain via the peer
        // replace_in_list), so the incoming pass sees no self-loops.
        for dir in [OUTGOING, INCOMING] {
            let other = 1 - dir;
            while let Some(eix) =
                EC::from_slot(unsafe { self.nodes.get_storage_unchecked(&node_ix) }.next[dir])
            {
                let (peer, next_this, next_other) = {
                    let s = unsafe { self.edges.get_storage_unchecked(&eix) };
                    (s.node[other], s.next[dir], s.next[other])
                };
                // eix is the head — advance directly.
                unsafe { self.nodes.get_storage_unchecked_mut(&node_ix) }.next[dir] = next_this;
                // Unlink eix from peer's `other` chain. For self-loops in the
                // outgoing pass, peer == node_ix and eix may sit anywhere in
                // node_ix's incoming chain.
                unsafe {
                    replace_in_list(
                        &mut self.nodes,
                        &mut self.edges,
                        peer,
                        eix,
                        next_other,
                        other,
                    )
                };
                let (_, _, swapped) = unsafe { self.edges.take_unchecked(&eix) };
                if let Some(old_last) = swapped {
                    let new_slot = EC::to_slot(eix);
                    let (m_out, m_in) = {
                        let s = unsafe { self.edges.get_storage_unchecked(&eix) };
                        (s.node[OUTGOING], s.node[INCOMING])
                    };
                    unsafe {
                        replace_in_list(
                            &mut self.nodes,
                            &mut self.edges,
                            m_out,
                            old_last,
                            new_slot,
                            OUTGOING,
                        );
                        replace_in_list(
                            &mut self.nodes,
                            &mut self.edges,
                            m_in,
                            old_last,
                            new_slot,
                            INCOMING,
                        );
                    }
                }
            }
        }
        unsafe { remove_node_inner(&mut self.nodes, &mut self.edges, node_ix) }
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
        use core::mem::MaybeUninit;
        use std::collections::{HashMap, HashSet};

        let explicit_nodes: Vec<NC::Index> = node_indices.into_iter().collect();
        let explicit_edges: Vec<EC::Index> = edge_indices.into_iter().collect();
        let node_result_len = explicit_nodes.len();
        let edge_result_len = explicit_edges.len();

        // Plan A — edge-only fast path. With no nodes to remove there is no
        // cascade and no node-relocation bookkeeping, so we skip the phased
        // algorithm's transient hash maps entirely and loop the scalar
        // `take_edge_unchecked`, which repairs adjacency with pointer writes
        // only (no hashing). Victims are processed in DESCENDING index order so
        // each `swap_remove` relocates only an already-final survivor, keeping
        // every queued index valid — the same invariant phase 3 relies on. For
        // map-backed `EC` the order is harmless (removal relocates nothing).
        // Output preserves the caller's input order to match the phased path.
        if node_result_len == 0 {
            let mut edges_with_slot: Vec<(usize, EC::Index)> =
                explicit_edges.into_iter().enumerate().collect();
            edges_with_slot.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
            #[cfg(debug_assertions)]
            for w in edges_with_slot.windows(2) {
                debug_assert!(w[0].1 != w[1].1, "duplicate edge index in batch removal");
            }
            let mut edges_out_buf: Vec<MaybeUninit<EC::Value>> = (0..edge_result_len)
                .map(|_| MaybeUninit::uninit())
                .collect();
            for (slot, eix) in edges_with_slot {
                // SAFETY: descending order keeps every queued index valid; no
                // duplicates (checked above), so each slot in
                // `0..edge_result_len` is written exactly once.
                let data = unsafe { <Self as RemoveEdge>::take_edge_unchecked(self, eix) };
                unsafe { edges_out_buf.get_unchecked_mut(slot).write(data) };
            }
            let mut edges_out = IE::default();
            edges_out
                .extend(IntoIterator::into_iter(edges_out_buf).map(|e| unsafe { e.assume_init() }));
            return (IN::default(), edges_out);
        }

        // Plan A — node-only fast path, gated on backend + sparsity. Loop the
        // scalar `take_node_unchecked` (cascades incident edges, pointer-only
        // repair, no hashing), victims in DESCENDING index order so each
        // `swap_remove` relocates only an already-final survivor. Each removal
        // walks every incident edge's PEER chain via `replace_in_list`, so the
        // loop is ~O(K·d²) (d = avg degree) vs the phased single sweep's
        // O(K·d). It wins only when (a) both collections are sequence-backed —
        // for map-backed each chain step costs an extra O(log n) lookup, so the
        // d² factor loses there despite the saved hashing — and (b) the graph
        // is sparse enough that d² stays below the phased path's saved
        // hash-probe cost (empirical crossover ≈ avg degree 16). Otherwise fall
        // through to the (correct) phased path; the avg-degree ratio is a cheap
        // O(1) global proxy for the dying subset's density (exact for random
        // victims).
        const SCALAR_NODE_MAX_AVG_DEGREE: usize = 16;
        if edge_result_len == 0
            && NC::dense_indices()
            && EC::dense_indices()
            && Collection::len(&self.edges) * 2
                <= SCALAR_NODE_MAX_AVG_DEGREE * Collection::len(&self.nodes)
        {
            let mut nodes_with_slot: Vec<(usize, NC::Index)> =
                explicit_nodes.into_iter().enumerate().collect();
            nodes_with_slot.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
            #[cfg(debug_assertions)]
            for w in nodes_with_slot.windows(2) {
                debug_assert!(w[0].1 != w[1].1, "duplicate node index in batch removal");
            }
            let mut nodes_out_buf: Vec<MaybeUninit<NC::Value>> = (0..node_result_len)
                .map(|_| MaybeUninit::uninit())
                .collect();
            for (slot, nix) in nodes_with_slot {
                // SAFETY: descending order keeps every queued index valid; no
                // duplicates (checked above), so each slot in
                // `0..node_result_len` is written exactly once.
                let data = unsafe { <Self as RemoveNode>::take_node_unchecked(self, nix) };
                unsafe { nodes_out_buf.get_unchecked_mut(slot).write(data) };
            }
            let mut nodes_out = IN::default();
            nodes_out
                .extend(IntoIterator::into_iter(nodes_out_buf).map(|n| unsafe { n.assume_init() }));
            return (nodes_out, IE::default());
        }

        // Nodes removed in phase 5. Edges incident on any of these need no
        // adjacency maintenance — the whole chain dies with the node.
        let removed_nodes: HashSet<NC::Index> = explicit_nodes.iter().copied().collect();
        debug_assert_eq!(
            removed_nodes.len(),
            node_result_len,
            "duplicate node index in batch removal"
        );

        // Output slot for each explicitly-requested edge. Cascade-only edges
        // do not appear in the map and skip the slot write.
        let mut edge_output_slot: HashMap<EC::Index, usize> =
            HashMap::with_capacity(edge_result_len);
        for (i, &eix) in explicit_edges.iter().enumerate() {
            edge_output_slot.entry(eix).or_insert(i);
        }
        debug_assert_eq!(
            edge_output_slot.len(),
            edge_result_len,
            "duplicate edge index in batch removal"
        );

        // Phase 1: the dying-edge set — explicit edges plus every edge
        // incident on a dying node. `dying_edges` deduplicates cascade edges
        // the caller also listed explicitly.
        let mut dying_edges: HashSet<EC::Index> = HashSet::with_capacity(edge_result_len);
        let mut all_edges: Vec<EC::Index> = Vec::with_capacity(edge_result_len);
        for &eix in &explicit_edges {
            if dying_edges.insert(eix) {
                all_edges.push(eix);
            }
        }
        for &nix in &explicit_nodes {
            let node_storage = unsafe { self.nodes.get_storage_unchecked(&nix) };
            let cascade = EdgeIndicesOf {
                edges: &self.edges,
                outgoing_next: node_storage.next[OUTGOING],
                incoming_next: node_storage.next[INCOMING],
                node_key: nix,
            };
            for eix in cascade {
                if dying_edges.insert(eix) {
                    all_edges.push(eix);
                }
            }
        }

        // Phase 2: unlink all dying edges from the chains of surviving
        // nodes, sweeping each affected (node, direction) chain exactly once
        // instead of once per dying edge. The per-chain counts let each
        // sweep stop at its last dying link.
        let mut chain_counts: HashMap<(NC::Index, usize), usize> = HashMap::new();
        for &eix in &all_edges {
            let storage = unsafe { self.edges.get_storage_unchecked(&eix) };
            for dir in [OUTGOING, INCOMING] {
                let endpoint = storage.node[dir];
                if !removed_nodes.contains(&endpoint) {
                    *chain_counts.entry((endpoint, dir)).or_insert(0) += 1;
                }
            }
        }
        for (&(nix, dir), &count) in &chain_counts {
            unsafe {
                unlink_dying_edges(
                    &mut self.nodes,
                    &mut self.edges,
                    nix,
                    &dying_edges,
                    count,
                    dir,
                )
            };
        }

        let mut edges_out_buf: Vec<MaybeUninit<EC::Value>> = (0..edge_result_len)
            .map(|_| MaybeUninit::uninit())
            .collect();
        let mut nodes_out_buf: Vec<MaybeUninit<NC::Value>> = (0..node_result_len)
            .map(|_| MaybeUninit::uninit())
            .collect();

        // Phase 3: physically remove every dying edge, highest index first
        // so a `swap_remove` only ever relocates a surviving edge (every
        // queued index keeps its position until processed). Relocations are
        // recorded instead of repaired one at a time; chained moves are
        // composed so each surviving edge gets a single entry keyed by the
        // position its predecessors still point at.
        all_edges.sort_unstable_by(|a, b| b.cmp(a));
        let mut edge_moves: HashMap<EC::Index, EC::Index> = HashMap::new();
        let mut move_origin: HashMap<EC::Index, EC::Index> = HashMap::new();
        for eix in all_edges {
            let (data, _storage, swapped) = unsafe { self.edges.take_unchecked(&eix) };
            if let Some(old_last) = swapped {
                let origin = move_origin.remove(&old_last).unwrap_or(old_last);
                edge_moves.insert(origin, eix);
                move_origin.insert(eix, origin);
            }
            if let Some(&slot) = edge_output_slot.get(&eix) {
                debug_assert!(slot < edge_result_len);
                unsafe { edges_out_buf.get_unchecked_mut(slot).write(data) };
            }
        }
        drop(move_origin);

        // Phase 4: repair the stale pointers left by phase 3 with one sweep
        // per affected (node, direction) chain. Surviving edges moved into
        // slots vacated by dying edges, which phase 2 unlinked from every
        // surviving chain, so original and current positions never collide
        // and each lookup resolves in one step.
        if !edge_moves.is_empty() {
            let moved_slots: HashMap<ESlot, ESlot> = edge_moves
                .iter()
                .map(|(&origin, &current)| (EC::to_slot(origin), EC::to_slot(current)))
                .collect();
            let mut stale_counts: HashMap<(NC::Index, usize), usize> = HashMap::new();
            for &current in edge_moves.values() {
                let storage = unsafe { self.edges.get_storage_unchecked(&current) };
                for dir in [OUTGOING, INCOMING] {
                    *stale_counts.entry((storage.node[dir], dir)).or_insert(0) += 1;
                }
            }
            for (&(nix, dir), &count) in &stale_counts {
                unsafe {
                    patch_moved_edges(
                        &mut self.nodes,
                        &mut self.edges,
                        nix,
                        &moved_slots,
                        count,
                        dir,
                    )
                };
            }
        }

        // Phase 5: remove the nodes, highest index first for the same
        // swap_remove reason. All incident edges are already gone; endpoint
        // rewrites for relocated survivors are batched into phase 6.
        let mut nodes_with_slot: Vec<(usize, NC::Index)> =
            explicit_nodes.into_iter().enumerate().collect();
        nodes_with_slot.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
        let mut node_moves: HashMap<NC::Index, NC::Index> = HashMap::new();
        let mut node_move_origin: HashMap<NC::Index, NC::Index> = HashMap::new();
        for (slot, nix) in nodes_with_slot {
            let (data, _storage, swapped) = unsafe { self.nodes.take_unchecked(&nix) };
            if let Some(old_last) = swapped {
                let origin = node_move_origin.remove(&old_last).unwrap_or(old_last);
                node_moves.insert(origin, nix);
                node_move_origin.insert(nix, origin);
            }
            debug_assert!(slot < node_result_len);
            unsafe { nodes_out_buf.get_unchecked_mut(slot).write(data) };
        }

        // Phase 6: every edge incident on a relocated node still stores the
        // node's original index; rewrite it by walking the relocated node's
        // two chains once each.
        for (&origin, &current) in &node_moves {
            for dir in [OUTGOING, INCOMING] {
                let head = unsafe { self.nodes.get_storage_unchecked(&current) }.next[dir];
                let mut cur = EC::from_slot(head);
                while let Some(eix) = cur {
                    let s = unsafe { self.edges.get_storage_unchecked_mut(&eix) };
                    debug_assert!(s.node[dir] == origin);
                    s.node[dir] = current;
                    cur = EC::from_slot(s.next[dir]);
                }
            }
        }

        let mut nodes_out = IN::default();
        let mut edges_out = IE::default();
        nodes_out
            .extend(IntoIterator::into_iter(nodes_out_buf).map(|n| unsafe { n.assume_init() }));
        edges_out
            .extend(IntoIterator::into_iter(edges_out_buf).map(|e| unsafe { e.assume_init() }));

        (nodes_out, edges_out)
    }
}

unsafe impl<NC, EC, ESlot> StableNode for LinkedAdjEdgeGraph<NC, EC>
where
    NC: StableCollection + RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
}

unsafe impl<NC, EC, ESlot> StableEdge for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: StableCollection + RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
{
}

impl<NC, EC, ESlot> UniqueNode for LinkedAdjEdgeGraph<NC, EC>
where
    NC: StableCollection + CollectionBiject + RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
    NC::Value: PartialEq,
{
    fn node_index(&self, node: impl Borrow<Self::Node>) -> Option<Self::NodeIx> {
        // SAFETY: CollectionBiject guarantees no two indices map to equal
        // values, so the lookup respects the uniqueness invariant.
        unsafe { self.nodes.value_to_key_unchecked(node.borrow()) }.copied()
    }
}

impl<NC, EC, ESlot> UniqueEdge for LinkedAdjEdgeGraph<NC, EC>
where
    NC: RandomAccess<Storage = NodeRepr<ESlot>>,
    EC: StableCollection
        + CollectionBiject
        + RandomAccess<Slot = ESlot, Storage = EdgeRepr<NC::Index, ESlot>>,
    ESlot: Copy + Eq + Hash,
    NC::Index: Display + Debug,
    EC::Index: Display + Debug,
    EC::Value: PartialEq,
{
    fn edge_index(&self, edge: impl Borrow<Self::Edge>) -> Option<Self::EdgeIx> {
        unsafe { self.edges.value_to_key_unchecked(edge.borrow()) }.copied()
    }
}

// GraphMap is only implemented for the Vec-backed shape (positions stay
// stable under value mapping); map-backed variants would need to rewire
// adjacency slot ids and are deferred.
impl<'r, N, E, NewN, NewE> GraphMap<'r, NewN, NewE>
    for LinkedAdjEdgeGraph<Vec<(N, NodeRepr<u32>)>, Vec<(E, EdgeRepr<u32, u32>)>>
{
    type Mapped = LinkedAdjEdgeGraph<Vec<(NewN, NodeRepr<u32>)>, Vec<(NewE, EdgeRepr<u32, u32>)>>;

    fn map<FN, FE>(self, mut fn_node: FN, mut fn_edge: FE) -> Self::Mapped
    where
        FN: FnMut(Self::Node) -> NewN,
        FE: FnMut(Self::Edge) -> NewE,
    {
        let nodes: Vec<(NewN, NodeRepr<u32>)> = self
            .nodes
            .into_iter()
            .map(|(v, storage)| (fn_node(v), storage))
            .collect();
        let edges: Vec<(NewE, EdgeRepr<u32, u32>)> = self
            .edges
            .into_iter()
            .map(|(v, storage)| (fn_edge(v), storage))
            .collect();
        LinkedAdjEdgeGraph { nodes, edges }
    }
}

/// `GraphMap` for the map-backed shapes (`BTreeMap` / `HashMap`).
///
/// Unlike the Vec-backed shape (positions stay stable under value mapping), here
/// the node/edge *value is its key*, so remapping the values rewires every stored
/// key reference: the adjacency head slots in each `NodeRepr`, and the endpoint
/// nodes plus next-edge slots in each `EdgeRepr`. The value maps must be
/// injective — two old keys mapping to one new key would silently collapse map
/// entries — so injectivity is checked and panics otherwise (matching the
/// Vec-backed positional guarantee).
macro_rules! impl_map_backed_graph_map {
    ($map:ident) => {
        impl<'r, N, E, NewN, NewE> GraphMap<'r, NewN, NewE>
            for LinkedAdjEdgeGraph<
                std::collections::$map<N, NodeRepr<Option<E>>>,
                std::collections::$map<E, EdgeRepr<N, Option<E>>>,
            >
        where
            N: Copy + Eq + Ord + Hash + Display + Debug,
            E: Copy + Eq + Ord + Hash + Display + Debug,
            NewN: Copy + Eq + Ord + Hash + Display + Debug,
            NewE: Copy + Eq + Ord + Hash + Display + Debug,
        {
            type Mapped = LinkedAdjEdgeGraph<
                std::collections::$map<NewN, NodeRepr<Option<NewE>>>,
                std::collections::$map<NewE, EdgeRepr<NewN, Option<NewE>>>,
            >;

            fn map<FN, FE>(self, mut fn_node: FN, mut fn_edge: FE) -> Self::Mapped
            where
                FN: FnMut(Self::Node) -> NewN,
                FE: FnMut(Self::Edge) -> NewE,
            {
                use std::collections::{HashMap, HashSet};

                // Build the key remaps once: `fn_node`/`fn_edge` are `FnMut`, so a
                // key reference must be looked up rather than recomputed at each of
                // its (possibly many) occurrences in the adjacency structure.
                let mut edge_remap: HashMap<E, NewE> = HashMap::new();
                let mut seen_e: HashSet<NewE> = HashSet::new();
                for &k in self.edges.keys() {
                    let nk = fn_edge(k);
                    assert!(seen_e.insert(nk), "fn_edge is not injective");
                    edge_remap.insert(k, nk);
                }
                let mut node_remap: HashMap<N, NewN> = HashMap::new();
                let mut seen_n: HashSet<NewN> = HashSet::new();
                for &k in self.nodes.keys() {
                    let nk = fn_node(k);
                    assert!(seen_n.insert(nk), "fn_node is not injective");
                    node_remap.insert(k, nk);
                }

                let nodes = self
                    .nodes
                    .into_iter()
                    .map(|(k, repr)| {
                        let next = repr.next.map(|s| s.map(|e| edge_remap[&e]));
                        (node_remap[&k], NodeRepr { next })
                    })
                    .collect();
                let edges = self
                    .edges
                    .into_iter()
                    .map(|(k, repr)| {
                        let next = repr.next.map(|s| s.map(|e| edge_remap[&e]));
                        let node = repr.node.map(|n| node_remap[&n]);
                        (edge_remap[&k], EdgeRepr { next, node })
                    })
                    .collect();
                LinkedAdjEdgeGraph { nodes, edges }
            }
        }
    };
}

impl_map_backed_graph_map!(BTreeMap);
impl_map_backed_graph_map!(HashMap);

#[cfg(test)]
mod fuzz_batch_removal {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    type G = LinkedAdjEdgeGraph<Vec<(u64, NodeRepr<u32>)>, Vec<(u64, EdgeRepr<u32, u32>)>>;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
    }

    // Walk a chain with safe indexing; panic on OOB pointer, cycle, or wrong
    // endpoint tag. Returns edge payloads in chain order.
    fn walk_chain(g: &G, nslot: usize, dir: usize, seed: u64) -> Vec<u64> {
        let mut out = Vec::new();
        let mut slot = g.nodes[nslot].1.next[dir];
        let mut steps = 0usize;
        while slot != u32::MAX {
            assert!(
                (slot as usize) < g.edges.len(),
                "seed {seed}: node slot {nslot} dir {dir}: dangling edge slot {slot} (len {})",
                g.edges.len()
            );
            let (payload, ref er) = g.edges[slot as usize];
            assert_eq!(
                er.node[dir] as usize, nslot,
                "seed {seed}: edge {payload} in chain ({nslot},{dir}) has wrong endpoint {}",
                er.node[dir]
            );
            out.push(payload);
            slot = er.next[dir];
            steps += 1;
            assert!(
                steps <= g.edges.len() + 1,
                "seed {seed}: cycle in chain ({nslot},{dir})"
            );
        }
        out
    }

    fn check_integrity(
        g: &G,
        model_nodes: &StdHashMap<u64, ()>,
        model_edges: &StdHashMap<u64, (u64, u64)>,
        seed: u64,
    ) {
        assert_eq!(g.nodes.len(), model_nodes.len(), "seed {seed}: node count");
        assert_eq!(g.edges.len(), model_edges.len(), "seed {seed}: edge count");
        for (np, _) in g.nodes.iter() {
            assert!(model_nodes.contains_key(np), "seed {seed}: ghost node {np}");
        }
        // Every edge endpoint index in range + matches model payloads.
        for (ep, er) in g.edges.iter() {
            let (mt, mh) = model_edges
                .get(ep)
                .unwrap_or_else(|| panic!("seed {seed}: ghost edge {ep}"));
            assert!(
                (er.node[OUTGOING] as usize) < g.nodes.len()
                    && (er.node[INCOMING] as usize) < g.nodes.len(),
                "seed {seed}: edge {ep} endpoint OOB {:?}",
                er.node
            );
            assert_eq!(
                g.nodes[er.node[OUTGOING] as usize].0, *mt,
                "seed {seed}: edge {ep} tail payload"
            );
            assert_eq!(
                g.nodes[er.node[INCOMING] as usize].0, *mh,
                "seed {seed}: edge {ep} head payload"
            );
        }
        // Chain coverage: every edge must appear exactly once in its tail's
        // outgoing chain and exactly once in its head's incoming chain.
        for dir in [OUTGOING, INCOMING] {
            let mut seen: StdHashMap<u64, usize> = StdHashMap::new();
            for nslot in 0..g.nodes.len() {
                for p in walk_chain(g, nslot, dir, seed) {
                    *seen.entry(p).or_insert(0) += 1;
                }
            }
            for (ep, _) in g.edges.iter() {
                assert_eq!(
                    seen.get(ep).copied().unwrap_or(0),
                    1,
                    "seed {seed}: edge {ep} appears wrong number of times in dir {dir} chains"
                );
            }
            assert_eq!(
                seen.values().sum::<usize>(),
                g.edges.len(),
                "seed {seed}: dir {dir} chain totals"
            );
        }
    }

    fn node_slot_of(g: &G, payload: u64) -> u32 {
        g.nodes.iter().position(|(p, _)| *p == payload).unwrap() as u32
    }
    fn edge_slot_of(g: &G, payload: u64) -> u32 {
        g.edges.iter().position(|(p, _)| *p == payload).unwrap() as u32
    }

    fn run(seed: u64) {
        let mut rng = Rng(seed);
        let mut g = G::default();
        let mut model_nodes: StdHashMap<u64, ()> = StdHashMap::new();
        let mut model_edges: StdHashMap<u64, (u64, u64)> = StdHashMap::new();
        let mut next_node: u64 = 1;
        let mut next_edge: u64 = 1_000_000;

        let n0 = 2 + rng.below(10);
        for _ in 0..n0 {
            unsafe { g.insert_node_unchecked(next_node).unwrap() };
            model_nodes.insert(next_node, ());
            next_node += 1;
        }
        let e0 = rng.below(30);
        for _ in 0..e0 {
            let keys: Vec<u64> = model_nodes.keys().copied().collect();
            let t = keys[rng.below(keys.len())];
            // 1 in 4: force self-loop; otherwise random (parallel edges arise
            // naturally from repetition).
            let h = if rng.below(4) == 0 {
                t
            } else {
                keys[rng.below(keys.len())]
            };
            let ts = node_slot_of(&g, t);
            let hs = node_slot_of(&g, h);
            unsafe { g.insert_edge_unchecked(next_edge, [ts, hs]).unwrap() };
            model_edges.insert(next_edge, (t, h));
            next_edge += 1;
        }
        check_integrity(&g, &model_nodes, &model_edges, seed);

        for _round in 0..6 {
            // victim selection: percentage varies wildly per round
            let npct = rng.below(101);
            let epct = rng.below(101);
            let node_victims: Vec<u64> = model_nodes
                .keys()
                .copied()
                .filter(|_| rng.below(100) < npct)
                .collect();
            let edge_victims: Vec<u64> = model_edges
                .keys()
                .copied()
                .filter(|_| rng.below(100) < epct)
                .collect();
            let node_ixs: Vec<u32> = node_victims.iter().map(|&p| node_slot_of(&g, p)).collect();
            let edge_ixs: Vec<u32> = edge_victims.iter().map(|&p| edge_slot_of(&g, p)).collect();

            let (got_n, got_e): (Vec<u64>, Vec<u64>) =
                unsafe { g.take_nodes_edges_unchecked(node_ixs, edge_ixs) };
            assert_eq!(got_n, node_victims, "seed {seed}: node payload order");
            assert_eq!(got_e, edge_victims, "seed {seed}: edge payload order");

            for p in &node_victims {
                model_nodes.remove(p);
            }
            model_edges
                .retain(|_, (t, h)| model_nodes.contains_key(t) && model_nodes.contains_key(h));
            for p in &edge_victims {
                model_edges.remove(p);
            }
            check_integrity(&g, &model_nodes, &model_edges, seed);

            // refill
            let add_n = rng.below(4);
            for _ in 0..add_n {
                unsafe { g.insert_node_unchecked(next_node).unwrap() };
                model_nodes.insert(next_node, ());
                next_node += 1;
            }
            if !model_nodes.is_empty() {
                let add_e = rng.below(8);
                for _ in 0..add_e {
                    let keys: Vec<u64> = model_nodes.keys().copied().collect();
                    let t = keys[rng.below(keys.len())];
                    let h = if rng.below(4) == 0 {
                        t
                    } else {
                        keys[rng.below(keys.len())]
                    };
                    let ts = node_slot_of(&g, t);
                    let hs = node_slot_of(&g, h);
                    unsafe { g.insert_edge_unchecked(next_edge, [ts, hs]).unwrap() };
                    model_edges.insert(next_edge, (t, h));
                    next_edge += 1;
                }
            }
            check_integrity(&g, &model_nodes, &model_edges, seed);
        }
    }

    #[test]
    fn fuzz_many_seeds() {
        for seed in 1..=5_000u64 {
            run(seed);
        }
    }
}
