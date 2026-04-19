//! Lifetime-erased item types for the `walks_*` iterators.
//!
//! A walk yields `(EdgeIx, &'r Edge, NodeIx)` (or node-first / mutable
//! variants). Naming that tuple directly in an associated-type bound requires
//! `Edge: 'r`, expressible only as a `where` clause on the associated-type
//! *definition* (stabilized in Rust 1.65) or, at the trait level, as
//! `Edge: 'static` (which the crate avoids, since `Graph: for<'r>
//! GraphOperation<'r>` would turn a trait-level `Edge: 'r` into `Edge:
//! 'static`).
//!
//! These wrappers sidestep both. The edge reference is stored as a raw pointer
//! plus a `PhantomData<&'r _>`, so the type is well-formed for *any* `Edge`
//! with no outlives bound — an associated type bounded by
//! `Iterator<Item = WalkItem<'r, …>>` needs no `where` clause. The borrowed
//! tuple is recovered two ways:
//!
//! - [`get`](WalkItem::get) (available once `Edge: 'r`) returns the borrowed
//!   `(EdgeIx, &'r Edge, NodeIx)` tuple at a borrow-scoped call site. (A
//!   `Deref`-to-tuple would have to *return a reference into `self`*, which
//!   forces transmuting between `(…, *const E, …)` and `(…, &'r E, …)` — not
//!   sound, as `repr(Rust)` tuple layout is unspecified. `get` builds the
//!   tuple from the parts instead, with no layout assumption.)
//! - [`into_parts`](WalkItem::into_parts) / [`from_parts`](WalkItem::from_parts)
//!   expose the raw pointer so wrappers can rebrand the indices or reproject the
//!   edge type *without* recovering the reference (hence without `Edge: 'r`).

use core::marker::PhantomData;

/// Item of [`walks_from`](crate::graph::Graph::walks_from) /
/// [`walks_of`](crate::graph::Graph::walks_of): `(EdgeIx, &'r Edge, NodeIx)`,
/// lifetime-erased. See the [module docs](self).
pub struct WalkItem<'r, EIx, E: ?Sized, NIx>((EIx, *const E, NIx), PhantomData<&'r ()>);

impl<'r, EIx, E: ?Sized, NIx> WalkItem<'r, EIx, E, NIx> {
    /// Build from a borrowed edge.
    #[inline]
    pub fn new(edge_ix: EIx, edge: &'r E, node_ix: NIx) -> Self {
        WalkItem((edge_ix, edge as *const E, node_ix), PhantomData)
    }

    /// Decompose into `(edge_ix, edge_ptr, node_ix)`. The pointer is valid for
    /// `'r`; this lets a wrapper rebrand the indices or reproject the edge
    /// without requiring `E: 'r`.
    #[inline]
    pub fn into_parts(self) -> (EIx, *const E, NIx) {
        self.0
    }

    /// Reassemble from parts.
    ///
    /// # Safety
    /// `edge_ptr` must point to a valid `E` for all of `'r` (e.g. it was
    /// derived from the pointer returned by [`into_parts`](Self::into_parts) of
    /// a `WalkItem` with the same `'r`).
    #[inline]
    pub unsafe fn from_parts(edge_ix: EIx, edge_ptr: *const E, node_ix: NIx) -> Self {
        WalkItem((edge_ix, edge_ptr, node_ix), PhantomData)
    }

    /// Recover the borrowed tuple `(edge_ix, &'r edge, node_ix)`.
    #[inline]
    pub fn get(self) -> (EIx, &'r E, NIx)
    where
        E: 'r,
    {
        let (edge_ix, edge_ptr, node_ix) = self.0;
        // SAFETY: `edge_ptr` came from a `&'r E` (via `new`/`from_parts`) and
        // `E: 'r`, so reborrowing it as `&'r E` is valid.
        (edge_ix, unsafe { &*edge_ptr }, node_ix)
    }
}

/// Mutable counterpart of [`WalkItem`]: `(EdgeIx, &'r mut Edge, NodeIx)`.
pub struct WalkItemMut<'r, EIx, E: ?Sized, NIx>((EIx, *mut E, NIx), PhantomData<&'r mut ()>);

impl<'r, EIx, E: ?Sized, NIx> WalkItemMut<'r, EIx, E, NIx> {
    /// Build from a mutably-borrowed edge.
    #[inline]
    pub fn new(edge_ix: EIx, edge: &'r mut E, node_ix: NIx) -> Self {
        WalkItemMut((edge_ix, edge as *mut E, node_ix), PhantomData)
    }

    /// Decompose into `(edge_ix, edge_ptr, node_ix)`; see
    /// [`WalkItem::into_parts`].
    #[inline]
    pub fn into_parts(self) -> (EIx, *mut E, NIx) {
        self.0
    }

    /// Reassemble from parts.
    ///
    /// # Safety
    /// `edge_ptr` must be uniquely valid (`&mut`) for all of `'r`.
    #[inline]
    pub unsafe fn from_parts(edge_ix: EIx, edge_ptr: *mut E, node_ix: NIx) -> Self {
        WalkItemMut((edge_ix, edge_ptr, node_ix), PhantomData)
    }

    /// Recover the borrowed tuple `(edge_ix, &'r mut edge, node_ix)`.
    #[inline]
    pub fn get_mut(self) -> (EIx, &'r mut E, NIx)
    where
        E: 'r,
    {
        let (edge_ix, edge_ptr, node_ix) = self.0;
        // SAFETY: `edge_ptr` came from a `&'r mut E` and is unique for `'r`.
        (edge_ix, unsafe { &mut *edge_ptr }, node_ix)
    }
}

/// Item of [`walks_to`](crate::graph::Graph::walks_to):
/// `(NodeIx, EdgeIx, &'r Edge)` (source node first), lifetime-erased.
pub struct WalkItemTo<'r, NIx, EIx, E: ?Sized>((NIx, EIx, *const E), PhantomData<&'r ()>);

impl<'r, NIx, EIx, E: ?Sized> WalkItemTo<'r, NIx, EIx, E> {
    /// Build from a borrowed edge.
    #[inline]
    pub fn new(node_ix: NIx, edge_ix: EIx, edge: &'r E) -> Self {
        WalkItemTo((node_ix, edge_ix, edge as *const E), PhantomData)
    }

    /// Decompose into `(node_ix, edge_ix, edge_ptr)`; see
    /// [`WalkItem::into_parts`].
    #[inline]
    pub fn into_parts(self) -> (NIx, EIx, *const E) {
        self.0
    }

    /// Reassemble from parts.
    ///
    /// # Safety
    /// `edge_ptr` must point to a valid `E` for all of `'r`.
    #[inline]
    pub unsafe fn from_parts(node_ix: NIx, edge_ix: EIx, edge_ptr: *const E) -> Self {
        WalkItemTo((node_ix, edge_ix, edge_ptr), PhantomData)
    }

    /// Recover the borrowed tuple `(node_ix, edge_ix, &'r edge)`.
    #[inline]
    pub fn get(self) -> (NIx, EIx, &'r E)
    where
        E: 'r,
    {
        let (node_ix, edge_ix, edge_ptr) = self.0;
        // SAFETY: see [`WalkItem::get`].
        (node_ix, edge_ix, unsafe { &*edge_ptr })
    }
}

/// Mutable counterpart of [`WalkItemTo`]: `(NodeIx, EdgeIx, &'r mut Edge)`.
pub struct WalkItemToMut<'r, NIx, EIx, E: ?Sized>((NIx, EIx, *mut E), PhantomData<&'r mut ()>);

impl<'r, NIx, EIx, E: ?Sized> WalkItemToMut<'r, NIx, EIx, E> {
    /// Build from a mutably-borrowed edge.
    #[inline]
    pub fn new(node_ix: NIx, edge_ix: EIx, edge: &'r mut E) -> Self {
        WalkItemToMut((node_ix, edge_ix, edge as *mut E), PhantomData)
    }

    /// Decompose into `(node_ix, edge_ix, edge_ptr)`.
    #[inline]
    pub fn into_parts(self) -> (NIx, EIx, *mut E) {
        self.0
    }

    /// Reassemble from parts.
    ///
    /// # Safety
    /// `edge_ptr` must be uniquely valid (`&mut`) for all of `'r`.
    #[inline]
    pub unsafe fn from_parts(node_ix: NIx, edge_ix: EIx, edge_ptr: *mut E) -> Self {
        WalkItemToMut((node_ix, edge_ix, edge_ptr), PhantomData)
    }

    /// Recover the borrowed tuple `(node_ix, edge_ix, &'r mut edge)`.
    #[inline]
    pub fn get_mut(self) -> (NIx, EIx, &'r mut E)
    where
        E: 'r,
    {
        let (node_ix, edge_ix, edge_ptr) = self.0;
        // SAFETY: `edge_ptr` came from a `&'r mut E` and is unique for `'r`.
        (node_ix, edge_ix, unsafe { &mut *edge_ptr })
    }
}
