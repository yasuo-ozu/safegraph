//! General-purpose collection traits.
//!
//! - [`Collection`] — consuming iteration, implemented for `Vec<(V, S)>`,
//!   `LinkedList<(V, S)>`, and `*Map<I, S>` types.
//! - [`RandomAccess`] — unchecked indexed read access by `Index`.
//! - [`RandomAccessRef`] — borrowing iteration of indices (no GATs; use the
//!   bound `for<'a> C: RandomAccessRef<'a>`).
//! - [`InsertableCollection`] — add a new (value, storage) entry; only on
//!   `Collection` (no `RandomAccess` requirement, so `LinkedList` implements).
//! - [`UpdatableRandomAccess`] — mutate the `Value` slot in place. Only
//!   implemented for collections where the value type is independent of the
//!   index (i.e. `Vec`); not implemented for map types whose keys ARE the
//!   values.
//! - [`RemovableRandomAccess`] — unsafe `take` (delete by index).
//! - [`StableCollection`] — marker: indices remain valid across mutations
//!   (impl'd only for the map types, not `Vec`).
//! - [`CollectionRef`] — borrowing iteration of values (no GATs; use the bound
//!   `for<'a> C: CollectionRef<'a>`).
//! - [`CollectionMut`] — mutable iteration of values; not implemented for
//!   map types (their keys are immutable).
//! - [`CollectionBiject`] — value→index reverse lookup; only `*Map` types.

use std::collections::{BTreeMap, HashMap, LinkedList};
use std::hash::Hash;

use crate::unwrap_unchecked;

/// Base collection: associated value/storage types and consuming iteration.
///
/// Associated types:
/// - `Value`   — logical value type (`V` for sequences, `I` for maps — Key=Value).
/// - `Storage` — per-entry auxiliary data (`S` for all impls).
pub trait Collection {
    type Value;
    type Storage;
    type IntoValues: Iterator<Item = Self::Value>;

    fn len(&self) -> usize;

    /// Returns `true` if the collection holds no entries.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of entries that can be held before the next reallocation, if the
    /// collection has such a notion. `Some(cap) with cap > len()` means an
    /// append will not reallocate. Returns `None` for collections without a
    /// meaningful pre-reallocation capacity (the default).
    #[inline]
    fn capacity(&self) -> Option<usize> {
        None
    }

    fn into_values(self) -> Self::IntoValues;
}

impl<V, S> Collection for Vec<(V, S)> {
    type Value = V;
    type Storage = S;
    type IntoValues = std::iter::Map<std::vec::IntoIter<(V, S)>, fn((V, S)) -> V>;

    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn capacity(&self) -> Option<usize> {
        Some(Vec::capacity(self))
    }

    #[inline]
    fn into_values(self) -> Self::IntoValues {
        IntoIterator::into_iter(self).map((|(v, _)| v) as fn((V, S)) -> V)
    }
}

impl<V, S> Collection for LinkedList<(V, S)> {
    type Value = V;
    type Storage = S;
    type IntoValues =
        std::iter::Map<std::collections::linked_list::IntoIter<(V, S)>, fn((V, S)) -> V>;

    #[inline]
    fn len(&self) -> usize {
        LinkedList::len(self)
    }

    #[inline]
    fn into_values(self) -> Self::IntoValues {
        IntoIterator::into_iter(self).map((|(v, _)| v) as fn((V, S)) -> V)
    }
}

impl<I: Copy + Eq + Ord + Hash, S> Collection for BTreeMap<I, S> {
    type Value = I;
    type Storage = S;
    type IntoValues = std::collections::btree_map::IntoKeys<I, S>;

    #[inline]
    fn len(&self) -> usize {
        BTreeMap::len(self)
    }

    #[inline]
    fn into_values(self) -> Self::IntoValues {
        BTreeMap::into_keys(self)
    }
}

impl<I: Copy + Eq + Ord + Hash, S> Collection for HashMap<I, S> {
    type Value = I;
    type Storage = S;
    type IntoValues = std::collections::hash_map::IntoKeys<I, S>;

    #[inline]
    fn len(&self) -> usize {
        HashMap::len(self)
    }

    #[inline]
    fn into_values(self) -> Self::IntoValues {
        HashMap::into_keys(self)
    }
}

impl<'b, V: Clone, S: Clone> Collection for &'b [(V, S)] {
    type Value = V;
    type Storage = S;
    type IntoValues = std::iter::Map<std::slice::Iter<'b, (V, S)>, fn(&'b (V, S)) -> V>;

    #[inline]
    fn len(&self) -> usize {
        <[(V, S)]>::len(self)
    }

    #[inline]
    fn into_values(self) -> Self::IntoValues {
        self.iter()
            .map((|r: &(V, S)| r.0.clone()) as fn(&'b (V, S)) -> V)
    }
}

impl<'b, V: Clone, S: Clone> Collection for &'b mut [(V, S)] {
    type Value = V;
    type Storage = S;
    type IntoValues = std::iter::Map<std::slice::Iter<'b, (V, S)>, fn(&'b (V, S)) -> V>;

    #[inline]
    fn len(&self) -> usize {
        <[(V, S)]>::len(self)
    }

    #[inline]
    fn into_values(self) -> Self::IntoValues {
        let s: &'b [(V, S)] = self;
        s.iter()
            .map((|r: &(V, S)| r.0.clone()) as fn(&'b (V, S)) -> V)
    }
}

/// Consume a collection yielding each `(value, storage)` entry.
///
/// Unlike [`Collection::into_values`] (which drops the `Storage`), this keeps the
/// per-entry storage so callers can recover data nested inside it — e.g.
/// [`FlatAdjEdgeGraph`](crate::raw_graph::flat_adj_edge::FlatAdjEdgeGraph), whose
/// edge payloads live inside each node's storage. Implemented only for the owned
/// backends: a borrowed slice cannot move its entries out.
pub trait DrainEntries: Collection {
    /// Iterator returned by [`into_entries`](Self::into_entries).
    type IntoEntries: Iterator<Item = (Self::Value, Self::Storage)>;

    /// Consume `self`, yielding `(value, storage)` for every entry.
    fn into_entries(self) -> Self::IntoEntries;
}

impl<V, S> DrainEntries for Vec<(V, S)> {
    type IntoEntries = std::vec::IntoIter<(V, S)>;
    #[inline]
    fn into_entries(self) -> Self::IntoEntries {
        IntoIterator::into_iter(self)
    }
}

impl<V, S> DrainEntries for LinkedList<(V, S)> {
    type IntoEntries = std::collections::linked_list::IntoIter<(V, S)>;
    #[inline]
    fn into_entries(self) -> Self::IntoEntries {
        IntoIterator::into_iter(self)
    }
}

impl<I: Copy + Eq + Ord + Hash, S> DrainEntries for BTreeMap<I, S> {
    type IntoEntries = std::collections::btree_map::IntoIter<I, S>;
    #[inline]
    fn into_entries(self) -> Self::IntoEntries {
        IntoIterator::into_iter(self)
    }
}

impl<I: Copy + Eq + Ord + Hash, S> DrainEntries for HashMap<I, S> {
    type IntoEntries = std::collections::hash_map::IntoIter<I, S>;
    #[inline]
    fn into_entries(self) -> Self::IntoEntries {
        IntoIterator::into_iter(self)
    }
}

/// Random access to values and storage by index.
///
/// Associated types:
/// - `Index`    — index/key type used to address entries.
/// - `Slot` — type of the end-of-list sentinel (`u32` for sequences,
///   `Option<Index>` for maps — always `None`).
pub trait RandomAccess: Collection {
    type Index: Copy + Eq + Ord + Hash;
    type Slot: Copy + Eq + Hash;

    /// Returns the sentinel/terminal index value used as an end-of-list marker.
    /// Sequence types return `u32::MAX`; map types return `None`.
    fn sentinel() -> Self::Slot;

    /// Wraps a valid index into the `Slot` representation.
    fn to_slot(ix: Self::Index) -> Self::Slot;

    /// Extracts an index from a `Slot` value, returning `None` for the terminal sentinel.
    fn from_slot(term: Self::Slot) -> Option<Self::Index>;

    /// Mutable counterpart of [`from_slot`](Self::from_slot): given a `&mut Slot`,
    /// returns `Some(&mut Index)` when the slot is non-sentinel, `None` otherwise.
    ///
    /// For sequence types (`Slot = Index = u32`) the reference aliases the slot
    /// directly, so writing through it updates the raw pointer in place.
    /// For map types (`Slot = Option<Index>`) this is `Option::as_mut`.
    fn from_slot_mut(slot: &mut Self::Slot) -> Option<&mut Self::Index>;

    /// Whether `Index` values form a dense `0..len()` range (sequence-backed),
    /// so removal relocates the last element (`swap_remove`) and indices double
    /// as array offsets. Map-backed collections keep arbitrary stable keys and
    /// return `false`. Used to pick removal fast paths that assume dense
    /// indices; the default is the conservative `false`.
    fn dense_indices() -> bool {
        false
    }

    fn contains_index(&self, ix: &Self::Index) -> bool;

    /// # Safety
    /// `ix` must be a valid index currently held by this collection.
    unsafe fn get_value_unchecked(&self, ix: &Self::Index) -> &Self::Value;

    /// # Safety
    /// `ix` must be a valid index currently held by this collection.
    unsafe fn get_storage_unchecked(&self, ix: &Self::Index) -> &Self::Storage;

    /// Fetch value and storage in one call. Useful when a hot iterator needs
    /// both — saves a second lookup. Default impl combines the two
    /// `get_*_unchecked` calls.
    ///
    /// # Safety
    /// `ix` must be a valid index currently held by this collection.
    #[inline]
    unsafe fn get_both_unchecked(&self, ix: &Self::Index) -> (&Self::Value, &Self::Storage) {
        let v = unsafe { self.get_value_unchecked(ix) };
        let s = unsafe { self.get_storage_unchecked(ix) };
        (v, s)
    }

    /// # Safety
    /// `ix` must be a valid index currently held by this collection.
    unsafe fn get_storage_unchecked_mut(&mut self, ix: &Self::Index) -> &mut Self::Storage;
}

impl<V, S> RandomAccess for Vec<(V, S)> {
    type Index = u32;
    type Slot = u32;

    #[inline]
    fn dense_indices() -> bool {
        true
    }

    #[inline]
    fn sentinel() -> u32 {
        u32::MAX
    }

    #[inline]
    fn to_slot(ix: u32) -> u32 {
        ix
    }

    #[inline]
    fn from_slot(term: u32) -> Option<u32> {
        if term == u32::MAX {
            None
        } else {
            Some(term)
        }
    }

    #[inline]
    fn from_slot_mut(slot: &mut u32) -> Option<&mut u32> {
        if *slot == u32::MAX {
            None
        } else {
            Some(slot)
        }
    }

    #[inline]
    fn contains_index(&self, ix: &u32) -> bool {
        (*ix as usize) < self.len()
    }
    #[inline]
    unsafe fn get_value_unchecked(&self, ix: &u32) -> &V {
        unsafe { &self.get_unchecked(*ix as usize).0 }
    }
    #[inline]
    unsafe fn get_storage_unchecked(&self, ix: &u32) -> &S {
        unsafe { &self.get_unchecked(*ix as usize).1 }
    }
    #[inline]
    unsafe fn get_both_unchecked(&self, ix: &u32) -> (&V, &S) {
        let r = unsafe { self.get_unchecked(*ix as usize) };
        (&r.0, &r.1)
    }
    #[inline]
    unsafe fn get_storage_unchecked_mut(&mut self, ix: &u32) -> &mut S {
        unsafe { &mut self.get_unchecked_mut(*ix as usize).1 }
    }
}

impl<I: Copy + Eq + Ord + Hash, S> RandomAccess for BTreeMap<I, S> {
    type Index = I;
    type Slot = Option<I>;

    #[inline]
    fn sentinel() -> Option<I> {
        None
    }

    #[inline]
    fn to_slot(ix: I) -> Option<I> {
        Some(ix)
    }

    #[inline]
    fn from_slot(term: Option<I>) -> Option<I> {
        term
    }

    #[inline]
    fn from_slot_mut(slot: &mut Option<I>) -> Option<&mut I> {
        slot.as_mut()
    }

    #[inline]
    fn contains_index(&self, ix: &I) -> bool {
        self.contains_key(ix)
    }
    #[inline]
    unsafe fn get_value_unchecked(&self, ix: &I) -> &I {
        unwrap_unchecked(self.get_key_value(ix)).0
    }
    #[inline]
    unsafe fn get_storage_unchecked(&self, ix: &I) -> &S {
        unwrap_unchecked(self.get(ix))
    }
    #[inline]
    unsafe fn get_both_unchecked(&self, ix: &I) -> (&I, &S) {
        unwrap_unchecked(self.get_key_value(ix))
    }
    #[inline]
    unsafe fn get_storage_unchecked_mut(&mut self, ix: &I) -> &mut S {
        unwrap_unchecked(self.get_mut(ix))
    }
}

impl<V: Clone, S: Clone> RandomAccess for &mut [(V, S)] {
    type Index = u32;
    type Slot = u32;

    #[inline]
    fn dense_indices() -> bool {
        true
    }

    #[inline]
    fn sentinel() -> u32 {
        u32::MAX
    }
    #[inline]
    fn to_slot(ix: u32) -> u32 {
        ix
    }
    #[inline]
    fn from_slot(term: u32) -> Option<u32> {
        if term == u32::MAX {
            None
        } else {
            Some(term)
        }
    }
    #[inline]
    fn from_slot_mut(slot: &mut u32) -> Option<&mut u32> {
        if *slot == u32::MAX {
            None
        } else {
            Some(slot)
        }
    }
    #[inline]
    fn contains_index(&self, ix: &u32) -> bool {
        (*ix as usize) < <[(V, S)]>::len(self)
    }
    #[inline]
    unsafe fn get_value_unchecked(&self, ix: &u32) -> &V {
        unsafe { &<[(V, S)]>::get_unchecked(self, *ix as usize).0 }
    }
    #[inline]
    unsafe fn get_storage_unchecked(&self, ix: &u32) -> &S {
        unsafe { &<[(V, S)]>::get_unchecked(self, *ix as usize).1 }
    }
    #[inline]
    unsafe fn get_both_unchecked(&self, ix: &u32) -> (&V, &S) {
        let r = unsafe { <[(V, S)]>::get_unchecked(self, *ix as usize) };
        (&r.0, &r.1)
    }
    #[inline]
    unsafe fn get_storage_unchecked_mut(&mut self, ix: &u32) -> &mut S {
        unsafe { &mut <[(V, S)]>::get_unchecked_mut(self, *ix as usize).1 }
    }
}

impl<I: Copy + Eq + Ord + Hash, S> RandomAccess for HashMap<I, S> {
    type Index = I;
    type Slot = Option<I>;

    #[inline]
    fn sentinel() -> Option<I> {
        None
    }

    #[inline]
    fn to_slot(ix: I) -> Option<I> {
        Some(ix)
    }

    #[inline]
    fn from_slot(term: Option<I>) -> Option<I> {
        term
    }

    #[inline]
    fn from_slot_mut(slot: &mut Option<I>) -> Option<&mut I> {
        slot.as_mut()
    }

    #[inline]
    fn contains_index(&self, ix: &I) -> bool {
        self.contains_key(ix)
    }
    #[inline]
    unsafe fn get_value_unchecked(&self, ix: &I) -> &I {
        unwrap_unchecked(self.get_key_value(ix)).0
    }
    #[inline]
    unsafe fn get_storage_unchecked(&self, ix: &I) -> &S {
        unwrap_unchecked(self.get(ix))
    }
    #[inline]
    unsafe fn get_both_unchecked(&self, ix: &I) -> (&I, &S) {
        unwrap_unchecked(self.get_key_value(ix))
    }
    #[inline]
    unsafe fn get_storage_unchecked_mut(&mut self, ix: &I) -> &mut S {
        unwrap_unchecked(self.get_mut(ix))
    }
}

/// Borrowing-iteration extension for [`Collection`] (shared values only).
pub trait CollectionRef<'a>: Collection
where
    Self: 'a,
    Self::Value: 'a,
{
    type IterValue: Iterator<Item = &'a Self::Value>;

    fn values(&'a self) -> Self::IterValue;
}

/// Mutable-iteration extension. NOT implemented for map types, because
/// map keys (which serve as `Value` in our Key=Value pattern) are
/// immutable.
pub trait CollectionMut<'a>: CollectionRef<'a>
where
    Self: 'a,
    Self::Value: 'a,
{
    type IterValueMut: Iterator<Item = &'a mut Self::Value>;

    fn values_mut(&'a mut self) -> Self::IterValueMut;
}

impl<'a, V: 'a, S: 'a> CollectionRef<'a> for Vec<(V, S)> {
    type IterValue = std::iter::Map<std::slice::Iter<'a, (V, S)>, fn(&'a (V, S)) -> &'a V>;

    #[inline]
    fn values(&'a self) -> Self::IterValue {
        self.iter()
            .map((|r: &(V, S)| &r.0) as fn(&'a (V, S)) -> &'a V)
    }
}

impl<'a, V: 'a, S: 'a> CollectionMut<'a> for Vec<(V, S)> {
    type IterValueMut =
        std::iter::Map<std::slice::IterMut<'a, (V, S)>, fn(&'a mut (V, S)) -> &'a mut V>;

    #[inline]
    fn values_mut(&'a mut self) -> Self::IterValueMut {
        self.iter_mut()
            .map((|r: &mut (V, S)| &mut r.0) as fn(&'a mut (V, S)) -> &'a mut V)
    }
}

impl<'a, V: 'a, S: 'a> CollectionRef<'a> for LinkedList<(V, S)> {
    type IterValue =
        std::iter::Map<std::collections::linked_list::Iter<'a, (V, S)>, fn(&'a (V, S)) -> &'a V>;

    #[inline]
    fn values(&'a self) -> Self::IterValue {
        self.iter()
            .map((|r: &(V, S)| &r.0) as fn(&'a (V, S)) -> &'a V)
    }
}

impl<'a, V: 'a, S: 'a> CollectionMut<'a> for LinkedList<(V, S)> {
    type IterValueMut = std::iter::Map<
        std::collections::linked_list::IterMut<'a, (V, S)>,
        fn(&'a mut (V, S)) -> &'a mut V,
    >;

    #[inline]
    fn values_mut(&'a mut self) -> Self::IterValueMut {
        self.iter_mut()
            .map((|r: &mut (V, S)| &mut r.0) as fn(&'a mut (V, S)) -> &'a mut V)
    }
}

impl<'a, I: Copy + Eq + Ord + Hash + 'a, S: 'a> CollectionRef<'a> for BTreeMap<I, S> {
    type IterValue = std::collections::btree_map::Keys<'a, I, S>;

    #[inline]
    fn values(&'a self) -> Self::IterValue {
        self.keys()
    }
}

impl<'a, I: Copy + Eq + Ord + Hash + 'a, S: 'a> CollectionRef<'a> for HashMap<I, S> {
    type IterValue = std::collections::hash_map::Keys<'a, I, S>;

    #[inline]
    fn values(&'a self) -> Self::IterValue {
        self.keys()
    }
}

/// Reverse lookup: given a value, find its index. O(n) linear scan.
/// Only implemented for `BTreeMap<I, S>` and `HashMap<I, S>`.
pub trait CollectionBiject: RandomAccess {
    /// # Safety
    /// The collection must uphold the bijection invariant: no two indices map
    /// to equal values.
    unsafe fn value_to_key_unchecked(&self, value: &Self::Value) -> Option<&Self::Index>
    where
        Self::Value: PartialEq;
}

impl<I: Copy + Eq + Ord + Hash, S> CollectionBiject for BTreeMap<I, S> {
    unsafe fn value_to_key_unchecked(&self, value: &I) -> Option<&I>
    where
        I: PartialEq,
    {
        self.get_key_value(value).map(|(k, _)| k)
    }
}

impl<I: Copy + Eq + Ord + Hash, S> CollectionBiject for HashMap<I, S> {
    unsafe fn value_to_key_unchecked(&self, value: &I) -> Option<&I>
    where
        I: PartialEq,
    {
        self.get_key_value(value).map(|(k, _)| k)
    }
}

/// Borrowing-iteration of indices for [`RandomAccess`].
///
/// Use the bound `for<'a> C: RandomAccessRef<'a>` in where clauses.
pub trait RandomAccessRef<'a>: RandomAccess
where
    Self: 'a,
{
    type Indices: Iterator<Item = Self::Index>;

    fn indices(&'a self) -> Self::Indices;
}

impl<'a, V: 'a, S: 'a> RandomAccessRef<'a> for Vec<(V, S)> {
    type Indices = std::iter::Map<std::ops::Range<usize>, fn(usize) -> u32>;

    #[inline]
    fn indices(&'a self) -> Self::Indices {
        (0..Vec::len(self)).map((|i| i as u32) as fn(usize) -> u32)
    }
}

impl<'a, 'b: 'a, V: Clone + 'a, S: Clone + 'a> RandomAccessRef<'a> for &'b mut [(V, S)] {
    type Indices = std::iter::Map<std::ops::Range<usize>, fn(usize) -> u32>;

    #[inline]
    fn indices(&'a self) -> Self::Indices {
        (0..<[(V, S)]>::len(self)).map((|i| i as u32) as fn(usize) -> u32)
    }
}

impl<'a, I: Copy + Eq + Ord + Hash + 'a, S: 'a> RandomAccessRef<'a> for BTreeMap<I, S> {
    type Indices = std::iter::Copied<std::collections::btree_map::Keys<'a, I, S>>;

    #[inline]
    fn indices(&'a self) -> Self::Indices {
        self.keys().copied()
    }
}

impl<'a, I: Copy + Eq + Ord + Hash + 'a, S: 'a> RandomAccessRef<'a> for HashMap<I, S> {
    type Indices = std::iter::Copied<std::collections::hash_map::Keys<'a, I, S>>;

    #[inline]
    fn indices(&'a self) -> Self::Indices {
        self.keys().copied()
    }
}

/// Insertion extension for [`Collection`]. Does not require [`RandomAccess`],
/// so collections like [`LinkedList`] can implement it.
///
/// - Sequence backends append and return the new positional index (or `()` for
///   non-indexed types).
/// - Map backends use the value as the key; returns `Err` on collision.
pub trait InsertableCollection: Collection {
    /// Index of the newly inserted entry. For collections that also implement
    /// [`RandomAccess`], this equals [`RandomAccess::Index`]. For non-indexed
    /// collections (e.g. [`LinkedList`]), this is `()`.
    type InsertedIndex;

    fn insert(
        &mut self,
        value: Self::Value,
        storage: Self::Storage,
    ) -> Result<Self::InsertedIndex, (Self::Value, Self::Storage)>;
}

impl<V, S> InsertableCollection for Vec<(V, S)> {
    type InsertedIndex = u32;

    #[inline]
    fn insert(&mut self, value: V, storage: S) -> Result<u32, (V, S)> {
        let ix = Vec::len(self) as u32;
        Vec::push(self, (value, storage));
        Ok(ix)
    }
}

impl<V, S> InsertableCollection for LinkedList<(V, S)> {
    type InsertedIndex = ();

    #[inline]
    fn insert(&mut self, value: V, storage: S) -> Result<(), (V, S)> {
        LinkedList::push_back(self, (value, storage));
        Ok(())
    }
}

impl<I: Copy + Eq + Ord + Hash, S> InsertableCollection for BTreeMap<I, S> {
    type InsertedIndex = I;

    #[inline]
    fn insert(&mut self, value: I, storage: S) -> Result<I, (I, S)> {
        if self.contains_key(&value) {
            return Err((value, storage));
        }
        BTreeMap::insert(self, value, storage);
        Ok(value)
    }
}

impl<I: Copy + Eq + Ord + Hash, S> InsertableCollection for HashMap<I, S> {
    type InsertedIndex = I;

    #[inline]
    fn insert(&mut self, value: I, storage: S) -> Result<I, (I, S)> {
        if self.contains_key(&value) {
            return Err((value, storage));
        }
        HashMap::insert(self, value, storage);
        Ok(value)
    }
}

/// In-place mutation of `Value` slots. Only implemented for collections where
/// the value is independent of the index (sequence-backed types). Map-backed
/// collections do not implement this trait: their `Value` is the key itself,
/// which is immutable.
pub trait UpdatableRandomAccess: RandomAccess {
    /// # Safety
    /// `ix` must be a valid index currently held by this collection.
    unsafe fn get_value_unchecked_mut(&mut self, ix: &Self::Index) -> &mut Self::Value;
}

impl<V, S> UpdatableRandomAccess for Vec<(V, S)> {
    #[inline]
    unsafe fn get_value_unchecked_mut(&mut self, ix: &u32) -> &mut V {
        unsafe { &mut self.get_unchecked_mut(*ix as usize).0 }
    }
}

impl<V: Clone, S: Clone> UpdatableRandomAccess for &mut [(V, S)] {
    #[inline]
    unsafe fn get_value_unchecked_mut(&mut self, ix: &u32) -> &mut V {
        unsafe { &mut <[(V, S)]>::get_unchecked_mut(self, *ix as usize).0 }
    }
}

/// Witness that this collection's [`RandomAccess::Index`] values stay valid
/// across mutations (insertion AND removal). Map-backed collections satisfy
/// this; sequence-backed ones (like `Vec`) do not, because `swap_remove`
/// relocates the last entry into the freed slot.
///
/// # Safety
/// Implementor must guarantee that an `Index` previously returned from this
/// collection continues to refer to the same entry until that specific entry
/// is removed.
pub unsafe trait StableCollection: RandomAccess {}

// SAFETY: auto impl
unsafe impl<I: Copy + Eq + Ord + Hash, S> StableCollection for BTreeMap<I, S> {}
// SAFETY: auto impl
unsafe impl<I: Copy + Eq + Ord + Hash, S> StableCollection for HashMap<I, S> {}

/// Removal extension for [`RandomAccess`].
///
/// `take_unchecked` removes an existing entry by index. Vec-backed types use
/// swap-remove and may return `Some(old_last)` indicating the index whose
/// entry was relocated into the freed slot. Map-backed types always return
/// `None`.
pub trait RemovableRandomAccess: RandomAccess {
    /// # Safety
    /// `ix` must be a valid index currently held by this collection.
    unsafe fn take_unchecked(
        &mut self,
        ix: &Self::Index,
    ) -> (Self::Value, Self::Storage, Option<Self::Index>);
}

impl<V, S> RemovableRandomAccess for Vec<(V, S)> {
    #[inline]
    unsafe fn take_unchecked(&mut self, ix: &u32) -> (V, S, Option<u32>) {
        let i = *ix as usize;
        let last = Vec::len(self) - 1;
        let (v, s) = Vec::swap_remove(self, i);
        let relocated = if i == last { None } else { Some(last as u32) };
        (v, s, relocated)
    }
}

impl<I: Copy + Eq + Ord + Hash, S> RemovableRandomAccess for BTreeMap<I, S> {
    #[inline]
    unsafe fn take_unchecked(&mut self, ix: &I) -> (I, S, Option<I>) {
        let (k, s) = unwrap_unchecked(BTreeMap::remove_entry(self, ix));
        (k, s, None)
    }
}

impl<I: Copy + Eq + Ord + Hash, S> RemovableRandomAccess for HashMap<I, S> {
    #[inline]
    unsafe fn take_unchecked(&mut self, ix: &I) -> (I, S, Option<I>) {
        let (k, s) = unwrap_unchecked(HashMap::remove_entry(self, ix));
        (k, s, None)
    }
}
