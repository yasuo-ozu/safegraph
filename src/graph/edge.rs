//! Edge endpoint types and their index-mapping support.
//!
//! An edge's **endpoints** describe which nodes it connects. The shape of
//! the endpoint collection varies by graph kind:
//!
//! | Graph kind | `Endpoints` type | Notes |
//! |---|---|---|
//! | Binary (ordinary) | `[NodeIx; 2]` | `[tail, head]` ordering |
//! | Hypergraph | `HashSet<NodeIx>` or `BTreeSet<NodeIx>` | Unordered multi-endpoint |
//!
//! The [`Endpoints`] trait abstracts over these shapes, and [`Map`] enables
//! graph wrappers (e.g. [`Context`](super::context::Context),
//! [`Stabilized`](super::stabilized::Stabilized)) to translate endpoint
//! index types without knowing the concrete container.

use std::collections::{BTreeSet, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// A collection of node indices that form an edge's endpoints.
///
/// Implemented for `[Nx; 2]` (binary edges), `HashSet<Nx>` and
/// `BTreeSet<Nx>` (hyperedges), and `Vec<Nx>`.
///
/// Library users rarely interact with this trait directly — it appears as
/// the [`GraphProperty::Endpoints`](super::GraphProperty::Endpoints)
/// associated type and is consumed/produced by
/// [`Graph::endpoints`](super::Graph::endpoints) and
/// [`Graph::insert_edge`](super::Graph::insert_edge).
pub trait Endpoints: Clone + IntoIterator<Item = Self::NodeIx> + Eq + Debug {
    /// The node index type stored in this endpoint collection.
    type NodeIx: Copy + Eq + Ord + Hash + Display + Debug;

    /// Returns an iterator over the contained node indices.
    fn iter(&self) -> Self::IntoIter {
        self.clone().into_iter()
    }

    /// Constructs an endpoint collection from a flat iterator of node
    /// indices. Returns `None` if the iterator has the wrong number of
    /// elements (e.g. `[Nx; 2]` requires exactly two).
    fn try_from_node_indices(nodes: impl IntoIterator<Item = Self::NodeIx>) -> Option<Self>;

    /// Constructs an endpoint collection from separate source (tail) and
    /// target (head) iterators. Returns `None` if the counts are invalid
    /// for this endpoint shape.
    fn try_from_sources_targets(
        source: impl IntoIterator<Item = Self::NodeIx>,
        target: impl IntoIterator<Item = Self::NodeIx>,
    ) -> Option<Self>;
}

/// Element-wise index translation for endpoint collections.
///
/// Graph wrappers use this to convert between their own index type and the
/// inner graph's index type. For example,
/// [`Stabilized`](super::stabilized::Stabilized) uses `map_forward` to
/// stamp version tags onto raw indices, and `map_backward` to strip them.
///
/// `BaseNodeIx` is the *target* index type for `map_forward` (and the
/// *source* for `map_backward`).
pub trait Map<BaseNodeIx>: Endpoints {
    /// The endpoint collection type after mapping, with `NodeIx =
    /// BaseNodeIx`.
    type Mapped: Endpoints<NodeIx = BaseNodeIx>;

    /// Applies `f` to every node index, producing a `Mapped` collection.
    fn map_forward(self, f: impl FnMut(Self::NodeIx) -> BaseNodeIx) -> Self::Mapped;

    /// The inverse of [`map_forward`](Self::map_forward): applies `f` to
    /// every index in `mapped` to recover the original index type.
    fn map_backward(mapped: Self::Mapped, f: impl FnMut(BaseNodeIx) -> Self::NodeIx) -> Self;
}

impl<Nx: Copy + Eq + Ord + Hash + Display + Debug> Endpoints for [Nx; 2] {
    type NodeIx = Nx;

    fn try_from_node_indices(nodes: impl IntoIterator<Item = Self::NodeIx>) -> Option<Self> {
        let mut iter = nodes.into_iter();
        let a = iter.next()?;
        let b = iter.next()?;
        if iter.next().is_some() {
            return None;
        }
        Some([a, b])
    }

    fn try_from_sources_targets(
        source: impl IntoIterator<Item = Self::NodeIx>,
        target: impl IntoIterator<Item = Self::NodeIx>,
    ) -> Option<Self> {
        let mut s = source.into_iter();
        let mut t = target.into_iter();
        let src = s.next()?;
        let dst = t.next()?;
        if s.next().is_some() || t.next().is_some() {
            return None;
        }
        Some([src, dst])
    }
}

impl<Nx, NewNx> Map<NewNx> for [Nx; 2]
where
    Nx: Copy + Eq + Ord + Hash + Display + Debug,
    NewNx: Copy + Eq + Ord + Hash + Display + Debug,
{
    type Mapped = [NewNx; 2];
    fn map_forward(self, mut f: impl FnMut(Nx) -> NewNx) -> [NewNx; 2] {
        [f(self[0]), f(self[1])]
    }

    fn map_backward(mapped: Self::Mapped, mut f: impl FnMut(NewNx) -> Self::NodeIx) -> Self {
        [f(mapped[0]), f(mapped[1])]
    }
}

impl<Nx: Copy + Eq + Ord + Hash + Display + Debug> Endpoints for HashSet<Nx> {
    type NodeIx = Nx;

    fn try_from_node_indices(nodes: impl IntoIterator<Item = Self::NodeIx>) -> Option<Self> {
        Some(nodes.into_iter().collect())
    }

    fn try_from_sources_targets(
        source: impl IntoIterator<Item = Self::NodeIx>,
        target: impl IntoIterator<Item = Self::NodeIx>,
    ) -> Option<Self> {
        Some(source.into_iter().chain(target).collect())
    }
}

impl<Nx, NewNx> Map<NewNx> for HashSet<Nx>
where
    Nx: Copy + Eq + Ord + Hash + Display + Debug,
    NewNx: Copy + Eq + Ord + Hash + Display + Debug,
{
    type Mapped = HashSet<NewNx>;
    fn map_forward(self, f: impl FnMut(Nx) -> NewNx) -> HashSet<NewNx> {
        self.into_iter().map(f).collect()
    }

    fn map_backward(mapped: Self::Mapped, f: impl FnMut(NewNx) -> Self::NodeIx) -> Self {
        mapped.into_iter().map(f).collect()
    }
}

impl<Nx: Copy + Eq + Ord + Hash + Display + Debug> Endpoints for BTreeSet<Nx> {
    type NodeIx = Nx;

    fn try_from_node_indices(nodes: impl IntoIterator<Item = Self::NodeIx>) -> Option<Self> {
        Some(nodes.into_iter().collect())
    }

    fn try_from_sources_targets(
        source: impl IntoIterator<Item = Self::NodeIx>,
        target: impl IntoIterator<Item = Self::NodeIx>,
    ) -> Option<Self> {
        Some(source.into_iter().chain(target).collect())
    }
}

impl<Nx, NewNx> Map<NewNx> for BTreeSet<Nx>
where
    Nx: Copy + Eq + Ord + Hash + Display + Debug,
    NewNx: Copy + Eq + Ord + Hash + Display + Debug,
{
    type Mapped = BTreeSet<NewNx>;
    fn map_forward(self, f: impl FnMut(Nx) -> NewNx) -> BTreeSet<NewNx> {
        self.into_iter().map(f).collect()
    }

    fn map_backward(mapped: Self::Mapped, f: impl FnMut(NewNx) -> Self::NodeIx) -> Self {
        mapped.into_iter().map(f).collect()
    }
}

impl<Nx: Copy + Eq + Ord + Hash + Display + Debug> Endpoints for Vec<Nx> {
    type NodeIx = Nx;

    fn try_from_node_indices(nodes: impl IntoIterator<Item = Nx>) -> Option<Self> {
        Some(nodes.into_iter().collect())
    }

    fn try_from_sources_targets(
        source: impl IntoIterator<Item = Nx>,
        target: impl IntoIterator<Item = Nx>,
    ) -> Option<Self> {
        Some(source.into_iter().chain(target).collect())
    }
}
