//! Storage-backend graph implementations built on the
//! [`crate::collection`] trait family.
//!
//! - [`linked_adj_edge`] — linked-list-adjacency graph backed by two independent
//!   `RandomAccess` collections (one for nodes, one for edges). Used by
//!   [`crate::VecGraph`], [`crate::BTreeGraph`], and [`crate::HashGraph`].
//! - [`flat_adj_edge`] — dense per-node adjacency-list graph (a single outer
//!   `RandomAccess` of nodes, each owning an inner `RandomAccess` of outgoing
//!   edges plus an optional incoming-edge reverse index `IS`). `IS = TNone`
//!   keeps only outgoing adjacency; a set `IS` adds an O(in-degree) reverse
//!   index.

pub mod flat_adj_edge;
pub mod hyper_edge;
pub mod linked_adj_edge;
#[cfg(feature = "matrix")]
pub mod matrix;
