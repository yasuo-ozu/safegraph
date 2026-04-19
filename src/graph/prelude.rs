//! Convenience re-exports for typical safegraph usage.
//!
//! Bringing this module into scope makes the [`Graph`] facade available, which
//! exposes every safe graph operation by method syntax without ambiguity.
//!
//! ```rust
//! use safegraph::graph::prelude::*;
//! ```
//!
//! [`GraphOperation`](super::GraphOperation) (the lifetime-parameterized,
//! per-backend impl trait) is intentionally *not* re-exported here: its safe
//! methods are all mirrored on [`Graph`], so importing both would make those
//! shared names ambiguous at the call site. Import it explicitly only when you
//! need its `unsafe …_unchecked` methods or associated iterator types.
//!
//! [`Graph`]: super::Graph

pub use super::{Graph, GraphMap, GraphProperty};
