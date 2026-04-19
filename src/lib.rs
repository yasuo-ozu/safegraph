#![doc = include_str!("../README.md")]
pub mod algo;
pub mod collection;
pub mod convert;
pub mod graph;
pub mod raw_graph;

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::raw_graph::linked_adj_edge::{EdgeRepr, LinkedAdjEdgeGraph, NodeRepr};

/// `Vec`-backed graph with `u32` node and edge indices.
///
/// # Examples
///
/// ```rust
/// use safegraph::graph::Graph;
/// use safegraph::VecGraph;
///
/// let g: VecGraph<&str, u32> = safegraph::graph!({"a"} -- {1} --> {"b"});
/// assert_eq!(g.len_node(), 2);
/// assert_eq!(g.len_edge(), 1);
/// ```
pub type VecGraph<N, E> = LinkedAdjEdgeGraph<Vec<(N, NodeRepr<u32>)>, Vec<(E, EdgeRepr<u32, u32>)>>;

/// `BTreeMap`-backed graph: node value IS the key (`N`), edge value IS the key (`E`).
///
/// Indices are stable across removals, so `insert_node` / `insert_edge` can be
/// called directly (no [`scope_mut`](crate::graph::Graph::scope_mut) needed).
///
/// # Examples
///
/// ```rust
/// use safegraph::graph::Graph;
/// use safegraph::BTreeGraph;
///
/// // the node value doubles as its index (the key)
/// let mut g = BTreeGraph::<u32, u32>::default();
/// g.insert_node(0).unwrap();
/// g.insert_node(1).unwrap();
/// g.insert_edge(10, [0, 1]).unwrap();
/// assert!(g.contains_edge_index(10));
/// ```
pub type BTreeGraph<N, E> =
    LinkedAdjEdgeGraph<BTreeMap<N, NodeRepr<Option<E>>>, BTreeMap<E, EdgeRepr<N, Option<E>>>>;

/// `HashMap`-backed graph: same Key=Value pattern as [`BTreeGraph`].
///
/// Indices are stable across removals, so `insert_node` / `insert_edge` can be
/// called directly.
///
/// # Examples
///
/// ```rust
/// use safegraph::graph::Graph;
/// use safegraph::HashGraph;
///
/// let mut g = HashGraph::<u32, u32>::default();
/// g.insert_node(0).unwrap();
/// g.insert_node(1).unwrap();
/// g.insert_edge(10, [0, 1]).unwrap();
/// assert_eq!(g.len_node(), 2);
/// ```
pub type HashGraph<N, E> =
    LinkedAdjEdgeGraph<HashMap<N, NodeRepr<Option<E>>>, HashMap<E, EdgeRepr<N, Option<E>>>>;

/// `Vec`-backed hypergraph (each edge may connect any number of nodes) with
/// `u32` indices and `HashSet`-based incidence/endpoint sets. The hypergraph
/// analog of [`VecGraph`]; indices are NOT stable across removals. See
/// [`raw_graph::hyper_edge`] for the stable map-backed variants.
///
/// Because indices are not stable, insertions go through
/// [`scope_mut`](crate::graph::Graph::scope_mut), which hands out scope-stable
/// indices for the duration of the closure.
///
/// # Examples
///
/// ```rust
/// use safegraph::graph::Graph;
/// use safegraph::graph::edge::Endpoints;
/// use safegraph::HyperGraph;
///
/// let mut g = HyperGraph::<u32, u32>::default();
/// g.scope_mut(|mut ctx| {
///     let a = ctx.insert_node(1).unwrap();
///     let b = ctx.insert_node(2).unwrap();
///     let c = ctx.insert_node(3).unwrap();
///     // a single hyperedge connects all three nodes
///     let ep = <_ as Endpoints>::try_from_node_indices([a, b, c]).unwrap();
///     ctx.insert_edge(10, ep).unwrap();
/// });
/// assert_eq!(g.len_node(), 3);
/// assert_eq!(g.len_edge(), 1);
/// ```
pub type HyperGraph<N, E> =
    raw_graph::hyper_edge::HyperGraph<Vec<(N, HashSet<u32>)>, Vec<(E, HashSet<u32>)>>;

/// Build graph nodes/edges with concise DSL syntax.
///
/// # Syntax
///
/// ## Expression mode
///
/// ```text
/// graph!(
///     node_spec (edge_op node_spec)+,
///     ...
/// )
/// ```
///
/// Builds and returns a new graph value (initialized with `Default::default()`).
///
/// ## Statement mode
///
/// ```text
/// graph!(
///     input_graph_expr =>
///     node_spec (edge_op node_spec)+,
///     ...
/// );
/// ```
///
/// Mutates `input_graph_expr` in place and does not return the graph.
/// Named bindings become available in the caller scope.
///
/// # `node_spec` / `edge_spec`
///
/// - `ident`: bind inserted node/edge index to `ident`, payload comes from that variable.
/// - `{expr}`: insert payload from expression, without binding name.
/// - `ident {expr}`: bind inserted index to `ident`, payload comes from `expr`.
/// - `edge_op` forms:
///   - Forward: `-- edge_spec -->`, `-->`
///   - Reverse: `<--`, `<-- edge_spec --`
/// - Empty edge payload (`-->`, `<--`) inserts `Default::default()`.
/// - Multiple edges can be chained in one line, for example:
///   - `A --> B --> C`
///   - `A --> B <-- C`
///   - `A --> B -- {expr} --> C --> D`
///
/// # Stability assertions
///
/// In statement mode (`input_graph_expr => ...`), using named node/edge bindings
/// (`ident` or `ident {expr}`) requires:
///
/// - node binding: `StableNode`
/// - edge binding: `StableEdge`
///
/// # Constraints
///
/// In one macro invocation, the same node/edge ident must not specify `{expr}`
/// more than once.
///
/// # Examples
///
/// Expression mode:
///
/// ```rust
/// use safegraph::graph::Graph;
/// use safegraph::VecGraph;
///
/// let g: VecGraph<u32, u32> = safegraph::graph!(
///     {0} --> {1} -- {11} --> {2},
///     {3} <-- {4},
/// );
/// assert_eq!(g.edges().count(), 3);
/// ```
///
/// Statement mode with bindings:
///
/// ```rust
/// use safegraph::graph::Graph;
/// use safegraph::VecGraph;
///
/// let mut g = VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
/// safegraph::graph!(
///     &mut g =>
///     a {(0, 1u32)} -- e {(0, 10u32)} --> b {(0, 2u32)}
/// );
///
/// assert_eq!(*g.node(a), (0, 1u32));
/// assert_eq!(*g.edge(e), (0, 10u32));
/// assert_eq!(*g.node(b), (0, 2u32));
/// ```
///
/// Statement mode inside `scope_mut()`:
///
/// ```rust
/// use safegraph::graph::Graph;
/// use safegraph::VecGraph;
///
/// let mut g = VecGraph::<(i64, u32), (i64, u32)>::default();
/// g.scope_mut(|mut ctx| {
///     safegraph::graph!(
///         &mut *ctx =>
///         n0 {(0i64, 1u32)} -- e0 {(0i64, 10u32)} --> n1 {(0, 2)}
///     );
///
///     assert_eq!(*ctx.node(n0), (0, 1));
///     assert_eq!(*ctx.edge(e0), (0, 10));
///     assert_eq!(*ctx.node(n1), (0, 2));
/// });
/// ```
#[macro_export]
macro_rules! graph {
    ($($tt:tt)*) => {
        safegraph_macros::graph!($crate, $($tt)*)
    };
}

pub use graph::Graph;

type Invariant<'a> = core::marker::PhantomData<fn(&'a ()) -> &'a ()>;

unsafe fn unwrap_unchecked<T>(input: Option<T>) -> T {
    match input {
        Some(out) => out,
        None => {
            #[cfg(not(debug_assertions))]
            core::hint::unreachable_unchecked();
            #[cfg(debug_assertions)]
            panic!();
        }
    }
}
