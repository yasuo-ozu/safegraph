# Safegraph - A graph manipulation library with fully type-safe, zero-cost, stable graph APIs [![Latest Version]][crates.io] [![Documentation]][docs.rs] [![GitHub Actions]][actions]

[Latest Version]: https://img.shields.io/crates/v/safegraph.svg
[crates.io]: https://crates.io/crates/safegraph
[Documentation]: https://img.shields.io/docsrs/safegraph
[docs.rs]: https://docs.rs/safegraph/latest/
[GitHub Actions]: https://github.com/yasuo-ozu/safegraph/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/yasuo-ozu/safegraph/actions/workflows/rust.yml

Safegraph is a graph library focused on type-safe, stable graph manipulation APIs.
It prevents common indexing mistakes using Rust's type system and zero-cost abstractions.

In this library, "stability" means an index keeps referring to the same logical
node or edge after it is obtained. Many graph libraries break this guarantee when
removing nodes or edges by compacting internal arrays, which can invalidate old indices.

```rust
use safegraph::graph::Graph;
use safegraph::VecGraph;

let g: VecGraph<u32, u32> = safegraph::graph!(
    {0} -- {10} --> {1},
    {1} -- {11} --> {2},
);
assert_eq!(g.len_node(), 4);
assert_eq!(g.len_edge(), 2);
```

Safegraph provides two main graph manipulation interfaces:

## scoped interface

This API provides stable indices inside a lexical scope. It uses a GhostCell-like
lifetime pattern so node/edge indices cannot escape their valid context. This gives
compile-time safety without runtime index validation overhead.

```rust
use safegraph::BTreeGraph;
use safegraph::graph::capability::UniqueEdge;
use safegraph::graph::Graph;

let g: BTreeGraph<u32, u32> = safegraph::graph!(
    {0} -- {10} --> {1},
);
g.scope(|ctx| {
    let e0 = ctx.edge_index(&10).unwrap();
    assert_eq!(*ctx.edge(e0), 10);
});
```

`scope_mut` allows mutation inside a scope. Combined with `graph!`, you can
insert nodes/edges and remove them with compile-time index safety:

```rust
use safegraph::graph::Graph;
use safegraph::VecGraph;

let mut g = VecGraph::<u32, u32>::default();
g.scope_mut(|mut ctx| {
    safegraph::graph!(
        &mut *ctx =>
        n0 {0} -- {10} --> n1 {1} -- e1 {11} --> n2 {2},
    );
    ctx.remove_nodes_edges([n1], [e1]);
});
assert_eq!(g.len_node(), 2);
assert_eq!(g.len_edge(), 0);
```

## `stabilize()` interface

This API converts a graph into a tombstone-versioned stable wrapper. It preserves
index identity across mutations and validates index availability at runtime.

```rust
use safegraph::graph::Graph;
use safegraph::VecGraph;

let mut g = VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
safegraph::graph!(
    &mut g =>
    n0 {(0, 1)} -- {(0, 10)} --> n1 {(0, 2)},
);
g.remove_node(n0);
assert!(!g.contains_node_index(n0));
assert!(g.contains_node_index(n1));

let n2 = g.insert_node((0, 3)).unwrap();
assert!(g.contains_node_index(n2));
```

### `graph!` Macro

`graph!` supports two modes:

1. Expression mode (no input graph):
   - `let g = graph!( ... );`
   - Builds and returns a new graph (`Default::default()` + insertions).
   - No `StableNode`/`StableEdge` assertion is emitted.

2. Statement mode (with input graph):
   - `graph!(input_graph_expr => ...);`
   - Mutates the provided graph expression in place.
   - Does not return the graph.
   - Exports named node/edge bindings (`ident` / `ident {expr}`) into the caller scope.
   - Emits `StableNode` / `StableEdge` assertions when named node/edge bindings are used.

Examples:

```rust
use safegraph::VecGraph;
use safegraph::graph::Graph;

// Expression mode: returns graph value
let g: VecGraph<u32, u32> = safegraph::graph!(
    {0} -- {10} --> {1},
    {1} -- {11} --> {2},
);
assert_eq!(g.nodes().count(), 4);
```

```rust
use safegraph::graph::Graph;
use safegraph::VecGraph;

// Statement mode: mutates existing graph and exports bound idents
let mut g = VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
safegraph::graph!(
    &mut g =>
    a {(0, 1)} -- e {(0, 10)} --> b {(0, 2)}
);
assert_eq!(*g.node(a), (0, 1));
assert_eq!(*g.edge(e), (0, 10));
```

`ident {expr}` rule:
- You can specify both a binding name and an explicit payload expression.
- The same node/edge ident must not specify `{expr}` more than once in one macro call.

## Core Concepts

### Stability

In safegraph, **stability** means:

- `StableNode`: a `NodeIx` keeps referring to the same logical node during the lifetime.
- `StableEdge`: an `EdgeIx` keeps referring to the same logical edge during the lifetime.

Without stability, indices are *unstable*: after insertion/removal, previously saved indices may no longer be valid, or refers different nodes / edges from it originally refers.

Unstable graph example:

```rust
use safegraph::graph::Graph;
use safegraph::VecGraph;

let mut g = VecGraph::<u32, u32>::default();
let n = unsafe { g.insert_node_unchecked(1).unwrap() };
g.remove_node(n);
assert!(!g.contains_node_index(n)); // old index is invalid
g.push(2); // insert another node
assert_eq!(g.node(n), &2); // refering another node
```

Stable wrapper example:

```rust
use safegraph::graph::Graph;
use safegraph::VecGraph;

let mut g = VecGraph::<u32, u32>::default().stabilize();
let n = g.insert_node(1).unwrap();
g.remove_node(n);
assert!(!g.contains_node_index(n)); // removed generation is invalid

let n2 = g.insert_node(2).unwrap();
assert!(g.contains_node_index(n2));
assert_eq!(*g.node(n2), 2); // new generation points to new payload
```

This is why some APIs/macros require `StableNode` / `StableEdge`: they need indices that remain meaningful under mutation.

## Usage Examples

### Creating and Manipulating Graphs

```rust
use safegraph::graph::Graph;
use safegraph::BTreeGraph;

let mut graph: BTreeGraph<&str, i32> = BTreeGraph::default();

graph.insert_node("Alice").unwrap();
graph.insert_node("Bob").unwrap();
graph.insert_node("Charlie").unwrap();

// Add weighted edges
graph.insert_edge(10, [&"Alice", &"Bob"]).unwrap();
graph.insert_edge(20, [&"Bob", &"Charlie"]).unwrap();
graph.insert_edge(5, [&"Alice", &"Charlie"]).unwrap();

// Query the graph
assert_eq!(graph.len_node(), 3);
assert_eq!(graph.len_edge(), 3);

// Iterate over outgoing edges
for edge_tag in graph.edge_indices_from(&"Alice") {
    let weight = graph.edge(edge_tag);
    let [from, to] = graph.endpoints(edge_tag);
    println!("Edge from {} to {} with weight {}", graph.node(from), graph.node(to), weight);
}
```

### Using Graph Algorithms

```rust
use safegraph::algo::connectivity::tarjan_scc;
use safegraph::algo::shortest_path::dijkstra;
use safegraph::algo::toposort::toposort;
use safegraph::BTreeGraph;
use safegraph::graph::Graph;

let mut graph = BTreeGraph::<&str, &str>::default();
graph.insert_node("A").unwrap();
graph.insert_node("B").unwrap();
graph.insert_node("C").unwrap();
graph.insert_edge("A->B", [&"A", &"B"]).unwrap();
graph.insert_edge("B->C", [&"B", &"C"]).unwrap();
graph.insert_edge("C->A", [&"C", &"A"]).unwrap();

// Find strongly connected components
let sccs: Vec<_> = tarjan_scc(&graph).collect();
println!("Found {} SCCs", sccs.len());

// Topological sort (returns Err for cyclic graphs)
assert!(toposort(&graph).is_err());

// Shortest paths
let dists = dijkstra(&graph, &"A", None, |_| 1u32);
```

## Benchmark

Three of the contenders are the same `VecGraph` accessed three safe ways:
`sg_vec_scoped` (via `scope()`/`scope_mut()`; `contains_*_index` is always
true, so no bounds check), `sg_vec_stabilized` (via `Graph::stabilize()`;
versioned/tombstoned stable indices, no scope), and `sg_vec_checked`
(graph-level `node()`/`edge()`, which assert `contains` = a real bounds
check, no scope). The rest: `sg_flat` (`FlatAdjEdgeGraph`), `sg_btree`
(payload-keyed `BTreeGraph`), `pg` (petgraph `DiGraph`), `pg_stable`
(petgraph `StableDiGraph`).

### creation (build n nodes + 5n edges)

| backend | 100 | 1 000 | 10 000 |
|---|--:|--:|--:|
| sg_vec_scoped | 1.95µs | 19.94µs | 249.66µs |
| sg_vec_stabilized | 3.82µs | 43.69µs | 472.48µs |
| sg_vec_checked | 1.89µs | 17.94µs | 222.23µs |
| sg_flat | 7.65µs | 82.53µs | 1.09ms |
| sg_btree | 64.10µs | 1.37ms | 19.48ms |
| pg | 1.71µs | 15.07µs | 197.41µs |
| pg_stable | 3.99µs | 37.17µs | 373.84µs |

### traversal (sum every node + outgoing edge payload)

| backend | 100 | 1 000 | 10 000 |
|---|--:|--:|--:|
| sg_vec_scoped | 293ns | 3.57µs | 300.33µs |
| sg_vec_stabilized | 326ns | 5.63µs | 277.03µs |
| sg_vec_checked | 215ns | 3.51µs | 255.94µs |
| sg_flat | 821ns | 9.82µs | 124.22µs |
| sg_btree | 6.00µs | 329.96µs | 5.60ms |
| pg | 247ns | 4.22µs | 303.04µs |
| pg_stable | 399ns | 6.25µs | 274.91µs |

### random_access (look up n nodes by index in a shuffled order)

| backend | 100 | 1 000 | 10 000 |
|---|--:|--:|--:|
| sg_vec_scoped | 35ns | 271ns | 3.26µs |
| sg_vec_stabilized | 59ns | 574ns | 6.95µs |
| sg_vec_checked | 49ns | 421ns | 4.36µs |
| sg_flat | 47ns | 489ns | 5.89µs |
| sg_btree | 686ns | 16.42µs | 937.67µs |
| pg | 45ns | 358ns | 6.26µs |
| pg_stable | 64ns | 573ns | 8.55µs |

### remove_edges (identify half the edges by predicate, then remove)

| backend | 100 | 1 000 | 10 000 |
|---|--:|--:|--:|
| sg_vec_scoped | 8.46µs | 84.91µs | 1.52ms |
| sg_vec_stabilized | 3.89µs | 30.08µs | 379.34µs |
| sg_vec_checked | 4.01µs | 35.23µs | 1.13ms |
| sg_flat | 7.65µs | 83.64µs | 1.19ms |
| sg_btree | 83.75µs | 1.50ms | 21.94ms |
| pg | 2.72µs | 32.12µs | 1.16ms |
| pg_stable | 3.31µs | 61.83µs | 1.31ms |
| pg_stable_loop | 3.39µs | 59.22µs | 1.50ms |

### remove_nodes (remove a quarter of the nodes, cascading their edges)

| backend | 100 | 1 000 | 10 000 |
|---|--:|--:|--:|
| sg_vec_scoped | 3.91µs | 63.52µs | 1.51ms |
| sg_vec_stabilized | 4.17µs | 62.30µs | 559.24µs |
| sg_vec_checked | 3.73µs | 73.27µs | 1.55ms |
| sg_flat | 57.22µs | 7.66ms | 734.78ms ⚠ |
| sg_btree | 85.16µs | 1.57ms | 18.69ms |
| pg | 4.22µs | 55.76µs | 1.25ms |
| pg_stable | 4.16µs | 22.45µs | 695.91µs |
| pg_stable_loop | 2.92µs | 29.26µs | — |

### memory (live heap bytes of one built graph)

| backend | 100 | 1 000 | 10 000 |
|---|--:|--:|--:|
| sg_vec_scoped | 14.0 KiB | 208.0 KiB | 1.75 MiB |
| sg_vec_stabilized | 19.0 KiB | 280.0 KiB | 2.38 MiB |
| sg_vec_checked | 14.0 KiB | 208.0 KiB | 1.75 MiB |
| sg_flat | 14.5 KiB | 137.7 KiB | 1.53 MiB |
| sg_btree | 59.5 KiB | 601.9 KiB | 5.88 MiB |
| pg | 14.0 KiB | 208.0 KiB | 1.75 MiB |
| pg_stable | 19.0 KiB | 280.0 KiB | 2.38 MiB |
