//! Sparse-adjacency-matrix graph backend (sprs).
//!
//! Layout:
//!
//! ```text
//! nodes: Vec<N>                  // node payloads indexed by VIx = u32
//! edges: sprs::CsMat<E>          // sparse adjacency matrix (CSR)
//!   - rows = head nodes
//!   - cols = tail nodes
//!   - entry [head, tail] = edge data
//! ```
//!
//! Edge indices are `EdgeIx(head, tail)` pairs. The matrix is built once
//! via [`CsMatGraph::from_triplets`] (or constructed from an existing
//! [`sprs::CsMat`] via [`CsMatGraph::from_parts`]); the graph is then
//! consumed as a read-only directed graph through [`GraphOperation`] /
//! [`Directed`].

use std::fmt::{self, Debug, Display, Formatter};

use sprs::{CsMat, TriMat};

use crate::graph::GraphProperty;
use crate::graph::capability::{Bigraph, Directed, StableEdge, StableNode};
use crate::graph::operation::GraphOperation;
use crate::graph::walk_item::{WalkItem, WalkItemTo};

/// Edge index = `(head, tail)` matrix coordinates.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EdgeIx(pub u32, pub u32);

impl Display for EdgeIx {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "e({},{})", self.0, self.1)
    }
}

/// Sparse-adjacency-matrix graph.
///
/// Read-only after construction; uses [`sprs::CsMat`] (CSR) internally
/// so per-row (outgoing) lookups are O(out-degree). Column (incoming)
/// lookups currently scan every row — O(nnz). A future enhancement can
/// store a CSC transpose for O(in-degree) incoming queries.
#[derive(Clone)]
pub struct CsMatGraph<N, E> {
    pub(crate) nodes: Vec<N>,
    pub(crate) edges: CsMat<E>,
}

impl<N: Debug, E> Debug for CsMatGraph<N, E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("CsMatGraph")
            .field("nodes", &self.nodes)
            .field("rows", &self.edges.rows())
            .field("cols", &self.edges.cols())
            .field("nnz", &self.edges.nnz())
            .finish()
    }
}

impl<N, E> CsMatGraph<N, E> {
    /// Construct from a node list and a pre-built [`sprs::CsMat`].
    ///
    /// Panics if `edges.rows() != edges.cols() != nodes.len()`.
    pub fn from_parts(nodes: Vec<N>, edges: CsMat<E>) -> Self {
        assert_eq!(edges.rows(), edges.cols(), "edges must be square");
        assert_eq!(
            edges.rows(),
            nodes.len(),
            "edges dimension must match node count"
        );
        Self { nodes, edges }
    }

    /// Consume and return `(nodes, edges)`.
    pub fn into_parts(self) -> (Vec<N>, CsMat<E>) {
        (self.nodes, self.edges)
    }
}

impl<N, E> CsMatGraph<N, E>
where
    E: Clone + std::ops::Add<Output = E>,
{
    /// Construct from a node list and an iterator of `(head, tail, edge_data)`
    /// triplets. Each `(head, tail)` cell may appear at most once — sparse
    /// matrices cannot represent parallel edges between the same nodes, so
    /// duplicate triplets are rejected with a panic.
    pub fn from_triplets<I>(nodes: Vec<N>, edges_iter: I) -> Self
    where
        I: IntoIterator<Item = (u32, u32, E)>,
    {
        let n = nodes.len();
        let mut tri: TriMat<E> = TriMat::new((n, n));
        let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for (row, col, val) in edges_iter {
            if !seen.insert((row, col)) {
                panic!(
                    "CsMatGraph::from_triplets: duplicate edge ({}, {}) — \
                     sparse matrices cannot store parallel edges between the same nodes",
                    row, col
                );
            }
            tri.add_triplet(row as usize, col as usize, val);
        }
        let edges = tri.to_csr();
        Self { nodes, edges }
    }
}

impl<N, E> GraphProperty for CsMatGraph<N, E> {
    type Node = N;
    type Edge = E;
    type NodeIx = u32;
    type EdgeIx = EdgeIx;
    type Endpoints = [u32; 2];
    const DIRECTED: bool = true;
}

impl<N, E> Bigraph for CsMatGraph<N, E> {
    fn endpoints_as_array(endpoints: Self::Endpoints) -> [Self::NodeIx; 2] {
        endpoints
    }
    fn endpoints_from_array(nodes: [Self::NodeIx; 2]) -> Self::Endpoints {
        nodes
    }
}

impl<'r, N: 'r, E: 'r> GraphOperation<'r> for CsMatGraph<N, E> {
    fn contains_node_index(&self, node_ix: Self::NodeIx) -> bool {
        (node_ix as usize) < self.nodes.len()
    }

    fn contains_edge_index(&self, edge_ix: Self::EdgeIx) -> bool {
        self.edges.get(edge_ix.0 as usize, edge_ix.1 as usize).is_some()
    }

    fn len_node(&self) -> usize {
        self.nodes.len()
    }

    fn len_edge(&self) -> usize {
        self.edges.nnz()
    }

    type NodeIndices = std::ops::Range<u32>;
    fn node_indices(&'r self) -> Self::NodeIndices {
        0..(self.nodes.len() as u32)
    }

    type EdgeIndices = std::vec::IntoIter<EdgeIx>;
    fn edge_indices(&'r self) -> Self::EdgeIndices {
        let mut out: Vec<EdgeIx> = Vec::with_capacity(self.edges.nnz());
        for (_, (r, c)) in self.edges.iter() {
            out.push(EdgeIx(r as u32, c as u32));
        }
        out.into_iter()
    }

    unsafe fn node_unchecked(&self, node_ix: Self::NodeIx) -> &Self::Node {
        // SAFETY: precondition.
        unsafe { self.nodes.get_unchecked(node_ix as usize) }
    }

    unsafe fn edge_unchecked(&self, edge_ix: Self::EdgeIx) -> &Self::Edge {
        // SAFETY: precondition — edge_ix is valid (i.e. nonzero in the matrix).
        self.edges
            .get(edge_ix.0 as usize, edge_ix.1 as usize)
            .expect("invalid EdgeIx passed to edge_unchecked")
    }

    unsafe fn endpoints_unchecked(&self, edge_ix: Self::EdgeIx) -> Self::Endpoints {
        [edge_ix.0, edge_ix.1]
    }

    type EdgeIndicesFrom = std::vec::IntoIter<EdgeIx>;
    unsafe fn edge_indices_from_unchecked(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesFrom {
        let row = node_ix as usize;
        let out: Vec<EdgeIx> = match self.edges.outer_view(row) {
            Some(view) => view
                .indices()
                .iter()
                .map(|&col| EdgeIx(row as u32, col as u32))
                .collect(),
            None => Vec::new(),
        };
        out.into_iter()
    }

    type EdgeIndicesOf = std::vec::IntoIter<EdgeIx>;
    unsafe fn edge_indices_of_unchecked(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesOf {
        let row = node_ix as usize;
        let mut out = Vec::new();
        if let Some(view) = self.edges.outer_view(row) {
            for &col in view.indices() {
                out.push(EdgeIx(row as u32, col as u32));
            }
        }
        for r in 0..self.edges.rows() {
            if r == row {
                continue;
            }
            if let Some(view) = self.edges.outer_view(r) {
                for &col in view.indices() {
                    if col == row {
                        out.push(EdgeIx(r as u32, col as u32));
                    }
                }
            }
        }
        out.into_iter()
    }

    type WalksFrom = std::vec::IntoIter<WalkItem<'r, EdgeIx, E, u32>>;
    unsafe fn walks_from_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksFrom {
        let row = node_ix as usize;
        let mut out = Vec::new();
        if let Some(view) = self.edges.outer_view(row) {
            // CsVecView::iter yields (col, &val).
            for (col, val) in view.iter() {
                let val_ref: &'r E = unsafe {
                    // SAFETY: the view borrows from self.edges with lifetime
                    // tied to 'r; the slice we project from is a sub-slice
                    // of self.edges' data() which also lives for 'r.
                    &*(val as *const E)
                };
                out.push(WalkItem::new(EdgeIx(row as u32, col as u32), val_ref, col as u32));
            }
        }
        out.into_iter()
    }

    type WalksOf = std::vec::IntoIter<WalkItem<'r, EdgeIx, E, u32>>;
    unsafe fn walks_of_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksOf {
        let row = node_ix as usize;
        let mut out = Vec::new();
        if let Some(view) = self.edges.outer_view(row) {
            for (col, val) in view.iter() {
                let val_ref: &'r E = unsafe { &*(val as *const E) };
                out.push(WalkItem::new(EdgeIx(row as u32, col as u32), val_ref, col as u32));
            }
        }
        for r in 0..self.edges.rows() {
            if r == row {
                continue;
            }
            if let Some(view) = self.edges.outer_view(r) {
                for (col, val) in view.iter() {
                    if col == row {
                        let val_ref: &'r E = unsafe { &*(val as *const E) };
                        out.push(WalkItem::new(EdgeIx(r as u32, col as u32), val_ref, r as u32));
                    }
                }
            }
        }
        out.into_iter()
    }

    type DrainNode = std::vec::IntoIter<N>;
    type DrainEdge = std::vec::IntoIter<E>;
    fn drain(self) -> (Self::DrainNode, Self::DrainEdge) {
        let edges: Vec<E> = self.edges.into_raw_storage().2;
        (self.nodes.into_iter(), edges.into_iter())
    }

    fn reverse(&mut self) {
        // Transpose the adjacency matrix in place: every edge `a→b` becomes
        // `b→a`. Implemented by moving each value into its transposed slot via a
        // counting sort — no `E: Clone + Add` (the bound a `TriMat::to_csr`
        // round-trip would need): `from_triplets` rejects duplicate `(r, c)`
        // entries, so no transposed cell ever collects two values to merge.
        let n = self.edges.rows();

        // CSR `iter()` order matches `into_raw_storage` data order, so the k-th
        // coordinate pairs with the k-th moved value. Indices are `usize` (Copy).
        let coords: Vec<(usize, usize)> = self.edges.iter().map(|(_, rc)| rc).collect();
        let placeholder = CsMat::new((n, n), vec![0usize; n + 1], Vec::new(), Vec::new());
        let data: Vec<E> = std::mem::replace(&mut self.edges, placeholder)
            .into_raw_storage()
            .2;

        // Transpose each coordinate (swap row/col), then order by the new
        // (row, col) to lay out CSR storage; values move along with their key.
        let mut triples: Vec<(usize, usize, E)> = coords
            .into_iter()
            .zip(data)
            .map(|((r, c), v)| (c, r, v))
            .collect();
        triples.sort_by_key(|&(nr, nc, _)| (nr, nc));

        let mut indptr = vec![0usize; n + 1];
        for &(nr, _, _) in &triples {
            indptr[nr + 1] += 1;
        }
        for r in 0..n {
            indptr[r + 1] += indptr[r];
        }
        let mut indices = Vec::with_capacity(triples.len());
        let mut data = Vec::with_capacity(triples.len());
        for (_, nc, v) in triples {
            indices.push(nc);
            data.push(v);
        }
        self.edges = CsMat::new((n, n), indptr, indices, data);
    }
}

impl<'r, N: 'r, E: 'r> Directed<'r> for CsMatGraph<N, E> {
    type EdgeIndicesTo = std::vec::IntoIter<EdgeIx>;
    unsafe fn edge_indices_to_unchecked(
        &'r self,
        node_ix: Self::NodeIx,
    ) -> Self::EdgeIndicesTo {
        // Scan every row for `col == node_ix`. O(nnz). A CSC transpose
        // view would make this O(in-degree) — left as a future enhancement.
        let target = node_ix as usize;
        let mut out = Vec::new();
        for r in 0..self.edges.rows() {
            if let Some(view) = self.edges.outer_view(r) {
                for &col in view.indices() {
                    if col == target {
                        out.push(EdgeIx(r as u32, col as u32));
                    }
                }
            }
        }
        out.into_iter()
    }

    type WalksTo = std::vec::IntoIter<WalkItemTo<'r, u32, EdgeIx, E>>;
    unsafe fn walks_to_unchecked(&'r self, node_ix: Self::NodeIx) -> Self::WalksTo {
        let target = node_ix as usize;
        let mut out = Vec::new();
        for r in 0..self.edges.rows() {
            if let Some(view) = self.edges.outer_view(r) {
                for (col, val) in view.iter() {
                    if col == target {
                        let val_ref: &'r E = unsafe { &*(val as *const E) };
                        out.push(WalkItemTo::new(r as u32, EdgeIx(r as u32, col as u32), val_ref));
                    }
                }
            }
        }
        out.into_iter()
    }

    type EdgeTailIndices = core::iter::Once<u32>;
    unsafe fn edge_tail_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeTailIndices {
        core::iter::once(edge_ix.0)
    }

    type EdgeHeadIndices = core::iter::Once<u32>;
    unsafe fn edge_head_indices_unchecked(
        &'r self,
        edge_ix: Self::EdgeIx,
    ) -> Self::EdgeHeadIndices {
        core::iter::once(edge_ix.1)
    }
}

// Vec node storage is stable across queries; the sparse matrix structure
// is fixed at construction, so `EdgeIx(row, col)` is also stable.
unsafe impl<N, E> StableNode for CsMatGraph<N, E> {}
unsafe impl<N, E> StableEdge for CsMatGraph<N, E> {}

