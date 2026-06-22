//! Criterion benchmark comparing basic graph-operation costs across the
//! safegraph backends and petgraph.
//!
//! Six groups — `creation`, `traversal`, `random_access`, `remove_edges`,
//! `remove_nodes` (criterion's default wall-clock measurement), plus `memory`
//! (a custom [`Measurement`] reading a tracking global allocator to report the
//! live heap bytes each built graph occupies). Every contender comes from
//! `adapters.rs`, run on the shared workloads from `common.rs`. A
//! `Once`-guarded [`sanity`] gate proves all contenders do equivalent work
//! before any measurement (a mismatch panics with the contender name).

// Benchmarks are a dev-only target and are not bound by the crate's MSRV, so
// they may use newer std APIs than `rust-version` allows.
#![allow(clippy::incompatible_msrv)]

mod adapters;
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Once;

use criterion::measurement::{Measurement, ValueFormatter};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};

use common::{
    edge_is_victim, generate_workload, node_is_victim, shuffled_ordinals, Workload, SEED, SIZES,
};

// ---------------------------------------------------------------------------
// Tracking global allocator + a criterion `Measurement` over live heap bytes.
//
// `LIVE` is net live bytes (allocated minus freed). The `memory` group brackets
// each graph build with `live_bytes()` to read the graph's heap footprint. The
// per-allocation atomic adds a small, uniform overhead to the wall-clock groups
// too — fine for backend-vs-backend comparison.
// ---------------------------------------------------------------------------

static LIVE: AtomicI64 = AtomicI64::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            LIVE.fetch_add(layout.size() as i64, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            LIVE.fetch_add(new_size as i64 - layout.size() as i64, Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn live_bytes() -> i64 {
    LIVE.load(Ordering::Relaxed)
}

/// Criterion measurement whose value is bytes (net live heap allocation).
struct AllocBytes;

static BYTE_FORMATTER: ByteFormatter = ByteFormatter;

impl Measurement for AllocBytes {
    type Intermediate = i64;
    type Value = u64;
    // start/end are unused here (the `memory` group uses `iter_custom`, which
    // returns the value directly), but the trait requires them.
    fn start(&self) -> i64 {
        live_bytes()
    }
    fn end(&self, i: i64) -> u64 {
        (live_bytes() - i).max(0) as u64
    }
    fn add(&self, a: &u64, b: &u64) -> u64 {
        a + b
    }
    fn zero(&self) -> u64 {
        0
    }
    fn to_f64(&self, v: &u64) -> f64 {
        *v as f64
    }
    fn formatter(&self) -> &dyn ValueFormatter {
        &BYTE_FORMATTER
    }
}

struct ByteFormatter;

impl ByteFormatter {
    fn scale(typical: f64, values: &mut [f64]) -> &'static str {
        let (div, unit) = if typical >= 1024.0 * 1024.0 {
            (1024.0 * 1024.0, "MiB")
        } else if typical >= 1024.0 {
            (1024.0, "KiB")
        } else {
            (1.0, "B")
        };
        for v in values.iter_mut() {
            *v /= div;
        }
        unit
    }
}

impl ValueFormatter for ByteFormatter {
    fn scale_values(&self, typical: f64, values: &mut [f64]) -> &'static str {
        Self::scale(typical, values)
    }
    fn scale_throughputs(
        &self,
        typical: f64,
        _throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        // Bytes have no meaningful throughput notion here; report bytes.
        Self::scale(typical, values)
    }
    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "bytes"
    }
}

// ---------------------------------------------------------------------------
// Sanity gate: workload equivalence across all contenders.
// ---------------------------------------------------------------------------

fn sanity() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Validate the two smaller sizes (skip 10k for startup time).
        for &(n, m) in &SIZES[..2] {
            run_sanity(&generate_workload(n, m, SEED));
        }
        // Self-loop + adjacent-edge micro-case: the self-loop must be counted
        // exactly once (outgoing-only) on every contender.
        run_sanity(&Workload {
            n: 2,
            edges: vec![(0, 0), (0, 1)],
        });
    });
}

fn run_sanity(w: &Workload) {
    let n = w.n;
    let m = w.edges.len();

    // Order/multiplicity-independent payload checksum.
    let node_sum = (0..n).fold(0usize, |a, i| a.wrapping_add(i));
    let edge_sum = (0..m).fold(0usize, |a, j| a.wrapping_add(j));
    let checksum = node_sum.wrapping_add(edge_sum);

    let edge_victims = (0..m).filter(|&j| edge_is_victim(j)).count();
    let node_victims = (0..n).filter(|&v| node_is_victim(v)).count();
    let edges_after_node_removal = w
        .edges
        .iter()
        .filter(|&&(f, t)| !node_is_victim(f as usize) && !node_is_victim(t as usize))
        .count();

    macro_rules! check {
        ($a:path, $name:literal) => {{
            use $a as ad;
            let g = ad::build(w);
            assert_eq!(ad::counts(&g), (n, m), concat!($name, " counts"));
            assert_eq!(ad::traverse_sum(&g), checksum, concat!($name, " checksum"));
            assert_eq!(
                ad::out_degree_sum(&g),
                m,
                concat!($name, " out-degree coverage")
            );

            let mut ge = g.clone();
            ad::remove_edge_set(&mut ge);
            assert_eq!(
                ad::counts(&ge),
                (n, m - edge_victims),
                concat!($name, " edge-remove counts")
            );

            let mut gn = g.clone();
            ad::remove_node_set(&mut gn);
            assert_eq!(
                ad::counts(&gn),
                (n - node_victims, edges_after_node_removal),
                concat!($name, " node-remove counts")
            );
        }};
    }

    check!(adapters::sg_vec_scoped, "sg_vec_scoped");
    check!(adapters::sg_vec_stabilized, "sg_vec_stabilized");
    check!(adapters::sg_vec_checked, "sg_vec_checked");
    check!(adapters::sg_flat, "sg_flat");
    check!(adapters::sg_btree, "sg_btree");
    check!(adapters::pg, "pg");
    check!(adapters::pg_stable, "pg_stable");

    // pg_stable loop rows must match its retain rows.
    {
        let g = adapters::pg_stable::build(w);
        let mut ge = g.clone();
        adapters::pg_stable::remove_edge_loop(&mut ge);
        assert_eq!(
            adapters::pg_stable::counts(&ge),
            (n, m - edge_victims),
            "pg_stable edge-loop counts"
        );
        let mut gn = g.clone();
        adapters::pg_stable::remove_node_loop(&mut gn);
        assert_eq!(
            adapters::pg_stable::counts(&gn),
            (n - node_victims, edges_after_node_removal),
            "pg_stable node-loop counts"
        );
    }
}

// ---------------------------------------------------------------------------
// Bench groups.
// ---------------------------------------------------------------------------

fn bench_creation(c: &mut Criterion) {
    sanity();
    let mut group = c.benchmark_group("creation");
    for &(n, m) in SIZES {
        let w = generate_workload(n, m, SEED);
        group.throughput(Throughput::Elements((n + m) as u64));
        macro_rules! row {
            ($a:path, $name:literal) => {{
                use $a as ad;
                group.bench_with_input(BenchmarkId::new($name, n), &w, |b, w| {
                    b.iter(|| black_box(ad::build(w)))
                });
            }};
        }
        row!(adapters::sg_vec_scoped, "sg_vec_scoped");
        row!(adapters::sg_vec_stabilized, "sg_vec_stabilized");
        row!(adapters::sg_vec_checked, "sg_vec_checked");
        row!(adapters::sg_flat, "sg_flat");
        row!(adapters::sg_btree, "sg_btree");
        row!(adapters::pg, "pg");
        row!(adapters::pg_stable, "pg_stable");
    }
    group.finish();
}

fn bench_traversal(c: &mut Criterion) {
    sanity();
    let mut group = c.benchmark_group("traversal");
    for &(n, m) in SIZES {
        let w = generate_workload(n, m, SEED);
        group.throughput(Throughput::Elements((n + m) as u64));
        macro_rules! row {
            ($a:path, $name:literal) => {{
                use $a as ad;
                let g = ad::build(&w);
                group.bench_with_input(BenchmarkId::new($name, n), &g, |b, g| {
                    b.iter(|| black_box(ad::traverse_sum(g)))
                });
            }};
        }
        row!(adapters::sg_vec_scoped, "sg_vec_scoped");
        row!(adapters::sg_vec_stabilized, "sg_vec_stabilized");
        row!(adapters::sg_vec_checked, "sg_vec_checked");
        row!(adapters::sg_flat, "sg_flat");
        row!(adapters::sg_btree, "sg_btree");
        row!(adapters::pg, "pg");
        row!(adapters::pg_stable, "pg_stable");
    }
    group.finish();
}

fn bench_random_access(c: &mut Criterion) {
    sanity();
    let mut group = c.benchmark_group("random_access");
    for &(n, _m) in SIZES {
        let w = generate_workload(n, _m, SEED);
        let order = shuffled_ordinals(n, SEED);
        group.throughput(Throughput::Elements(n as u64));
        // Scoped contenders run the whole access inside one `scope()` (prep
        // untimed, lookups timed) so the checked accessor pays no bounds check.
        macro_rules! row_scoped {
            ($a:path, $name:literal) => {{
                use $a as ad;
                let g = ad::build(&w);
                group.bench_with_input(BenchmarkId::new($name, n), &g, |b, g| {
                    ad::bench_random_access(g, &order, b)
                });
            }};
        }
        // Precompute-then-access contenders: indices are built untimed, the
        // sum loop is timed.
        macro_rules! row {
            ($a:path, $name:literal) => {{
                use $a as ad;
                let g = ad::build(&w);
                let ixs = ad::access_indices(&g, &order);
                group.bench_with_input(BenchmarkId::new($name, n), &(g, ixs), |b, (g, ixs)| {
                    b.iter(|| black_box(ad::access_sum(g, ixs)))
                });
            }};
        }
        row_scoped!(adapters::sg_vec_scoped, "sg_vec_scoped");
        row!(adapters::sg_vec_stabilized, "sg_vec_stabilized");
        row!(adapters::sg_vec_checked, "sg_vec_checked");
        row_scoped!(adapters::sg_flat, "sg_flat");
        row_scoped!(adapters::sg_btree, "sg_btree");
        row!(adapters::pg, "pg");
        row!(adapters::pg_stable, "pg_stable");
    }
    group.finish();
}

fn bench_remove_edges(c: &mut Criterion) {
    sanity();
    let mut group = c.benchmark_group("remove_edges");
    group.sample_size(30);
    for &(n, m) in SIZES {
        let w = generate_workload(n, m, SEED);
        group.throughput(Throughput::Elements(m as u64));
        let bs = if n <= 1_000 {
            BatchSize::SmallInput
        } else {
            BatchSize::LargeInput
        };
        macro_rules! row {
            ($a:path, $name:literal) => {{
                use $a as ad;
                let g0 = ad::build(&w);
                group.bench_with_input(BenchmarkId::new($name, n), &g0, |b, g0| {
                    b.iter_batched(
                        || g0.clone(),
                        |mut g| {
                            ad::remove_edge_set(&mut g);
                            g
                        },
                        bs,
                    )
                });
            }};
        }
        row!(adapters::sg_vec_scoped, "sg_vec_scoped");
        row!(adapters::sg_vec_stabilized, "sg_vec_stabilized");
        row!(adapters::sg_vec_checked, "sg_vec_checked");
        row!(adapters::sg_flat, "sg_flat");
        row!(adapters::sg_btree, "sg_btree");
        row!(adapters::pg, "pg");
        row!(adapters::pg_stable, "pg_stable");

        // One-at-a-time removal — valid only on the stable backend.
        let g0 = adapters::pg_stable::build(&w);
        group.bench_with_input(BenchmarkId::new("pg_stable_loop", n), &g0, |b, g0| {
            b.iter_batched(
                || g0.clone(),
                |mut g| {
                    adapters::pg_stable::remove_edge_loop(&mut g);
                    g
                },
                bs,
            )
        });
    }
    group.finish();
}

fn bench_remove_nodes(c: &mut Criterion) {
    sanity();
    let mut group = c.benchmark_group("remove_nodes");
    group.sample_size(30);
    for &(n, m) in SIZES {
        let w = generate_workload(n, m, SEED);
        group.throughput(Throughput::Elements(n as u64));
        let bs = if n <= 1_000 {
            BatchSize::SmallInput
        } else {
            BatchSize::LargeInput
        };
        macro_rules! row {
            ($a:path, $name:literal) => {{
                use $a as ad;
                let g0 = ad::build(&w);
                group.bench_with_input(BenchmarkId::new($name, n), &g0, |b, g0| {
                    b.iter_batched(
                        || g0.clone(),
                        |mut g| {
                            ad::remove_node_set(&mut g);
                            g
                        },
                        bs,
                    )
                });
            }};
        }
        row!(adapters::sg_vec_scoped, "sg_vec_scoped");
        row!(adapters::sg_vec_stabilized, "sg_vec_stabilized");
        row!(adapters::sg_vec_checked, "sg_vec_checked");
        row!(adapters::sg_flat, "sg_flat");
        row!(adapters::sg_btree, "sg_btree");
        row!(adapters::pg, "pg");
        row!(adapters::pg_stable, "pg_stable");

        // One-at-a-time node removal is O(victims · deg); cap to the two
        // smaller sizes so it does not dominate wall time at 10k.
        if n <= 1_000 {
            let g0 = adapters::pg_stable::build(&w);
            group.bench_with_input(BenchmarkId::new("pg_stable_loop", n), &g0, |b, g0| {
                b.iter_batched(
                    || g0.clone(),
                    |mut g| {
                        adapters::pg_stable::remove_node_loop(&mut g);
                        g
                    },
                    bs,
                )
            });
        }
    }
    group.finish();
}

fn bench_memory(c: &mut Criterion<AllocBytes>) {
    sanity();
    let mut group = c.benchmark_group("memory");
    for &(n, m) in SIZES {
        let w = generate_workload(n, m, SEED);
        group.throughput(Throughput::Elements((n + m) as u64));
        macro_rules! row {
            ($a:path, $name:literal) => {{
                use $a as ad;
                group.bench_with_input(BenchmarkId::new($name, n), &w, |b, w| {
                    // Live-heap delta across one build = the graph's footprint.
                    // Each graph is dropped immediately so peak stays bounded.
                    b.iter_custom(|iters| {
                        let mut total = 0u64;
                        for _ in 0..iters {
                            let before = live_bytes();
                            let g = ad::build(w);
                            let after = live_bytes();
                            total += (after - before).max(0) as u64;
                            drop(black_box(g));
                        }
                        total
                    })
                });
            }};
        }
        row!(adapters::sg_vec_scoped, "sg_vec_scoped");
        row!(adapters::sg_vec_stabilized, "sg_vec_stabilized");
        row!(adapters::sg_vec_checked, "sg_vec_checked");
        row!(adapters::sg_flat, "sg_flat");
        row!(adapters::sg_btree, "sg_btree");
        row!(adapters::pg, "pg");
        row!(adapters::pg_stable, "pg_stable");
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_creation,
    bench_traversal,
    bench_random_access,
    bench_remove_edges,
    bench_remove_nodes
);
criterion_group! {
    name = memory;
    // Memory is deterministic (zero variance); disable plot generation, whose
    // KDE divides by the standard deviation and would NaN-panic.
    config = Criterion::default().with_measurement(AllocBytes).without_plots();
    targets = bench_memory
}
criterion_main!(benches, memory);
