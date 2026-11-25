//! Dinic's maximum flow, its minimum cuts and the minimum vertex cut, checked
//! against Edmonds-Karp, petgraph and brute force on random graphs.

use std::collections::VecDeque;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use safegraph::algo::max_flow::{
    dinic, dinic_min_cut, dinic_with_flows, edmonds_karp, min_vertex_cut,
};
use safegraph::graph::capability::{InsertEdge, InsertNode};
use safegraph::graph::Graph;
use safegraph::{HashGraph, VecGraph};

/// A flow network: `n` nodes `0..n`, and edges `(tail, head, capacity)`
/// whose edge index is their position.
#[derive(Debug)]
struct Network {
    n: usize,
    edges: Vec<(u32, u32, u32)>,
}

impl Network {
    /// `n` nodes and `m` edges with random ends (self-loops and parallel
    /// edges allowed) and capacities in `0..=max_cap`.
    fn random(rng: &mut StdRng, n: usize, m: usize, max_cap: u32) -> Self {
        let edges = (0..m)
            .map(|_| {
                let u = rng.gen_range(0..n) as u32;
                let v = rng.gen_range(0..n) as u32;
                (u, v, rng.gen_range(0..=max_cap))
            })
            .collect();
        Network { n, edges }
    }

    fn to_vec_graph(&self) -> VecGraph<(), u32> {
        let mut g = VecGraph::<(), u32>::default();
        unsafe {
            for i in 0..self.n {
                assert_eq!(
                    InsertNode::insert_node_unchecked(&mut g, ()).unwrap(),
                    i as u32
                );
            }
            for (k, &(u, v, c)) in self.edges.iter().enumerate() {
                assert_eq!(
                    InsertEdge::insert_edge_unchecked(&mut g, c, [u, v]).unwrap(),
                    k as u32
                );
            }
        }
        g
    }

    fn to_petgraph(&self) -> petgraph::Graph<(), u32> {
        let mut g = petgraph::Graph::<(), u32>::new();
        let nodes: Vec<_> = (0..self.n).map(|_| g.add_node(())).collect();
        for &(u, v, c) in &self.edges {
            g.add_edge(nodes[u as usize], nodes[v as usize], c);
        }
        g
    }

    /// Two different random nodes.
    fn terminals(&self, rng: &mut StdRng) -> (u32, u32) {
        let s = rng.gen_range(0..self.n) as u32;
        let mut t = rng.gen_range(0..self.n - 1) as u32;
        if t >= s {
            t += 1;
        }
        (s, t)
    }

    /// The nodes that `start` reaches (`forward`) or that reach `start`, in
    /// the residual network of a flow given per edge.
    fn residual_reach(&self, flow: &[u32], start: u32, forward: bool) -> Vec<bool> {
        let mut arcs: Vec<Vec<u32>> = vec![Vec::new(); self.n];
        for (k, &(u, v, c)) in self.edges.iter().enumerate() {
            let mut arc = |from: u32, to: u32| match forward {
                true => arcs[from as usize].push(to),
                false => arcs[to as usize].push(from),
            };
            if flow[k] < c {
                arc(u, v);
            }
            if flow[k] > 0 {
                arc(v, u);
            }
        }
        reach(&arcs, &[start])
    }
}

/// The nodes reachable from `starts` by `arcs`.
fn reach(arcs: &[Vec<u32>], starts: &[u32]) -> Vec<bool> {
    let mut seen = vec![false; arcs.len()];
    let mut queue: VecDeque<u32> = VecDeque::new();
    for &s in starts {
        if !seen[s as usize] {
            seen[s as usize] = true;
            queue.push_back(s);
        }
    }
    while let Some(v) = queue.pop_front() {
        for &w in &arcs[v as usize] {
            if !seen[w as usize] {
                seen[w as usize] = true;
                queue.push_back(w);
            }
        }
    }
    seen
}

/// Random networks of every size class used below.
fn random_networks(seed: u64) -> impl Iterator<Item = (Network, u32, u32)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..400).map(move |round| {
        let (n, m, max_cap) = match round % 4 {
            0 => (rng.gen_range(2..=5), rng.gen_range(0..=10), 3),
            1 => (rng.gen_range(2..=10), rng.gen_range(0..=30), 8),
            2 => (rng.gen_range(10..=30), rng.gen_range(0..=120), 1),
            _ => (rng.gen_range(10..=40), rng.gen_range(0..=200), 100),
        };
        let network = Network::random(&mut rng, n, m, max_cap);
        let (s, t) = network.terminals(&mut rng);
        (network, s, t)
    })
}

// ---------------------------------------------------------------------------
// Flow value
// ---------------------------------------------------------------------------

#[test]
fn dinic_matches_edmonds_karp_and_petgraph() {
    for (network, s, t) in random_networks(1) {
        let g = network.to_vec_graph();
        let flow = dinic(&g, s, t, |&c| c);
        assert_eq!(
            flow,
            edmonds_karp(&g, s, t, |&c| c),
            "{:?} {} -> {}",
            network,
            s,
            t
        );
        let pg = network.to_petgraph();
        let (pg_flow, _) = petgraph::algo::ford_fulkerson(
            &pg,
            petgraph::graph::NodeIndex::new(s as usize),
            petgraph::graph::NodeIndex::new(t as usize),
        );
        assert_eq!(flow, pg_flow, "{:?} {} -> {}", network, s, t);
    }
}

#[test]
fn dinic_with_flows_is_a_maximum_flow() {
    for (network, s, t) in random_networks(2) {
        let g = network.to_vec_graph();
        let g = unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node() };
        let (flow, flows) = dinic_with_flows(g, s, t, |&c| c);
        assert_eq!(flows.len(), network.edges.len());
        let mut net = vec![0i64; network.n];
        for (k, &(u, v, c)) in network.edges.iter().enumerate() {
            let f = flows[&(k as u32)];
            assert!(f <= c, "edge {} carries {} over its capacity {}", k, f, c);
            net[u as usize] -= f as i64;
            net[v as usize] += f as i64;
        }
        for (v, &balance) in net.iter().enumerate() {
            let expected = match v as u32 {
                v if v == s => -(flow as i64),
                v if v == t => flow as i64,
                _ => 0,
            };
            assert_eq!(balance, expected, "flow is not conserved at node {}", v);
        }
        // No augmenting path is left: the flow is maximum.
        let per_edge: Vec<u32> = (0..network.edges.len())
            .map(|k| flows[&(k as u32)])
            .collect();
        assert!(!network.residual_reach(&per_edge, s, true)[t as usize]);
    }
}

// ---------------------------------------------------------------------------
// Minimum cuts
// ---------------------------------------------------------------------------

#[test]
fn both_min_cuts_have_the_capacity_of_the_flow_and_separate() {
    for (network, s, t) in random_networks(3) {
        let g = network.to_vec_graph();
        let g = unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node() };
        let (flow, cut) = dinic_min_cut(g, s, t, |&c| c);
        assert!(cut.source_side(s) && !cut.source_side(t));
        assert!(cut.sink_side(t) && !cut.sink_side(s));
        for v in 0..network.n as u32 {
            assert!(!(cut.source_side(v) && cut.sink_side(v)));
        }
        let cuts: [Vec<u32>; 2] = [
            cut.source_cut_edges().copied().collect(),
            cut.sink_cut_edges().copied().collect(),
        ];
        for edges in &cuts {
            let mut sorted = edges.clone();
            sorted.sort_unstable();
            assert_eq!(edges, &sorted, "cut edges are in index order");
            let capacity: u32 = edges.iter().map(|&k| network.edges[k as usize].2).sum();
            assert_eq!(capacity, flow, "{:?} {} -> {}", network, s, t);
            // Without the cut edges, the source cannot reach the sink at all.
            let mut arcs = vec![Vec::new(); network.n];
            for (k, &(u, v, _)) in network.edges.iter().enumerate() {
                if !edges.contains(&(k as u32)) {
                    arcs[u as usize].push(v);
                }
            }
            assert!(!reach(&arcs, &[s])[t as usize]);
        }
        let source_side: Vec<u32> = cut.source_side_nodes().copied().collect();
        let expected: Vec<u32> = (0..network.n as u32)
            .filter(|&v| cut.source_side(v))
            .collect();
        assert_eq!(source_side, expected);
        let sink_side: Vec<u32> = cut.sink_side_nodes().copied().collect();
        let expected: Vec<u32> = (0..network.n as u32)
            .filter(|&v| cut.sink_side(v))
            .collect();
        assert_eq!(sink_side, expected);
    }
}

/// The two sides do not depend on which maximum flow was found: petgraph's
/// flow (Edmonds-Karp on its own edge order) leaves the same residual
/// reachability as Dinic's.
#[test]
fn cut_sides_are_the_same_for_every_maximum_flow() {
    for (network, s, t) in random_networks(4) {
        let g = network.to_vec_graph();
        let g = unsafe { g.unsafe_assert_stable_edge().unsafe_assert_stable_node() };
        let (_, cut) = dinic_min_cut(g, s, t, |&c| c);
        let (_, pg_flows) = petgraph::algo::ford_fulkerson(
            &network.to_petgraph(),
            petgraph::graph::NodeIndex::new(s as usize),
            petgraph::graph::NodeIndex::new(t as usize),
        );
        let (_, dinic_flows) = dinic_with_flows(g, s, t, |&c| c);
        let dinic_flows: Vec<u32> = (0..network.edges.len())
            .map(|k| dinic_flows[&(k as u32)])
            .collect();
        for flows in &[pg_flows, dinic_flows] {
            let from_source = network.residual_reach(flows, s, true);
            let to_sink = network.residual_reach(flows, t, false);
            for v in 0..network.n {
                assert_eq!(
                    cut.source_side(v as u32),
                    from_source[v],
                    "{:?} {} -> {}",
                    network,
                    s,
                    t
                );
                assert_eq!(
                    cut.sink_side(v as u32),
                    to_sink[v],
                    "{:?} {} -> {}",
                    network,
                    s,
                    t
                );
            }
        }
    }
}

/// Same graph, different `HashMap` iteration orders: the same flow on every
/// edge.
#[test]
fn dinic_does_not_depend_on_iteration_order() {
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..50 {
        let network = Network::random(&mut rng, 12, 40, 5);
        let (s, t) = network.terminals(&mut rng);
        // `HashGraph` indices are the node and edge data: node `v` is `v`,
        // edge `k` is `k`.
        let build = |reversed: bool| {
            let mut g = HashGraph::<u32, u32>::default();
            let mut nodes: Vec<u32> = (0..network.n as u32).collect();
            let mut edges: Vec<usize> = (0..network.edges.len()).collect();
            if reversed {
                nodes.reverse();
                edges.reverse();
            }
            for v in nodes {
                g.insert_node(v).unwrap();
            }
            for k in edges {
                let (u, v, _) = network.edges[k];
                g.insert_edge(k as u32, [u, v]).unwrap();
            }
            g
        };
        let capacity = |&k: &u32| network.edges[k as usize].2;
        let (a, b) = (build(false), build(true));
        let (flow_a, flows_a) = dinic_with_flows(&a, s, t, capacity);
        let (flow_b, flows_b) = dinic_with_flows(&b, s, t, capacity);
        assert_eq!(flow_a, flow_b);
        assert_eq!(flows_a, flows_b);
        let (_, cut_a) = dinic_min_cut(&a, s, t, capacity);
        let (_, cut_b) = dinic_min_cut(&b, s, t, capacity);
        assert!(cut_a.sink_cut_edges().eq(cut_b.sink_cut_edges()));
        assert!(cut_a.source_cut_edges().eq(cut_b.source_cut_edges()));
    }
}

// ---------------------------------------------------------------------------
// Bipartite matching: unit capacities
// ---------------------------------------------------------------------------

#[test]
fn unit_capacity_bipartite_matching() {
    let mut rng = StdRng::seed_from_u64(6);
    for round in 0..60 {
        let (left, right) = if round < 50 {
            (rng.gen_range(1..=12), rng.gen_range(1..=12))
        } else {
            (300, 300)
        };
        let pairs: Vec<(u32, u32)> = (0..left * right / 3)
            .map(|_| (rng.gen_range(0..left), rng.gen_range(0..right)))
            .collect();
        // Nodes: s = 0, t = 1, left `2..2 + left`, right after them.
        let l = |i: u32| 2 + i;
        let r = |j: u32| 2 + left + j;
        let mut edges: Vec<(u32, u32, u32)> = Vec::new();
        edges.extend((0..left).map(|i| (0, l(i), 1)));
        edges.extend(pairs.iter().map(|&(i, j)| (l(i), r(j), 1)));
        edges.extend((0..right).map(|j| (r(j), 1, 1)));
        let network = Network {
            n: (2 + left + right) as usize,
            edges,
        };
        let g = network.to_vec_graph();
        let flow = dinic(&g, 0, 1, |&c| c);

        let mut bipartite = petgraph::graph::UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..left + right).map(|_| bipartite.add_node(())).collect();
        for &(i, j) in &pairs {
            bipartite.add_edge(nodes[i as usize], nodes[(left + j) as usize], ());
        }
        let matching = petgraph::algo::maximum_matching(&bipartite).edges().count();
        assert_eq!(flow as usize, matching);
        if round < 50 {
            assert_eq!(flow, edmonds_karp(&g, 0, 1, |&c| c));
        }
    }
}

// ---------------------------------------------------------------------------
// Minimum vertex cut
// ---------------------------------------------------------------------------

/// A small random graph with node capacities (`None`: cannot be cut).
struct VertexCase {
    capacity: Vec<Option<u32>>,
    edges: Vec<(u32, u32)>,
    sources: Vec<u32>,
    sinks: Vec<u32>,
}

impl VertexCase {
    fn random(rng: &mut StdRng) -> Self {
        let n = rng.gen_range(1..=8);
        let capacity = (0..n)
            .map(|_| match rng.gen_range(0..5) {
                0 => None,
                _ => Some(rng.gen_range(0..=3)),
            })
            .collect();
        let m = rng.gen_range(0..=2 * n);
        let edges = (0..m)
            .map(|_| (rng.gen_range(0..n) as u32, rng.gen_range(0..n) as u32))
            .collect();
        let mut pick = |p: f64| {
            (0..n as u32)
                .filter(|_| rng.gen_bool(p))
                .collect::<Vec<_>>()
        };
        let (sources, sinks) = (pick(0.3), pick(0.3));
        VertexCase {
            capacity,
            edges,
            sources,
            sinks,
        }
    }

    fn n(&self) -> usize {
        self.capacity.len()
    }

    /// The nodes outside `cut` that reach a sink outside `cut` avoiding it.
    fn reaching_sinks(&self, cut: &[bool]) -> Vec<bool> {
        let mut into = vec![Vec::new(); self.n()];
        for &(u, v) in &self.edges {
            if !cut[u as usize] && !cut[v as usize] {
                into[v as usize].push(u);
            }
        }
        let sinks: Vec<u32> = self
            .sinks
            .iter()
            .copied()
            .filter(|&t| !cut[t as usize])
            .collect();
        reach(&into, &sinks)
    }

    fn separates(&self, cut: &[bool]) -> bool {
        let reaching = self.reaching_sinks(cut);
        self.sources
            .iter()
            .all(|&s| cut[s as usize] || !reaching[s as usize])
    }

    fn cost(&self, cut: &[bool]) -> Option<u32> {
        let mut cost = 0;
        for (v, &on) in cut.iter().enumerate() {
            if on {
                cost += self.capacity[v]?;
            }
        }
        Some(cost)
    }
}

#[test]
fn min_vertex_cut_matches_brute_force() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..1500 {
        let case = VertexCase::random(&mut rng);
        let n = case.n();
        let mut g = VecGraph::<Option<u32>, ()>::default();
        unsafe {
            for &c in &case.capacity {
                InsertNode::insert_node_unchecked(&mut g, c).unwrap();
            }
            for &(u, v) in &case.edges {
                InsertEdge::insert_edge_unchecked(&mut g, (), [u, v]).unwrap();
            }
        }
        let g = unsafe { g.unsafe_assert_stable_node() };
        let found = min_vertex_cut(g, case.sources.clone(), case.sinks.clone(), |&c| c);

        // Every separating set of cuttable nodes, with its cost.
        let separating: Vec<(u32, Vec<bool>)> = (0u32..1 << n)
            .map(|mask| (0..n).map(|v| mask & (1 << v) != 0).collect::<Vec<bool>>())
            .filter_map(|cut| Some((case.cost(&cut)?, cut)))
            .filter(|(_, cut)| case.separates(cut))
            .collect();
        let best = match separating.iter().map(|&(cost, _)| cost).min() {
            None => {
                assert_eq!(found, None);
                continue;
            }
            Some(best) => best,
        };
        let (cost, nodes) = found.expect("a separating set exists");
        assert_eq!(cost, best);
        let mut sorted = nodes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(nodes, sorted, "cut nodes are in index order");
        let mut cut = vec![false; n];
        for &v in &nodes {
            cut[v as usize] = true;
        }
        assert_eq!(case.cost(&cut), Some(best));
        assert!(case.separates(&cut));
        // Closest to the sinks: every minimum cut keeps at least as many
        // nodes on the sinks' side (cut, or still reaching a sink).
        let sink_part = |cut: &[bool]| -> Vec<bool> {
            let reaching = case.reaching_sinks(cut);
            (0..n).map(|v| cut[v] || reaching[v]).collect()
        };
        let ours = sink_part(&cut);
        for (_, other) in separating.iter().filter(|&&(cost, _)| cost == best) {
            let theirs = sink_part(other);
            assert!(
                (0..n).all(|v| !ours[v] || theirs[v]),
                "not the cut closest to the sinks"
            );
        }
    }
}

/// type-leak's shape: many roots, each a tuple `(R<i>, Hub)` holding its own
/// leaf `R<i>` and one shared `Hub`. Every leaf and the hub cost 1 to cut; the
/// roots cost `root_cost`.
fn shared_leaf(n: u32, root_cost: Option<u32>) -> (VecGraph<Option<u32>, ()>, Vec<u32>, Vec<u32>) {
    let mut g = VecGraph::<Option<u32>, ()>::default();
    unsafe {
        let hub = InsertNode::insert_node_unchecked(&mut g, Some(1)).unwrap();
        let mut roots = Vec::new();
        let mut leaves = vec![hub];
        for _ in 0..n {
            let root = InsertNode::insert_node_unchecked(&mut g, root_cost).unwrap();
            let leaf = InsertNode::insert_node_unchecked(&mut g, Some(1)).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, (), [root, leaf]).unwrap();
            InsertEdge::insert_edge_unchecked(&mut g, (), [root, hub]).unwrap();
            roots.push(root);
            leaves.push(leaf);
        }
        (g, roots, leaves)
    }
}

#[test]
fn shared_leaf_vertex_cut() {
    let n = 2000;
    // Cutting every root (n) beats cutting every leaf and the hub (n + 1).
    let (g, roots, leaves) = shared_leaf(n, Some(1));
    let g = unsafe { g.unsafe_assert_stable_node() };
    assert_eq!(
        min_vertex_cut(g, roots.clone(), leaves.clone(), |&c| c),
        Some((n, roots.clone()))
    );
    // Nothing can be cut.
    assert_eq!(
        min_vertex_cut(g, roots.clone(), leaves.clone(), |_| None::<u32>),
        None
    );
    // With the roots uncuttable, the leaves and the hub are cut.
    let (g, roots, mut leaves) = shared_leaf(n, None);
    let g = unsafe { g.unsafe_assert_stable_node() };
    leaves.sort_unstable();
    assert_eq!(
        min_vertex_cut(g, roots, leaves.clone(), |&c| c),
        Some((n + 1, leaves))
    );
}
