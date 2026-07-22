use safegraph::convert::{to_aa_with, AaLabel};
use safegraph::graph::Graph;
use safegraph::BTreeGraph;

fn main() {
    let nodes = [
        "Gateway", "Jobs", "API", "Queue", "Worker-A", "Auth", "Metrics", "Search", "Worker-B",
        "DB", "Cache", "Isolated",
    ];
    let edges = [
        (0, 2, "HTTPS"),
        (3, 4, "consume-A"),
        (3, 8, "consume-B"),
        (2, 3, "enqueue"),
        (7, 9, "fallback"),
        (1, 1, "heartbeat"),
        (2, 7, "index"),
        (9, 10, "invalidate"),
        (2, 5, "login"),
        (2, 9, "mirror"),
        (4, 9, "persist-A"),
        (8, 9, "persist-B"),
        (2, 9, "query"),
        (10, 9, "refresh"),
        (4, 3, "retry"),
        (1, 3, "schedule"),
        (5, 10, "session"),
        (2, 6, "telemetry"),
    ];
    let mut graph = BTreeGraph::<usize, usize>::default();
    for node in 0..nodes.len() {
        graph.insert_node(node).unwrap();
    }
    for (edge, &(from, to, _)) in edges.iter().enumerate() {
        graph.insert_edge(edge, [from, to]).unwrap();
    }
    print!(
        "{}",
        to_aa_with(
            &graph,
            72,
            |node| AaLabel::plain(nodes[*node]),
            |edge| AaLabel::plain(edges[*edge].2),
        )
    );
}
