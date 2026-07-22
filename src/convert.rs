//! Conversion utilities for exporting graphs into other formats.

use std::fmt::Display;

use crate::graph::capability::Bigraph;
use crate::graph::Graph;

mod aa;
pub use aa::{
    to_aa, to_aa_with, to_ascii, to_ascii_art, to_ascii_art_with, to_ascii_with, AaLabel, AnsiColor,
};

/// Convert a graph into Mermaid `flowchart LR` source.
///
/// Handles both directed and undirected graphs from one entrypoint: the link
/// style is chosen from [`GraphProperty::DIRECTED`](crate::graph::GraphProperty::DIRECTED)
/// — directed graphs use the arrow link `-->`, undirected graphs (e.g. an
/// [`Undirected`](crate::graph::undirected::Undirected) view) use the open link
/// `---`.
///
/// Each node is rendered as `index["node data"]` and each edge as
/// `from <link>|edge data| to`: the node/edge *indices* form the (unique)
/// Mermaid node ids and edge endpoints, while the node/edge *data* is shown as
/// the label — hence the `Display` bounds on `G::Node` / `G::Edge`. Labels are
/// run through `escape_label` so arbitrary data cannot break the Mermaid
/// syntax.
pub fn to_mermaid<G>(graph: &G) -> String
where
    G: Graph + Bigraph + ?Sized,
    G::Node: Display,
    G::Edge: Display,
    for<'scope> crate::graph::context::Context<'scope, G>: Graph<Node = G::Node, Edge = G::Edge>,
{
    // Directed → arrow link, undirected → open link.
    let connector = if G::DIRECTED { "-->" } else { "---" };
    graph.scope(|ctx| {
        let mut out = String::from("flowchart LR\n");
        for n in ctx.node_indices() {
            // id = index (links edges), label = escaped node data
            let label = escape_label(&ctx.node(n).to_string());
            out.push_str(&format!("    {}[\"{}\"]\n", n, label));
        }
        for e in ctx.edge_indices() {
            let mut it = ctx.endpoints(e).into_iter();
            let from = it.next().unwrap();
            let to = it.next().unwrap();
            // label = escaped edge data, endpoints = node indices
            let label = escape_label(&ctx.edge(e).to_string());
            out.push_str(&format!("    {} {}|{}| {}\n", from, connector, label, to));
        }
        out
    })
}

/// Escape a label for safe inclusion in a Mermaid node (`["…"]`) or edge
/// (`|…|`) label.
///
/// Mermaid interprets `#nnn;` numeric character entities, so `#` is escaped
/// first (otherwise the escapes we emit would be re-interpreted), then the
/// characters that would terminate a label (`"`, `|`); newlines are folded to a
/// space so they cannot split the line-based output.
fn escape_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '#' => out.push_str("#35;"),
            '"' => out.push_str("#34;"),
            '|' => out.push_str("#124;"),
            '\n' | '\r' => out.push(' '),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{escape_label, to_aa, to_aa_with, to_mermaid, AnsiColor};
    use crate::graph::Graph;
    use crate::BTreeGraph;

    #[test]
    fn aa_directed_and_isolated_nodes() {
        let mut g = BTreeGraph::<&str, &str>::default();
        g.insert_node("A").unwrap();
        g.insert_node("B").unwrap();
        g.insert_node("C").unwrap();
        g.insert_edge("road", ["A", "B"]).unwrap();

        let aa = to_aa(&g, 80);
        assert_eq!(aa.matches("| A |").count(), 1, "{aa}");
        assert_eq!(aa.matches("| B |").count(), 1, "{aa}");
        assert_eq!(aa.matches("| C |").count(), 1, "{aa}");
        assert!(aa.contains("road"), "{aa}");
        assert!(
            aa.chars().any(|ch| matches!(ch, '>' | '<' | '^' | 'v')),
            "{aa}"
        );
        assert!(
            !aa.contains("+v"),
            "arrowhead must have a straight lead-in:\n{aa}"
        );
    }

    #[test]
    fn aa_undirected_has_no_arrowhead() {
        let mut g = BTreeGraph::<&str, &str>::default();
        g.insert_node("A").unwrap();
        g.insert_node("B").unwrap();
        g.insert_edge("road", ["A", "B"]).unwrap();

        let aa = to_aa(&g.undirected(), 80);
        assert_eq!(aa.matches("| A |").count(), 1, "{aa}");
        assert_eq!(aa.matches("| B |").count(), 1, "{aa}");
        assert!(aa.contains("road"), "{aa}");
        assert!(
            !aa.chars().any(|ch| matches!(ch, '>' | '<' | '^' | 'v')),
            "{aa}"
        );
    }

    #[test]
    fn aa_custom_labels_colors_and_width() {
        let mut g = BTreeGraph::<&str, &str>::default();
        g.insert_node("alpha").unwrap();
        g.insert_node("beta").unwrap();
        g.insert_edge("long edge", ["alpha", "beta"]).unwrap();

        let aa = to_aa_with(
            &g,
            17,
            |node| (node.to_uppercase(), AnsiColor::Green),
            |edge| ((*edge).to_owned(), AnsiColor::Blue),
        );
        let uncolored = aa
            .replace("\x1b[32m", "")
            .replace("\x1b[34m", "")
            .replace("\x1b[0m", "");
        assert_eq!(uncolored.matches("ALPHA").count(), 1, "{uncolored}");
        assert_eq!(uncolored.matches("BETA").count(), 1, "{uncolored}");
        assert!(uncolored.contains("long edge"), "{uncolored}");
        assert!(uncolored
            .lines()
            .all(|line| unicode_width::UnicodeWidthStr::width(line) <= 17));
        assert!(aa.contains("\x1b[32m"));
        assert!(aa.contains("\x1b[34m"));
    }

    #[test]
    fn aa_tiny_and_zero_widths_are_honored() {
        let mut g = BTreeGraph::<&str, &str>::default();
        g.insert_node("日本語").unwrap();

        assert_eq!(to_aa(&g, 0), "");
        for width in 1..7 {
            let aa = to_aa(&g, width);
            assert!(aa
                .lines()
                .all(|line| unicode_width::UnicodeWidthStr::width(line) <= width));
        }
    }

    #[test]
    fn aa_draws_loops_and_parallel_edges_without_repeating_nodes() {
        let mut g = BTreeGraph::<&str, &str>::default();
        g.insert_node("A").unwrap();
        g.insert_node("B").unwrap();
        g.insert_edge("first", ["A", "B"]).unwrap();
        g.insert_edge("loop", ["A", "A"]).unwrap();
        g.insert_edge("second", ["A", "B"]).unwrap();

        let aa = to_aa(&g, 50);
        assert_eq!(aa.matches("| A |").count(), 1, "{aa}");
        assert_eq!(aa.matches("| B |").count(), 1, "{aa}");
        assert!(aa.contains("first"), "{aa}");
        assert!(aa.contains("loop"), "{aa}");
        assert!(aa.contains("second"), "{aa}");
    }

    #[test]
    fn aa_directed_dag_uses_top_to_bottom_layers() {
        let mut g = BTreeGraph::<&str, &str>::default();
        for node in ["A", "B", "C", "D"] {
            g.insert_node(node).unwrap();
        }
        g.insert_edge("ab", ["A", "B"]).unwrap();
        g.insert_edge("ac", ["A", "C"]).unwrap();
        g.insert_edge("bd", ["B", "D"]).unwrap();
        g.insert_edge("cd", ["C", "D"]).unwrap();

        let aa = to_aa(&g, 60);
        let row = |label: &str| {
            aa.lines()
                .position(|line| line.contains(&format!("| {} |", label)))
                .unwrap()
        };
        assert!(row("A") < row("B"), "{aa}");
        assert_eq!(row("B"), row("C"), "{aa}");
        assert!(row("C") < row("D"), "{aa}");
    }

    #[test]
    fn aa_keeps_strongly_connected_nodes_together() {
        let mut g = BTreeGraph::<&str, &str>::default();
        for node in ["A", "B", "C"] {
            g.insert_node(node).unwrap();
        }
        g.insert_edge("ab", ["A", "B"]).unwrap();
        g.insert_edge("ba", ["B", "A"]).unwrap();
        g.insert_edge("bc", ["B", "C"]).unwrap();

        let aa = to_aa(&g, 50);
        let row = |label: &str| {
            aa.lines()
                .position(|line| line.contains(&format!("| {} |", label)))
                .unwrap()
        };
        assert_eq!(row("A"), row("B"), "{aa}");
        assert!(row("B") < row("C"), "{aa}");
    }

    #[test]
    fn aa_dense_graph_uses_edge_label_legend_without_endpoints() {
        let mut g = BTreeGraph::<&str, &str>::default();
        g.insert_node("A").unwrap();
        g.insert_node("B").unwrap();
        for edge in ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"] {
            g.insert_edge(edge, ["A", "B"]).unwrap();
        }

        let aa = to_aa(&g, 50);
        assert!(aa.contains("edge labels (x = crossing):"), "{aa}");
        for number in 1..=8 {
            let mapping = format!("#{} e{}", number, number);
            assert!(
                aa.contains(&mapping),
                "missing edge mapping {mapping}:\n{aa}"
            );
        }
        assert!(!aa.contains("A -> B"), "legend repeated endpoints:\n{aa}");
        assert!(aa
            .lines()
            .all(|line| unicode_width::UnicodeWidthStr::width(line) <= 50));
    }

    #[test]
    fn aa_expands_high_degree_nodes_for_separated_ports() {
        let mut g = BTreeGraph::<&str, &str>::default();
        for node in ["A", "B", "C", "D", "E", "F", "DB"] {
            g.insert_node(node).unwrap();
        }
        for (edge, source) in [
            ("a", "A"),
            ("b", "B"),
            ("c", "C"),
            ("d", "D"),
            ("e", "E"),
            ("f", "F"),
        ] {
            g.insert_edge(edge, [source, "DB"]).unwrap();
        }

        let aa = to_aa(&g, 80);
        assert!(
            aa.contains("| DB        |"),
            "high-degree node was not widened:\n{aa}"
        );
    }

    #[test]
    fn aa_self_loop_has_compact_route_and_destination() {
        let mut g = BTreeGraph::<&str, &str>::default();
        g.insert_node("A").unwrap();
        g.insert_edge("loop", ["A", "A"]).unwrap();

        let aa = to_aa(&g, 40);
        assert_eq!(aa.matches("| A |").count(), 1, "{aa}");
        assert!(aa.contains("| A |-"), "{aa}");
        assert!(aa.contains("loop"), "{aa}");
        assert!(aa.contains('^'), "{aa}");
    }

    #[test]
    fn mermaid_empty_graph() {
        let g = BTreeGraph::<u32, u32>::default();
        assert_eq!(to_mermaid(&g), "flowchart LR\n");
    }

    #[test]
    fn mermaid_btreegraph_renders_key_indices() {
        let mut g = BTreeGraph::<u32, u32>::default();
        g.insert_node(10).unwrap();
        g.insert_node(20).unwrap();
        g.insert_edge(99, [10, 20]).unwrap();

        let src = to_mermaid(&g);
        assert!(src.starts_with("flowchart LR\n"));
        assert!(src.contains("    10[\"10\"]\n"));
        assert!(src.contains("    20[\"20\"]\n"));
        assert!(src.contains("    10 -->|99| 20\n"));
    }

    #[test]
    fn mermaid_btreegraph_keeps_sorted_key_order() {
        let mut g = BTreeGraph::<u32, u32>::default();
        g.insert_node(30).unwrap();
        g.insert_node(10).unwrap();
        g.insert_node(20).unwrap();
        g.insert_edge(2, [20, 30]).unwrap();
        g.insert_edge(1, [10, 20]).unwrap();

        let src = to_mermaid(&g);
        let expected = concat!(
            "flowchart LR\n",
            "    10[\"10\"]\n",
            "    20[\"20\"]\n",
            "    30[\"30\"]\n",
            "    10 -->|1| 20\n",
            "    20 -->|2| 30\n",
        );
        assert_eq!(src, expected);
    }

    // Same entrypoint, undirected graph: the `Undirected` view (`DIRECTED = false`)
    // makes `to_mermaid` emit the open link `---` instead of `-->`.
    #[test]
    fn mermaid_undirected_uses_open_links() {
        let mut g = BTreeGraph::<u32, u32>::default();
        g.insert_node(10).unwrap();
        g.insert_node(20).unwrap();
        g.insert_edge(99, [10, 20]).unwrap();

        let src = to_mermaid(&g.undirected());
        assert!(src.contains("    10[\"10\"]\n"), "{src}");
        assert!(src.contains("    20[\"20\"]\n"), "{src}");
        // open link, no arrowhead
        assert!(src.contains("    10 ---|99| 20\n"), "{src}");
        assert!(!src.contains("-->"), "{src}");
    }

    // Node/edge *data* (distinct from the positional indices) is shown as the
    // label, and breaking characters are escaped.
    #[test]
    fn mermaid_shows_escaped_data() {
        use crate::VecGraph;

        // VecGraph indices are positional (0, 1); the data is separate.
        // Built with the safe `push` / `push_edge` (no `unsafe`).
        let mut g = VecGraph::<&str, &str>::default();
        g.push("A \"x\"").unwrap();
        g.push("B|C").unwrap();
        g.push_edge("e#1", [0, 1]).unwrap();

        let src = to_mermaid(&g);
        // id = positional index, label = escaped node data
        assert!(src.contains("    0[\"A #34;x#34;\"]\n"), "{src}");
        assert!(src.contains("    1[\"B#124;C\"]\n"), "{src}");
        // edge label = escaped edge data, endpoints = indices
        assert!(src.contains("    0 -->|e#35;1| 1\n"), "{src}");
    }

    #[test]
    fn escape_label_handles_breaking_chars() {
        assert_eq!(escape_label("plain"), "plain");
        assert_eq!(escape_label("a\"b"), "a#34;b");
        assert_eq!(escape_label("a|b"), "a#124;b");
        assert_eq!(escape_label("a#b"), "a#35;b");
        // `#` is escaped first, so an emitted entity is not re-escaped.
        assert_eq!(escape_label("\""), "#34;");
        assert_eq!(escape_label("line1\nline2"), "line1 line2");
    }
}
