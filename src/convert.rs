//! Conversion utilities for exporting graphs into other formats.

use std::fmt::Display;

use crate::graph::capability::Bigraph;
use crate::graph::Graph;

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
    use super::{escape_label, to_mermaid};
    use crate::graph::Graph;
    use crate::BTreeGraph;

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
