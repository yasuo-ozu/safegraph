use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt::Display;

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::graph::capability::Bigraph;
use crate::graph::Graph;

/// A terminal color used by [`to_aa_with`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnsiColor {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
    /// A color from the terminal's 256-color palette.
    Fixed(u8),
    /// A 24-bit terminal color.
    Rgb(u8, u8, u8),
}

impl AnsiColor {
    fn prefix(self) -> String {
        let code = match self {
            Self::Black => "30".to_owned(),
            Self::Red => "31".to_owned(),
            Self::Green => "32".to_owned(),
            Self::Yellow => "33".to_owned(),
            Self::Blue => "34".to_owned(),
            Self::Magenta => "35".to_owned(),
            Self::Cyan => "36".to_owned(),
            Self::White => "37".to_owned(),
            Self::BrightBlack => "90".to_owned(),
            Self::BrightRed => "91".to_owned(),
            Self::BrightGreen => "92".to_owned(),
            Self::BrightYellow => "93".to_owned(),
            Self::BrightBlue => "94".to_owned(),
            Self::BrightMagenta => "95".to_owned(),
            Self::BrightCyan => "96".to_owned(),
            Self::BrightWhite => "97".to_owned(),
            Self::Fixed(n) => format!("38;5;{}", n),
            Self::Rgb(r, g, b) => format!("38;2;{};{};{}", r, g, b),
        };
        format!("\x1b[{}m", code)
    }
}

/// Text and optional terminal color returned by an AA formatting closure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AaLabel {
    pub text: String,
    pub color: Option<AnsiColor>,
}

impl AaLabel {
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            color: None,
        }
    }

    pub fn colored(text: impl Into<String>, color: AnsiColor) -> Self {
        Self {
            text: text.into(),
            color: Some(color),
        }
    }
}

impl<T: Display> From<(T, AnsiColor)> for AaLabel {
    fn from((text, color): (T, AnsiColor)) -> Self {
        Self::colored(text.to_string(), color)
    }
}

impl<T: Display> From<(T, Option<AnsiColor>)> for AaLabel {
    fn from((text, color): (T, Option<AnsiColor>)) -> Self {
        Self {
            text: text.to_string(),
            color,
        }
    }
}

/// Draw an arbitrary binary graph on a width-constrained virtual canvas.
///
/// Every node is placed exactly once. Connected nodes are kept close by a
/// deterministic breadth-first layout, then edges are routed orthogonally
/// around node boxes. The router supports cycles, self-loops, parallel edges,
/// disconnected components and both directed and undirected graphs. Dense
/// drawings use compact `#N` route markers and an edge-label legend;
/// an arrowhead marks each directed edge's destination, and `x` marks a
/// geometric crossing where the two edges do not join.
pub fn to_aa<G>(graph: &G, maximum_width: usize) -> String
where
    G: Graph + Bigraph + ?Sized,
    G::Node: Display,
    G::Edge: Display,
    for<'scope> crate::graph::context::Context<'scope, G>: Graph<Node = G::Node, Edge = G::Edge>,
{
    to_aa_with(
        graph,
        maximum_width,
        |node| AaLabel::plain(node.to_string()),
        |edge| AaLabel::plain(edge.to_string()),
    )
}

/// Draw a graph with custom node and edge text/colors.
///
/// Each closure is invoked exactly once per corresponding graph item and may
/// return an [`AaLabel`], `(text, AnsiColor)`, or
/// `(text, Option<AnsiColor>)`.
pub fn to_aa_with<G, FN, FE, NL, EL>(
    graph: &G,
    maximum_width: usize,
    mut node_label: FN,
    mut edge_label: FE,
) -> String
where
    G: Graph + Bigraph + ?Sized,
    FN: FnMut(&G::Node) -> NL,
    FE: FnMut(&G::Edge) -> EL,
    NL: Into<AaLabel>,
    EL: Into<AaLabel>,
    for<'scope> crate::graph::context::Context<'scope, G>: Graph<Node = G::Node, Edge = G::Edge>,
{
    if maximum_width == 0 {
        return String::new();
    }

    graph.scope(|ctx| {
        let indices: Vec<_> = ctx.node_indices().collect();
        let index_map: HashMap<_, _> = indices
            .iter()
            .copied()
            .enumerate()
            .map(|(position, index)| (index, position))
            .collect();
        let nodes: Vec<_> = indices
            .iter()
            .map(|&index| sanitize(node_label(ctx.node(index)).into()))
            .collect();
        let mut edges = Vec::new();
        for edge_index in ctx.edge_indices() {
            let mut endpoints = ctx.endpoints(edge_index).into_iter();
            let from = index_map[&endpoints.next().unwrap()];
            let to = index_map[&endpoints.next().unwrap()];
            edges.push((from, to, sanitize(edge_label(ctx.edge(edge_index)).into())));
        }
        render(nodes, edges, G::DIRECTED, maximum_width)
    })
}

pub use to_aa as to_ascii;
pub use to_aa as to_ascii_art;
pub use to_aa_with as to_ascii_with;
pub use to_aa_with as to_ascii_art_with;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Point {
    x: usize,
    y: usize,
}

#[derive(Clone, Copy)]
struct Cell {
    ch: char,
    color: Option<AnsiColor>,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            color: None,
        }
    }
}

struct Canvas {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
    blocked: Vec<bool>,
    route_mask: Vec<u8>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell::default(); width * height],
            blocked: vec![false; width * height],
            route_mask: vec![0; width * height],
        }
    }

    fn index(&self, point: Point) -> usize {
        point.y * self.width + point.x
    }

    fn put(&mut self, point: Point, ch: char, color: Option<AnsiColor>) {
        let index = self.index(point);
        self.cells[index] = Cell { ch, color };
    }

    fn put_text(&mut self, mut point: Point, text: &str, color: Option<AnsiColor>) {
        for ch in text.chars() {
            let width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if point.x + width > self.width {
                break;
            }
            self.put(point, ch, color);
            for offset in 1..width {
                self.put(
                    Point {
                        x: point.x + offset,
                        y: point.y,
                    },
                    '\0',
                    color,
                );
            }
            point.x += width;
        }
    }

    fn block_rect(&mut self, x: usize, y: usize, width: usize, height: usize) {
        for row in y..y + height {
            for column in x..x + width {
                let index = self.index(Point { x: column, y: row });
                self.blocked[index] = true;
            }
        }
    }

    fn render(&self) -> String {
        let first = (0..self.height).find(|&y| self.row_has_content(y));
        let last = (0..self.height).rfind(|&y| self.row_has_content(y));
        let (first, last) = match (first, last) {
            (Some(first), Some(last)) => (first, last),
            _ => return String::new(),
        };

        let mut output = String::new();
        for y in first..=last {
            let end = (0..self.width)
                .rfind(|&x| self.cells[self.index(Point { x, y })].ch != ' ')
                .map_or(0, |x| x + 1);
            let mut active_color = None;
            for x in 0..end {
                let cell = self.cells[self.index(Point { x, y })];
                if cell.ch == '\0' {
                    continue;
                }
                if cell.color != active_color {
                    if active_color.is_some() {
                        output.push_str("\x1b[0m");
                    }
                    if let Some(color) = cell.color {
                        output.push_str(&color.prefix());
                    }
                    active_color = cell.color;
                }
                output.push(cell.ch);
            }
            if active_color.is_some() {
                output.push_str("\x1b[0m");
            }
            output.push('\n');
        }
        output
    }

    fn row_has_content(&self, y: usize) -> bool {
        (0..self.width).any(|x| self.cells[self.index(Point { x, y })].ch != ' ')
    }

    fn grow(&mut self, rows: usize) {
        self.cells
            .extend(std::iter::repeat(Cell::default()).take(self.width * rows));
        self.blocked
            .extend(std::iter::repeat(false).take(self.width * rows));
        self.route_mask
            .extend(std::iter::repeat(0).take(self.width * rows));
        self.height += rows;
    }
}

#[derive(Clone)]
struct PlacedNode {
    label: AaLabel,
    x: usize,
    y: usize,
    width: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy)]
struct Port {
    point: Point,
    arrow: char,
    side: Side,
}

struct RoutedEdge {
    path: Vec<Point>,
    arrow: char,
    label: AaLabel,
    number: usize,
}

fn render(
    mut nodes: Vec<AaLabel>,
    edges: Vec<(usize, usize, AaLabel)>,
    directed: bool,
    maximum_width: usize,
) -> String {
    if nodes.is_empty() {
        return String::new();
    }
    if maximum_width < 7 {
        return render_tiny(nodes, maximum_width);
    }

    let label_limit = maximum_width.saturating_sub(6).max(1);
    for node in &mut nodes {
        node.text = truncate(&node.text, label_limit);
    }
    let mut widths: Vec<_> = nodes
        .iter()
        .map(|node| UnicodeWidthStr::width(node.text.as_str()) + 4)
        .collect();
    let mut incoming = vec![0usize; nodes.len()];
    let mut outgoing = vec![0usize; nodes.len()];
    for &(from, to, _) in &edges {
        if from == to {
            continue;
        }
        outgoing[from] += 1;
        incoming[to] += 1;
        if !directed {
            outgoing[to] += 1;
            incoming[from] += 1;
        }
    }
    for (node, width) in widths.iter_mut().enumerate() {
        let port_demand = incoming[node].max(outgoing[node]);
        let port_width = port_demand.saturating_mul(2).saturating_add(1);
        *width = (*width).max(port_width).min(maximum_width);
    }
    let (positions, height) = layered_layout(nodes.len(), &widths, &edges, directed, maximum_width);
    let mut canvas = Canvas::new(maximum_width, height);
    let mut placed: Vec<Option<PlacedNode>> = vec![None; nodes.len()];

    for node_index in 0..nodes.len() {
        let label = nodes[node_index].clone();
        let width = widths[node_index];
        let (x, y) = positions[node_index];
        let node = PlacedNode { label, x, y, width };
        draw_node(&mut canvas, &node);
        placed[node_index] = Some(node);
    }

    let placed: Vec<_> = placed.into_iter().map(Option::unwrap).collect();
    let dense = edges.len() >= 8 && edges.len() > nodes.len();
    let mut route_order: Vec<_> = edges.into_iter().enumerate().collect();
    route_order.sort_by_key(|(_, (from, to, _))| {
        let source = &placed[*from];
        let target = &placed[*to];
        (
            usize::from(from != to),
            absolute_difference(source.y, target.y) + absolute_difference(source.x, target.x),
        )
    });
    let mut routed = Vec::new();
    for (number, (from, to, label)) in route_order {
        let route_label = if dense {
            AaLabel {
                text: format!("#{}", number + 1),
                color: label.color,
            }
        } else {
            label.clone()
        };
        if let Some((path, arrow, _route_label)) =
            route_edge(&mut canvas, &placed[from], &placed[to], route_label)
        {
            routed.push(RoutedEdge {
                path,
                arrow,
                label,
                number: number + 1,
            });
        }
    }

    let mut legend = Vec::new();
    if dense {
        routed.sort_by_key(|edge| edge.number);
        for edge in &routed {
            let marker = AaLabel {
                text: format!("#{}", edge.number),
                color: edge.label.color,
            };
            place_edge_label(&mut canvas, &edge.path, &marker);
            legend.push((
                edge.number,
                edge.label.clone(),
                format!("#{} {}", edge.number, edge.label.text),
            ));
        }
    } else {
        for edge in &routed {
            if !place_edge_label(&mut canvas, &edge.path, &edge.label)
                && !edge.label.text.is_empty()
            {
                let number = legend.len() + 1;
                let marker = AaLabel {
                    text: format!("#{}", number),
                    color: edge.label.color,
                };
                place_edge_label(&mut canvas, &edge.path, &marker);
                legend.push((
                    number,
                    edge.label.clone(),
                    format!("#{} {}", number, edge.label.text),
                ));
            }
        }
    }

    // Arrowheads are drawn last so labels and route markers cannot erase them.
    if directed {
        for edge in &routed {
            canvas.put(*edge.path.last().unwrap(), edge.arrow, edge.label.color);
        }
    }

    if !legend.is_empty() {
        let first_row = (0..canvas.height)
            .rfind(|&y| canvas.row_has_content(y))
            .map_or(0, |y| y + 2);
        let required_height = first_row + legend.len() + 1;
        if required_height > canvas.height {
            canvas.grow(required_height - canvas.height);
        }
        canvas.put_text(
            Point { x: 0, y: first_row },
            "edge labels (x = crossing):",
            None,
        );
        for (offset, (_, label, description)) in legend.into_iter().enumerate() {
            let text = truncate(&description, canvas.width);
            canvas.put_text(
                Point {
                    x: 0,
                    y: first_row + offset + 1,
                },
                &text,
                label.color,
            );
        }
    }
    canvas.render()
}

fn render_tiny(nodes: Vec<AaLabel>, maximum_width: usize) -> String {
    let mut output = String::new();
    for node in nodes {
        output.push_str(&truncate(&node.text, maximum_width));
        output.push('\n');
    }
    output
}

fn adjacency(node_count: usize, edges: &[(usize, usize, AaLabel)]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(from, to, _) in edges {
        if !adjacency[from].contains(&to) {
            adjacency[from].push(to);
        }
        if !adjacency[to].contains(&from) {
            adjacency[to].push(from);
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    adjacency
}

fn weak_components(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let node_count = adjacency.len();
    let mut seen = vec![false; node_count];
    let mut components = Vec::new();
    while seen.iter().any(|seen| !seen) {
        let root = (0..node_count)
            .filter(|&node| !seen[node])
            .max_by_key(|&node| (adjacency[node].len(), Reverse(node)))
            .unwrap();
        seen[root] = true;
        let mut queue = VecDeque::from(vec![root]);
        let mut component = Vec::new();
        while let Some(node) = queue.pop_front() {
            component.push(node);
            for &neighbor in &adjacency[node] {
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        components.push(component);
    }
    components
}

fn layered_layout(
    node_count: usize,
    widths: &[usize],
    edges: &[(usize, usize, AaLabel)],
    directed: bool,
    maximum_width: usize,
) -> (Vec<(usize, usize)>, usize) {
    let adjacency = adjacency(node_count, edges);
    let components = weak_components(&adjacency);
    let mut positions = vec![(0, 0); node_count];
    let mut y = 2;

    for component in components {
        let (mut layers, cyclic_groups) = if directed {
            directed_layers(&component, node_count, edges)
        } else {
            (undirected_layers(&component, &adjacency), Vec::new())
        };
        reduce_crossings(&mut layers, &adjacency, node_count);
        keep_groups_together(&mut layers, &cyclic_groups);

        let layer_count = layers.len();
        for (layer_index, layer) in layers.into_iter().enumerate() {
            let rows = pack_layer(&layer, widths, maximum_width);
            for row in rows {
                let total_width = row.iter().map(|&node| widths[node]).sum::<usize>()
                    + row.len().saturating_sub(1) * 4;
                let mut x = (maximum_width.saturating_sub(total_width)) / 2;
                x = x.max(1);
                for node in row {
                    positions[node] = (x, y);
                    x += widths[node] + 4;
                }
                y += 5;
            }
            if layer_index + 1 < layer_count {
                // This clear channel is where most inter-layer edges are routed.
                y += 3;
            }
        }
    }
    (positions, y + 2)
}

fn pack_layer(layer: &[usize], widths: &[usize], maximum_width: usize) -> Vec<Vec<usize>> {
    let available = maximum_width.saturating_sub(2);
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut row = Vec::new();
    let mut used = 0;
    for &node in layer {
        let addition = widths[node] + usize::from(!row.is_empty()) * 4;
        if !row.is_empty() && used + addition > available {
            rows.push(row);
            row = Vec::new();
            used = 0;
        }
        used += widths[node] + usize::from(!row.is_empty()) * 4;
        row.push(node);
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
}

fn undirected_layers(component: &[usize], adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let root = *component
        .iter()
        .max_by_key(|&&node| (adjacency[node].len(), Reverse(node)))
        .unwrap();
    let mut level = vec![usize::MAX; adjacency.len()];
    level[root] = 0;
    let mut queue = VecDeque::from(vec![root]);
    while let Some(node) = queue.pop_front() {
        for &neighbor in &adjacency[node] {
            if level[neighbor] == usize::MAX {
                level[neighbor] = level[node] + 1;
                queue.push_back(neighbor);
            }
        }
    }
    levels_from_assignment(component, &level)
}

fn directed_layers(
    component: &[usize],
    node_count: usize,
    edges: &[(usize, usize, AaLabel)],
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut outgoing = vec![Vec::new(); node_count];
    let mut incoming = vec![Vec::new(); node_count];
    for &(from, to, _) in edges {
        outgoing[from].push(to);
        incoming[to].push(from);
    }
    let mut allowed = vec![false; node_count];
    for &node in component {
        allowed[node] = true;
    }

    // Iterative Kosaraju avoids recursion limits on large graphs.
    let mut seen = vec![false; node_count];
    let mut finish_order = Vec::with_capacity(component.len());
    for &start in component {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((node, next_index)) = stack.last_mut() {
            if *next_index < outgoing[*node].len() {
                let next = outgoing[*node][*next_index];
                *next_index += 1;
                if allowed[next] && !seen[next] {
                    seen[next] = true;
                    stack.push((next, 0));
                }
            } else {
                finish_order.push(*node);
                stack.pop();
            }
        }
    }

    let mut component_of = vec![usize::MAX; node_count];
    let mut strongly_connected = Vec::<Vec<usize>>::new();
    for &start in finish_order.iter().rev() {
        if component_of[start] != usize::MAX {
            continue;
        }
        let id = strongly_connected.len();
        component_of[start] = id;
        let mut members = Vec::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            members.push(node);
            for &next in &incoming[node] {
                if allowed[next] && component_of[next] == usize::MAX {
                    component_of[next] = id;
                    stack.push(next);
                }
            }
        }
        members.sort_unstable();
        strongly_connected.push(members);
    }

    let scc_count = strongly_connected.len();
    let mut successors = vec![Vec::new(); scc_count];
    let mut indegree = vec![0usize; scc_count];
    for &(from, to, _) in edges {
        if !allowed[from] || !allowed[to] {
            continue;
        }
        let source = component_of[from];
        let target = component_of[to];
        if source != target && !successors[source].contains(&target) {
            successors[source].push(target);
            indegree[target] += 1;
        }
    }
    let mut queue: VecDeque<_> = (0..scc_count).filter(|&scc| indegree[scc] == 0).collect();
    let mut scc_level = vec![0usize; scc_count];
    while let Some(source) = queue.pop_front() {
        for &target in &successors[source] {
            scc_level[target] = scc_level[target].max(scc_level[source] + 1);
            indegree[target] -= 1;
            if indegree[target] == 0 {
                queue.push_back(target);
            }
        }
    }

    let mut level = vec![0usize; node_count];
    for (scc, members) in strongly_connected.iter().enumerate() {
        for &node in members {
            level[node] = scc_level[scc];
        }
    }
    let cyclic_groups = strongly_connected
        .into_iter()
        .filter(|members| members.len() > 1)
        .collect();
    (levels_from_assignment(component, &level), cyclic_groups)
}

fn keep_groups_together(layers: &mut [Vec<usize>], groups: &[Vec<usize>]) {
    for group in groups {
        for layer in layers.iter_mut() {
            let first = layer.iter().position(|node| group.contains(node));
            let first = match first {
                Some(first) => first,
                None => continue,
            };
            let mut members: Vec<_> = layer
                .iter()
                .copied()
                .filter(|node| group.contains(node))
                .collect();
            if members.len() < 2 {
                break;
            }
            layer.retain(|node| !group.contains(node));
            for (offset, member) in members.drain(..).enumerate() {
                layer.insert(first + offset, member);
            }
            break;
        }
    }
}

fn levels_from_assignment(component: &[usize], level: &[usize]) -> Vec<Vec<usize>> {
    let level_count = component.iter().map(|&node| level[node]).max().unwrap_or(0) + 1;
    let mut layers = vec![Vec::new(); level_count];
    for &node in component {
        layers[level[node]].push(node);
    }
    layers
}

fn reduce_crossings(layers: &mut [Vec<usize>], adjacency: &[Vec<usize>], node_count: usize) {
    if layers.len() < 2 {
        return;
    }
    for _ in 0..4 {
        for layer in 1..layers.len() {
            let reference = layer_positions(&layers[layer - 1], node_count);
            sort_by_barycenter(&mut layers[layer], &reference, adjacency);
        }
        for layer in (0..layers.len() - 1).rev() {
            let reference = layer_positions(&layers[layer + 1], node_count);
            sort_by_barycenter(&mut layers[layer], &reference, adjacency);
        }
    }
}

fn layer_positions(layer: &[usize], node_count: usize) -> Vec<usize> {
    let mut positions = vec![usize::MAX; node_count];
    for (position, &node) in layer.iter().enumerate() {
        positions[node] = position;
    }
    positions
}

fn sort_by_barycenter(layer: &mut [usize], reference: &[usize], adjacency: &[Vec<usize>]) {
    let original: HashMap<_, _> = layer
        .iter()
        .copied()
        .enumerate()
        .map(|(position, node)| (node, position))
        .collect();
    layer.sort_by(|left, right| {
        let score = |node: usize| {
            adjacency[node]
                .iter()
                .filter_map(|&neighbor| match reference[neighbor] {
                    usize::MAX => None,
                    position => Some(position),
                })
                .fold((0usize, 0usize), |(sum, count), value| {
                    (sum + value, count + 1)
                })
        };
        let (left_sum, left_count) = score(*left);
        let (right_sum, right_count) = score(*right);
        match (left_count, right_count) {
            (0, 0) => original[left].cmp(&original[right]),
            (0, _) => std::cmp::Ordering::Greater,
            (_, 0) => std::cmp::Ordering::Less,
            _ => ((left_sum as u128) * (right_count as u128))
                .cmp(&((right_sum as u128) * (left_count as u128)))
                .then_with(|| original[left].cmp(&original[right])),
        }
    });
}

fn absolute_difference(left: usize, right: usize) -> usize {
    if left >= right {
        left - right
    } else {
        right - left
    }
}

fn draw_node(canvas: &mut Canvas, node: &PlacedNode) {
    let color = node.label.color;
    canvas.put_text(
        Point {
            x: node.x,
            y: node.y,
        },
        &format!("+{}+", "-".repeat(node.width - 2)),
        color,
    );
    let label_width = UnicodeWidthStr::width(node.label.text.as_str());
    let padding = node.width - label_width - 2;
    canvas.put_text(
        Point {
            x: node.x,
            y: node.y + 1,
        },
        &format!("| {}{}|", node.label.text, " ".repeat(padding - 1)),
        color,
    );
    canvas.put_text(
        Point {
            x: node.x,
            y: node.y + 2,
        },
        &format!("+{}+", "-".repeat(node.width - 2)),
        color,
    );
    canvas.block_rect(node.x, node.y, node.width, 3);
}

fn ports(node: &PlacedNode, canvas: &Canvas) -> Vec<Port> {
    let mut ports = Vec::new();
    if node.x > 0 {
        ports.push(Port {
            point: Point {
                x: node.x - 1,
                y: node.y + 1,
            },
            arrow: '>',
            side: Side::Left,
        });
    }
    if node.x + node.width < canvas.width {
        ports.push(Port {
            point: Point {
                x: node.x + node.width,
                y: node.y + 1,
            },
            arrow: '<',
            side: Side::Right,
        });
    }
    if node.y > 0 {
        for offset in port_offsets(node.width) {
            ports.push(Port {
                point: Point {
                    x: node.x + offset,
                    y: node.y - 1,
                },
                arrow: 'v',
                side: Side::Top,
            });
        }
    }
    if node.y + 3 < canvas.height {
        for offset in port_offsets(node.width) {
            ports.push(Port {
                point: Point {
                    x: node.x + offset,
                    y: node.y + 3,
                },
                arrow: '^',
                side: Side::Bottom,
            });
        }
    }
    ports
}

fn port_offsets(width: usize) -> Vec<usize> {
    let mut offsets: Vec<_> = (1..width - 1).step_by(2).collect();
    if let Some(last) = offsets.last_mut() {
        *last = width - 2;
    }
    offsets.dedup();
    offsets
}

fn route_edge(
    canvas: &mut Canvas,
    from: &PlacedNode,
    to: &PlacedNode,
    mut label: AaLabel,
) -> Option<(Vec<Point>, char, AaLabel)> {
    label.text = truncate(&label.text, (canvas.width / 3).max(1));
    if std::ptr::eq(from, to) {
        if let Some(path) = self_loop_path(canvas, from) {
            draw_path(canvas, &path, label.color);
            return Some((path, '^', label));
        }
    }
    let from_ports = ports(from, canvas);
    let to_ports = ports(to, canvas);
    let mut best: Option<(usize, Vec<Point>, char)> = None;
    let (preferred_start, preferred_end) = preferred_sides(from, to);
    for preferred_only in [true, false] {
        for &start in &from_ports {
            if preferred_only && start.side != preferred_start {
                continue;
            }
            for &end in &to_ports {
                if preferred_only && end.side != preferred_end {
                    continue;
                }
                if start.point == end.point {
                    continue;
                }
                if let Some((cost, path)) = route_between_ports(canvas, start, end) {
                    let cost = cost
                        + port_penalty(from, to, start.side, true)
                        + port_penalty(from, to, end.side, false)
                        + endpoint_occupancy_penalty(canvas, start.point)
                        + endpoint_occupancy_penalty(canvas, end.point);
                    if best
                        .as_ref()
                        .map_or(true, |(best_cost, _, _)| cost < *best_cost)
                    {
                        best = Some((cost, path, end.arrow));
                    }
                }
            }
        }
        if best.is_some() {
            break;
        }
    }
    let (_, path, arrow) = best?;
    draw_path(canvas, &path, label.color);
    Some((path, arrow, label))
}

fn endpoint_occupancy_penalty(canvas: &Canvas, point: Point) -> usize {
    usize::from(canvas.route_mask[canvas.index(point)] != 0) * 40
}

fn route_between_ports(canvas: &Canvas, start: Port, end: Port) -> Option<(usize, Vec<Point>)> {
    let start_outer = outward_point(canvas, start)?;
    let end_outer = outward_point(canvas, end)?;
    let (cost, middle) = shortest_path(canvas, start_outer, end_outer)?;
    let mut path = vec![start.point];
    append_segment(&mut path, start_outer);
    path.extend(middle.into_iter().skip(1));
    append_segment(&mut path, end.point);
    Some((cost, path))
}

fn outward_point(canvas: &Canvas, port: Port) -> Option<Point> {
    let point = match port.side {
        Side::Left if port.point.x > 0 => Point {
            x: port.point.x - 1,
            y: port.point.y,
        },
        Side::Right if port.point.x + 1 < canvas.width => Point {
            x: port.point.x + 1,
            y: port.point.y,
        },
        Side::Top if port.point.y > 0 => Point {
            x: port.point.x,
            y: port.point.y - 1,
        },
        Side::Bottom if port.point.y + 1 < canvas.height => Point {
            x: port.point.x,
            y: port.point.y + 1,
        },
        _ => return None,
    };
    if canvas.blocked[canvas.index(point)] {
        None
    } else {
        Some(point)
    }
}

fn self_loop_path(canvas: &Canvas, node: &PlacedNode) -> Option<Vec<Point>> {
    let bottom_y = node.y + 4;
    if bottom_y >= canvas.height {
        return None;
    }

    let candidates = [
        if node.x + node.width + 5 < canvas.width {
            Some((
                Point {
                    x: node.x + node.width,
                    y: node.y + 1,
                },
                Point {
                    x: node.x + node.width + 5,
                    y: node.y + 1,
                },
                Point {
                    x: node.x + node.width - 2,
                    y: node.y + 3,
                },
            ))
        } else {
            None
        },
        if node.x >= 6 {
            Some((
                Point {
                    x: node.x - 1,
                    y: node.y + 1,
                },
                Point {
                    x: node.x - 6,
                    y: node.y + 1,
                },
                Point {
                    x: node.x + 1,
                    y: node.y + 3,
                },
            ))
        } else {
            None
        },
    ];

    for candidate in candidates.iter().flatten() {
        let (start, outside, target) = *candidate;
        let corner = Point {
            x: outside.x,
            y: bottom_y,
        };
        let below_target = Point {
            x: target.x,
            y: bottom_y,
        };
        let mut path = vec![start];
        append_segment(&mut path, outside);
        append_segment(&mut path, corner);
        append_segment(&mut path, below_target);
        append_segment(&mut path, target);
        if path
            .iter()
            .all(|&point| !canvas.blocked[canvas.index(point)])
        {
            return Some(path);
        }
    }
    None
}

fn append_segment(path: &mut Vec<Point>, target: Point) {
    let mut point = *path.last().unwrap();
    while point != target {
        if point.x < target.x {
            point.x += 1;
        } else if point.x > target.x {
            point.x -= 1;
        } else if point.y < target.y {
            point.y += 1;
        } else {
            point.y -= 1;
        }
        path.push(point);
    }
}

fn port_penalty(from: &PlacedNode, to: &PlacedNode, side: Side, source: bool) -> usize {
    if std::ptr::eq(from, to) {
        return match side {
            Side::Right | Side::Bottom => 0,
            Side::Left | Side::Top => 6,
        };
    }
    let (start, end) = preferred_sides(from, to);
    let preferred = if source { start } else { end };
    if side == preferred {
        0
    } else {
        8
    }
}

fn preferred_sides(from: &PlacedNode, to: &PlacedNode) -> (Side, Side) {
    if to.y > from.y {
        (Side::Bottom, Side::Top)
    } else if to.y < from.y {
        (Side::Top, Side::Bottom)
    } else if to.x > from.x {
        (Side::Right, Side::Left)
    } else {
        (Side::Left, Side::Right)
    }
}

fn shortest_path(canvas: &Canvas, start: Point, end: Point) -> Option<(usize, Vec<Point>)> {
    const DIRECTIONS: [(isize, isize); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    let states = canvas.width * canvas.height * 4;
    let mut distance = vec![usize::MAX; states];
    let mut previous = vec![usize::MAX; states];
    let mut heap = BinaryHeap::new();
    let start_cell = canvas.index(start);
    for direction in 0..4 {
        let state = start_cell * 4 + direction;
        distance[state] = 0;
        heap.push(Reverse((0usize, state)));
    }

    let mut final_state = None;
    while let Some(Reverse((cost, state))) = heap.pop() {
        if cost != distance[state] {
            continue;
        }
        let cell = state / 4;
        let direction = state % 4;
        let point = Point {
            x: cell % canvas.width,
            y: cell / canvas.width,
        };
        if point == end {
            final_state = Some(state);
            break;
        }
        for (next_direction, &(dx, dy)) in DIRECTIONS.iter().enumerate() {
            let x = point.x as isize + dx;
            let y = point.y as isize + dy;
            if x < 0 || y < 0 || x >= canvas.width as isize || y >= canvas.height as isize {
                continue;
            }
            let next = Point {
                x: x as usize,
                y: y as usize,
            };
            let next_cell = canvas.index(next);
            if canvas.blocked[next_cell] {
                continue;
            }
            let occupied = usize::from(canvas.route_mask[next_cell] != 0);
            let nearby_parallel_routes =
                adjacent_parallel_route_count(canvas, next, next_direction);
            let bend = usize::from(direction != next_direction);
            // A route in the immediately adjacent row/column is technically
            // unambiguous, but parallel stems then look like one thick line.
            // Charge enough clearance cost to make the router leave a blank
            // lane whenever the available canvas permits it.
            let next_cost = cost + 1 + occupied * 15 + nearby_parallel_routes * 12 + bend * 3;
            let next_state = next_cell * 4 + next_direction;
            if next_cost < distance[next_state] {
                distance[next_state] = next_cost;
                previous[next_state] = state;
                heap.push(Reverse((next_cost, next_state)));
            }
        }
    }

    let final_state = final_state?;
    let cost = distance[final_state];
    let mut state = final_state;
    let mut path = Vec::new();
    loop {
        let cell = state / 4;
        path.push(Point {
            x: cell % canvas.width,
            y: cell / canvas.width,
        });
        let prior = previous[state];
        if prior == usize::MAX {
            break;
        }
        state = prior;
    }
    path.reverse();
    path.dedup();
    Some((cost, path))
}

fn adjacent_parallel_route_count(canvas: &Canvas, point: Point, direction: usize) -> usize {
    let (neighbors, route_bit) = if direction == 0 || direction == 2 {
        let right = if point.x + 1 < canvas.width {
            Some(Point {
                x: point.x + 1,
                y: point.y,
            })
        } else {
            None
        };
        (
            [
                point.x.checked_sub(1).map(|x| Point { x, y: point.y }),
                right,
            ],
            2,
        )
    } else {
        let below = if point.y + 1 < canvas.height {
            Some(Point {
                x: point.x,
                y: point.y + 1,
            })
        } else {
            None
        };
        (
            [
                point.y.checked_sub(1).map(|y| Point { x: point.x, y }),
                below,
            ],
            1,
        )
    };
    neighbors
        .into_iter()
        .flatten()
        .filter(|&neighbor| canvas.route_mask[canvas.index(neighbor)] & route_bit != 0)
        .count()
}

fn draw_path(canvas: &mut Canvas, path: &[Point], color: Option<AnsiColor>) {
    for (index, &point) in path.iter().enumerate() {
        let prior = index.checked_sub(1).map(|i| path[i]);
        let next = path.get(index + 1).copied();
        let horizontal =
            prior.map_or(false, |p| p.y == point.y) || next.map_or(false, |p| p.y == point.y);
        let vertical =
            prior.map_or(false, |p| p.x == point.x) || next.map_or(false, |p| p.x == point.x);
        let new_mask = u8::from(horizontal) | (u8::from(vertical) << 1);
        let ch = match (horizontal, vertical) {
            (true, true) => '+',
            (true, false) => '-',
            (false, true) => '|',
            (false, false) => '+',
        };
        let cell_index = canvas.index(point);
        let old = canvas.cells[cell_index];
        let old_mask = canvas.route_mask[cell_index];
        let perpendicular_crossing =
            (old_mask == 1 && new_mask == 2) || (old_mask == 2 && new_mask == 1);
        let combined_mask = old_mask | new_mask;
        let merged = if perpendicular_crossing {
            'x'
        } else if old_mask == 0 {
            ch
        } else if combined_mask == 1 {
            '-'
        } else if combined_mask == 2 {
            '|'
        } else {
            '+'
        };
        let merged_color = if old_mask == 0 || old.color == color {
            color
        } else {
            None
        };
        canvas.route_mask[cell_index] = combined_mask;
        canvas.cells[cell_index] = Cell {
            ch: merged,
            color: merged_color,
        };
    }
}

fn place_edge_label(canvas: &mut Canvas, path: &[Point], label: &AaLabel) -> bool {
    let width = UnicodeWidthStr::width(label.text.as_str());
    if width == 0 {
        return true;
    }
    // Keep the two cells nearest each node free for an unmistakable port stub
    // and (for directed graphs) an arrowhead.
    let path = if path.len() > 6 {
        &path[2..path.len() - 2]
    } else if path.len() > 2 {
        &path[1..path.len() - 1]
    } else {
        &path[0..0]
    };

    let mut best_run = None;
    let mut start = 0;
    while start < path.len() {
        let mut end = start + 1;
        while end < path.len() && path[end].y == path[start].y {
            end += 1;
        }
        let usable = path[start..end].iter().all(|&point| {
            let cell = canvas.cells[canvas.index(point)];
            !canvas.blocked[canvas.index(point)]
                && matches!(cell.ch, '-' | '|' | '+' | '<' | '>' | '^' | 'v')
        });
        if end - start >= width + 2 && usable {
            let length = end - start;
            if best_run.map_or(true, |(_, best_length)| length > best_length) {
                best_run = Some((start, length));
            }
        }
        start = end;
    }
    if let Some((start, length)) = best_run {
        // Preserve a visible route cell on both sides of an inline label so
        // the text cannot look fused to a bend, crossing, or nearby stem.
        let offset = 1 + (length - width - 2) / 2;
        let segment = &path[start..start + length];
        let x = segment.iter().map(|point| point.x).min().unwrap() + offset;
        canvas.put_text(Point { x, y: segment[0].y }, &label.text, label.color);
        return true;
    }

    // Prefer a detached side label with one blank cell before falling back to
    // a compact placement on exceptionally crowded canvases.
    for padding in [1usize, 0] {
        for &point in path {
            let candidates = [
                point.x.checked_add(1 + padding),
                point.x.checked_sub(width + padding),
            ];
            for x in candidates.iter().flatten().copied() {
                if x + width > canvas.width {
                    continue;
                }
                let start = x.saturating_sub(padding);
                let end = (x + width + padding).min(canvas.width);
                let free = (start..end).all(|column| {
                    let index = canvas.index(Point {
                        x: column,
                        y: point.y,
                    });
                    !canvas.blocked[index] && canvas.cells[index].ch == ' '
                });
                if free {
                    canvas.put_text(Point { x, y: point.y }, &label.text, label.color);
                    return true;
                }
            }
        }
    }

    for &point in path {
        for y in [point.y.checked_sub(1), point.y.checked_add(1)]
            .iter()
            .flatten()
            .copied()
        {
            let x = point.x.saturating_sub(width / 2);
            if y >= canvas.height || x + width > canvas.width {
                continue;
            }
            let free = (x..x + width).all(|column| {
                let index = canvas.index(Point { x: column, y });
                !canvas.blocked[index] && canvas.cells[index].ch == ' '
            });
            if free {
                canvas.put_text(Point { x, y }, &label.text, label.color);
                return true;
            }
        }
    }
    false
}

fn sanitize(mut label: AaLabel) -> AaLabel {
    label.text = label
        .text
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect();
    label
}

fn truncate(text: &str, maximum_width: usize) -> String {
    if UnicodeWidthStr::width(text) <= maximum_width {
        return text.to_owned();
    }
    if maximum_width == 0 {
        return String::new();
    }
    let mut output = String::new();
    let mut width = 0;
    for character in text.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if width + character_width > maximum_width - 1 {
            break;
        }
        output.push(character);
        width += character_width;
    }
    output.push('~');
    output
}

#[cfg(test)]
mod tests {
    use super::{draw_path, port_offsets, shortest_path, Canvas, Point};

    #[test]
    fn perpendicular_routes_are_crossings_not_junctions() {
        let mut canvas = Canvas::new(7, 7);
        draw_path(
            &mut canvas,
            &[
                Point { x: 1, y: 3 },
                Point { x: 2, y: 3 },
                Point { x: 3, y: 3 },
                Point { x: 4, y: 3 },
                Point { x: 5, y: 3 },
            ],
            None,
        );
        draw_path(
            &mut canvas,
            &[
                Point { x: 3, y: 1 },
                Point { x: 3, y: 2 },
                Point { x: 3, y: 3 },
                Point { x: 3, y: 4 },
                Point { x: 3, y: 5 },
            ],
            None,
        );
        assert_eq!(canvas.cells[canvas.index(Point { x: 3, y: 3 })].ch, 'x');
    }

    #[test]
    fn parallel_routes_leave_a_clear_lane_when_space_exists() {
        let mut canvas = Canvas::new(13, 9);
        let existing: Vec<_> = (1..=7).map(|y| Point { x: 7, y }).collect();
        draw_path(&mut canvas, &existing, None);

        let (_, path) = shortest_path(&canvas, Point { x: 6, y: 1 }, Point { x: 6, y: 7 }).unwrap();

        assert!(
            path[2..path.len() - 2].iter().all(|point| point.x <= 5),
            "parallel route did not open a clear lane: {path:?}"
        );
    }

    #[test]
    fn node_ports_have_a_clear_column_between_them() {
        let offsets = port_offsets(13);
        assert_eq!(offsets, vec![1, 3, 5, 7, 9, 11]);
        assert!(offsets.windows(2).all(|pair| pair[1] - pair[0] >= 2));
    }
}
