extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::{Spacing, TokenStream as TokenStream2, TokenTree};
use proc_macro_error::{abort_if_dirty, emit_error, proc_macro_error, set_dummy};
use syn::parse::{Parse, ParseStream};
use syn::{braced, Expr, Ident, Path, Token};
use template_quote::{quote, ToTokens};

// ---------------------------------------------------------------------------
// AST types
// ---------------------------------------------------------------------------

/// A node specification: either a named identifier or an anonymous expression.
#[derive(Clone)]
enum NodeSpec {
    Named(Ident),
    NamedExpr(Ident, Expr),
    Anon(Expr),
}

/// An edge specification: named ident, anonymous expression, or empty (unit).
#[derive(Clone)]
enum EdgeSpec {
    Named(Ident),
    NamedExpr(Ident, Expr),
    Anon(Expr),
    Empty,
}

/// One edge declaration: `node_spec -- edge_spec --> node_spec`
struct EdgeDecl {
    src: NodeSpec,
    edge: EdgeSpec,
    dst: NodeSpec,
}

struct DoubleDash;
struct RightArrow;
struct LeftArrow;

/// The full macro input.
struct GraphInput {
    crate_path: Path,
    initial: Option<Expr>,
    decls: Vec<EdgeDecl>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

impl Parse for NodeSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(syn::token::Brace) {
            let content;
            braced!(content in input);
            let expr: Expr = content.parse()?;
            Ok(NodeSpec::Anon(expr))
        } else {
            let ident: Ident = input.parse()?;
            if input.peek(syn::token::Brace) {
                let content;
                braced!(content in input);
                let expr: Expr = content.parse()?;
                Ok(NodeSpec::NamedExpr(ident, expr))
            } else {
                Ok(NodeSpec::Named(ident))
            }
        }
    }
}

impl Parse for EdgeSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Peek ahead: if the next tokens are `-->`, the edge spec is empty.
        if input.peek(Token![-]) && input.peek2(Token![-]) {
            return Ok(EdgeSpec::Empty);
        }
        if input.peek(syn::token::Brace) {
            let content;
            braced!(content in input);
            let expr: Expr = content.parse()?;
            Ok(EdgeSpec::Anon(expr))
        } else {
            let ident: Ident = input.parse()?;
            if input.peek(syn::token::Brace) {
                let content;
                braced!(content in input);
                let expr: Expr = content.parse()?;
                Ok(EdgeSpec::NamedExpr(ident, expr))
            } else {
                Ok(EdgeSpec::Named(ident))
            }
        }
    }
}

impl Parse for DoubleDash {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // the first minus should be Joint
        let minus = input.parse::<TokenTree>()?;
        if !matches!(minus, TokenTree::Punct(p) if p.as_char() == '-' && p.spacing() == Spacing::Joint)
        {
            return Err(syn::Error::new(
                input.span(),
                "expected `-` with joint spacing",
            ));
        }
        input.parse::<Token![-]>()?;
        Ok(Self)
    }
}

impl Parse for RightArrow {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<DoubleDash>()?;
        input.parse::<Token![>]>()?;
        Ok(Self)
    }
}

impl Parse for LeftArrow {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![<]>()?;
        input.parse::<DoubleDash>()?;
        Ok(Self)
    }
}

fn is_arrow_only(input: ParseStream) -> bool {
    let fork = input.fork();
    let first = fork.parse::<TokenTree>();
    let second = fork.parse::<TokenTree>();
    let third = fork.parse::<TokenTree>();
    if let (Ok(TokenTree::Punct(a)), Ok(TokenTree::Punct(b)), Ok(TokenTree::Punct(c))) =
        (first, second, third)
    {
        if a.as_char() == '-'
            && b.as_char() == '-'
            && c.as_char() == '>'
            && a.spacing() == Spacing::Joint
            && b.spacing() == Spacing::Joint
        {
            return true;
        }
    }
    false
}

fn parse_arrow_only(input: ParseStream) -> syn::Result<bool> {
    if is_arrow_only(input) {
        let _: TokenTree = input.parse()?;
        let _: TokenTree = input.parse()?;
        let _: TokenTree = input.parse()?;
        return Ok(true);
    }
    Ok(false)
}

fn parse_forward_segment(input: ParseStream, src: NodeSpec) -> syn::Result<(EdgeDecl, NodeSpec)> {
    if parse_arrow_only(input)? {
        let dst: NodeSpec = input.parse()?;
        let decl = EdgeDecl {
            src,
            edge: EdgeSpec::Empty,
            dst: dst.clone(),
        };
        return Ok((decl, dst));
    }

    let _: DoubleDash = input.parse()?;
    if is_arrow_only(input) {
        return Err(syn::Error::new(
            input.span(),
            "empty forward edge must use `-->` (not `-- -->`)",
        ));
    }
    let edge: EdgeSpec = input.parse()?;
    let _: RightArrow = input.parse()?;
    let dst: NodeSpec = input.parse()?;
    let decl = EdgeDecl {
        src,
        edge,
        dst: dst.clone(),
    };
    Ok((decl, dst))
}

fn parse_reverse_segment(input: ParseStream, left: NodeSpec) -> syn::Result<(EdgeDecl, NodeSpec)> {
    let _: LeftArrow = input.parse()?;

    // `<--`
    if !input.peek(syn::token::Brace) && !input.peek(Ident) && !input.peek(Token![-]) {
        return Err(syn::Error::new(input.span(), "expected node after `<--`"));
    }

    // `<-- node`
    if input.peek(syn::token::Brace) || input.peek(Ident) {
        let fork = input.fork();
        let maybe_spec = fork.parse::<EdgeSpec>();
        if let Ok(spec) = maybe_spec {
            if !matches!(spec, EdgeSpec::Empty) && fork.parse::<DoubleDash>().is_ok() {
                // `<-- edge_spec -- node`
                let edge: EdgeSpec = input.parse()?;
                let _: DoubleDash = input.parse()?;
                let right: NodeSpec = input.parse()?;
                let decl = EdgeDecl {
                    src: right.clone(),
                    edge,
                    dst: left,
                };
                return Ok((decl, right));
            }
        }

        // `<-- node`
        let right: NodeSpec = input.parse()?;
        let decl = EdgeDecl {
            src: right.clone(),
            edge: EdgeSpec::Empty,
            dst: left,
        };
        return Ok((decl, right));
    }

    // `<-- -- node` is intentionally disallowed; use `<-- node`.
    let _: DoubleDash = input.parse()?;
    Err(syn::Error::new(
        input.span(),
        "empty reverse edge must use `<--` (not `<-- --`)",
    ))
}

impl Parse for GraphInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let crate_path: Path = input.parse()?;
        input.parse::<Token![,]>()?;

        // Try to parse `initial =>` prefix.
        let initial = if input.peek2(Token![=>]) || {
            // Look further: could be a longer expression before `=>`
            let fork = input.fork();
            fork.parse::<Expr>().is_ok() && fork.peek(Token![=>])
        } {
            let expr: Expr = input.parse()?;
            input.parse::<Token![=>]>()?;
            Some(expr)
        } else {
            None
        };

        let mut decls = Vec::new();
        while !input.is_empty() {
            let mut cur: NodeSpec = input.parse()?;
            while !input.is_empty() && !input.peek(Token![,]) {
                if input.peek(Token![<]) {
                    let (decl, next) = parse_reverse_segment(input, cur)?;
                    decls.push(decl);
                    cur = next;
                } else {
                    let (decl, next) = parse_forward_segment(input, cur)?;
                    decls.push(decl);
                    cur = next;
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(GraphInput {
            crate_path,
            initial,
            decls,
        })
    }
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

/// Generate code for a single node insertion.
/// Returns `(binding_tokens, is_named)`.
fn gen_node_insert(
    spec: &NodeSpec,
    crate_path: &Path,
    graph_access: &TokenStream2,
    counter: &mut usize,
) -> (TokenStream2, bool) {
    match spec {
        NodeSpec::Named(ident) => {
            let expr_ident = ident.clone();
            (
                quote! {
                    let #expr_ident = unsafe {
                        #crate_path::graph::capability::InsertNode::insert_node_unchecked(
                            #graph_access, #expr_ident
                        ).unwrap()
                    };
                },
                true,
            )
        }
        NodeSpec::NamedExpr(ident, expr) => {
            let binding_ident = ident.clone();
            (
                quote! {
                    let #binding_ident = unsafe {
                        #crate_path::graph::capability::InsertNode::insert_node_unchecked(
                            #graph_access, #expr
                        ).unwrap()
                    };
                },
                true,
            )
        }
        NodeSpec::Anon(expr) => {
            let tmp = syn::Ident::new(
                &format!("__node_{}", counter),
                proc_macro2::Span::call_site(),
            );
            *counter += 1;
            (
                quote! {
                    let #tmp = unsafe {
                        #crate_path::graph::capability::InsertNode::insert_node_unchecked(
                            #graph_access, #expr
                        ).unwrap()
                    };
                },
                false,
            )
        }
    }
}

#[proc_macro]
#[proc_macro_error]
pub fn graph(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as GraphInput);
    let crate_path = &input.crate_path;
    let has_initial = input.initial.is_some();
    let graph_access = if has_initial {
        quote! { &mut *__graph }
    } else {
        quote! { &mut __graph }
    };

    let node_payload_sig = |spec: &NodeSpec| -> Option<String> {
        match spec {
            NodeSpec::Named(id) => Some(format!("named:{}", id)),
            NodeSpec::NamedExpr(_, expr) => Some(format!("expr:{}", expr.to_token_stream())),
            NodeSpec::Anon(_) => None,
        }
    };
    let edge_payload_sig = |spec: &EdgeSpec| -> String {
        match spec {
            EdgeSpec::Named(id) => format!("named:{}", id),
            EdgeSpec::NamedExpr(_, expr) => format!("expr:{}", expr.to_token_stream()),
            EdgeSpec::Anon(expr) => format!("anon:{}", expr.to_token_stream()),
            EdgeSpec::Empty => "empty".to_string(),
        }
    };

    // Generate let-bindings for named node values (before graph creation).
    // This moves/copies the outer scope variables into the block.
    let mut named_let_keys = Vec::<String>::new();
    let mut named_lets = Vec::<TokenStream2>::new();
    for decl in &input.decls {
        for spec in [&decl.src, &decl.dst] {
            match spec {
                NodeSpec::Named(ident) => {
                    let key = ident.to_string();
                    if !named_let_keys.contains(&key) {
                        let id = ident.clone();
                        named_lets.push(quote! { let #id = #id; });
                        named_let_keys.push(key);
                    }
                }
                NodeSpec::NamedExpr(_, _) | NodeSpec::Anon(_) => {}
            }
        }
    }

    // Also capture named edge values before graph creation.
    let mut named_edge_let_keys = Vec::<String>::new();
    let mut named_edge_lets = Vec::<TokenStream2>::new();
    for decl in &input.decls {
        match &decl.edge {
            EdgeSpec::Named(ident) => {
                let key = ident.to_string();
                if !named_edge_let_keys.contains(&key) {
                    let id = ident.clone();
                    named_edge_lets.push(quote! { let #id = #id; });
                    named_edge_let_keys.push(key);
                }
            }
            EdgeSpec::NamedExpr(_, _) | EdgeSpec::Anon(_) | EdgeSpec::Empty => {}
        }
    }

    // Node insertion statements and per-declaration endpoint vars.
    let mut node_stmts = Vec::<TokenStream2>::new();
    let mut decl_endpoint_vars = Vec::<(syn::Ident, syn::Ident)>::new();
    let mut named_node_defined = std::collections::HashMap::<String, String>::new();
    let mut named_node_span = std::collections::HashMap::<String, proc_macro2::Span>::new();
    let mut node_counter: usize = 0;

    let mut ensure_node = |spec: &NodeSpec| -> syn::Ident {
        match spec {
            NodeSpec::Named(ident) | NodeSpec::NamedExpr(ident, _) => {
                let key = ident.to_string();
                let sig = node_payload_sig(spec).unwrap();
                if let Some(prev) = named_node_defined.get(&key) {
                    if prev != &sig {
                        let first_span = named_node_span[&key];
                        emit_error!(
                            ident.span(),
                            "node `{}` has conflicting definitions",
                            key;
                            help = first_span => "first definition for node `{}` is here", key
                        );
                    }
                    ident.clone()
                } else {
                    let (stmt, _) =
                        gen_node_insert(spec, crate_path, &graph_access, &mut node_counter);
                    node_stmts.push(stmt);
                    named_node_defined.insert(key.clone(), sig);
                    named_node_span.insert(key, ident.span());
                    ident.clone()
                }
            }
            NodeSpec::Anon(_) => {
                let tmp = syn::Ident::new(
                    &format!("__node_{}", node_counter),
                    proc_macro2::Span::call_site(),
                );
                let (stmt, _) = gen_node_insert(spec, crate_path, &graph_access, &mut node_counter);
                node_stmts.push(stmt);
                tmp
            }
        }
    };

    for decl in &input.decls {
        let src_var = ensure_node(&decl.src);
        let dst_var = ensure_node(&decl.dst);
        decl_endpoint_vars.push((src_var, dst_var));
    }

    // Edge insertion statements.
    let mut edge_stmts = Vec::<TokenStream2>::new();
    let mut any_named_node = false;
    let mut any_named_edge = false;
    let mut named_node_span: Option<proc_macro2::Span> = None;
    let mut named_edge_span: Option<proc_macro2::Span> = None;
    let mut edge_counter: usize = 0;
    let mut named_edge_defined =
        std::collections::HashMap::<String, (String, proc_macro2::Span)>::new();

    for (decl_ix, decl) in input.decls.iter().enumerate() {
        let (src_var, dst_var) = &decl_endpoint_vars[decl_ix];

        // Track named flags.
        if matches!(&decl.src, NodeSpec::Named(_) | NodeSpec::NamedExpr(_, _))
            || matches!(&decl.dst, NodeSpec::Named(_) | NodeSpec::NamedExpr(_, _))
        {
            any_named_node = true;
            if named_node_span.is_none() {
                named_node_span = match (&decl.src, &decl.dst) {
                    (NodeSpec::Named(id), _) => Some(id.span()),
                    (_, NodeSpec::Named(id)) => Some(id.span()),
                    (NodeSpec::NamedExpr(id, _), _) => Some(id.span()),
                    (_, NodeSpec::NamedExpr(id, _)) => Some(id.span()),
                    _ => None,
                };
            }
        }

        match &decl.edge {
            EdgeSpec::Named(ident) => {
                any_named_edge = true;
                if named_edge_span.is_none() {
                    named_edge_span = Some(ident.span());
                }
                let sig = format!("{}|{}|{}", src_var, edge_payload_sig(&decl.edge), dst_var);
                let key = ident.to_string();
                if let Some((prev_sig, first_span)) = named_edge_defined.get(&key) {
                    if prev_sig != &sig {
                        emit_error!(
                            ident.span(),
                            "edge `{}` is used for different edges",
                            key;
                            help = *first_span => "first definition for edge `{}` is here", key
                        );
                    }
                    continue;
                }
                named_edge_defined.insert(key, (sig, ident.span()));
                let id = ident.clone();
                edge_stmts.push(quote! {
                    let #id = unsafe {
                        #crate_path::graph::capability::InsertEdge::insert_edge_unchecked(
                            #graph_access, #id, __directed_endpoints(#src_var, #dst_var)
                        ).unwrap()
                    };
                });
            }
            EdgeSpec::NamedExpr(ident, expr) => {
                any_named_edge = true;
                if named_edge_span.is_none() {
                    named_edge_span = Some(ident.span());
                }
                let sig = format!("{}|{}|{}", src_var, edge_payload_sig(&decl.edge), dst_var);
                let key = ident.to_string();
                if let Some((prev_sig, first_span)) = named_edge_defined.get(&key) {
                    if prev_sig != &sig {
                        emit_error!(
                            ident.span(),
                            "edge `{}` is used for different edges",
                            key;
                            help = *first_span => "first definition for edge `{}` is here", key
                        );
                    }
                    continue;
                }
                named_edge_defined.insert(key, (sig, ident.span()));
                let id = ident.clone();
                edge_stmts.push(quote! {
                    let #id = unsafe {
                        #crate_path::graph::capability::InsertEdge::insert_edge_unchecked(
                            #graph_access, #expr, __directed_endpoints(#src_var, #dst_var)
                        ).unwrap()
                    };
                });
            }
            EdgeSpec::Anon(expr) => {
                let tmp = syn::Ident::new(
                    &format!("__edge_{}", edge_counter),
                    proc_macro2::Span::call_site(),
                );
                edge_counter += 1;
                edge_stmts.push(quote! {
                    let #tmp = unsafe {
                        #crate_path::graph::capability::InsertEdge::insert_edge_unchecked(
                            #graph_access, #expr, __directed_endpoints(#src_var, #dst_var)
                        ).unwrap()
                    };
                });
            }
            EdgeSpec::Empty => {
                let tmp = syn::Ident::new(
                    &format!("__edge_{}", edge_counter),
                    proc_macro2::Span::call_site(),
                );
                edge_counter += 1;
                edge_stmts.push(quote! {
                    let #tmp = unsafe {
                        #crate_path::graph::capability::InsertEdge::insert_edge_unchecked(
                            #graph_access, Default::default(), __directed_endpoints(#src_var, #dst_var)
                        ).unwrap()
                    };
                });
            }
        }
    }

    let assert_node_ident = syn::Ident::new(
        "__assert_stable_node",
        named_node_span.unwrap_or(proc_macro2::Span::call_site()),
    );
    let assert_edge_ident = syn::Ident::new(
        "__assert_stable_edge",
        named_edge_span.unwrap_or(proc_macro2::Span::call_site()),
    );

    let output = if let Some(init_expr) = &input.initial {
        // With input graph:
        // - emit statements in caller scope (export named node/edge idents)
        // - do not return graph instance
        // - enforce StableNode/StableEdge only in this mode
        quote! {
            #(#named_lets)*
            #(#named_edge_lets)*
            fn __directed_endpoints<E: #crate_path::graph::edge::Endpoints>(
                source: E::NodeIx,
                target: E::NodeIx,
            ) -> E {
                E::try_from_sources_targets(core::iter::once(source), core::iter::once(target))
                    .expect("failed to construct endpoints from source/target")
            }
            let mut __graph = #init_expr;
            #(#node_stmts)*
            #(#edge_stmts)*
            #(if any_named_node) {
                fn #assert_node_ident<G: #crate_path::graph::capability::StableNode>(_: &G) {}
                #assert_node_ident(&*__graph);
            }
            #(if any_named_edge) {
                fn #assert_edge_ident<G: #crate_path::graph::capability::StableEdge>(_: &G) {}
                #assert_edge_ident(&*__graph);
            }
        }
    } else {
        // Without input graph, behave as expression and return the created graph.
        quote! {
            {
                #(#named_lets)*
                #(#named_edge_lets)*
                fn __directed_endpoints<E: #crate_path::graph::edge::Endpoints>(
                    source: E::NodeIx,
                    target: E::NodeIx,
                ) -> E {
                    E::try_from_sources_targets(core::iter::once(source), core::iter::once(target))
                        .expect("failed to construct endpoints from source/target")
                }
                let mut __graph = Default::default();
                #(#node_stmts)*
                #(#edge_stmts)*
                __graph
            }
        }
    };

    // Use the full expansion as dummy output when emitting errors.
    set_dummy(output.clone());

    abort_if_dirty();

    output.into()
}
