//! Expand `__builtin_spawn` before VLIR lowering: outer `for` over blocks (`__spawn_b`)
//! with `T` sequential thread copies using `name_t` locals (`name_0`, `name_1`, …).
//! Grid builtins become `__spawn_b`, thread constants, and `block_dim` constant.
//! `__builtin_sync()` splits the target body into barrier-separated segments.

use std::collections::{HashMap, HashSet};

use lang_c::ast::{
    BinaryOperator, BinaryOperatorExpression, BlockItem, CallExpression, Constant, Declaration,
    DeclarationSpecifier, Declarator, DeclaratorKind, DerivedDeclarator, Expression,
    ExternalDeclaration, ForInitializer, ForStatement, FunctionDefinition, Identifier,
    InitDeclarator, Initializer, Integer, IntegerBase, IntegerSize, IntegerSuffix, Statement,
    TranslationUnit, TypeSpecifier, UnaryOperator, UnaryOperatorExpression,
};
use lang_c::span::{Node, Span};

const BLOCK_IDX_VAR: &str = "__spawn_b";

#[derive(Debug)]
pub enum SpawnExpandError {
    Msg(String),
}

impl std::fmt::Display for SpawnExpandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpawnExpandError::Msg(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for SpawnExpandError {}

fn err(s: impl Into<String>) -> SpawnExpandError {
    SpawnExpandError::Msg(s.into())
}

fn span() -> Span {
    Span::none()
}

/// Run on the parsed translation unit (before lowering). Rewrites `kernel()` only.
pub fn expand_spawn_in_kernel(unit: &mut TranslationUnit) -> Result<(), SpawnExpandError> {
    let fns = collect_function_definitions(unit);
    let globals = collect_global_constants(unit);
    for ext in &mut unit.0 {
        if let ExternalDeclaration::FunctionDefinition(def) = &mut ext.node {
            let name = declarator_name(&def.node.declarator)?;
            if name != "kernel" {
                continue;
            }
            let new_stmt = expand_spawns_in_statement(&def.node.statement.node, &fns, &globals)?;
            def.node.statement = Node::new(new_stmt, def.node.statement.span);
        }
    }
    Ok(())
}

fn collect_function_definitions(
    unit: &TranslationUnit,
) -> HashMap<String, Node<FunctionDefinition>> {
    let mut m = HashMap::new();
    for ext in &unit.0 {
        if let ExternalDeclaration::FunctionDefinition(def) = &ext.node {
            if let Ok(name) = declarator_name(&def.node.declarator) {
                m.insert(name, def.clone());
            }
        }
    }
    m
}

fn collect_global_constants(unit: &TranslationUnit) -> HashMap<String, i32> {
    let mut out = HashMap::new();
    for ext in &unit.0 {
        if let ExternalDeclaration::Declaration(d) = &ext.node {
            for idecl in &d.node.declarators {
                let Ok(name) = declarator_name(&idecl.node.declarator) else {
                    continue;
                };
                let Some(init) = &idecl.node.initializer else {
                    continue;
                };
                let Initializer::Expression(e) = &init.node else {
                    continue;
                };
                if let Some(v) = try_compile_time_i32(&e.node) {
                    out.insert(name, v);
                }
            }
        }
    }
    out
}

fn declarator_name(d: &Node<Declarator>) -> Result<String, SpawnExpandError> {
    match &d.node.kind.node {
        DeclaratorKind::Identifier(id) => Ok(id.node.name.clone()),
        DeclaratorKind::Declarator(inner) => declarator_name(inner),
        DeclaratorKind::Abstract => Err(err("abstract declarator")),
    }
}

fn expand_spawns_in_statement(
    stmt: &Statement,
    fns: &HashMap<String, Node<FunctionDefinition>>,
    globals: &HashMap<String, i32>,
) -> Result<Statement, SpawnExpandError> {
    match stmt {
        Statement::Compound(items) => {
            let mut out: Vec<Node<BlockItem>> = Vec::new();
            for item in items {
                match &item.node {
                    BlockItem::Statement(s) => {
                        if let Some(blocks) = try_expand_spawn_statement(&s.node, fns, globals)? {
                            for bi in blocks {
                                out.push(bi);
                            }
                        } else {
                            let u = expand_spawns_in_statement(&s.node, fns, globals)?;
                            out.push(Node::new(
                                BlockItem::Statement(Node::new(u, s.span)),
                                item.span,
                            ));
                        }
                    }
                    BlockItem::Declaration(d) => {
                        out.push(Node::new(
                            BlockItem::Declaration(expand_spawns_in_declaration(d, fns)?),
                            item.span,
                        ));
                    }
                    BlockItem::StaticAssert(_s) => out.push(item.clone()),
                }
            }
            Ok(Statement::Compound(out))
        }
        Statement::If(i) => {
            let then_s = expand_spawns_in_statement(&i.node.then_statement.node, fns, globals)?;
            let else_s = i
                .node
                .else_statement
                .as_ref()
                .map(|e| {
                    expand_spawns_in_statement(&e.node, fns, globals)
                        .map(|s| Box::new(Node::new(s, e.span)))
                })
                .transpose()?;
            Ok(Statement::If(Node::new(
                lang_c::ast::IfStatement {
                    condition: i.node.condition.clone(),
                    then_statement: Box::new(Node::new(then_s, i.node.then_statement.span)),
                    else_statement: else_s,
                },
                i.span,
            )))
        }
        Statement::For(f) => {
            let st = expand_spawns_in_statement(&f.node.statement.node, fns, globals)?;
            Ok(Statement::For(Node::new(
                ForStatement {
                    initializer: f.node.initializer.clone(),
                    condition: f.node.condition.clone(),
                    step: f.node.step.clone(),
                    statement: Box::new(Node::new(st, f.node.statement.span)),
                },
                f.span,
            )))
        }
        Statement::Expression(opt) => Ok(Statement::Expression(
            opt.as_ref()
                .map(|e| {
                    let ex = expand_spawns_in_expr(&e.node, fns)?;
                    Ok(Box::new(Node::new(ex, e.span)))
                })
                .transpose()?,
        )),
        Statement::Return(opt) => Ok(Statement::Return(
            opt.as_ref()
                .map(|e| {
                    let ex = expand_spawns_in_expr(&e.node, fns)?;
                    Ok(Box::new(Node::new(ex, e.span)))
                })
                .transpose()?,
        )),
        _ => Ok(stmt.clone()),
    }
}

fn expand_spawns_in_declaration(
    d: &Node<Declaration>,
    fns: &HashMap<String, Node<FunctionDefinition>>,
) -> Result<Node<Declaration>, SpawnExpandError> {
    let mut decls = Vec::new();
    for idecl in &d.node.declarators {
        let init = idecl
            .node
            .initializer
            .as_ref()
            .map(|i| match &i.node {
                Initializer::Expression(e) => {
                    let ex = expand_spawns_in_expr(&e.node, fns)?;
                    Ok(Node::new(
                        Initializer::Expression(Box::new(Node::new(ex, e.span))),
                        i.span,
                    ))
                }
                Initializer::List(_) => Err(err("spawn expand: list initializer not supported")),
            })
            .transpose()?;
        decls.push(Node::new(
            InitDeclarator {
                declarator: idecl.node.declarator.clone(),
                initializer: init,
            },
            idecl.span,
        ));
    }
    Ok(Node::new(
        Declaration {
            specifiers: d.node.specifiers.clone(),
            declarators: decls,
        },
        d.span,
    ))
}

fn expand_spawns_in_expr(
    e: &Expression,
    fns: &HashMap<String, Node<FunctionDefinition>>,
) -> Result<Expression, SpawnExpandError> {
    match e {
        Expression::Call(c) => Ok(Expression::Call(Box::new(Node::new(
            CallExpression {
                callee: Box::new(Node::new(
                    expand_spawns_in_expr(&c.node.callee.node, fns)?,
                    c.node.callee.span,
                )),
                arguments: c
                    .node
                    .arguments
                    .iter()
                    .map(|a| expand_spawns_in_expr(&a.node, fns).map(|x| Node::new(x, a.span)))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            c.span,
        )))),
        Expression::BinaryOperator(b) => Ok(Expression::BinaryOperator(Box::new(Node::new(
            BinaryOperatorExpression {
                operator: b.node.operator.clone(),
                lhs: Box::new(Node::new(
                    expand_spawns_in_expr(&b.node.lhs.node, fns)?,
                    b.node.lhs.span,
                )),
                rhs: Box::new(Node::new(
                    expand_spawns_in_expr(&b.node.rhs.node, fns)?,
                    b.node.rhs.span,
                )),
            },
            b.span,
        )))),
        Expression::UnaryOperator(u) => Ok(Expression::UnaryOperator(Box::new(Node::new(
            UnaryOperatorExpression {
                operator: u.node.operator.clone(),
                operand: Box::new(Node::new(
                    expand_spawns_in_expr(&u.node.operand.node, fns)?,
                    u.node.operand.span,
                )),
            },
            u.span,
        )))),
        Expression::Cast(c) => Ok(Expression::Cast(Box::new(Node::new(
            lang_c::ast::CastExpression {
                type_name: c.node.type_name.clone(),
                expression: Box::new(Node::new(
                    expand_spawns_in_expr(&c.node.expression.node, fns)?,
                    c.node.expression.span,
                )),
            },
            c.span,
        )))),
        Expression::Conditional(c) => Ok(Expression::Conditional(Box::new(Node::new(
            lang_c::ast::ConditionalExpression {
                condition: Box::new(Node::new(
                    expand_spawns_in_expr(&c.node.condition.node, fns)?,
                    c.node.condition.span,
                )),
                then_expression: Box::new(Node::new(
                    expand_spawns_in_expr(&c.node.then_expression.node, fns)?,
                    c.node.then_expression.span,
                )),
                else_expression: Box::new(Node::new(
                    expand_spawns_in_expr(&c.node.else_expression.node, fns)?,
                    c.node.else_expression.span,
                )),
            },
            c.span,
        )))),
        _ => Ok(e.clone()),
    }
}

fn try_expand_spawn_statement(
    stmt: &Statement,
    fns: &HashMap<String, Node<FunctionDefinition>>,
    globals: &HashMap<String, i32>,
) -> Result<Option<Vec<Node<BlockItem>>>, SpawnExpandError> {
    let Statement::Expression(Some(e)) = stmt else {
        return Ok(None);
    };
    let Expression::Call(c) = &e.node else {
        return Ok(None);
    };
    let Expression::Identifier(id) = &c.node.callee.node else {
        return Ok(None);
    };
    if id.node.name != "__builtin_spawn" {
        return Ok(None);
    }
    Ok(Some(expand_spawn_call(c.as_ref(), fns, globals)?))
}

fn expand_spawn_call(
    c: &Node<CallExpression>,
    fns: &HashMap<String, Node<FunctionDefinition>>,
    globals: &HashMap<String, i32>,
) -> Result<Vec<Node<BlockItem>>, SpawnExpandError> {
    let args = &c.node.arguments;
    if args.len() < 3 {
        return Err(err("__builtin_spawn: expected blocks, threads, fn, ..."));
    }
    let num_blocks = try_compile_time_i32(&args[0].node)
        .ok_or_else(|| err("__builtin_spawn: block count must be compile-time constant"))?;
    let num_threads = try_compile_time_i32(&args[1].node)
        .ok_or_else(|| err("__builtin_spawn: thread count must be compile-time constant"))?;
    if num_blocks < 0 || num_threads < 1 || num_threads > 64 {
        return Err(err("__builtin_spawn: invalid block or thread count"));
    }
    let b_max = num_blocks as usize;
    let t_max = num_threads as usize;

    let target_name = match &args[2].node {
        Expression::Identifier(id) => id.node.name.clone(),
        _ => {
            return Err(err(
                "__builtin_spawn: third argument must be the target function name",
            ));
        }
    };
    let fn_def = fns
        .get(&target_name)
        .ok_or_else(|| err(format!("unknown spawn target `{target_name}`")))?;

    let params = function_parameters(&fn_def.node)?;
    if args.len() != 3 + params.len() {
        return Err(err(
            "__builtin_spawn: argument count must match target function parameters",
        ));
    }

    let mut param_subst: HashMap<String, Node<Expression>> = HashMap::new();
    for (i, (pname, _is_vec)) in params.iter().enumerate() {
        param_subst.insert(pname.clone(), args[3 + i].clone());
    }

    let unrolled_body = crate::pass::loop_unroll::unroll_statement(&fn_def.node.statement.node, globals);
    let body_items = function_body_to_block_items(&unrolled_body)?;
    let vector_param_count = params.iter().filter(|(_, is_vec)| *is_vec).count();
    // Interleaving threads across sync-separated segments maximizes ILP but keeps per-thread state
    // live across all segments. For spawn targets with many vector params this can exceed scratch.
    let interleave_across_sync = vector_param_count <= 8;
    let segments = if interleave_across_sync {
        split_top_level_at_sync(&body_items)?
    } else {
        vec![body_items]
    };
    let mut local_names: HashSet<String> = HashSet::new();
    collect_local_names_in_statement(&fn_def.node.statement.node, &mut local_names);

    let mut out_items: Vec<Node<BlockItem>> = Vec::new();
    for (seg_i, seg) in segments.iter().enumerate() {
        if seg.is_empty() {
            continue;
        }
        let seg_stmt = Statement::Compound(seg.clone());
        let mut thread_items: Vec<Node<BlockItem>> = Vec::new();
        for t in 0..t_max {
            let mut s = seg_stmt.clone();
            s = subst_params_in_statement(&s, &param_subst)?;
            s = subst_grid_builtins_in_statement(&s, num_threads, t as i32)?;
            s = rename_locals_in_statement(&s, &local_names, t as i32)?;
            match s {
                Statement::Compound(items) => {
                    for bi in items {
                        thread_items.push(bi);
                    }
                }
                other => thread_items.push(Node::new(
                    BlockItem::Statement(Node::new(other, span())),
                    span(),
                )),
            }
        }
        let inner = Statement::Compound(thread_items);
        let for_stmt = make_block_for_loop(b_max, inner)?;
        out_items.push(Node::new(
            BlockItem::Statement(Node::new(for_stmt, span())),
            span(),
        ));
        if interleave_across_sync && seg_i + 1 < segments.len() {
            out_items.push(sync_expr_statement()?);
        }
    }
    Ok(out_items)
}

fn make_block_for_loop(num_blocks: usize, body: Statement) -> Result<Statement, SpawnExpandError> {
    if num_blocks == 0 {
        return Ok(Statement::Compound(vec![]));
    }
    let init_decl = uint32_decl_with_init(BLOCK_IDX_VAR, 0)?;
    let cond = Expression::BinaryOperator(Box::new(Node::new(
        BinaryOperatorExpression {
            operator: Node::new(BinaryOperator::Less, span()),
            lhs: ident_expr(BLOCK_IDX_VAR),
            rhs: Box::new(Node::new(int_expr(num_blocks as i32), span())),
        },
        span(),
    )));
    let step = Expression::UnaryOperator(Box::new(Node::new(
        UnaryOperatorExpression {
            operator: Node::new(UnaryOperator::PostIncrement, span()),
            operand: ident_expr(BLOCK_IDX_VAR),
        },
        span(),
    )));
    Ok(Statement::For(Node::new(
        ForStatement {
            initializer: Node::new(ForInitializer::Declaration(init_decl), span()),
            condition: Some(Box::new(Node::new(cond, span()))),
            step: Some(Box::new(Node::new(step, span()))),
            statement: Box::new(Node::new(body, span())),
        },
        span(),
    )))
}

fn uint32_decl_with_init(name: &str, val: i32) -> Result<Node<Declaration>, SpawnExpandError> {
    let spec = Node::new(
        DeclarationSpecifier::TypeSpecifier(Node::new(
            TypeSpecifier::TypedefName(Node::new(
                Identifier {
                    name: "uint32_t".into(),
                },
                span(),
            )),
            span(),
        )),
        span(),
    );
    let decl = Declarator {
        kind: Node::new(
            DeclaratorKind::Identifier(Node::new(Identifier { name: name.into() }, span())),
            span(),
        ),
        derived: vec![],
        extensions: vec![],
    };
    let init = InitDeclarator {
        declarator: Node::new(decl, span()),
        initializer: Some(Node::new(
            Initializer::Expression(Box::new(Node::new(int_expr(val), span()))),
            span(),
        )),
    };
    Ok(Node::new(
        Declaration {
            specifiers: vec![spec],
            declarators: vec![Node::new(init, span())],
        },
        span(),
    ))
}

fn ident_expr(name: &str) -> Box<Node<Expression>> {
    Box::new(Node::new(
        Expression::Identifier(Box::new(Node::new(
            Identifier { name: name.into() },
            span(),
        ))),
        span(),
    ))
}

fn int_expr(v: i32) -> Expression {
    Expression::Constant(Box::new(Node::new(
        Constant::Integer(Integer {
            base: IntegerBase::Decimal,
            number: v.to_string().into_boxed_str(),
            suffix: IntegerSuffix {
                size: IntegerSize::Int,
                unsigned: false,
                imaginary: false,
            },
        }),
        span(),
    )))
}

fn sync_expr_statement() -> Result<Node<BlockItem>, SpawnExpandError> {
    let call = Expression::Call(Box::new(Node::new(
        CallExpression {
            callee: Box::new(Node::new(
                Expression::Identifier(Box::new(Node::new(
                    Identifier {
                        name: "__builtin_sync".into(),
                    },
                    span(),
                ))),
                span(),
            )),
            arguments: vec![],
        },
        span(),
    )));
    Ok(Node::new(
        BlockItem::Statement(Node::new(
            Statement::Expression(Some(Box::new(Node::new(call, span())))),
            span(),
        )),
        span(),
    ))
}

fn function_body_to_block_items(
    stmt: &Statement,
) -> Result<Vec<Node<BlockItem>>, SpawnExpandError> {
    fn flatten_items(items: &[Node<BlockItem>], out: &mut Vec<Node<BlockItem>>) {
        for item in items {
            match &item.node {
                BlockItem::Statement(s) => match &s.node {
                    Statement::Compound(sub) => flatten_items(sub, out),
                    _ => out.push(item.clone()),
                },
                _ => out.push(item.clone()),
            }
        }
    }

    match stmt {
        Statement::Compound(items) => {
            let mut out = Vec::new();
            flatten_items(items, &mut out);
            Ok(out)
        }
        _ => Ok(vec![Node::new(
            BlockItem::Statement(Node::new(stmt.clone(), span())),
            span(),
        )]),
    }
}

/// Split top-level compound items at `__builtin_sync()` expression statements (sync removed from output).
fn split_top_level_at_sync(
    items: &[Node<BlockItem>],
) -> Result<Vec<Vec<Node<BlockItem>>>, SpawnExpandError> {
    let mut segments: Vec<Vec<Node<BlockItem>>> = vec![vec![]];
    for item in items {
        if is_sync_expr_statement(item) {
            if segments.last().map(|s| s.is_empty()).unwrap_or(false) && segments.len() == 1 {
                return Err(err(
                    "__builtin_sync: barrier at start of function body is invalid",
                ));
            }
            segments.push(vec![]);
            continue;
        }
        segments.last_mut().unwrap().push(item.clone());
    }
    // if segments.iter().any(|s| s.is_empty()) {
    //     return Err(err(
    //         "__builtin_sync: empty segment (duplicate or trailing sync?)",
    //     ));
    // }
    Ok(segments)
}

fn is_sync_expr_statement(item: &Node<BlockItem>) -> bool {
    matches!(
        &item.node,
        BlockItem::Statement(s)
            if matches!(
                &s.node,
                Statement::Expression(Some(e))
                    if matches!(
                        &e.node,
                        Expression::Call(c)
                            if matches!(
                                &c.node.callee.node,
                                Expression::Identifier(id) if id.node.name == "__builtin_sync"
                            )
                    )
            )
    )
}

fn function_parameters(def: &FunctionDefinition) -> Result<Vec<(String, bool)>, SpawnExpandError> {
    let decl = &def.declarator.node;
    for d in &decl.derived {
        if let DerivedDeclarator::Function(f) = &d.node {
            let mut params = Vec::new();
            for p in &f.node.parameters {
                let name = match &p.node.declarator {
                    Some(pd) => declarator_name(pd)?,
                    None => {
                        return Err(err("spawn target: parameters must be named"));
                    }
                };
                let is_vec = p.node.specifiers.iter().any(|spec| {
                    matches!(
                        &spec.node,
                        DeclarationSpecifier::TypeSpecifier(ts)
                            if matches!(
                                &ts.node,
                                TypeSpecifier::TypedefName(id) if id.node.name == "vec8_t"
                            )
                    )
                });
                params.push((name, is_vec));
            }
            return Ok(params);
        }
    }
    Err(err(
        "spawn target: expected function declarator with parameters",
    ))
}

fn collect_local_names_in_statement(stmt: &Statement, acc: &mut HashSet<String>) {
    match stmt {
        Statement::Compound(items) => {
            for item in items {
                match &item.node {
                    BlockItem::Declaration(d) => {
                        for idecl in &d.node.declarators {
                            if let Ok(n) = declarator_name(&idecl.node.declarator) {
                                acc.insert(n);
                            }
                        }
                    }
                    BlockItem::Statement(s) => collect_local_names_in_statement(&s.node, acc),
                    BlockItem::StaticAssert(_) => {}
                }
            }
        }
        Statement::For(f) => {
            match &f.node.initializer.node {
                ForInitializer::Declaration(d) => {
                    for idecl in &d.node.declarators {
                        if let Ok(n) = declarator_name(&idecl.node.declarator) {
                            acc.insert(n);
                        }
                    }
                }
                _ => {}
            }
            collect_local_names_in_statement(&f.node.statement.node, acc);
        }
        Statement::If(i) => {
            collect_local_names_in_statement(&i.node.then_statement.node, acc);
            if let Some(es) = &i.node.else_statement {
                collect_local_names_in_statement(&es.node, acc);
            }
        }
        Statement::While(w) => collect_local_names_in_statement(&w.node.statement.node, acc),
        Statement::DoWhile(d) => collect_local_names_in_statement(&d.node.statement.node, acc),
        Statement::Switch(sw) => collect_local_names_in_statement(&sw.node.statement.node, acc),
        _ => {}
    }
}

fn try_compile_time_i32(e: &Expression) -> Option<i32> {
    match e {
        Expression::Constant(c) => parse_i32_constant(&c.node).ok(),
        Expression::Cast(c) => try_compile_time_i32(&c.node.expression.node),
        _ => None,
    }
}

fn parse_i32_constant(c: &Constant) -> Result<i32, ()> {
    match c {
        Constant::Integer(i) => {
            let s = i.number.as_ref();
            if s.starts_with("0x") || s.starts_with("0X") {
                i64::from_str_radix(&s[2..], 16)
                    .map(|v| v as i32)
                    .map_err(|_| ())
            } else {
                s.parse::<i64>().map(|v| v as i32).map_err(|_| ())
            }
        }
        _ => Err(()),
    }
}

fn subst_params_in_statement(
    stmt: &Statement,
    subst: &HashMap<String, Node<Expression>>,
) -> Result<Statement, SpawnExpandError> {
    match stmt {
        Statement::Compound(items) => {
            let mut v = Vec::new();
            for item in items {
                v.push(Node::new(
                    match &item.node {
                        BlockItem::Statement(s) => BlockItem::Statement(Node::new(
                            subst_params_in_statement(&s.node, subst)?,
                            s.span,
                        )),
                        BlockItem::Declaration(d) => {
                            BlockItem::Declaration(subst_params_in_declaration(d, subst)?)
                        }
                        BlockItem::StaticAssert(s) => BlockItem::StaticAssert(s.clone()),
                    },
                    item.span,
                ));
            }
            Ok(Statement::Compound(v))
        }
        Statement::Expression(e) => Ok(Statement::Expression(
            e.as_ref()
                .map(|x| {
                    Ok(Box::new(Node::new(
                        subst_params_in_expr(&x.node, subst)?,
                        x.span,
                    )))
                })
                .transpose()?,
        )),
        Statement::If(i) => Ok(Statement::If(Node::new(
            lang_c::ast::IfStatement {
                condition: Box::new(Node::new(
                    subst_params_in_expr(&i.node.condition.node, subst)?,
                    i.node.condition.span,
                )),
                then_statement: Box::new(Node::new(
                    subst_params_in_statement(&i.node.then_statement.node, subst)?,
                    i.node.then_statement.span,
                )),
                else_statement: i
                    .node
                    .else_statement
                    .as_ref()
                    .map(|e| {
                        subst_params_in_statement(&e.node, subst)
                            .map(|s| Box::new(Node::new(s, e.span)))
                    })
                    .transpose()?,
            },
            i.span,
        ))),
        Statement::For(f) => Ok(Statement::For(Node::new(
            ForStatement {
                initializer: subst_params_in_for_init(&f.node.initializer, subst)?,
                condition: f
                    .node
                    .condition
                    .as_ref()
                    .map(|c| {
                        subst_params_in_expr(&c.node, subst).map(|e| Box::new(Node::new(e, c.span)))
                    })
                    .transpose()?,
                step: f
                    .node
                    .step
                    .as_ref()
                    .map(|s| {
                        subst_params_in_expr(&s.node, subst).map(|e| Box::new(Node::new(e, s.span)))
                    })
                    .transpose()?,
                statement: Box::new(Node::new(
                    subst_params_in_statement(&f.node.statement.node, subst)?,
                    f.node.statement.span,
                )),
            },
            f.span,
        ))),
        Statement::Return(e) => Ok(Statement::Return(
            e.as_ref()
                .map(|x| {
                    Ok(Box::new(Node::new(
                        subst_params_in_expr(&x.node, subst)?,
                        x.span,
                    )))
                })
                .transpose()?,
        )),
        _ => Err(err("spawn expand: unsupported statement in spawn target")),
    }
}

fn subst_params_in_for_init(
    init: &Node<ForInitializer>,
    subst: &HashMap<String, Node<Expression>>,
) -> Result<Node<ForInitializer>, SpawnExpandError> {
    Ok(Node::new(
        match &init.node {
            ForInitializer::Empty => ForInitializer::Empty,
            ForInitializer::Expression(e) => ForInitializer::Expression(Box::new(Node::new(
                subst_params_in_expr(&e.node, subst)?,
                e.span,
            ))),
            ForInitializer::Declaration(d) => {
                ForInitializer::Declaration(subst_params_in_declaration(d, subst)?)
            }
            ForInitializer::StaticAssert(_) => {
                return Err(err("spawn expand: static assert in for init not supported"));
            }
        },
        init.span,
    ))
}

fn subst_params_in_declaration(
    d: &Node<Declaration>,
    subst: &HashMap<String, Node<Expression>>,
) -> Result<Node<Declaration>, SpawnExpandError> {
    let mut decls = Vec::new();
    for idecl in &d.node.declarators {
        let init = idecl
            .node
            .initializer
            .as_ref()
            .map(|i| match &i.node {
                Initializer::Expression(e) => Ok(Node::new(
                    Initializer::Expression(Box::new(Node::new(
                        subst_params_in_expr(&e.node, subst)?,
                        e.span,
                    ))),
                    i.span,
                )),
                Initializer::List(_) => Err(err("spawn expand: list initializer not supported")),
            })
            .transpose()?;
        decls.push(Node::new(
            InitDeclarator {
                declarator: idecl.node.declarator.clone(),
                initializer: init,
            },
            idecl.span,
        ));
    }
    Ok(Node::new(
        Declaration {
            specifiers: d.node.specifiers.clone(),
            declarators: decls,
        },
        d.span,
    ))
}

fn subst_params_in_expr(
    e: &Expression,
    subst: &HashMap<String, Node<Expression>>,
) -> Result<Expression, SpawnExpandError> {
    Ok(match e {
        Expression::Identifier(id) => {
            if let Some(rep) = subst.get(&id.node.name) {
                return Ok(rep.node.clone());
            }
            e.clone()
        }
        Expression::Call(c) => Expression::Call(Box::new(Node::new(
            CallExpression {
                callee: Box::new(Node::new(
                    subst_params_in_expr(&c.node.callee.node, subst)?,
                    c.node.callee.span,
                )),
                arguments: c
                    .node
                    .arguments
                    .iter()
                    .map(|a| subst_params_in_expr(&a.node, subst).map(|x| Node::new(x, a.span)))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            c.span,
        ))),
        Expression::BinaryOperator(b) => Expression::BinaryOperator(Box::new(Node::new(
            BinaryOperatorExpression {
                operator: b.node.operator.clone(),
                lhs: Box::new(Node::new(
                    subst_params_in_expr(&b.node.lhs.node, subst)?,
                    b.node.lhs.span,
                )),
                rhs: Box::new(Node::new(
                    subst_params_in_expr(&b.node.rhs.node, subst)?,
                    b.node.rhs.span,
                )),
            },
            b.span,
        ))),
        Expression::UnaryOperator(u) => Expression::UnaryOperator(Box::new(Node::new(
            UnaryOperatorExpression {
                operator: u.node.operator.clone(),
                operand: Box::new(Node::new(
                    subst_params_in_expr(&u.node.operand.node, subst)?,
                    u.node.operand.span,
                )),
            },
            u.span,
        ))),
        Expression::Cast(c) => Expression::Cast(Box::new(Node::new(
            lang_c::ast::CastExpression {
                type_name: c.node.type_name.clone(),
                expression: Box::new(Node::new(
                    subst_params_in_expr(&c.node.expression.node, subst)?,
                    c.node.expression.span,
                )),
            },
            c.span,
        ))),
        Expression::Conditional(c) => Expression::Conditional(Box::new(Node::new(
            lang_c::ast::ConditionalExpression {
                condition: Box::new(Node::new(
                    subst_params_in_expr(&c.node.condition.node, subst)?,
                    c.node.condition.span,
                )),
                then_expression: Box::new(Node::new(
                    subst_params_in_expr(&c.node.then_expression.node, subst)?,
                    c.node.then_expression.span,
                )),
                else_expression: Box::new(Node::new(
                    subst_params_in_expr(&c.node.else_expression.node, subst)?,
                    c.node.else_expression.span,
                )),
            },
            c.span,
        ))),
        _ => e.clone(),
    })
}

fn subst_grid_builtins_in_statement(
    stmt: &Statement,
    block_dim: i32,
    thread_idx: i32,
) -> Result<Statement, SpawnExpandError> {
    match stmt {
        Statement::Compound(items) => {
            let mut v = Vec::new();
            for item in items {
                v.push(Node::new(
                    match &item.node {
                        BlockItem::Statement(s) => BlockItem::Statement(Node::new(
                            subst_grid_builtins_in_statement(&s.node, block_dim, thread_idx)?,
                            s.span,
                        )),
                        BlockItem::Declaration(d) => BlockItem::Declaration(
                            subst_grid_builtins_in_declaration(d, block_dim, thread_idx)?,
                        ),
                        BlockItem::StaticAssert(s) => BlockItem::StaticAssert(s.clone()),
                    },
                    item.span,
                ));
            }
            Ok(Statement::Compound(v))
        }
        Statement::Expression(e) => Ok(Statement::Expression(
            e.as_ref()
                .map(|x| {
                    Ok(Box::new(Node::new(
                        subst_grid_builtins_in_expr(&x.node, block_dim, thread_idx)?,
                        x.span,
                    )))
                })
                .transpose()?,
        )),
        Statement::If(i) => Ok(Statement::If(Node::new(
            lang_c::ast::IfStatement {
                condition: Box::new(Node::new(
                    subst_grid_builtins_in_expr(&i.node.condition.node, block_dim, thread_idx)?,
                    i.node.condition.span,
                )),
                then_statement: Box::new(Node::new(
                    subst_grid_builtins_in_statement(
                        &i.node.then_statement.node,
                        block_dim,
                        thread_idx,
                    )?,
                    i.node.then_statement.span,
                )),
                else_statement: i
                    .node
                    .else_statement
                    .as_ref()
                    .map(|e| {
                        subst_grid_builtins_in_statement(&e.node, block_dim, thread_idx)
                            .map(|s| Box::new(Node::new(s, e.span)))
                    })
                    .transpose()?,
            },
            i.span,
        ))),
        Statement::For(f) => Ok(Statement::For(Node::new(
            ForStatement {
                initializer: subst_grid_builtins_in_for_init(
                    &f.node.initializer,
                    block_dim,
                    thread_idx,
                )?,
                condition: f
                    .node
                    .condition
                    .as_ref()
                    .map(|c| {
                        subst_grid_builtins_in_expr(&c.node, block_dim, thread_idx)
                            .map(|e| Box::new(Node::new(e, c.span)))
                    })
                    .transpose()?,
                step: f
                    .node
                    .step
                    .as_ref()
                    .map(|s| {
                        subst_grid_builtins_in_expr(&s.node, block_dim, thread_idx)
                            .map(|e| Box::new(Node::new(e, s.span)))
                    })
                    .transpose()?,
                statement: Box::new(Node::new(
                    subst_grid_builtins_in_statement(
                        &f.node.statement.node,
                        block_dim,
                        thread_idx,
                    )?,
                    f.node.statement.span,
                )),
            },
            f.span,
        ))),
        Statement::Return(e) => Ok(Statement::Return(
            e.as_ref()
                .map(|x| {
                    Ok(Box::new(Node::new(
                        subst_grid_builtins_in_expr(&x.node, block_dim, thread_idx)?,
                        x.span,
                    )))
                })
                .transpose()?,
        )),
        _ => Err(err("spawn expand: unsupported statement in spawn target")),
    }
}

fn subst_grid_builtins_in_for_init(
    init: &Node<ForInitializer>,
    block_dim: i32,
    thread_idx: i32,
) -> Result<Node<ForInitializer>, SpawnExpandError> {
    Ok(Node::new(
        match &init.node {
            ForInitializer::Empty => ForInitializer::Empty,
            ForInitializer::Expression(e) => ForInitializer::Expression(Box::new(Node::new(
                subst_grid_builtins_in_expr(&e.node, block_dim, thread_idx)?,
                e.span,
            ))),
            ForInitializer::Declaration(d) => ForInitializer::Declaration(
                subst_grid_builtins_in_declaration(d, block_dim, thread_idx)?,
            ),
            ForInitializer::StaticAssert(_) => {
                return Err(err("spawn expand: static assert in for init not supported"));
            }
        },
        init.span,
    ))
}

fn subst_grid_builtins_in_declaration(
    d: &Node<Declaration>,
    block_dim: i32,
    thread_idx: i32,
) -> Result<Node<Declaration>, SpawnExpandError> {
    let mut decls = Vec::new();
    for idecl in &d.node.declarators {
        let init = idecl
            .node
            .initializer
            .as_ref()
            .map(|i| match &i.node {
                Initializer::Expression(e) => Ok(Node::new(
                    Initializer::Expression(Box::new(Node::new(
                        subst_grid_builtins_in_expr(&e.node, block_dim, thread_idx)?,
                        e.span,
                    ))),
                    i.span,
                )),
                Initializer::List(_) => Err(err("spawn expand: list initializer not supported")),
            })
            .transpose()?;
        decls.push(Node::new(
            InitDeclarator {
                declarator: idecl.node.declarator.clone(),
                initializer: init,
            },
            idecl.span,
        ));
    }
    Ok(Node::new(
        Declaration {
            specifiers: d.node.specifiers.clone(),
            declarators: decls,
        },
        d.span,
    ))
}

fn subst_grid_builtins_in_expr(
    e: &Expression,
    block_dim: i32,
    thread_idx: i32,
) -> Result<Expression, SpawnExpandError> {
    if let Expression::Call(c) = e {
        if let Expression::Identifier(id) = &c.node.callee.node {
            match id.node.name.as_str() {
                "__builtin_block_dim" => {
                    if !c.node.arguments.is_empty() {
                        return Err(err("__builtin_block_dim takes no arguments"));
                    }
                    return Ok(int_expr(block_dim));
                }
                "__builtin_block_idx" => {
                    if !c.node.arguments.is_empty() {
                        return Err(err("__builtin_block_idx takes no arguments"));
                    }
                    return Ok(Expression::Identifier(Box::new(Node::new(
                        Identifier {
                            name: BLOCK_IDX_VAR.into(),
                        },
                        span(),
                    ))));
                }
                "__builtin_thread_idx" => {
                    if !c.node.arguments.is_empty() {
                        return Err(err("__builtin_thread_idx takes no arguments"));
                    }
                    return Ok(int_expr(thread_idx));
                }
                _ => {}
            }
        }
    }
    Ok(match e {
        Expression::Call(c) => Expression::Call(Box::new(Node::new(
            CallExpression {
                callee: Box::new(Node::new(
                    subst_grid_builtins_in_expr(&c.node.callee.node, block_dim, thread_idx)?,
                    c.node.callee.span,
                )),
                arguments: c
                    .node
                    .arguments
                    .iter()
                    .map(|a| {
                        subst_grid_builtins_in_expr(&a.node, block_dim, thread_idx)
                            .map(|x| Node::new(x, a.span))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
            c.span,
        ))),
        Expression::BinaryOperator(b) => Expression::BinaryOperator(Box::new(Node::new(
            BinaryOperatorExpression {
                operator: b.node.operator.clone(),
                lhs: Box::new(Node::new(
                    subst_grid_builtins_in_expr(&b.node.lhs.node, block_dim, thread_idx)?,
                    b.node.lhs.span,
                )),
                rhs: Box::new(Node::new(
                    subst_grid_builtins_in_expr(&b.node.rhs.node, block_dim, thread_idx)?,
                    b.node.rhs.span,
                )),
            },
            b.span,
        ))),
        Expression::UnaryOperator(u) => Expression::UnaryOperator(Box::new(Node::new(
            UnaryOperatorExpression {
                operator: u.node.operator.clone(),
                operand: Box::new(Node::new(
                    subst_grid_builtins_in_expr(&u.node.operand.node, block_dim, thread_idx)?,
                    u.node.operand.span,
                )),
            },
            u.span,
        ))),
        Expression::Cast(c) => Expression::Cast(Box::new(Node::new(
            lang_c::ast::CastExpression {
                type_name: c.node.type_name.clone(),
                expression: Box::new(Node::new(
                    subst_grid_builtins_in_expr(&c.node.expression.node, block_dim, thread_idx)?,
                    c.node.expression.span,
                )),
            },
            c.span,
        ))),
        Expression::Conditional(c) => Expression::Conditional(Box::new(Node::new(
            lang_c::ast::ConditionalExpression {
                condition: Box::new(Node::new(
                    subst_grid_builtins_in_expr(&c.node.condition.node, block_dim, thread_idx)?,
                    c.node.condition.span,
                )),
                then_expression: Box::new(Node::new(
                    subst_grid_builtins_in_expr(
                        &c.node.then_expression.node,
                        block_dim,
                        thread_idx,
                    )?,
                    c.node.then_expression.span,
                )),
                else_expression: Box::new(Node::new(
                    subst_grid_builtins_in_expr(
                        &c.node.else_expression.node,
                        block_dim,
                        thread_idx,
                    )?,
                    c.node.else_expression.span,
                )),
            },
            c.span,
        ))),
        _ => e.clone(),
    })
}

fn rename_locals_in_statement(
    stmt: &Statement,
    local_names: &HashSet<String>,
    thread_idx: i32,
) -> Result<Statement, SpawnExpandError> {
    match stmt {
        Statement::Compound(items) => {
            let mut v = Vec::new();
            for item in items {
                v.push(Node::new(
                    match &item.node {
                        BlockItem::Statement(s) => BlockItem::Statement(Node::new(
                            rename_locals_in_statement(&s.node, local_names, thread_idx)?,
                            s.span,
                        )),
                        BlockItem::Declaration(d) => BlockItem::Declaration(
                            rename_locals_in_declaration(d, local_names, thread_idx)?,
                        ),
                        BlockItem::StaticAssert(s) => BlockItem::StaticAssert(s.clone()),
                    },
                    item.span,
                ));
            }
            Ok(Statement::Compound(v))
        }
        Statement::Expression(e) => Ok(Statement::Expression(
            e.as_ref()
                .map(|x| {
                    Ok(Box::new(Node::new(
                        rename_locals_in_expr(&x.node, local_names, thread_idx)?,
                        x.span,
                    )))
                })
                .transpose()?,
        )),
        Statement::If(i) => Ok(Statement::If(Node::new(
            lang_c::ast::IfStatement {
                condition: Box::new(Node::new(
                    rename_locals_in_expr(&i.node.condition.node, local_names, thread_idx)?,
                    i.node.condition.span,
                )),
                then_statement: Box::new(Node::new(
                    rename_locals_in_statement(
                        &i.node.then_statement.node,
                        local_names,
                        thread_idx,
                    )?,
                    i.node.then_statement.span,
                )),
                else_statement: i
                    .node
                    .else_statement
                    .as_ref()
                    .map(|e| {
                        rename_locals_in_statement(&e.node, local_names, thread_idx)
                            .map(|s| Box::new(Node::new(s, e.span)))
                    })
                    .transpose()?,
            },
            i.span,
        ))),
        Statement::For(f) => Ok(Statement::For(Node::new(
            ForStatement {
                initializer: rename_locals_in_for_init(
                    &f.node.initializer,
                    local_names,
                    thread_idx,
                )?,
                condition: f
                    .node
                    .condition
                    .as_ref()
                    .map(|c| {
                        rename_locals_in_expr(&c.node, local_names, thread_idx)
                            .map(|e| Box::new(Node::new(e, c.span)))
                    })
                    .transpose()?,
                step: f
                    .node
                    .step
                    .as_ref()
                    .map(|s| {
                        rename_locals_in_expr(&s.node, local_names, thread_idx)
                            .map(|e| Box::new(Node::new(e, s.span)))
                    })
                    .transpose()?,
                statement: Box::new(Node::new(
                    rename_locals_in_statement(&f.node.statement.node, local_names, thread_idx)?,
                    f.node.statement.span,
                )),
            },
            f.span,
        ))),
        Statement::Return(e) => Ok(Statement::Return(
            e.as_ref()
                .map(|x| {
                    Ok(Box::new(Node::new(
                        rename_locals_in_expr(&x.node, local_names, thread_idx)?,
                        x.span,
                    )))
                })
                .transpose()?,
        )),
        _ => Err(err("spawn expand: unsupported statement in spawn target")),
    }
}

fn rename_locals_in_for_init(
    init: &Node<ForInitializer>,
    local_names: &HashSet<String>,
    thread_idx: i32,
) -> Result<Node<ForInitializer>, SpawnExpandError> {
    Ok(Node::new(
        match &init.node {
            ForInitializer::Empty => ForInitializer::Empty,
            ForInitializer::Expression(e) => ForInitializer::Expression(Box::new(Node::new(
                rename_locals_in_expr(&e.node, local_names, thread_idx)?,
                e.span,
            ))),
            ForInitializer::Declaration(d) => ForInitializer::Declaration(
                rename_locals_in_declaration(d, local_names, thread_idx)?,
            ),
            ForInitializer::StaticAssert(_) => {
                return Err(err("spawn expand: static assert in for init not supported"));
            }
        },
        init.span,
    ))
}

fn rename_locals_in_declaration(
    d: &Node<Declaration>,
    local_names: &HashSet<String>,
    thread_idx: i32,
) -> Result<Node<Declaration>, SpawnExpandError> {
    let mut decls = Vec::new();
    for idecl in &d.node.declarators {
        let new_decl = rename_declarator_tree(&idecl.node.declarator, local_names, thread_idx)?;
        let init = idecl
            .node
            .initializer
            .as_ref()
            .map(|i| match &i.node {
                Initializer::Expression(e) => Ok(Node::new(
                    Initializer::Expression(Box::new(Node::new(
                        rename_locals_in_expr(&e.node, local_names, thread_idx)?,
                        e.span,
                    ))),
                    i.span,
                )),
                Initializer::List(_) => Err(err("spawn expand: list initializer not supported")),
            })
            .transpose()?;
        decls.push(Node::new(
            InitDeclarator {
                declarator: new_decl,
                initializer: init,
            },
            idecl.span,
        ));
    }
    Ok(Node::new(
        Declaration {
            specifiers: d.node.specifiers.clone(),
            declarators: decls,
        },
        d.span,
    ))
}

fn rename_declarator_tree(
    d: &Node<Declarator>,
    local_names: &HashSet<String>,
    thread_idx: i32,
) -> Result<Node<Declarator>, SpawnExpandError> {
    Ok(Node::new(
        Declarator {
            kind: Node::new(
                match &d.node.kind.node {
                    DeclaratorKind::Identifier(id) => {
                        let name = if local_names.contains(&id.node.name) {
                            format!("{}_{}", id.node.name, thread_idx)
                        } else {
                            id.node.name.clone()
                        };
                        DeclaratorKind::Identifier(Node::new(Identifier { name }, id.span))
                    }
                    DeclaratorKind::Declarator(inner) => DeclaratorKind::Declarator(Box::new(
                        rename_declarator_tree(inner, local_names, thread_idx)?,
                    )),
                    DeclaratorKind::Abstract => DeclaratorKind::Abstract,
                },
                d.node.kind.span,
            ),
            derived: d.node.derived.clone(),
            extensions: d.node.extensions.clone(),
        },
        d.span,
    ))
}

fn rename_locals_in_expr(
    e: &Expression,
    local_names: &HashSet<String>,
    thread_idx: i32,
) -> Result<Expression, SpawnExpandError> {
    Ok(match e {
        Expression::Identifier(id) => {
            if local_names.contains(&id.node.name) {
                Expression::Identifier(Box::new(Node::new(
                    Identifier {
                        name: format!("{}_{}", id.node.name, thread_idx),
                    },
                    id.span,
                )))
            } else {
                e.clone()
            }
        }
        Expression::Call(c) => Expression::Call(Box::new(Node::new(
            CallExpression {
                callee: Box::new(Node::new(
                    rename_locals_in_expr(&c.node.callee.node, local_names, thread_idx)?,
                    c.node.callee.span,
                )),
                arguments: c
                    .node
                    .arguments
                    .iter()
                    .map(|a| {
                        rename_locals_in_expr(&a.node, local_names, thread_idx)
                            .map(|x| Node::new(x, a.span))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
            c.span,
        ))),
        Expression::BinaryOperator(b) => Expression::BinaryOperator(Box::new(Node::new(
            BinaryOperatorExpression {
                operator: b.node.operator.clone(),
                lhs: Box::new(Node::new(
                    rename_locals_in_expr(&b.node.lhs.node, local_names, thread_idx)?,
                    b.node.lhs.span,
                )),
                rhs: Box::new(Node::new(
                    rename_locals_in_expr(&b.node.rhs.node, local_names, thread_idx)?,
                    b.node.rhs.span,
                )),
            },
            b.span,
        ))),
        Expression::UnaryOperator(u) => Expression::UnaryOperator(Box::new(Node::new(
            UnaryOperatorExpression {
                operator: u.node.operator.clone(),
                operand: Box::new(Node::new(
                    rename_locals_in_expr(&u.node.operand.node, local_names, thread_idx)?,
                    u.node.operand.span,
                )),
            },
            u.span,
        ))),
        Expression::Cast(c) => Expression::Cast(Box::new(Node::new(
            lang_c::ast::CastExpression {
                type_name: c.node.type_name.clone(),
                expression: Box::new(Node::new(
                    rename_locals_in_expr(&c.node.expression.node, local_names, thread_idx)?,
                    c.node.expression.span,
                )),
            },
            c.span,
        ))),
        Expression::Conditional(c) => Expression::Conditional(Box::new(Node::new(
            lang_c::ast::ConditionalExpression {
                condition: Box::new(Node::new(
                    rename_locals_in_expr(&c.node.condition.node, local_names, thread_idx)?,
                    c.node.condition.span,
                )),
                then_expression: Box::new(Node::new(
                    rename_locals_in_expr(&c.node.then_expression.node, local_names, thread_idx)?,
                    c.node.then_expression.span,
                )),
                else_expression: Box::new(Node::new(
                    rename_locals_in_expr(&c.node.else_expression.node, local_names, thread_idx)?,
                    c.node.else_expression.span,
                )),
            },
            c.span,
        ))),
        _ => e.clone(),
    })
}
