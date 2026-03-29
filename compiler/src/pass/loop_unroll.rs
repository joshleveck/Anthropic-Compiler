//! Full loop unrolling when the trip count is known at compile time.
//!
//! Replaces `for (init; cond; step) body` with a straight-line sequence of
//! `body` copies where the loop induction variable is replaced by a constant
//! on each iteration. Loops whose bounds depend on runtime values (e.g. loads
//! from memory) are left unchanged (still lowered as CFG with branches).

use std::collections::HashMap;

use lang_c::ast::{
    BinaryOperator, BinaryOperatorExpression, BlockItem, Constant, Declaration, Declarator,
    Designator, Expression, ForInitializer, ForStatement, InitDeclarator, Initializer,
    InitializerListItem, Integer, IntegerBase, IntegerSize, IntegerSuffix, Statement,
    UnaryOperator, UnaryOperatorExpression,
};
use lang_c::span::{Node, Span};

/// Hard cap so we never explode IR size by accident.
const MAX_UNROLL_ITERATIONS: i32 = 1_000_000;

/// Recursively unroll `for` loops where bounds and step are compile-time known.
pub fn unroll_statement(stmt: &Statement, globals: &HashMap<String, i32>) -> Statement {
    unroll_statement_with_env(stmt, globals)
}

fn unroll_statement_with_env(stmt: &Statement, env: &HashMap<String, i32>) -> Statement {
    match stmt {
        Statement::Compound(items) => {
            let mut env_local = env.clone();
            let mut out: Vec<Node<BlockItem>> = Vec::new();
            for item in items {
                let node = match &item.node {
                    BlockItem::Statement(s) => {
                        let u = unroll_statement_with_env(&s.node, &env_local);
                        BlockItem::Statement(Node::new(u, s.span))
                    }
                    BlockItem::Declaration(d) => {
                        for idecl in &d.node.declarators {
                            if let Some(name) = extract_decl_name(&idecl.node.declarator)
                                && let Some(init) = &idecl.node.initializer
                                && let Initializer::Expression(e) = &init.node
                                && let Ok(v) = const_eval_expr(&e.node, &env_local)
                            {
                                env_local.insert(name, v);
                            }
                        }
                        BlockItem::Declaration(d.clone())
                    }
                    BlockItem::StaticAssert(s) => BlockItem::StaticAssert(s.clone()),
                };
                out.push(Node::new(node, item.span));
            }
            Statement::Compound(out)
        }
        Statement::For(f) => {
            if let Some(unrolled) = try_unroll_for(&f.node, env) {
                unrolled
            } else {
                Statement::For(f.clone())
            }
        }
        Statement::If(i) => {
            // After loop substitution, many conditions (e.g. r == 0 / r == 1) become
            // compile-time constants. Folding them here avoids CFG merges for purely
            // static control in sample.c.
            if let Ok(v) = const_eval_expr(&i.node.condition.node, env) {
                if v != 0 {
                    return unroll_statement_with_env(&i.node.then_statement.node, env);
                }
                if let Some(es) = &i.node.else_statement {
                    return unroll_statement_with_env(&es.node, env);
                }
                return Statement::Compound(vec![]);
            }

            let then_s = unroll_statement_with_env(&i.node.then_statement.node, env);
            let else_s = i
                .node
                .else_statement
                .as_ref()
                .map(|e| Box::new(Node::new(unroll_statement_with_env(&e.node, env), e.span)));
            Statement::If(Node::new(
                lang_c::ast::IfStatement {
                    condition: i.node.condition.clone(),
                    then_statement: Box::new(Node::new(then_s, i.node.then_statement.span)),
                    else_statement: else_s,
                },
                i.span,
            ))
        }
        _ => stmt.clone(),
    }
}

fn try_unroll_for(f: &ForStatement, globals: &HashMap<String, i32>) -> Option<Statement> {
    let (var, start) = parse_for_init(&f.initializer.node, globals).ok()?;
    let cond = f.condition.as_ref()?;
    let (lhs, cmp, limit) = parse_for_condition(&cond.node, globals).ok()?;
    if lhs != var {
        return None;
    }
    let step = parse_for_step(f.step.as_ref()?, &var, globals).ok()?;

    let mut v = start;
    let mut blocks: Vec<Node<BlockItem>> = Vec::new();
    let mut iters = 0;
    while cmp_holds(v, limit, cmp) {
        iters += 1;
        if iters > MAX_UNROLL_ITERATIONS {
            return None;
        }
        let body = subst_statement(&f.statement.node, &var, v);
        let body = unroll_statement(&body, globals);
        blocks.push(Node::new(
            BlockItem::Statement(Node::new(body, f.statement.span)),
            f.statement.span,
        ));
        v = v.checked_add(step)?;
    }

    if blocks.is_empty() {
        return Some(Statement::Compound(vec![]));
    }
    Some(Statement::Compound(blocks))
}

fn cmp_holds(v: i32, limit: i32, cmp: Cmp) -> bool {
    match cmp {
        Cmp::Lt => v < limit,
        Cmp::Le => v <= limit,
        Cmp::Gt => v > limit,
        Cmp::Ge => v >= limit,
        Cmp::Ne => v != limit,
        Cmp::Eq => v == limit,
    }
}

#[derive(Clone, Copy)]
enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Ne,
    Eq,
}

fn parse_for_init(init: &ForInitializer, globals: &HashMap<String, i32>) -> Result<(String, i32), ()> {
    match init {
        ForInitializer::Empty => Err(()),
        ForInitializer::Expression(e) => {
            let b = match &e.node {
                Expression::BinaryOperator(b) => &b.node,
                _ => return Err(()),
            };
            if !matches!(b.operator.node, BinaryOperator::Assign) {
                return Err(());
            }
            let name = match &b.lhs.node {
                Expression::Identifier(id) => id.node.name.clone(),
                _ => return Err(()),
            };
            let val = const_eval_expr(&b.rhs.node, globals)?;
            Ok((name, val))
        }
        ForInitializer::Declaration(d) => parse_decl_init(&d.node, globals),
        ForInitializer::StaticAssert(_) => Err(()),
    }
}

fn parse_decl_init(decl: &Declaration, globals: &HashMap<String, i32>) -> Result<(String, i32), ()> {
    let idecl = decl.declarators.first().ok_or(())?;
    let name = extract_decl_name(&idecl.node.declarator).ok_or(())?;
    let init = idecl.node.initializer.as_ref().ok_or(())?;
    match &init.node {
        Initializer::Expression(e) => {
            let v = const_eval_expr(&e.node, globals)?;
            Ok((name, v))
        }
        _ => Err(()),
    }
}

fn extract_decl_name(decl: &Node<Declarator>) -> Option<String> {
    match &decl.node.kind.node {
        lang_c::ast::DeclaratorKind::Identifier(id) => Some(id.node.name.clone()),
        lang_c::ast::DeclaratorKind::Declarator(inner) => extract_decl_name(inner),
        lang_c::ast::DeclaratorKind::Abstract => None,
    }
}

fn parse_for_condition(expr: &Expression, globals: &HashMap<String, i32>) -> Result<(String, Cmp, i32), ()> {
    let b = match expr {
        Expression::BinaryOperator(b) => &b.node,
        _ => return Err(()),
    };
    let lhs = match &b.lhs.node {
        Expression::Identifier(id) => id.node.name.clone(),
        _ => return Err(()),
    };
    let limit = const_eval_expr(&b.rhs.node, globals)?;
    let cmp = match b.operator.node {
        BinaryOperator::Less => Cmp::Lt,
        BinaryOperator::LessOrEqual => Cmp::Le,
        BinaryOperator::Greater => Cmp::Gt,
        BinaryOperator::GreaterOrEqual => Cmp::Ge,
        BinaryOperator::NotEquals => Cmp::Ne,
        BinaryOperator::Equals => Cmp::Eq,
        _ => return Err(()),
    };
    Ok((lhs, cmp, limit))
}

fn parse_for_step(expr: &Node<Expression>, var: &str, globals: &HashMap<String, i32>) -> Result<i32, ()> {
    match &expr.node {
        Expression::UnaryOperator(u) => match u.node.operator.node {
            UnaryOperator::PostIncrement | UnaryOperator::PreIncrement => {
                match &u.node.operand.node {
                    Expression::Identifier(id) if id.node.name == var => Ok(1),
                    _ => Err(()),
                }
            }
            UnaryOperator::PostDecrement | UnaryOperator::PreDecrement => {
                match &u.node.operand.node {
                    Expression::Identifier(id) if id.node.name == var => Ok(-1),
                    _ => Err(()),
                }
            }
            _ => Err(()),
        },
        Expression::BinaryOperator(b) => {
            if !matches!(b.node.operator.node, BinaryOperator::AssignPlus) {
                return Err(());
            }
            match &b.node.lhs.node {
                Expression::Identifier(id) if id.node.name == var => {
                    const_eval_expr(&b.node.rhs.node, globals)
                }
                _ => Err(()),
            }
        }
        _ => Err(()),
    }
}

fn const_eval_expr(e: &Expression, globals: &HashMap<String, i32>) -> Result<i32, ()> {
    match e {
        Expression::Constant(c) => parse_int_const(&c.node),
        Expression::Identifier(id) => globals.get(&id.node.name).copied().ok_or(()),
        Expression::Cast(c) => const_eval_expr(&c.node.expression.node, globals),
        Expression::UnaryOperator(u) => match u.node.operator.node {
            UnaryOperator::Plus => const_eval_expr(&u.node.operand.node, globals),
            UnaryOperator::Minus => const_eval_expr(&u.node.operand.node, globals).map(|v| -v),
            _ => Err(()),
        },
        Expression::BinaryOperator(b) => {
            let lhs = const_eval_expr(&b.node.lhs.node, globals)?;
            let rhs = const_eval_expr(&b.node.rhs.node, globals)?;
            use BinaryOperator::*;
            match b.node.operator.node {
                Plus => Ok(lhs.wrapping_add(rhs)),
                Minus => Ok(lhs.wrapping_sub(rhs)),
                Multiply => Ok(lhs.wrapping_mul(rhs)),
                Divide => {
                    if rhs == 0 {
                        Err(())
                    } else {
                        Ok(lhs.wrapping_div(rhs))
                    }
                }
                Modulo => {
                    if rhs == 0 {
                        Err(())
                    } else {
                        Ok(lhs.wrapping_rem(rhs))
                    }
                }
                ShiftLeft => Ok(lhs.wrapping_shl(rhs as u32)),
                ShiftRight => Ok(lhs.wrapping_shr(rhs as u32)),
                BitwiseAnd => Ok(lhs & rhs),
                BitwiseOr => Ok(lhs | rhs),
                BitwiseXor => Ok(lhs ^ rhs),
                Equals => Ok((lhs == rhs) as i32),
                NotEquals => Ok((lhs != rhs) as i32),
                Less => Ok((lhs < rhs) as i32),
                LessOrEqual => Ok((lhs <= rhs) as i32),
                Greater => Ok((lhs > rhs) as i32),
                GreaterOrEqual => Ok((lhs >= rhs) as i32),
                _ => Err(()),
            }
        }
        _ => Err(()),
    }
}

fn parse_int_const(c: &Constant) -> Result<i32, ()> {
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

fn int_const_expr(val: i32) -> Box<Node<Constant>> {
    Box::new(Node::new(
        Constant::Integer(Integer {
            base: IntegerBase::Decimal,
            number: format!("{val}").into_boxed_str(),
            suffix: IntegerSuffix {
                size: IntegerSize::Int,
                unsigned: false,
                imaginary: false,
            },
        }),
        Span::none(),
    ))
}

fn subst_declaration(decl: &Declaration, var: &str, val: i32) -> Declaration {
    Declaration {
        specifiers: decl.specifiers.clone(),
        declarators: decl
            .declarators
            .iter()
            .map(|n| Node::new(subst_init_declarator(&n.node, var, val), n.span))
            .collect(),
    }
}

fn subst_init_declarator(id: &InitDeclarator, var: &str, val: i32) -> InitDeclarator {
    InitDeclarator {
        declarator: id.declarator.clone(),
        initializer: id.initializer.as_ref().map(|n| {
            Node::new(subst_initializer(&n.node, var, val), n.span)
        }),
    }
}

fn subst_initializer(init: &Initializer, var: &str, val: i32) -> Initializer {
    match init {
        Initializer::Expression(e) => Initializer::Expression(Box::new(Node::new(
            subst_expr(&e.node, var, val),
            e.span,
        ))),
        Initializer::List(items) => Initializer::List(
            items
                .iter()
                .map(|item| {
                    let designation: Vec<Node<Designator>> = item
                        .node
                        .designation
                        .iter()
                        .map(|d| {
                            let nd = match &d.node {
                                Designator::Index(e) => Designator::Index(Node::new(
                                    subst_expr(&e.node, var, val),
                                    e.span,
                                )),
                                Designator::Member(m) => Designator::Member(m.clone()),
                                Designator::Range(r) => Designator::Range(r.clone()),
                            };
                            Node::new(nd, d.span)
                        })
                        .collect();
                    Node::new(
                        InitializerListItem {
                            designation,
                            initializer: Box::new(Node::new(
                                subst_initializer(&item.node.initializer.node, var, val),
                                item.node.initializer.span,
                            )),
                        },
                        item.span,
                    )
                })
                .collect(),
        ),
    }
}

fn subst_statement(stmt: &Statement, var: &str, val: i32) -> Statement {
    match stmt {
        Statement::Compound(items) => {
            let mapped: Vec<Node<BlockItem>> = items
                .iter()
                .map(|item| {
                    let n = match &item.node {
                        BlockItem::Statement(s) => {
                            BlockItem::Statement(Node::new(subst_statement(&s.node, var, val), s.span))
                        }
                        BlockItem::Declaration(d) => BlockItem::Declaration(Node::new(
                            subst_declaration(&d.node, var, val),
                            d.span,
                        )),
                        BlockItem::StaticAssert(s) => BlockItem::StaticAssert(s.clone()),
                    };
                    Node::new(n, item.span)
                })
                .collect();
            Statement::Compound(mapped)
        }
        Statement::Expression(opt) => Statement::Expression(opt.as_ref().map(|e| {
            Box::new(Node::new(subst_expr(&e.node, var, val), e.span))
        })),
        Statement::If(i) => {
            let cond = subst_expr(&i.node.condition.node, var, val);
            let then_s = subst_statement(&i.node.then_statement.node, var, val);
            let else_s = i
                .node
                .else_statement
                .as_ref()
                .map(|e| Box::new(Node::new(subst_statement(&e.node, var, val), e.span)));
            Statement::If(Node::new(
                lang_c::ast::IfStatement {
                    condition: Box::new(Node::new(cond, i.node.condition.span)),
                    then_statement: Box::new(Node::new(then_s, i.node.then_statement.span)),
                    else_statement: else_s,
                },
                i.span,
            ))
        }
        Statement::For(f) => {
            let init = subst_for_init(&f.node.initializer.node, var, val);
            let cond = f
                .node
                .condition
                .as_ref()
                .map(|c| Box::new(Node::new(subst_expr(&c.node, var, val), c.span)));
            let step = f
                .node
                .step
                .as_ref()
                .map(|s| Node::new(subst_expr(&s.node, var, val), s.span));
            Statement::For(Node::new(
                ForStatement {
                    initializer: Node::new(init, f.node.initializer.span),
                    condition: cond,
                    step: step.map(Box::new),
                    statement: Box::new(Node::new(
                        subst_statement(&f.node.statement.node, var, val),
                        f.node.statement.span,
                    )),
                },
                f.span,
            ))
        }
        _ => stmt.clone(),
    }
}

fn subst_for_init(init: &ForInitializer, var: &str, val: i32) -> ForInitializer {
    match init {
        ForInitializer::Empty => ForInitializer::Empty,
        ForInitializer::Expression(e) => {
            ForInitializer::Expression(Box::new(Node::new(subst_expr(&e.node, var, val), e.span)))
        }
        ForInitializer::Declaration(d) => ForInitializer::Declaration(Node::new(
            subst_declaration(&d.node, var, val),
            d.span,
        )),
        ForInitializer::StaticAssert(s) => ForInitializer::StaticAssert(s.clone()),
    }
}

fn subst_expr(e: &Expression, var: &str, val: i32) -> Expression {
    match e {
        Expression::Identifier(id) if id.node.name == var => {
            Expression::Constant(int_const_expr(val))
        }
        Expression::BinaryOperator(b) => Expression::BinaryOperator(Box::new(Node::new(
            BinaryOperatorExpression {
                operator: b.node.operator.clone(),
                lhs: Box::new(Node::new(
                    subst_expr(&b.node.lhs.node, var, val),
                    b.node.lhs.span,
                )),
                rhs: Box::new(Node::new(
                    subst_expr(&b.node.rhs.node, var, val),
                    b.node.rhs.span,
                )),
            },
            b.span,
        ))),
        Expression::UnaryOperator(u) => Expression::UnaryOperator(Box::new(Node::new(
            UnaryOperatorExpression {
                operator: u.node.operator.clone(),
                operand: Box::new(Node::new(
                    subst_expr(&u.node.operand.node, var, val),
                    u.node.operand.span,
                )),
            },
            u.span,
        ))),
        Expression::Call(c) => Expression::Call(Box::new(Node::new(
            lang_c::ast::CallExpression {
                callee: Box::new(Node::new(
                    subst_expr(&c.node.callee.node, var, val),
                    c.node.callee.span,
                )),
                arguments: c
                    .node
                    .arguments
                    .iter()
                    .map(|a| Node::new(subst_expr(&a.node, var, val), a.span))
                    .collect(),
            },
            c.span,
        ))),
        Expression::Cast(c) => Expression::Cast(Box::new(Node::new(
            lang_c::ast::CastExpression {
                type_name: c.node.type_name.clone(),
                expression: Box::new(Node::new(
                    subst_expr(&c.node.expression.node, var, val),
                    c.node.expression.span,
                )),
            },
            c.span,
        ))),
        Expression::Conditional(c) => Expression::Conditional(Box::new(Node::new(
            lang_c::ast::ConditionalExpression {
                condition: Box::new(Node::new(
                    subst_expr(&c.node.condition.node, var, val),
                    c.node.condition.span,
                )),
                then_expression: Box::new(Node::new(
                    subst_expr(&c.node.then_expression.node, var, val),
                    c.node.then_expression.span,
                )),
                else_expression: Box::new(Node::new(
                    subst_expr(&c.node.else_expression.node, var, val),
                    c.node.else_expression.span,
                )),
            },
            c.span,
        ))),
        Expression::Member(m) => Expression::Member(Box::new(Node::new(
            lang_c::ast::MemberExpression {
                operator: m.node.operator.clone(),
                expression: Box::new(Node::new(
                    subst_expr(&m.node.expression.node, var, val),
                    m.node.expression.span,
                )),
                identifier: m.node.identifier.clone(),
            },
            m.span,
        ))),
        Expression::Comma(cs) => Expression::Comma(Box::new(
            cs.iter()
                .map(|n| Node::new(subst_expr(&n.node, var, val), n.span))
                .collect(),
        )),
        _ => e.clone(),
    }
}
