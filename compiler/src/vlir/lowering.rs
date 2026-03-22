use std::collections::{HashMap, HashSet};

use lang_c::ast::{
    BinaryOperator, BlockItem, Constant, Declaration, DeclarationSpecifier, Declarator,
    DeclaratorKind, Expression, ExternalDeclaration, ForInitializer, FunctionDefinition,
    InitDeclarator, Statement, TranslationUnit, TypeSpecifier, UnaryOperator,
};
use lang_c::span::Node;

use crate::pass::loop_unroll;
use crate::vlir::alu::{AluInst, AluOp};
use crate::vlir::flow::{FlowInst, Terminator};
use crate::vlir::load::{LoadInst, LoadKind};
use crate::vlir::store::{StoreInst, StoreKind};
use crate::vlir::valu::{ValuInst, ValuOp};
use crate::vlir::{
    BlockId, Function, InstrKind, Operand, Program, RegisterId, UnitClass, ValueType,
};

#[derive(Debug)]
pub enum AstLoweringError {
    Unsupported(&'static str),
    UnknownVariable(String),
    MissingFunctionBody(String),
    ExpectedIdentifier,
    IntegerParse(String),
}

pub fn lower_translation_unit(unit: &TranslationUnit) -> Result<Program, AstLoweringError> {
    let globals = collect_global_constants(unit)?;
    let mut functions = Vec::new();
    for ext in &unit.0 {
        if let ExternalDeclaration::FunctionDefinition(def) = &ext.node {
            let name = declarator_name(&def.node.declarator)?;
            if name == "kernel" {
                functions.push(lower_function_definition(&def.node, &globals)?);
            }
        }
    }
    Ok(Program { functions })
}

fn lower_function_definition(
    def: &FunctionDefinition,
    globals: &HashMap<String, i32>,
) -> Result<Function, AstLoweringError> {
    let name = declarator_name(&def.declarator)?;
    let mut cx = FuncLowering::new(Function::new(name));
    for (name, value) in globals {
        let r = cx.emit_const(*value);
        cx.vars.insert(name.clone(), r);
        cx.func.set_reg_name(r, name.clone());
    }
    let body = loop_unroll::unroll_statement(&def.statement.node, globals);
    cx.lower_statement(&body)?;
    if cx.current_terminator_is_unreachable() {
        cx.func
            .set_terminator(cx.current, Terminator::Return { value: None });
    }
    Ok(cx.func)
}

fn collect_global_constants(
    unit: &TranslationUnit,
) -> Result<HashMap<String, i32>, AstLoweringError> {
    let mut out = HashMap::new();
    for ext in &unit.0 {
        if let ExternalDeclaration::Declaration(d) = &ext.node {
            for idecl in &d.node.declarators {
                let name = declarator_name(&idecl.node.declarator)?;
                let Some(init) = &idecl.node.initializer else {
                    continue;
                };
                let val = match &init.node {
                    lang_c::ast::Initializer::Expression(e) => parse_const_expr(&e.node)?,
                    _ => continue,
                };
                out.insert(name, val);
            }
        }
    }
    Ok(out)
}

struct FuncLowering {
    func: Function,
    vars: HashMap<String, RegisterId>,
    current: BlockId,
    /// Cached scalar 0 for vector lane copy (`+` with zero).
    zero_scalar: Option<RegisterId>,
    /// One `const` register per immediate used by `myhash` / `vhash` (shared across all calls).
    hash_const_cache: HashMap<i32, RegisterId>,
}

impl FuncLowering {
    fn new(func: Function) -> Self {
        let current = func.entry;
        Self {
            func,
            vars: HashMap::new(),
            current,
            zero_scalar: None,
            hash_const_cache: HashMap::new(),
        }
    }

    /// Materialize a hash-stage immediate once per function; reuse the same scratch for every `myhash`/`vhash` call.
    fn hash_const_reg(&mut self, v: i32) -> RegisterId {
        if let Some(&r) = self.hash_const_cache.get(&v) {
            return r;
        }
        let r = self.emit_const(v);
        self.hash_const_cache.insert(v, r);
        r
    }

    fn zero_reg(&mut self) -> RegisterId {
        if let Some(z) = self.zero_scalar {
            return z;
        }
        let z = self.emit_const(0);
        self.zero_scalar = Some(z);
        z
    }

    fn current_terminator_is_unreachable(&self) -> bool {
        self.func
            .blocks
            .iter()
            .find(|b| b.id == self.current)
            .map(|b| matches!(b.terminator, Terminator::Unreachable))
            .unwrap_or(false)
    }

    fn emit(&mut self, kind: InstrKind, unit: UnitClass) {
        self.func.append_instruction(self.current, kind, unit, 1);
    }

    fn new_temp(&mut self, ty: ValueType) -> RegisterId {
        self.func.new_register(ty)
    }

    fn emit_const(&mut self, value: i32) -> RegisterId {
        let dst = self.new_temp(ValueType::Scalar);
        self.emit(InstrKind::Const { dst, value }, UnitClass::LoadStore);
        dst
    }

    fn emit_binary(
        &mut self,
        op: BinaryOperator,
        lhs: RegisterId,
        rhs: RegisterId,
        ty: ValueType,
    ) -> RegisterId {
        let dst = self.new_temp(ty);
        match ty {
            ValueType::Vector => {
                let lhs = self.ensure_vector(lhs);
                let rhs = self.ensure_vector(rhs);
                self.emit(
                    InstrKind::Valu(ValuInst {
                        op: map_valu_op(op),
                        dst,
                        src1: lhs,
                        src2: rhs,
                        src3: None,
                    }),
                    UnitClass::VectorAlu,
                );
            }
            _ => {
                self.emit(
                    InstrKind::Alu(AluInst {
                        op: map_alu_op(op),
                        dst,
                        lhs: Operand::Reg(lhs),
                        rhs: Operand::Reg(rhs),
                    }),
                    UnitClass::ScalarAlu,
                );
            }
        }
        dst
    }

    fn emit_vbroadcast(&mut self, src: RegisterId) -> RegisterId {
        let dst = self.new_temp(ValueType::Vector);
        self.emit(
            InstrKind::Valu(ValuInst {
                op: ValuOp::Broadcast,
                dst,
                src1: src,
                src2: src,
                src3: None,
            }),
            UnitClass::VectorAlu,
        );
        dst
    }

    fn ensure_vector(&mut self, reg: RegisterId) -> RegisterId {
        if self.reg_ty(reg) == ValueType::Vector {
            reg
        } else {
            self.emit_vbroadcast(reg)
        }
    }

    fn lower_statement(&mut self, stmt: &Statement) -> Result<(), AstLoweringError> {
        match stmt {
            Statement::Compound(items) => {
                for item in items {
                    match &item.node {
                        BlockItem::Declaration(d) => self.lower_declaration(&d.node)?,
                        BlockItem::Statement(s) => self.lower_statement(&s.node)?,
                        BlockItem::StaticAssert(_) => {}
                    }
                }
                Ok(())
            }
            Statement::Expression(Some(expr)) => {
                let _ = self.lower_expr(&expr.node)?;
                Ok(())
            }
            Statement::Expression(None) => Ok(()),
            Statement::If(ifs) => {
                let cond = self.lower_expr(&ifs.node.condition.node)?;
                let then_bb = self.func.new_block();
                let else_bb = self.func.new_block();
                let exit_bb = self.func.new_block();
                // Snapshot before either branch so the else branch sees the same bindings as
                // the then branch (then must not clobber names the else branch reads).
                let pre_branch = self.vars.clone();

                self.func.set_terminator(
                    self.current,
                    Terminator::Branch {
                        cond,
                        then_bb,
                        else_bb,
                    },
                );

                self.current = then_bb;
                self.lower_statement(&ifs.node.then_statement.node)?;
                if self.current_terminator_is_unreachable() {
                    self.func
                        .set_terminator(self.current, Terminator::Jump { target: exit_bb });
                }
                let post_then = self.vars.clone();

                self.vars = pre_branch.clone();
                self.current = else_bb;
                if let Some(es) = &ifs.node.else_statement {
                    self.lower_statement(&es.node)?;
                }
                if self.current_terminator_is_unreachable() {
                    self.func
                        .set_terminator(self.current, Terminator::Jump { target: exit_bb });
                }
                let post_else = self.vars.clone();

                self.current = exit_bb;
                self.merge_vars_after_if(cond, &pre_branch, &post_then, &post_else)?;
                Ok(())
            }
            Statement::For(f) => {
                match &f.node.initializer.node {
                    ForInitializer::Empty => {}
                    ForInitializer::Expression(e) => {
                        let _ = self.lower_expr(&e.node)?;
                    }
                    ForInitializer::Declaration(d) => self.lower_declaration(&d.node)?,
                    ForInitializer::StaticAssert(_) => {}
                }

                let cond_bb = self.func.new_block();
                let body_bb = self.func.new_block();
                let step_bb = self.func.new_block();
                let exit_bb = self.func.new_block();

                self.func
                    .set_terminator(self.current, Terminator::Jump { target: cond_bb });

                self.current = cond_bb;
                if let Some(cond) = &f.node.condition {
                    let c = self.lower_expr(&cond.node)?;
                    self.func.set_terminator(
                        self.current,
                        Terminator::Branch {
                            cond: c,
                            then_bb: body_bb,
                            else_bb: exit_bb,
                        },
                    );
                } else {
                    self.func
                        .set_terminator(self.current, Terminator::Jump { target: body_bb });
                }

                self.current = body_bb;
                self.lower_statement(&f.node.statement.node)?;
                if self.current_terminator_is_unreachable() {
                    self.func
                        .set_terminator(self.current, Terminator::Jump { target: step_bb });
                }

                self.current = step_bb;
                if let Some(step) = &f.node.step {
                    let _ = self.lower_expr(&step.node)?;
                }
                self.func
                    .set_terminator(self.current, Terminator::Jump { target: cond_bb });
                self.current = exit_bb;
                Ok(())
            }
            Statement::Return(Some(e)) => {
                let v = self.lower_expr(&e.node)?;
                self.func
                    .set_terminator(self.current, Terminator::Return { value: Some(v) });
                Ok(())
            }
            Statement::Return(None) => {
                self.func
                    .set_terminator(self.current, Terminator::Return { value: None });
                Ok(())
            }
            _ => Err(AstLoweringError::Unsupported("statement kind")),
        }
    }

    fn lower_declaration(&mut self, decl: &Declaration) -> Result<(), AstLoweringError> {
        let decl_ty = declaration_type(decl);
        for d in &decl.declarators {
            self.lower_init_declarator(&d.node, decl_ty)?;
        }
        Ok(())
    }

    fn lower_init_declarator(
        &mut self,
        d: &InitDeclarator,
        decl_ty: ValueType,
    ) -> Result<(), AstLoweringError> {
        let name = declarator_name(&d.declarator)?;
        let ty = if declarator_type(&d.declarator) == ValueType::Vector
            || decl_ty == ValueType::Vector
        {
            ValueType::Vector
        } else {
            ValueType::Scalar
        };
        let reg = self.new_temp(ty);
        self.vars.insert(name.clone(), reg);

        if let Some(init) = &d.initializer {
            let rhs = match &init.node {
                lang_c::ast::Initializer::Expression(e) => self.lower_expr(&e.node)?,
                _ => return Err(AstLoweringError::Unsupported("non-expression initializer")),
            };
            self.vars.insert(name.clone(), rhs);
            self.func.set_reg_name(rhs, name);
        } else {
            self.func.set_reg_name(reg, name);
        }
        Ok(())
    }

    fn lower_expr(&mut self, e: &Expression) -> Result<RegisterId, AstLoweringError> {
        match e {
            Expression::Identifier(id) => self.lookup(&id.node.name),
            Expression::Constant(c) => {
                let v = parse_i32_constant(&c.node)?;
                Ok(self.emit_const(v))
            }
            Expression::UnaryOperator(u) => self.lower_unary_expr(&u.node),
            Expression::BinaryOperator(b) => self.lower_binary_expr(&b.node),
            Expression::Call(c) => self.lower_call(c),
            _ => Err(AstLoweringError::Unsupported("expression kind")),
        }
    }

    fn lower_unary_expr(
        &mut self,
        u: &lang_c::ast::UnaryOperatorExpression,
    ) -> Result<RegisterId, AstLoweringError> {
        match u.operator.node {
            UnaryOperator::PostIncrement | UnaryOperator::PreIncrement => {
                let one = self.emit_const(1);
                let cur = self.lower_expr(&u.operand.node)?;
                let out = self.emit_binary(BinaryOperator::Plus, cur, one, self.reg_ty(cur));
                self.assign_lvalue(&u.operand.node, out)?;
                Ok(out)
            }
            UnaryOperator::PostDecrement | UnaryOperator::PreDecrement => {
                let one = self.emit_const(1);
                let cur = self.lower_expr(&u.operand.node)?;
                let out = self.emit_binary(BinaryOperator::Minus, cur, one, self.reg_ty(cur));
                self.assign_lvalue(&u.operand.node, out)?;
                Ok(out)
            }
            UnaryOperator::Plus => self.lower_expr(&u.operand.node),
            UnaryOperator::Minus => {
                let zero = self.emit_const(0);
                let cur = self.lower_expr(&u.operand.node)?;
                Ok(self.emit_binary(BinaryOperator::Minus, zero, cur, self.reg_ty(cur)))
            }
            _ => Err(AstLoweringError::Unsupported("unary operator")),
        }
    }

    fn lower_binary_expr(
        &mut self,
        b: &lang_c::ast::BinaryOperatorExpression,
    ) -> Result<RegisterId, AstLoweringError> {
        use BinaryOperator::*;
        match b.operator.node {
            Assign => {
                let rhs = self.lower_expr(&b.rhs.node)?;
                self.assign_lvalue(&b.lhs.node, rhs)?;
                Ok(rhs)
            }
            AssignPlus | AssignMinus | AssignMultiply | AssignDivide | AssignModulo
            | AssignBitwiseAnd | AssignBitwiseOr | AssignBitwiseXor | AssignShiftLeft
            | AssignShiftRight => {
                let lhs_reg = self.lower_expr(&b.lhs.node)?;
                let rhs_reg = self.lower_expr(&b.rhs.node)?;
                let base = assign_to_base_op(b.operator.node.clone());
                let ty = self.reg_ty(lhs_reg);
                let out = self.emit_binary(base, lhs_reg, rhs_reg, ty);
                self.assign_lvalue(&b.lhs.node, out)?;
                Ok(out)
            }
            Index => {
                let lane = try_compile_time_lane_index(&b.rhs.node).ok_or(
                    AstLoweringError::Unsupported("vector index must be compile-time constant"),
                )?;
                if !(0..8).contains(&lane) {
                    return Err(AstLoweringError::Unsupported(
                        "vector lane index must be in 0..8",
                    ));
                }
                let vec_reg = self.lower_expr(&b.lhs.node)?;
                if self.reg_ty(vec_reg) != ValueType::Vector {
                    return Err(AstLoweringError::Unsupported(
                        "indexed read requires a vector",
                    ));
                }
                let z = self.zero_reg();
                let dst = self.new_temp(ValueType::Scalar);
                self.emit(
                    InstrKind::VectorLaneLoad {
                        dst,
                        vec: vec_reg,
                        lane: lane as u8,
                        zero: z,
                    },
                    UnitClass::ScalarAlu,
                );
                Ok(dst)
            }
            _ => {
                let lhs = self.lower_expr(&b.lhs.node)?;
                let rhs = self.lower_expr(&b.rhs.node)?;
                let ty = if self.reg_ty(lhs) == ValueType::Vector
                    || self.reg_ty(rhs) == ValueType::Vector
                {
                    ValueType::Vector
                } else {
                    ValueType::Scalar
                };
                Ok(self.emit_binary(b.operator.node.clone(), lhs, rhs, ty))
            }
        }
    }

    fn lower_call(
        &mut self,
        c: &Node<lang_c::ast::CallExpression>,
    ) -> Result<RegisterId, AstLoweringError> {
        let fname = match &c.node.callee.node {
            Expression::Identifier(id) => id.node.name.as_str(),
            _ => return Err(AstLoweringError::Unsupported("indirect call")),
        };
        match fname {
            "load" => {
                let addr = self.lower_expr(&c.node.arguments[0].node)?;
                let dst = self.new_temp(ValueType::Scalar);
                self.emit(
                    InstrKind::Load(LoadInst {
                        kind: LoadKind::I32,
                        dst,
                        base_ptr: addr,
                        offset: 0,
                    }),
                    UnitClass::LoadStore,
                );
                Ok(dst)
            }
            "vload" => {
                let addr = self.lower_expr(&c.node.arguments[0].node)?;
                let dst = self.new_temp(ValueType::Vector);
                self.emit(
                    InstrKind::Load(LoadInst {
                        kind: LoadKind::Vec128,
                        dst,
                        base_ptr: addr,
                        offset: 0,
                    }),
                    UnitClass::LoadStore,
                );
                Ok(dst)
            }
            "store" => {
                let addr = self.lower_expr(&c.node.arguments[0].node)?;
                let src = self.lower_expr(&c.node.arguments[1].node)?;
                self.emit(
                    InstrKind::Store(StoreInst {
                        kind: StoreKind::I32,
                        base_ptr: addr,
                        offset: 0,
                        src,
                    }),
                    UnitClass::LoadStore,
                );
                Ok(src)
            }
            "vstore" => {
                let addr = self.lower_expr(&c.node.arguments[0].node)?;
                let src = self.lower_expr(&c.node.arguments[1].node)?;
                self.emit(
                    InstrKind::Store(StoreInst {
                        kind: StoreKind::Vec128,
                        base_ptr: addr,
                        offset: 0,
                        src,
                    }),
                    UnitClass::LoadStore,
                );
                Ok(src)
            }
            "vbroadcast" => {
                let s = self.lower_expr(&c.node.arguments[0].node)?;
                Ok(self.emit_vbroadcast(s))
            }
            "vselect" => {
                let cond = self.lower_expr(&c.node.arguments[0].node)?;
                let a = self.lower_expr(&c.node.arguments[1].node)?;
                let b = self.lower_expr(&c.node.arguments[2].node)?;
                let dst = self.new_temp(ValueType::Vector);
                self.emit(
                    InstrKind::Flow(FlowInst::VSelect { dst, cond, a, b }),
                    UnitClass::LoadStore,
                );
                Ok(dst)
            }
            // Inline vhash(a) for sample.c as repeated vector ops.
            "vhash" => {
                let mut a = self.lower_expr(&c.node.arguments[0].node)?;
                let stages: [(&str, i32, &str, &str, i32); 6] = [
                    ("+", 0x7ED55D16_u32 as i32, "+", "<<", 12),
                    ("^", 0xC761C23C_u32 as i32, "^", ">>", 19),
                    ("+", 0x165667B1, "+", "<<", 5),
                    ("+", 0xD3A2646C_u32 as i32, "^", "<<", 9),
                    ("+", 0xFD7046C5_u32 as i32, "+", "<<", 3),
                    ("^", 0xB55A4F09_u32 as i32, "^", ">>", 16),
                ];
                for (op1, v1, op2, op3, v3) in stages {
                    let c1 = self.hash_const_reg(v1);
                    let c3 = self.hash_const_reg(v3);
                    let t1 = self.emit_binary(op_text_to_bin(op1), a, c1, ValueType::Vector);
                    let t2 = self.emit_binary(op_text_to_bin(op3), a, c3, ValueType::Vector);
                    a = self.emit_binary(op_text_to_bin(op2), t1, t2, ValueType::Vector);
                }
                Ok(a)
            }
            // Scalar myhash — matches problem.myhash / reference_kernel2.
            "myhash" => {
                let mut a = self.lower_expr(&c.node.arguments[0].node)?;
                let stages: [(&str, i32, &str, &str, i32); 6] = [
                    ("+", 0x7ED55D16_u32 as i32, "+", "<<", 12),
                    ("^", 0xC761C23C_u32 as i32, "^", ">>", 19),
                    ("+", 0x165667B1, "+", "<<", 5),
                    ("+", 0xD3A2646C_u32 as i32, "^", "<<", 9),
                    ("+", 0xFD7046C5_u32 as i32, "+", "<<", 3),
                    ("^", 0xB55A4F09_u32 as i32, "^", ">>", 16),
                ];
                for (op1, v1, op2, op3, v3) in stages {
                    let c1 = self.hash_const_reg(v1);
                    let c3 = self.hash_const_reg(v3);
                    let t1 = self.emit_binary(op_text_to_bin(op1), a, c1, ValueType::Scalar);
                    let t2 = self.emit_binary(op_text_to_bin(op3), a, c3, ValueType::Scalar);
                    a = self.emit_binary(op_text_to_bin(op2), t1, t2, ValueType::Scalar);
                }
                Ok(a)
            }
            "debug" => {
                if c.node.arguments.len() != 4 {
                    return Err(AstLoweringError::Unsupported(
                        "debug(value, round, batch, tag) expects 4 arguments",
                    ));
                }
                let v = self.lower_expr(&c.node.arguments[0].node)?;
                let round = try_compile_time_i32(&c.node.arguments[1].node).ok_or(
                    AstLoweringError::Unsupported("debug: round must be compile-time constant"),
                )?;
                let batch = try_compile_time_i32(&c.node.arguments[2].node).ok_or(
                    AstLoweringError::Unsupported("debug: batch must be compile-time constant"),
                )?;
                let tag = try_compile_time_i32(&c.node.arguments[3].node).ok_or(
                    AstLoweringError::Unsupported("debug: tag must be compile-time constant"),
                )?;
                if !(0..=5).contains(&tag) {
                    return Err(AstLoweringError::Unsupported(
                        "debug: tag must be 0..=5 (idx,val,node_val,hashed_val,next_idx,wrapped_idx)",
                    ));
                }
                self.emit(
                    InstrKind::DebugCompare {
                        value: v,
                        round,
                        batch,
                        tag: tag as u8,
                    },
                    UnitClass::Debug,
                );
                Ok(v)
            }
            "flow_pause" => {
                if !c.node.arguments.is_empty() {
                    return Err(AstLoweringError::Unsupported(
                        "flow_pause() takes no arguments",
                    ));
                }
                self.emit(InstrKind::Flow(FlowInst::Pause), UnitClass::LoadStore);
                Ok(self.zero_reg())
            }
            "sync" => {
                if !c.node.arguments.is_empty() {
                    return Err(AstLoweringError::Unsupported("sync() takes no arguments"));
                }
                self.emit(InstrKind::Flow(FlowInst::Sync), UnitClass::LoadStore);
                Ok(self.zero_reg())
            }
            _ => Err(AstLoweringError::Unsupported("call target")),
        }
    }

    fn lookup(&self, name: &str) -> Result<RegisterId, AstLoweringError> {
        self.vars
            .get(name)
            .copied()
            .ok_or_else(|| AstLoweringError::UnknownVariable(name.to_string()))
    }

    fn assign_lvalue(&mut self, lhs: &Expression, rhs: RegisterId) -> Result<(), AstLoweringError> {
        match lhs {
            Expression::Identifier(id) => {
                self.vars.insert(id.node.name.clone(), rhs);
                Ok(())
            }
            Expression::BinaryOperator(b)
                if matches!(b.node.operator.node, BinaryOperator::Index) =>
            {
                let arr_name = match &b.node.lhs.node {
                    Expression::Identifier(id) => id.node.name.clone(),
                    _ => {
                        return Err(AstLoweringError::Unsupported(
                            "indexed assignment base must be an identifier",
                        ));
                    }
                };
                let arr_reg = self.lookup(&arr_name)?;
                if self.reg_ty(arr_reg) != ValueType::Vector {
                    return Err(AstLoweringError::Unsupported(
                        "indexed assignment requires a vector lvalue",
                    ));
                }
                let lane = try_compile_time_lane_index(&b.node.rhs.node).ok_or(
                    AstLoweringError::Unsupported(
                        "indexed vector assignment requires a compile-time constant index",
                    ),
                )?;
                if !(0..8).contains(&lane) {
                    return Err(AstLoweringError::Unsupported(
                        "vector lane index must be in 0..8",
                    ));
                }
                if self.reg_ty(rhs) != ValueType::Scalar {
                    return Err(AstLoweringError::Unsupported(
                        "vector lane store value must be scalar",
                    ));
                }
                let z = self.zero_reg();
                self.emit(
                    InstrKind::VectorLaneStore {
                        vec: arr_reg,
                        lane: lane as u8,
                        src: rhs,
                        zero: z,
                    },
                    UnitClass::ScalarAlu,
                );
                Ok(())
            }
            _ => Err(AstLoweringError::Unsupported(
                "only identifier or compile-time-indexed vector assignments supported",
            )),
        }
    }

    /// Merge `vars` maps after if/else using flow select when a name maps to different registers.
    fn merge_vars_after_if(
        &mut self,
        cond: RegisterId,
        pre: &HashMap<String, RegisterId>,
        post_then: &HashMap<String, RegisterId>,
        post_else: &HashMap<String, RegisterId>,
    ) -> Result<(), AstLoweringError> {
        let mut names = HashSet::new();
        for k in pre.keys() {
            names.insert(k.clone());
        }
        for k in post_then.keys() {
            names.insert(k.clone());
        }
        for k in post_else.keys() {
            names.insert(k.clone());
        }

        let mut merged = HashMap::new();
        for name in names {
            let r_then = post_then
                .get(&name)
                .copied()
                .or_else(|| pre.get(&name).copied());
            let r_else = post_else
                .get(&name)
                .copied()
                .or_else(|| pre.get(&name).copied());
            let (Some(a), Some(b)) = (r_then, r_else) else {
                continue;
            };
            if a == b {
                merged.insert(name, a);
                continue;
            }
            let ty_a = self.reg_ty(a);
            let ty_b = self.reg_ty(b);
            if ty_a != ty_b {
                return Err(AstLoweringError::Unsupported(
                    "if/else merge: branch values have mismatched types",
                ));
            }
            let dst = self.new_temp(ty_a);
            match ty_a {
                ValueType::Vector => {
                    self.emit(
                        InstrKind::Flow(FlowInst::VSelect { dst, cond, a, b }),
                        UnitClass::LoadStore,
                    );
                }
                _ => {
                    self.emit(
                        InstrKind::Flow(FlowInst::Select { dst, cond, a, b }),
                        UnitClass::LoadStore,
                    );
                }
            }
            merged.insert(name, dst);
        }
        self.vars = merged;
        Ok(())
    }

    fn reg_ty(&self, reg: RegisterId) -> ValueType {
        self.func
            .reg_types
            .get(&reg)
            .copied()
            .unwrap_or(ValueType::Scalar)
    }
}

fn declarator_name(decl: &Node<Declarator>) -> Result<String, AstLoweringError> {
    extract_decl_name(decl).ok_or(AstLoweringError::ExpectedIdentifier)
}

fn extract_decl_name(decl: &Node<Declarator>) -> Option<String> {
    match &decl.node.kind.node {
        DeclaratorKind::Identifier(id) => Some(id.node.name.clone()),
        DeclaratorKind::Declarator(inner) => extract_decl_name(inner),
        DeclaratorKind::Abstract => None,
    }
}

fn declarator_type(decl: &Node<Declarator>) -> ValueType {
    for d in &decl.node.derived {
        if let lang_c::ast::DerivedDeclarator::Array(_) = d.node {
            return ValueType::Vector;
        }
    }
    ValueType::Scalar
}

fn declaration_type(decl: &Declaration) -> ValueType {
    for spec in &decl.specifiers {
        if let DeclarationSpecifier::TypeSpecifier(ts) = &spec.node {
            if let TypeSpecifier::TypedefName(id) = &ts.node {
                if id.node.name == "vec8_t" {
                    return ValueType::Vector;
                }
            }
        }
    }
    ValueType::Scalar
}

fn parse_i32_constant(c: &Constant) -> Result<i32, AstLoweringError> {
    match c {
        Constant::Integer(i) => {
            let s = i.number.as_ref();
            if s.starts_with("0x") || s.starts_with("0X") {
                i64::from_str_radix(&s[2..], 16)
                    .map(|v| v as i32)
                    .map_err(|_| AstLoweringError::IntegerParse(s.to_string()))
            } else {
                s.parse::<i64>()
                    .map(|v| v as i32)
                    .map_err(|_| AstLoweringError::IntegerParse(s.to_string()))
            }
        }
        _ => Err(AstLoweringError::Unsupported("non-integer constants")),
    }
}

fn parse_const_expr(e: &Expression) -> Result<i32, AstLoweringError> {
    match e {
        Expression::Constant(c) => parse_i32_constant(&c.node),
        Expression::Cast(c) => parse_const_expr(&c.node.expression.node),
        _ => Err(AstLoweringError::Unsupported("global constant initializer")),
    }
}

/// Integer constant for unrolled loop substitution, `debug(..., round, batch, tag)`, etc.
fn try_compile_time_i32(e: &Expression) -> Option<i32> {
    match e {
        Expression::Constant(c) => parse_i32_constant(&c.node).ok(),
        Expression::Cast(c) => try_compile_time_i32(&c.node.expression.node),
        _ => None,
    }
}

/// Lane index for `v[i]` when `i` is compile-time known (unrolled loops, literals).
fn try_compile_time_lane_index(e: &Expression) -> Option<i32> {
    try_compile_time_i32(e)
}

fn assign_to_base_op(op: BinaryOperator) -> BinaryOperator {
    match op {
        BinaryOperator::AssignPlus => BinaryOperator::Plus,
        BinaryOperator::AssignMinus => BinaryOperator::Minus,
        BinaryOperator::AssignMultiply => BinaryOperator::Multiply,
        BinaryOperator::AssignDivide => BinaryOperator::Divide,
        BinaryOperator::AssignModulo => BinaryOperator::Modulo,
        BinaryOperator::AssignBitwiseAnd => BinaryOperator::BitwiseAnd,
        BinaryOperator::AssignBitwiseOr => BinaryOperator::BitwiseOr,
        BinaryOperator::AssignBitwiseXor => BinaryOperator::BitwiseXor,
        BinaryOperator::AssignShiftLeft => BinaryOperator::ShiftLeft,
        BinaryOperator::AssignShiftRight => BinaryOperator::ShiftRight,
        _ => panic!("Unsupported operator: {:?}", op),
    }
}

fn map_alu_op(op: BinaryOperator) -> AluOp {
    match op {
        BinaryOperator::Plus => AluOp::Add,
        BinaryOperator::Minus => AluOp::Sub,
        BinaryOperator::Multiply => AluOp::Mul,
        BinaryOperator::Divide => AluOp::Div,
        BinaryOperator::Modulo => AluOp::Mod,
        BinaryOperator::BitwiseAnd => AluOp::And,
        BinaryOperator::BitwiseOr => AluOp::Or,
        BinaryOperator::BitwiseXor => AluOp::Xor,
        BinaryOperator::ShiftLeft => AluOp::Shl,
        BinaryOperator::ShiftRight => AluOp::Shr,
        BinaryOperator::Equals => AluOp::CmpEq,
        BinaryOperator::Less => AluOp::CmpLt,
        _ => panic!("Unsupported operator: {:?}", op),
    }
}

fn map_valu_op(op: BinaryOperator) -> ValuOp {
    match op {
        BinaryOperator::Plus => ValuOp::Add,
        BinaryOperator::Minus => ValuOp::Sub,
        BinaryOperator::Multiply => ValuOp::Mul,
        BinaryOperator::Divide => ValuOp::Div,
        BinaryOperator::Modulo => ValuOp::Mod,
        BinaryOperator::BitwiseAnd => ValuOp::And,
        BinaryOperator::BitwiseOr => ValuOp::Or,
        BinaryOperator::BitwiseXor => ValuOp::Xor,
        BinaryOperator::ShiftLeft => ValuOp::Shl,
        BinaryOperator::ShiftRight => ValuOp::Shr,
        BinaryOperator::Equals => ValuOp::CmpEq,
        BinaryOperator::Less => ValuOp::CmpLt,
        _ => panic!("Unsupported operator: {:?}", op),
    }
}

fn op_text_to_bin(s: &str) -> BinaryOperator {
    match s {
        "+" => BinaryOperator::Plus,
        "-" => BinaryOperator::Minus,
        "*" => BinaryOperator::Multiply,
        "//" => BinaryOperator::Divide,
        "%" => BinaryOperator::Modulo,
        "&" => BinaryOperator::BitwiseAnd,
        "|" => BinaryOperator::BitwiseOr,
        "^" => BinaryOperator::BitwiseXor,
        "<<" => BinaryOperator::ShiftLeft,
        ">>" => BinaryOperator::ShiftRight,
        "==" => BinaryOperator::Equals,
        "<" => BinaryOperator::Less,
        _ => panic!("Unsupported operator: {}", s),
    }
}
