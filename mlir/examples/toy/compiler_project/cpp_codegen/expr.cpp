#include "cpp_codegen/expr.h"
#include <iostream>
#include <sstream>
#include "mlir/MLIRGen.h"

using namespace mlir;

namespace gen {

void LiteralExpr::Visit(std::ostream *os) const {
  switch (Op()) {
    case ExprType::Int:*os << std::get<int64_t>(val_);
      break;
    case ExprType::String:*os << '"' << std::get<std::string_view>(val_) << '"';
      break;
    case ExprType::Float:*os << std::get<double>(val_);
      break;
    case ExprType::Char:*os << "'" << (std::get<char>(val_)) << "'";
      break;
    case ExprType::Bool:*os << (std::get<bool>(val_) ? "true" : "false");
      break;
    default:
      std::cout << "Not a literal expression!" << std::endl;
      abort();
  }
}

mlir::Value LiteralExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  switch (Op()) {
    case ExprType::Int: {
      auto val = std::get<int64_t>(val_);
      auto mlir_attr = mlir_gen->Builder()->getI64IntegerAttr(val);//mlir_gen->Builder()->getIntegerAttr(type, val);
      return mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), mlir_attr.getType(), mlir_attr);
    }
    case ExprType::Float: {
      mlir::Type type = mlir_gen->Builder()->getF64Type();
      auto val = std::get<double>(val_);
      auto mlir_attr = mlir_gen->Builder()->getFloatAttr(type, val);
      return mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), type, mlir_attr);
    }
    default:
      std::cout << "Not a literal expression!" << std::endl;
      abort();
  }
}

void ColumnIdExpr::Visit(std::ostream *os) const {
  *os <<"ColumnID "<< GetColumnID();
}

mlir::Value ColumnIdExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  llvm::StringRef callee("getcolumn");
  auto location = mlir_gen->Loc();

  // Codegen the operands first.
  SmallVector<mlir::Value, 2> operands;
  auto mlir_attr = mlir_gen->Builder()->getI64IntegerAttr(this->GetTableID());//mlir_gen->Builder()->getIntegerAttr(type, table_id_);
  auto arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), mlir_attr.getType(), mlir_attr);
  operands.push_back(arg);

  mlir_attr = mlir_gen->Builder()->getI64IntegerAttr(this->GetColumnID());//mlir_gen->Builder()->getIntegerAttr(type, id);
  arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), mlir_attr.getType(), mlir_attr);
  operands.push_back(arg);

  auto op = mlir_gen->Builder()->create<mlir::sqlir::GetColumnOp>(location, mlir_attr.getType(), operands[0], operands[1]);
  return op.getResult();
}




mlir::Value SelectExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  auto curr_block = mlir_gen->Builder()->getBlock();
  auto parent = curr_block->getParent();
  // First allocate a temp table.
  auto temp_table_id = mlir_gen->Builder()->create<mlir::sqlir::NewTempTableOp>(mlir_gen->Loc(), mlir_gen->Builder()->getIntegerType(64));


  auto top_block = mlir_gen->Builder()->createBlock(parent, parent->end());
  auto exit_block = mlir_gen->Builder()->createBlock(parent, parent->end());
  // Jump into this new block.
  mlir_gen->Builder()->setInsertionPointToEnd(curr_block);
  mlir_gen->Builder()->create<mlir::BranchOp>(mlir_gen->Loc(), top_block);
  mlir_gen->Builder()->setInsertionPointToStart(top_block);

  // Advance the table
  mlir::Value has_next;
  {
    auto mlir_attr = mlir_gen->Builder()->getI64IntegerAttr(table_id_);
    auto arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), mlir_attr.getType(), mlir_attr);
    has_next = mlir_gen->Builder()->create<mlir::sqlir::TableNextOp>(mlir_gen->Loc(), mlir_gen->Builder()->getI1Type(), arg);
  }

  // Filters
  auto false_target = exit_block;
  mlir::Value curr_jump_cond = has_next;
  {
    for(auto filter_expression: filters_) {
      auto curr_block = mlir_gen->Builder()->getBlock();
      auto parent = curr_block->getParent();
      auto new_filter_block = mlir_gen->Builder()->createBlock(exit_block);

      // Jump into this new block.
      mlir_gen->Builder()->setInsertionPointToEnd(curr_block);
      mlir_gen->Builder()->create<mlir::CondBranchOp>(mlir_gen->Loc(), curr_jump_cond, new_filter_block, false_target);

      mlir_gen->Builder()->setInsertionPointToStart(new_filter_block);

      auto expression_variable_mlir = filter_expression->Visit(mlir_gen);
      curr_jump_cond = expression_variable_mlir;
      false_target = top_block;
    }
  }

  // Projection
  {
    auto curr_block = mlir_gen->Builder()->getBlock();
    auto parent = curr_block->getParent();
    auto project_block = mlir_gen->Builder()->createBlock(exit_block);

    // Jump into this new block.
    mlir_gen->Builder()->setInsertionPointToEnd(curr_block);
    mlir_gen->Builder()->create<mlir::CondBranchOp>(mlir_gen->Loc(), curr_jump_cond, project_block, false_target);

    mlir_gen->Builder()->setInsertionPointToStart(project_block);

    for(auto projection_expression: projections_) {
      auto expression_variable_mlir = projection_expression->Visit(mlir_gen);
      mlir_gen->Builder()->create<mlir::sqlir::FillResultOp>(mlir_gen->Loc(), expression_variable_mlir, temp_table_id);
    }
    mlir_gen->Builder()->create<mlir::BranchOp>(mlir_gen->Loc(), top_block);
  }

  // Exit
  mlir_gen->Builder()->setInsertionPointToEnd(exit_block);
  mlir_gen->Builder()->create<mlir::ReturnOp>(mlir_gen->Loc(), llvm::makeArrayRef(temp_table_id.getResult()));
  return nullptr;
}

mlir::Value JoinExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  auto curr_block = mlir_gen->Builder()->getBlock();

  mlir::Value temp_table_id;
  for(unsigned i = 1; i<table_ids_.size(); i++) {

    // Codegen the operands first.
    SmallVector<mlir::Value, 2> operands;

    if(i == 1) {
      auto mlir_attr = mlir_gen->Builder()->getI64IntegerAttr(table_ids_[0]);
      auto arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), mlir_attr.getType(), mlir_attr);
      operands.push_back(arg);

      mlir_attr = mlir_gen->Builder()->getI64IntegerAttr(table_ids_[1]);
      arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), mlir_attr.getType(), mlir_attr);
      operands.push_back(arg);

    } else {
      auto mlir_attr = mlir_gen->Builder()->getI64IntegerAttr(table_ids_[i]);
      auto arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), mlir_attr.getType(), mlir_attr);
      operands.push_back(arg);
      operands.push_back(temp_table_id);
    }

    /*Create join op*/
    auto op = mlir_gen->Builder()->create<mlir::sqlir::JoinOp>(mlir_gen->Loc(), operands[0].getType(), operands[0], operands[1]);
    temp_table_id = op.getResult();
  }

  mlir_gen->Builder()->create<mlir::ReturnOp>(mlir_gen->Loc(), llvm::makeArrayRef(temp_table_id));
  return nullptr;
}



mlir::Value IdentExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  if (auto variable = mlir_gen->SymTable()->lookup(symbol_.ident_))
    return variable;

  emitError(mlir_gen->Loc(), "error: unknown variable '")
      << std::string(symbol_.ident_) << "'";
  return nullptr;
}

mlir::Value FetchValueExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  if (auto temp_id = mlir_gen->SymTable()->lookup(symbol_.ident_)) {
    auto i64_type = mlir_gen->Builder()->getIntegerType(64);
    auto row_idx_attr = mlir_gen->Builder()->getIntegerAttr(i64_type, row_idx_);
    auto col_idx_attr = mlir_gen->Builder()->getIntegerAttr(i64_type, row_idx_);
    auto row_idx_arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), row_idx_attr.getType(), row_idx_attr);
    auto col_idx_arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), col_idx_attr.getType(), col_idx_attr);

    return mlir_gen->Builder()->create<mlir::sqlir::FetchValueOp>(mlir_gen->Loc(), i64_type, temp_id, row_idx_arg.getResult(), col_idx_arg.getResult());
  }

  emitError(mlir_gen->Loc(), "error: unknown variable '")
      << std::string(symbol_.ident_) << "'";
  return nullptr;
}


void AssignOp::Visit(std::ostream *os) const {
  // Gen member access
  auto lhs = Child(0);
  auto rhs = Child(1);
  lhs->Visit(os);
  switch (Op()) {
    case ExprType::Assign:*os << " = ";
      break;
    case ExprType::PlusEqual:*os << " += ";
      break;
    case ExprType::MinusEqual:*os << " -= ";
      break;
    case ExprType::MulEqual:*os << " *= ";
      break;
    case ExprType::DivEqual:*os << " /= ";
      break;
    case ExprType::ModEqual:*os << " %= ";
      break;
    case ExprType::ShrEqual:*os << " >>= ";
      break;
    case ExprType::ShlEqual:*os << " <<= ";
      break;
    case ExprType::BitAndEqual:*os << " &= ";
      break;
    case ExprType::BitOrEqual:*os << " |= ";
      break;
    case ExprType::BitXorEqual:*os << " ^= ";
      break;
    default:
      std::cout << "Not an assign op!" << std::endl;
      abort();
  }
  rhs->Visit(os);
}


void MemberOp::Visit(std::ostream *os) const {
  // Gen member access
  auto lhs = Child(0);
  auto rhs = Child(1);
  lhs->Visit(os);
  switch (Op()) {
    case ExprType::Dot:*os << ".";
      break;
    case ExprType::Arrow:*os << "->";
      break;
    default:
      std::cout << "Not a call expr!" << std::endl;
      abort();
  }
  rhs->Visit(os);
}


void CallOp::Visit(std::ostream *os) const {
  // Gen function
  auto fn = Child(0);
  fn->Visit(os);
  // Add args
  *os << "(";
  auto num_children = NumChildren();
  for (uint32_t i = 1; i < num_children; i++) {
    auto child = Child(i);
    child->Visit(os);
    if (i < num_children - 1) {
      *os << ", ";
    }
  }
  *os << ")";
}

mlir::Value CallOp::Visit(mlirgen::MLIRGen *mlir_gen) const {
  auto fn = Child(0);
  // Print name into string
  std::stringstream ss{};
  fn->Visit(&ss);
  std::string callee_name(ss.str());
  llvm::StringRef callee(callee_name);
  auto location = mlir_gen->Loc();
  llvm::errs() << callee;

  // Codegen the operands first.
  SmallVector<mlir::Value, 2> operands;
  auto num_children = NumChildren();
  for (uint32_t i = 1; i < num_children; i++) {
    auto child = Child(i);
    auto arg = child->Visit(mlir_gen);
    operands.push_back(arg);
  }
  llvm::errs() << callee << "  ASDA\n";
  auto op = mlir_gen->Builder()->create<mlir::CallOp>(location, callee, ret_type_->Visit(mlir_gen), operands);
  return op.getResult(0);
  return nullptr;
}


void TemplateCallOp::Visit(std::ostream *os) const {
  // Gen function
  auto fn = Child(0);
  fn->Visit(os);

  // Add Types
  *os << "<";
  for (uint32_t i = 0; i < types_.size(); i++) {
    types_[i]->Visit(os);
    if (i < types_.size() - 1) {
      *os << ", ";
    }
  }
  *os << ">";

  // Add args
  *os << "(";
  auto num_children = NumChildren();
  for (uint32_t i = 1; i < num_children; i++) {
    auto child = Child(i);
    child->Visit(os);
    if (i < num_children - 1) {
      *os << ", ";
    }
  }
  *os << ")";
}


void SubscriptOp::Visit(std::ostream *os) const {
  // Open paren
  *os << "(";

  // Gen subscript op.
  auto lhs = Child(0);
  auto rhs = Child(1);
  lhs->Visit(os);
  *os << "[";
  rhs->Visit(os);
  *os << "]";

  // Close paren
  *os << ")";
}


void PointerOp::Visit(std::ostream *os) const {
  // Open paren
  *os << '(';

  // Gen pointer op
  auto operand = Child(0);
  switch (Op()) {
    case ExprType::Ref:*os << '&';
      break;
    case ExprType::Deref:*os << '*';
      break;
    default:
      std::cout << "Not a pointer operation" << std::endl;
      abort();
  }
  operand->Visit(os);

  // Close paren
  *os << ')';
}


void UnaryOp::Visit(std::ostream *os) const {
  // Open paren
  *os << '(';

  // Gen unary op
  auto operand = Child(0);
  switch (Op()) {
    case ExprType::Plus:*os << "+";
      operand->Visit(os);
      break;
    case ExprType::Minus:*os << "-";
      operand->Visit(os);
      break;
    case ExprType::Not:*os << "!";
      operand->Visit(os);
      break;
    case ExprType::BitNot:*os << "~";
      operand->Visit(os);
      break;
    case ExprType::PreIncr:*os << "++";
      operand->Visit(os);
      break;
    case ExprType::PreDecr:*os << "--";
      operand->Visit(os);
      break;
    case ExprType::PostIncr:operand->Visit(os);
      *os << "++";
      break;
    case ExprType::PostDecr:operand->Visit(os);
      *os << "--";
      break;
    default:
      std::cout << "Not a unary expr!" << std::endl;
      abort();
  }

  // Close paren
  *os << ')';
}


void BinaryOp::Visit(std::ostream *os) const {
  // Open paren
  *os << '(';

  // Gen binary op
  auto lhs = Child(0);
  auto rhs = Child(1);
  lhs->Visit(os);
  switch (Op()) {
    case ExprType::IAdd:
    case ExprType::FAdd:
      *os << "+";
      break;
    case ExprType::ISub:
    case ExprType::FSub:
      *os << "-";
      break;
    case ExprType::IMul:
    case ExprType::FMul:
      *os << "*";
      break;
    case ExprType::IDiv:*os << "/";
      break;
    case ExprType::IMod:*os << "%";
      break;
    case ExprType::Lt:*os << "<";
      break;
    case ExprType::Le:*os << "<=";
      break;
    case ExprType::Gt:*os << ">";
      break;
    case ExprType::Ge:*os << ">=";
      break;
    case ExprType::Eq:*os << "==";
      break;
    case ExprType::Neq:*os << "!=";
      break;
    case ExprType::And:*os << "&&";
      break;
    case ExprType::Or:*os << "||";
      break;
    case ExprType::BitAnd:*os << "&";
      break;
    case ExprType::BitOr:*os << "|";
      break;
    case ExprType::BitXor:*os << "^";
      break;
    case ExprType::Shr:*os << ">>";
      break;
    case ExprType::Shl:*os << "<<";
      break;
    default:
      std::cout << "Not binary op!" << std::endl;
      abort();
  }
  rhs->Visit(os);

  // Close paren
  *os << ')';
}

mlir::Value BinaryOp::Visit(mlirgen::MLIRGen *mlir_gen) const {
  // Gen binary op
  auto lhs = Child(0)->Visit(mlir_gen);
  auto rhs = Child(1)->Visit(mlir_gen);

  switch (Op()) {
    case ExprType::IAdd: {
      // Otherwise, this return operation has zero operands.
      return mlir_gen->Builder()->create<mlir::AddIOp>(mlir_gen->Loc(), lhs, rhs);
    }
    case ExprType::FAdd: {
      // Otherwise, this return operation has zero operands.
      return mlir_gen->Builder()->create<mlir::AddFOp>(mlir_gen->Loc(), lhs, rhs);
    }
    case ExprType::IMul: {
      // Otherwise, this return operation has zero operands.
      return mlir_gen->Builder()->create<mlir::MulIOp>(mlir_gen->Loc(), lhs, rhs);
    }
    case ExprType::FMul: {
      // Otherwise, this return operation has zero operands.
      return mlir_gen->Builder()->create<mlir::MulFOp>(mlir_gen->Loc(), lhs, rhs);
    }
    case ExprType::Lt: {
      return mlir_gen->Builder()->create<CmpIOp>(mlir_gen->Loc(), CmpIPredicate::slt, lhs, rhs);
    }
    case ExprType::Gt: {
      return mlir_gen->Builder()->create<CmpIOp>(mlir_gen->Loc(), CmpIPredicate::sgt, lhs, rhs);
    }
    default: {
      std::cout << "Unsupported Binary Op" << std::endl;
      return nullptr;
    }
  }

}

}
