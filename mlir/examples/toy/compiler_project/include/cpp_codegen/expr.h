#pragma once

#include <ostream>
#include <vector>
#include <string_view>
#include <variant>
#include "symbol.h"
#include "type.h"
#include "mlir/IR/Value.h"

namespace mlirgen {
  class MLIRGen;
}

namespace gen {

// Type of expressions.
enum class ExprType : int {
  // Binary Arithmetic
  IAdd, ISub, IMul, IDiv, IMod, FAdd, FMul, FSub,

  // Binary Comparison
  Lt, Le, Gt, Ge, Eq, Neq,

  // Binary Logical
  And, Or,

  // Binary Bitwise
  BitAnd, BitOr, BitXor, Shr, Shl,

  // Unary Arithmetic
  Plus, Minus, PreIncr, PreDecr, PostIncr, PostDecr,

  // Unary Logical and Bitwise
  Not, BitNot,

  // Pointer Operations
  Ref, Deref,

  // Struct operations
  Dot, Arrow,

  // Array operations
  Subscript,

  // Calls
  Call, TemplateCall,

  // Assignments
  Assign, PlusEqual, MinusEqual, MulEqual, DivEqual, ModEqual,
  ShrEqual, ShlEqual, BitAndEqual, BitOrEqual, BitXorEqual,

  // Identifiers & Literal
  Ident, Int, Float, Bool, String, Char,

  // ColumnID Expression
  ColumnId,

  // Select
  Select, Join, FetchValue
};

// Generic expression
class Expr {
 public:
  // Constructor
  Expr(ExprType expr_type, std::vector<const Expr *> &&children)
      : expr_type_(expr_type), type_(nullptr), children_(std::move(children)) {
  }

  virtual ~Expr() = default;

  // Start visit
  virtual void Visit(std::ostream *os) const = 0;

  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const {
    return nullptr;
  };

  [[nodiscard]] const Type* GetType() {
    return type_;
  }

  // Get the expression type
  [[nodiscard]] ExprType Op() const {
    return expr_type_;
  }

  // Get the child at the given index
  [[nodiscard]] const Expr *Child(uint32_t idx) const {
    return children_[idx];
  }

  // Get the number of children
  [[nodiscard]] uint32_t NumChildren() const {
    return static_cast<uint32_t>(children_.size());
  }
 private:
  ExprType expr_type_;
  const Type* type_;
  std::vector<const Expr *> children_;
};

// An expression that's just an identifier
class IdentExpr : public Expr {
 public:
  // Constructor
  explicit IdentExpr(Symbol symbol)
      : Expr(ExprType::Ident, {}), symbol_(symbol) {}

  virtual ~IdentExpr() = default;


  void Visit(std::ostream *os) const override {
    *os << symbol_.ident_;
  }

  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const override;

 private:
  Symbol symbol_;
};

// Fetch the value of the return of a select
class FetchValueExpr : public Expr {
 public:
  // Constructor
  explicit FetchValueExpr(Symbol symbol, uint64_t row_idx, uint64_t col_idx)
      : Expr(ExprType::FetchValue, {}), symbol_(symbol), row_idx_{row_idx}, col_idx_{col_idx} {}

  virtual ~FetchValueExpr() = default;


  void Visit(std::ostream *os) const override {
    *os << symbol_.ident_;
  }

  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const override;

 private:
  Symbol symbol_;
  uint64_t row_idx_;
  uint64_t col_idx_;
};

// Represents a literal
class LiteralExpr : public Expr {
 public:
  // Integer literal
  explicit LiteralExpr(int64_t val)
      : Expr(ExprType::Int, {}), val_(val) {}

  virtual ~LiteralExpr() = default;

  // String literal
  explicit LiteralExpr(std::string_view val)
      : Expr(ExprType::String, {}), val_(val) {}

  // Float literal
  explicit LiteralExpr(double val)
      : Expr(ExprType::Float, {}), val_(val) {}

  // Character literal
  explicit LiteralExpr(char val)
      : Expr(ExprType::Char, {}), val_(val) {}

  // Bool literal
  explicit LiteralExpr(bool val)
      : Expr(ExprType::Bool, {}), val_(val) {}

  void Visit(std::ostream *os) const override;
  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const override;

 private:
  std::variant<std::string_view, int64_t, double, char, bool> val_{};
};

class ColumnIdExpr: public Expr {
  public:
  explicit ColumnIdExpr(uint64_t column_id, uint64_t table_id):
  Expr(ExprType::ColumnId, {}),
  column_id_(column_id),
  table_id_(table_id) {}
  virtual ~ColumnIdExpr() = default;

  uint64_t GetColumnID() const {return column_id_;}
  uint64_t GetTableID() const {return table_id_;}


  void Visit(std::ostream *os) const override;
  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const override;

  private:
  uint64_t column_id_;
  uint64_t table_id_;
};


class SelectExpr : public Expr {
 public:
  // Integer literal
  SelectExpr(std::vector<uint64_t> &&column_ids,
    std::vector<const Expr*> &&projections,
    std::vector<const Expr*> &&filters,
    uint64_t table_id) : Expr(ExprType::Select, {}),
    column_ids_(std::move(column_ids)),
    projections_(std::move(projections)),
    filters_(std::move(filters)),
    table_id_(table_id) {}

  void Visit(std::ostream *os) const override {};
  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const override;

 private:
  std::vector<uint64_t> column_ids_;
  std::vector<const Expr*> projections_;
  std::vector<const Expr*> filters_;
  uint64_t table_id_{37};
};

class JoinExpr : public Expr {
public:
  // Integer literal
  JoinExpr(std::vector<uint64_t> &&tableids) : Expr(ExprType::Join, {}),
                                  table_ids_(std::move(tableids))
                                   {}

  void Visit(std::ostream *os) const override {};
  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const override;

private:
  std::vector<uint64_t> table_ids_;
};

// An assignment or compound assignment.
class AssignOp : public Expr {
 public:
  AssignOp(ExprType expr_type, const Expr *lhs, const Expr *rhs)
      : Expr(expr_type, {lhs, rhs}) {}

  virtual ~AssignOp() = default;


  void Visit(std::ostream *os) const override;
};

// lhs.rhs or lhs->rhs
class MemberOp : public Expr {
 public:
  // Constructor
  MemberOp(ExprType expr_type, const Expr *lhs, const Expr *rhs)
      : Expr(expr_type, {lhs, rhs}) {}

  virtual ~MemberOp() = default;

  void Visit(std::ostream *os) const override;
};

// Represents fn(child1, child2, ...)
class CallOp : public Expr {
 public:
  // Constructor
  CallOp(const Expr *fn, std::vector<const Expr *> &&args, const Type* ret_type = nullptr)
      : Expr(ExprType::Call, AllChildren(fn, std::move(args))), ret_type_(ret_type) {}

  virtual ~CallOp() = default;

  void Visit(std::ostream *os) const override;
  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const override;

  // Make the vector of all children. The function is the first child.
  static std::vector<const Expr *> AllChildren(const Expr *fn, std::vector<const Expr *> &&args) {
    args.insert(args.begin(), fn);
    return std::move(args);
  }

private:
  const Type* ret_type_;
};

// Represents fn<types...>(args...)
class TemplateCallOp : public Expr {
 public:
  // Constructor
  TemplateCallOp(const Expr *fn, std::vector<const Type *> &&types, std::vector<const Expr *> &&args)
      : Expr(ExprType::Call, AllChildren(fn, std::move(args))), types_(std::move(types)) {}

  virtual ~TemplateCallOp() = default;

  void Visit(std::ostream *os) const override;

  // Make the vector of all children. The function is the first child.
  static std::vector<const Expr *> AllChildren(const Expr *fn, std::vector<const Expr *> &&args) {
    args.insert(args.begin(), fn);
    return std::move(args);
  }

 private:
  std::vector<const Type *> types_;
};

// Represents lhs[rhs]
class SubscriptOp : public Expr {
 public:
  // Constructor
  SubscriptOp(const Expr *lhs, const Expr *rhs)
      : Expr(ExprType::Subscript, {lhs, rhs}) {}

  virtual ~SubscriptOp() = default;

  void Visit(std::ostream *os) const override;
};

// Represents *operand or &operand.
class PointerOp : public Expr {
 public:
  // Constructor
  PointerOp(ExprType op, const Expr *operand)
      : Expr(op, {operand}) {}

  virtual ~PointerOp() = default;

  void Visit(std::ostream *os) const override;
};

// Reprents a unary operation.
class UnaryOp : public Expr {
 public:
  UnaryOp(ExprType op, const Expr *operand)
      : Expr(op, {operand}) {}

  virtual ~UnaryOp() = default;

  void Visit(std::ostream *os) const override;
};

// Represents a binary operations
class BinaryOp : public Expr {
 public:
  // Constructor
  BinaryOp(ExprType op, const Expr *lhs, const Expr *rhs)
      : Expr(op, {lhs, rhs}) {}

  virtual ~BinaryOp() = default;

  // Output Binary operation.
  void Visit(std::ostream *os) const override;

  // MLIR
  virtual mlir::Value Visit(mlirgen::MLIRGen *mlir_gen) const override;
};
}
