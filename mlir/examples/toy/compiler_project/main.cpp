// CppCodegen.cpp : Defines the entry point for the application.
//

#include "cpp_codegen/ast.h"
#include "cpp_codegen/node_builders.h"
#include "cpp_codegen/expr_builder.h"
#include "cpp_codegen/type_builder.h"
#include "cpp_codegen/compile.h"
#include <memory>
#include <iostream>
#include <cassert>

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Dialect.h"
#include "mlir/MLIRGen.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"


using namespace std;

void DumpMLIR(std::vector<const gen::Node*> & nodes) {
  // TODO(Create Dialect);
  mlir::registerDialect<mlir::sqlir::SqlIRDialect>();
  mlir::registerDialect<mlir::StandardOpsDialect>();

  mlir::MLIRContext context;
  mlirgen::MLIRGen mlir_gen(context);
  mlir::OwningModuleRef module = mlir_gen.mlirGen(nodes);

  module->dump();


  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  // applyPassManagerCLOptions(pm);

  // Inline all functions into main and then delete them.
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::sqlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.run(*module);
  module->dump();
}


int main() {
  gen::CodegenContext cg;
  gen::NodeFactory N(&cg);
  gen::ExprBuilder E(&cg);
  gen::TypeBuilder T(&cg);
  gen::Compiler comp;

  // Needed expressions.
  auto main_arg_ident = cg.GetSymbol("main_arg");
  auto main_arg = E.MakeExpr(main_arg_ident);
  auto table_id_ident = cg.GetSymbol("main_arg");
  auto table_id = E.MakeExpr(table_id_ident);

  // Make the file
  gen::FileBuilder file_builder(&cg);
  file_builder.Include(cg.GetSymbol("iostream"), true);

  // Start the main function
  gen::FunctionBuilder main_fn(&cg);
  main_fn.SetName(cg.GetSymbol("Main"));
  main_fn.AddField({T.GetType(gen::PrimType::I64), main_arg_ident});
  main_fn.SetRetType(T.GetType(gen::PrimType::I64));

  // Make the select fn
  gen::FunctionBuilder select_fn(&cg);
  auto select_sym = cg.GetSymbol("Select");
  select_fn.SetName(select_sym);
  select_fn.AddField({T.GetType(gen::PrimType::I64), table_id_ident});
  select_fn.SetRetType(T.GetType(gen::PrimType::I64));

  // Make the join fn
  gen::FunctionBuilder join_fn(&cg);
  auto join_sym = cg.GetSymbol("Join");
  join_fn.SetName(join_sym);
  join_fn.SetRetType(T.GetType(gen::PrimType::I64));



  auto table_id_ident_get_column = cg.GetSymbol("table_id_arg");
  auto column_id_ident = cg.GetSymbol("column_id_arg");
  auto column_id = E.MakeExpr(column_id_ident);
    // Make the GetColumnId fn
  gen::FunctionBuilder get_column(&cg);
  get_column.SetName(cg.GetSymbol("getcolumn"));
  get_column.AddField({T.GetType(gen::PrimType::I64), table_id_ident_get_column});
  get_column.AddField({T.GetType(gen::PrimType::I64), column_id_ident});
  get_column.SetRetType(T.GetType(gen::PrimType::I64));
  get_column.Return(column_id);


  // Gen Simple Select
  std::vector<uint64_t> select_column_ids{1,2};
  auto col1 = E.ColumnId(1, 37);
  auto col2 = E.ColumnId(2, 37);
  auto col1_mul_col2 = E.IMul(col1, col2);
  auto col1_sum_col2 = E.IAdd(col1, col2);
  auto const_col = E.IMul(table_id, table_id);

  auto col1_less_col2 = E.Lt(col1_sum_col2, col2);
  auto col1_greater_col2= E.Gt(col1, col2);
  auto const_comp = E.Gt(E.IntLiteral(37), E.IntLiteral(73));

  std::vector<const gen::Expr *> select_projection_expressions{ const_col, col1_mul_col2};
  std::vector<const gen::Expr *> select_filter_expressions{ col1_less_col2, col1_greater_col2, const_comp };


  auto sel = E.Select(std::move(select_column_ids),
           std::move(select_projection_expressions),
           std::move(select_filter_expressions),
           37);
  select_fn.Add(sel);
  //select_fn.Return(E.IntLiteral(10000));

  // Create Join Expression
  std::vector<uint64_t> join_table_ids{1,2,3,4,5,6,7};
  auto join = E.Join(std::move(join_table_ids));
  join_fn.Add(join);

  // Get Rate
  auto rate_sym = cg.GetSymbol("rate");
  auto select_res_sym = cg.GetSymbol("select_res");
  auto join_res_sym = cg.GetSymbol("join_res");
  auto rate = E.MakeExpr(rate_sym);
  auto select_res = E.MakeExpr(select_res_sym);
  auto join_res = E.MakeExpr(join_res_sym);
  main_fn.Declare(rate_sym, E.FloatLiteral(37.77));
  main_fn.Declare(select_res_sym, E.Call(E.MakeExpr(select_sym), {E.IntLiteral(37)}, T.GetType(gen::PrimType::I64)));
  main_fn.Declare(join_res_sym, E.Call(E.MakeExpr(join_sym), {}, T.GetType(gen::PrimType::I64)));
  main_fn.Return(E.IMul(E.FetchValue(select_res_sym, 0, 0), E.FetchValue(join_res_sym, 0, 0)));

  auto main_node = main_fn.Finish();
  auto select_node = select_fn.Finish();
  auto join_node = join_fn.Finish();
  auto get_column_node = get_column.Finish();
  (void)main_node;

  std::vector<const gen::Node*> nodes;
  // nodes.emplace_back(get_column_node);
  nodes.emplace_back(select_node);
  nodes.emplace_back(join_node);
  nodes.emplace_back(main_node);

  DumpMLIR(nodes);
  return 0;
}
