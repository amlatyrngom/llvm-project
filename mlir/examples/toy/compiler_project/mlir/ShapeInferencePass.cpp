//===- ShapeInferencePass.cpp - Shape Inference ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include <unordered_map>
#include <map>
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect.h"
#include "mlir/Passes.h"
#include "mlir/ShapeInferenceInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace sqlir;

/// Include the auto-generated definitions for the shape inference interfaces.
#include "mlir/ShapeInferenceOpInterfaces.cpp.inc"
#include "mlir/MLIRGen.h"

namespace {
/// The ShapeInferencePass is a FunctionPass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
class ShapeInferencePass : public mlir::FunctionPass<ShapeInferencePass> {
public:

  bool IsSelectInvariant(Operation *op, function_ref<bool(Value)> definedOutside) {
    // Check that dependencies are defined outside of loop.
    llvm::outs() << "CHECKING: " << *op << "\n";
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;
    llvm::outs() << "DONE\n";

    if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!memInterface.hasNoEffect())
        return false;
      // If the operation doesn't have side effects and it doesn't recursively
      // have side effects, it can always be hoisted.
      if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>())
        return true;

      // Otherwise, if the operation doesn't provide the memory effect interface
      // and it doesn't have recursive side effects we treat it conservatively as
      // side-effecting.
    } else if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
      return false;
    }
    return true;
  }


  void runOnFunction() override {
    auto f = getFunction();

    OpBuilder builder(f);

    Block* pre_block = nullptr;
    Block* start_block = nullptr;
    std::vector<Block*> filter_blocks;
    std::unordered_map<Block*, double> filter_ranks;
    Block* project_block = nullptr;
    Block* exit_block = nullptr;
    Block* join_block = nullptr;

    // Gather function arguments.
    SmallPtrSet<Value, 8> args;
    for (auto &arg : f.getArguments()) {
      args.insert(arg);
    }

    // Helper to check whether an operation is loop invariant wrt. SSA properties.
    SmallPtrSet<Operation *, 8> willBeMovedSet;
    SmallVector<Operation *, 8> opsToMove;
    auto isDefinedOutsideOfBody = [&](Value value) {
      auto definingOp = value.getDefiningOp();
      // Function arguments are always outside the loop.
      if (!!args.count(value)) {
        return true;
      }
      // Don't know what this means.
      if (definingOp == nullptr) {
        return false;
      }
      // Operand Already moved.
      if (!!willBeMovedSet.count(definingOp)) {
        return true;
      }

      // Check if defined outside select body.
      auto parent_block = definingOp->getBlock();
      if (parent_block == start_block || parent_block == project_block) {
        return false;
      }
      for (const auto& filter_block: filter_blocks) {
        if (parent_block == filter_block) return false;
      }
      return true;
    };

    auto findInvariants = [&](Block* block) {
      for (Operation &op : block->without_terminator()) {
        if (IsSelectInvariant(&op, isDefinedOutsideOfBody)) {
          opsToMove.push_back(&op);
          willBeMovedSet.insert(&op);
        }
      }
    };


    for (Block &block : f) {
      if (pre_block == nullptr) pre_block = &block;
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (isa<TableNextOp>(op)) {
          start_block = &block;
        }
        if (isa<JoinOp>(op)) {
          join_block = &block;
        }
        if (isa<FillResultOp>(op)) {
          project_block = &block;
        }
      }
      if (isa<mlir::ReturnOp>(block.getTerminator())) {
        exit_block = &block;
      }
      if (start_block != nullptr && &block != project_block && &block != start_block) {
        if (!isa<mlir::ReturnOp>(block.getTerminator())) {
          filter_blocks.emplace_back(&block);
        }
      }
    }

    // Not a select function
    if (start_block == nullptr && join_block == nullptr) return;
    if(join_block != nullptr) {
      int i = 0;
      std::vector<Value> join_table_operands;
      // Do Join Reordering
      for (Operation &op : llvm::make_early_inc_range(*join_block)) {
        if (isa<JoinOp>(op)) {

          if(i==0) {
            join_table_operands.push_back(op.getOperand(0));
            join_table_operands.push_back(op.getOperand(1));
          } else {
            join_table_operands.push_back(op.getOperand(0));
          }
          i++;
        }
      }

      std::map<int, Value> costs;
      EstimateCosts(costs, join_table_operands);
      join_table_operands.clear();
      for(auto &it: costs) {
        join_table_operands.push_back(it.second);
      }

      i = 0;
      // Do Join Reordering
      for (Operation &op : llvm::make_early_inc_range(*join_block)) {
        if (isa<JoinOp>(op)) {

          if(i==0) {
            op.setOperand(0, join_table_operands[i]);
            op.setOperand(1, join_table_operands[i+1]);
            i+=2;

          } else {
            op.setOperand(0, join_table_operands[i]);
            i++;
          }
        }
      }
      return;
    }
    // Impossible filters
    if (project_block == nullptr) {
      auto curr_term = pre_block->getTerminator();
      builder.setInsertionPointToEnd(pre_block);
      builder.create<mlir::BranchOp>(curr_term->getLoc(), exit_block);
      curr_term->erase();
      return;
    }
    // No filters
    if (filter_blocks.empty()) return;


    // Rank the filters.
    for (const auto& filter_block: filter_blocks) {
      filter_ranks.emplace(filter_block, EstimateRank(filter_block));
    }
    // Reorder
    std::sort(filter_blocks.begin(), filter_blocks.end(), [&](const auto& b1, const auto& b2) {
      return filter_ranks[b1] > filter_ranks[b2];
    });

    // Change jump targets.
    // Target of start block.
    {
      builder.setInsertionPointToEnd(start_block);
      auto term = start_block->getTerminator();
      llvm::outs() << "Start Terminator: " << *term << "\n";
      if (isa<mlir::CondBranchOp>(*term)) {
        auto branch_inst = dyn_cast<mlir::CondBranchOp>(*term);
        llvm::outs() << "Start Branch: " << *branch_inst << "\n";
        auto false_dest = branch_inst.getFalseDest();
        auto cond = branch_inst.getCondition();
        builder.create<mlir::CondBranchOp>(branch_inst.getLoc(), cond, filter_blocks[0], false_dest);
        branch_inst.erase();
      }
    }

    // Change filter targets
    {
      for (uint32_t i = 0; i < filter_blocks.size(); i++) {
        auto curr_block = filter_blocks[i];
        builder.setInsertionPointToEnd(curr_block);
        auto term = curr_block->getTerminator();
        if (isa<mlir::CondBranchOp>(*term)) {
          auto branch_inst = dyn_cast<mlir::CondBranchOp>(*term);
          auto false_dest = branch_inst.getFalseDest();
          auto true_dest = (i == filter_blocks.size() - 1) ? project_block : filter_blocks[i+1];
          auto cond = branch_inst.getCondition();
          builder.create<mlir::CondBranchOp>(branch_inst.getLoc(), cond, true_dest, false_dest);
          branch_inst.erase();
        }
      }
    }

    // Find the invariants.
    findInvariants(start_block);
    findInvariants(project_block);
    for (auto& filter_block: filter_blocks) {
      findInvariants(filter_block);
    }

    // Now move out
    for (const auto& op: opsToMove) {
      op->moveBefore(pre_block->getTerminator());
    }
  }

 private:
  double EstimateRank(Block* block) {
    uint32_t num_insts_{0};
    block->walk([&](mlir::Operation *op) {
      // TODO: estimate cost of op.
      // TODO: If op is terminate, estimate selectivity.
      num_insts_++;
    });
    return 1.0 / num_insts_;
  }

  void EstimateCosts(std::map<int,Value> & costs, std::vector<Value> &values) {
    for(auto value: values) {
      costs.insert(std::make_pair(rand()%10000, value));
    }
  }
};
} // end anonymous namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::sqlir::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
