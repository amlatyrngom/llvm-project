//===- Ops.td - Toy dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the SQLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SQLIR_OPS
#define SQLIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def SqlIR_Dialect : Dialect {
  let name = "sqlir";
  let cppNamespace = "sqlir";
}

// Base class for SQLIR dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class SQLIR_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SqlIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// SQLIR Operations
//===----------------------------------------------------------------------===//

def GetColumnOp : SQLIR_Op<"GetColumn"> {
  let summary = "print operation";
  let description = [{
    The "GetColumn" builtin operation prints a given input tableid and columnid, and produces
    a SignlessIntegerLike.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike);
}


#endif // SQLIR_OPS
