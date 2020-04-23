//===- Ops.td - SQLIR dialect operation definitions ----------*- tablegen -*-===//
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

// Provide a definition of the 'SQLIR' dialect in the ODS framework so that we
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
  let summary = "getcolumn operation";
  let description = [{
    The "GetColumn" builtin operation prints a given input tableid and columnid, and produces
    a SignlessIntegerLike.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike:$res);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def JoinOp : SQLIR_Op<"Join"> {
  let summary = "join operation";
  let description = [{
    The "Join" builtin operation takes table_id of two tables and returns table oid of a temp
    table representing the "Natural Join" of the input tables
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike:$res);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def TableNextOp : SQLIR_Op<"TableNext"> {
  let summary = "hasnext operation";
  let description = [{
    The "TableNextOp" builtin op.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins SignlessIntegerLike:$input);
  let results = (outs I1:$res);

  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
}

def NewTempTableOp : SQLIR_Op<"NewTempTable"> {
  let summary = "new temp table operation";
  let description = [{
  The "NewTempTable" builtin op.
  }];

  let arguments = (ins);
  let results = (outs I64:$res);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def FetchValueOp : SQLIR_Op<"FetchValue"> {
  let summary = "fetch value operation";
  let description = [{
    The "FetchValueOp" builtin op.
  }];

  let arguments = (ins SignlessIntegerLike:$temp_id, SignlessIntegerLike: $row_idx, SignlessIntegerLike: $col_idx);
  let results = (outs SignlessIntegerLike:$res);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def FillResultOp : SQLIR_Op<"FillResult"> {
  let summary = "fill result operation";
  let description = [{
    The "fill result" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins SignlessIntegerLike:$input, SignlessIntegerLike:$temp);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def PrintOp : SQLIR_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins SignlessIntegerLike:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // SQLIR_OPS
