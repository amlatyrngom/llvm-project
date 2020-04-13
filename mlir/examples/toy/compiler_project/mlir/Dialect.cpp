#include "mlir/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir::sqlir {
  SqlIRDialect::SqlIRDialect(mlir::MLIRContext *ctx) : mlir::Dialect("sqlir", ctx) {
  	  addOperations<
		#define GET_OP_LIST
		#include "mlir/Ops.cpp.inc"
      >();
  }
}
