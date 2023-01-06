//===- AffineRaiseReduction.cpp -affine raise reduction pass-----*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass matches affine loop carried load/store and transforms it to
/// iter_args
///
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
// #include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"

namespace imex {
#define GEN_PASS_DEF_AFFINERAISEREDUCTION
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;

namespace {
struct AffineRaiseReduction
    : public imex::impl::AffineRaiseReductionBase<AffineRaiseReduction> {
  void runOnOperation() override {
    auto &postDom = getAnalysis<PostDominanceInfo>();
    auto func = getOperation();
    func.walk([&](AffineForOp forOp) {
      // collect memref
      llvm::DenseSet<mlir::Value> memrefs;
      forOp.walk([&](Operation *op) {
        mlir::Value memref;
        if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
          memref = loadOp.getMemref();
        } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
          memref = storeOp.getMemref();
        } else {
          return;
        }
        memrefs.insert(memref);
      });
      // process each memref
      for (auto memref : memrefs) {
        SmallVector<AffineLoadOp, 4> loads;
        AffineStoreOp store;
        bool supported = true;
        // collect load/store
        for (auto user : memref.getUsers()) {
          if (!forOp->isProperAncestor(user)) {
            continue;
          }
          if (isa<AffineLoadOp>(user)) {
            loads.push_back(cast<AffineLoadOp>(user));
          } else if (isa<AffineStoreOp>(user)) {
            // multi store not supported
            if (store) {
              supported = false;
              break;
            }
            store = cast<AffineStoreOp>(user);
          } else {
            // other memref uses not supported
            supported = false;
            break;
          }
        }
        if (!supported || !store) {
          continue;
        }
        MemRefAccess storeAccess(store);
        AffineValueMap storeMap;
        storeAccess.getAccessMap(&storeMap);
        AffineMap map = storeMap.getAffineMap();
        llvm::ArrayRef operands = storeMap.getOperands();
        // match reduction:
        // store should post dominate all loads
        // all access should be the same
        // all access should be loop invariant
        if (store->getParentOp() == forOp &&
            llvm::all_of(loads,
                         [&](AffineLoadOp load) {
                           return postDom.postDominates(store, load);
                         }) &&
            !llvm::is_contained(operands, forOp.getInductionVar()) &&
            llvm::all_of(loads, [&](AffineLoadOp load) {
              MemRefAccess loadAccess(load);
              return loadAccess == storeAccess;
            })) {
          mlir::OpBuilder b(forOp);
          auto loc = forOp.getLoc();
          auto init = b.create<AffineLoadOp>(loc, memref, map, operands);
          AffineForOp newLoop = replaceForOpWithNewYields(
              b, forOp, init.getResult(), store.getValue(), init.getResult(),
              /*replaceLoopResults*/ true);
          b.setInsertionPointAfter(newLoop);
          b.create<AffineStoreOp>(loc, newLoop.getResults().back(), memref, map,
                                  operands);
          // clean up
          forOp.erase();
          for (auto load : loads) {
            load.getResult().replaceAllUsesWith(
                newLoop.getRegionIterArgs().back());
            load.erase();
          }
          store.erase();
        }
      }
    });
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createAffineRaiseReductionPass() {
  return std::make_unique<AffineRaiseReduction>();
}
} // namespace imex
