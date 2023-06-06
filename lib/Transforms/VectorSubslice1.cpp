//===- Vectorsubslice.cpp - subslice vector operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass to subslice vectors into smaller ones which are
// more close to hardware vector size.
//
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "vector-subslice1"
// namespace mlir {
// namespace vector {
//#define GEN_PASS_DEF_VECTORSUBSLICE1
//#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
//} // namespace vector
//} // namespace mlir

namespace imex {
#define GEN_PASS_DEF_VECTORSUBSLICE1
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace mlir::vector;
using namespace imex;

namespace {
/// Collect perfectly nested loops starting from `rootForOps`.  Loops are
/// perfectly nested if each loop is the first and only non-terminator operation
/// in the parent loop.  Collect at most `maxLoops` loops and append them to
/// `forOps`.
template <typename T>
static void getPerfectlyNestedLoopsImpl(
    SmallVectorImpl<T> &forOps, T rootForOp,
    unsigned maxLoops = std::numeric_limits<unsigned>::max()) {
  for (unsigned i = 0; i < maxLoops; ++i) {
    forOps.push_back(rootForOp);
    Block &body = rootForOp.getRegion().front();
    if (body.begin() != std::prev(body.end(), 2))
      return;

    rootForOp = dyn_cast<T>(&body.front());
    if (!rootForOp)
      return;
  }
}
void getPerfectlyNestedLoops(SmallVectorImpl<AffineParallelOp> &nestedLoops,
                             AffineParallelOp root) {
  getPerfectlyNestedLoopsImpl(nestedLoops, root);
}

struct TransferReadOpPattern : public OpConversionPattern<TransferReadOp> {
  TransferReadOpPattern(TypeConverter &converter, MLIRContext *context,
                        bool enableSIMT)
      : OpConversionPattern<TransferReadOp>(converter, context),
        enableSIMT(enableSIMT) {}
  using OpConversionPattern<TransferReadOp>::OpConversionPattern;
  bool enableSIMT;
  LogicalResult
  matchAndRewrite(TransferReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string mode = enableSIMT ? "simt" : "simd";
    op->setAttr("mode", rewriter.getStringAttr(mode));
    Location loc = op.getLoc();
    auto indices = op.getIndices();
    auto size = indices.size();
    auto indVar = cast<AffineParallelOp>(op->getParentOp()).getIVs()[0];
    auto map = AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1));
    auto newIndex = rewriter.create<AffineApplyOp>(
        loc, map, ValueRange{indices[size - 1], indVar});
    auto newType = getTypeConverter()->convertType(op->getResultTypes()[0]);
    SmallVector<Value, 4> operands(op->getOperands());
    operands[1 + size - 1] = newIndex;
    rewriter.replaceOpWithNewOp<TransferReadOp>(op, newType, operands,
                                                op->getAttrs());
    return success();
  }
};
struct TransferWriteOpPattern : public OpConversionPattern<TransferWriteOp> {
  TransferWriteOpPattern(TypeConverter &converter, MLIRContext *context,
                         bool enableSIMT)
      : OpConversionPattern<TransferWriteOp>(converter, context),
        enableSIMT(enableSIMT) {}
  using OpConversionPattern<TransferWriteOp>::OpConversionPattern;
  bool enableSIMT;
  LogicalResult
  matchAndRewrite(TransferWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string mode = enableSIMT ? "simt" : "simd";
    op->setAttr("mode", rewriter.getStringAttr(mode));
    Location loc = op.getLoc();
    auto indices = op.getIndices();
    auto size = indices.size();
    auto indVar = cast<AffineParallelOp>(op->getParentOp()).getIVs()[0];
    auto map = AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1));
    auto newIndex = rewriter.create<AffineApplyOp>(
        loc, map, ValueRange{indices[size - 1], indVar});
    rewriter.updateRootInPlace(op, [&] {
      op.setOperand(0, adaptor.getVector());
      op.setOperand(2 + size - 1, newIndex);
    });
    return success();
  }
};
struct ReductionOpPattern : public OpConversionPattern<ReductionOp> {
  using OpConversionPattern<ReductionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReductionOp redOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = redOp->getLoc();
    // rewriter.updateRootInPlace(redOp, [&]() {
    //   redOp.setOperand(0, adaptor.getVector());
    //   redOp.setResultType(adaptor.getVector().getType());
    // });
    auto red =
        rewriter.create<ReductionOp>(loc, redOp.getKind(), adaptor.getVector());
    auto kind = redOp.getKind();
    gpu::AllReduceOperation newKind;
    switch (kind) {
    case CombiningKind::ADD:
      // addi tbd
      newKind = gpu::AllReduceOperation::ADD;
      break;
    case CombiningKind::MAXF:
      newKind = gpu::AllReduceOperation::MAX;
      break;
    default:
      break;
    }
    auto redAttr =
        gpu::AllReduceOperationAttr::get(rewriter.getContext(), newKind);
    auto uniformAttr = rewriter.getUnitAttr();
    auto op =
        rewriter.create<gpu::SubgroupReduceOp>(loc, red, redAttr, uniformAttr);
    rewriter.replaceOp(redOp, op->getResults());
    return success();
  }
};
struct BroadcastOpPattern : public OpConversionPattern<BroadcastOp> {
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<BroadcastOp>(op, newType,
                                             adaptor.getOperands());
    return success();
  }
};
template <typename OpTy>
struct ComputeOpPattern : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<OpTy>(op, adaptor.getOperands());
    return success();
  }
};
// a new pass
struct VectorSubslice1Pass
    : public imex::impl::VectorSubslice1Base<VectorSubslice1Pass> {
  // using Base::Base;
  using VectorSubslice1Base::VectorSubslice1Base;

  void runOnOperation() override {
    auto func = getOperation();
    AffineParallelOp root;
    for (auto &op : func.getBody().front()) {
      if (auto loop = dyn_cast<AffineParallelOp>(op)) {
        root = loop;
        // FIXME: 1 top paralleOp for now
        break;
      }
    }
    SmallVector<AffineParallelOp, 4> loops;
    getPerfectlyNestedLoops(loops, root);
    auto readOp =
        llvm::to_vector(loops.back().getOps<TransferReadOp>()).front();
    auto vecType = readOp.getResult().getType().cast<VectorType>();
    assert(vecType.getRank() == 1 && "support 1D vector for now");
    // unsigned vectorSize = vecType.getShape()[0];
    subsliceLoop(loops.back(), vecType, simdSize);
  }

  void subsliceLoop(AffineParallelOp loop, VectorType vecType,
                    unsigned simdSize) {
    unsigned vectorSize = vecType.getShape()[0];
    unsigned subVectorSize = vectorSize / simdSize;
    auto *ctx = &getContext();

    OpBuilder b(loop);
    SmallVector<Type, 4> types;
    SmallVector<arith::AtomicRMWKind, 4> reductions;
    SmallVector<int64_t, 4> ranges{simdSize};
    auto innerLoop = b.create<AffineParallelOp>(loop.getLoc(), TypeRange{},
                                                reductions, ranges);
    auto yield = loop.getBody()->getTerminator();
    innerLoop.getBody()->getOperations().splice(
        std::prev(innerLoop.getBody()->end()), loop.getBody()->getOperations());
    // loop.getBody()->moveBefore()
    Block *body = loop.getBody();
    innerLoop->moveBefore(body, body->end());
    yield->moveBefore(body, body->end());

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([&](VectorType type) {
      assert(type.getRank() == 1 && "n-D shape TBD");
      if (vectorSize == type.getShape()[0]) {
        return VectorType::get(subVectorSize, type.getElementType());
      } else {
        return type;
      }
    });
    RewritePatternSet patterns(ctx);
    populateSubslicePatterns(converter, patterns, enableSIMT);
    ConversionTarget target(*ctx);
    target.addLegalOp<AffineApplyOp>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                                      vector::VectorDialect>(
        [&](Operation *op) { return converter.isLegal(op); });
    if (failed(applyPartialConversion(innerLoop, target, std::move(patterns))))
      signalPassFailure();
  }
  void populateSubslicePatterns(TypeConverter &converter,
                                RewritePatternSet &patterns, bool enableSIMT) {
    patterns.add<
        /*TransferReadOpPattern, TransferWriteOpPattern,*/ ReductionOpPattern,
        BroadcastOpPattern, ComputeOpPattern<arith::SubFOp>,
        ComputeOpPattern<arith::DivFOp>, ComputeOpPattern<math::ExpOp>>(
        converter, patterns.getContext());
    patterns.add<TransferReadOpPattern, TransferWriteOpPattern>(
        converter, patterns.getContext(), enableSIMT);
  }
};
} // namespace

std::unique_ptr<Pass> imex::createVectorSubslice1Pass() {
  return std::make_unique<VectorSubslice1Pass>();
}
