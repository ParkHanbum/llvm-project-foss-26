//===- EvaluatorTest.cpp - Runtime and initializer evaluation
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Evaluator.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class EvaluatorTest : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;

  void SetUp() override {
    SMDiagnostic Error;
    M = parseAssemblyString(R"IR(
      target datalayout = "e-p:64:64"
      @state = global i32 7
      declare void @opaque()
      declare i8 @llvm.vector.reduce.or.v2i8(<2 x i8>)
      declare i1 @llvm.is.constant.i32(i32)

      define internal i1 @constant_query(i32 %x) {
        %r = call i1 @llvm.is.constant.i32(i32 %x)
        ret i1 %r
      }
      define internal i32 @pure(i32 %x) {
        %r = mul i32 %x, 3
        ret i32 %r
      }
      define internal range(i32 9, 10) i32 @return_range(i32 %x) {
        %r = mul i32 %x, 3
        ret i32 %r
      }
      define internal i32 @call_return_range(i32 %x) {
        %r = call range(i32 9, 10) i32 @pure(i32 %x)
        ret i32 %r
      }
      define internal i32 @call_parameter_range(i32 %x) {
        %r = call i32 @pure(i32 range(i32 2, 4) %x)
        ret i32 %r
      }
      define internal <2 x i8> @vector_parameter_range(<2 x i8> range(i8 2, 4) %x) {
        ret <2 x i8> %x
      }
      define internal i32 @read(i32 %x) {
        %r = load i32, ptr @state
        ret i32 %r
      }
      define internal i32 @write(i32 %x) {
        store i32 %x, ptr @state
        ret i32 %x
      }
      define internal i32 @local(i32 %x) {
        %slot = alloca i32, align 4
        store i32 %x, ptr %slot, align 4
        ret i32 9
      }
      define internal i32 @oversized_store(i32 %x) {
        %slot = alloca i8, align 4
        store i32 %x, ptr %slot, align 4
        ret i32 9
      }
      define internal i32 @atomic_store(i32 %x) {
        %slot = alloca i32, align 4
        store atomic i32 %x, ptr %slot release, align 4
        ret i32 9
      }
      define internal i32 @volatile_store(i32 %x) {
        %slot = alloca i32, align 4
        store volatile i32 %x, ptr %slot, align 4
        ret i32 9
      }
      define internal i32 @path(i32 %x) {
        %ok = icmp sge i32 %x, 0
        br i1 %ok, label %pure, label %effect
      pure:
        ret i32 %x
      effect:
        call void @opaque()
        ret i32 14
      }
      define internal i32 @nested(i32 %x) {
        %r = call i32 @pure(i32 %x)
        ret i32 %r
      }
      define internal i32 @recursive(i32 %x) {
        %r = call i32 @recursive(i32 %x)
        ret i32 %r
      }
      define internal i32 @loop(i32 %x) {
        br label %body
      body:
        br label %body
      }
      define weak i32 @interposable(i32 %x) {
        ret i32 %x
      }
      define internal i8 @overflow(i8 %x) {
        %r = add nsw i8 %x, 1
        ret i8 %r
      }
      define internal i16 @negative(i8 %x) {
        %r = zext nneg i8 %x to i16
        ret i16 %r
      }
      define internal i8 @exact(i8 %x) {
        %r = udiv exact i8 %x, 2
        ret i8 %r
      }
      define internal i8 @zero_divisor(i8 %x) {
        %r = urem i8 %x, 0
        ret i8 %r
      }
      define internal <2 x i8> @vector_overflow() {
        %r = add nsw <2 x i8> <i8 1, i8 127>, <i8 1, i8 1>
        ret <2 x i8> %r
      }
      define internal i8 @reduce() {
        %r = call i8 @llvm.vector.reduce.or.v2i8(<2 x i8> <i8 1, i8 2>)
        ret i8 %r
      }
      define internal i32 @polynomial(i32 %x, i32 %y) {
        %p = mul nsw i32 %x, %y
        %r = add nsw i32 %p, 1
        ret i32 %r
      }
      define i32 @expression(i32 %x) {
        %y = add nsw i32 %x, 2
        %r = call i32 @polynomial(i32 %x, i32 %y)
        ret i32 %r
      }
      define i32 @phi_expression(i1 %c, i1 %d, i32 %x) {
        br i1 %c, label %left, label %right
      left:
        %left.value = add i32 %x, 1
        br label %merge
      right:
        %right.value = sub i32 %x, 1
        br label %merge
      merge:
        %p = phi i32 [ %left.value, %left ], [ %right.value, %right ]
        %n = select i1 %d, i32 2, i32 3
        %r = mul i32 %p, %n
        ret i32 %r
      }
    )IR",
                            Error, Context);
    ASSERT_TRUE(M);
  }

  bool evaluate(StringRef Name, ArrayRef<int64_t> Inputs, Constant *&Result,
                unsigned Steps = 256, unsigned Depth = 4) {
    Function *F = M->getFunction(Name);
    SmallVector<Constant *, 4> Args;
    for (auto [Arg, Input] : zip(F->args(), Inputs))
      Args.push_back(ConstantInt::getSigned(Arg.getType(), Input));
    Evaluator Eval(M->getDataLayout(), nullptr, Evaluator::Mode::Runtime, Steps,
                   Depth);
    return Eval.EvaluateFunction(F, Result, Args);
  }
};

TEST_F(EvaluatorTest, RuntimeArithmetic) {
  Constant *Result = nullptr;
  ASSERT_TRUE(evaluate("pure", {3}, Result));
  EXPECT_EQ(cast<ConstantInt>(Result)->getSExtValue(), 9);
}

TEST_F(EvaluatorTest, RuntimeRejectsMutableMemory) {
  Constant *Result = nullptr;
  EXPECT_FALSE(evaluate("read", {2}, Result));
  EXPECT_FALSE(evaluate("write", {2}, Result));
  EXPECT_EQ(cast<ConstantInt>(M->getNamedGlobal("state")->getInitializer())
                ->getSExtValue(),
            7);
}

TEST_F(EvaluatorTest, InitializerModeRemainsTheDefault) {
  Evaluator Eval(M->getDataLayout(), nullptr);
  Constant *Result = nullptr;
  SmallVector<Constant *, 1> Args{
      ConstantInt::get(Type::getInt32Ty(Context), 2)};
  ASSERT_TRUE(Eval.EvaluateFunction(M->getFunction("read"), Result, Args));
  EXPECT_EQ(cast<ConstantInt>(Result)->getSExtValue(), 7);
}

TEST_F(EvaluatorTest, InitializerWritesAreSimulated) {
  Evaluator Eval(M->getDataLayout(), nullptr);
  Constant *Result = nullptr;
  SmallVector<Constant *, 1> Args{
      ConstantInt::get(Type::getInt32Ty(Context), 2)};
  ASSERT_TRUE(Eval.EvaluateFunction(M->getFunction("write"), Result, Args));
  auto Writes = Eval.getMutatedInitializers();
  ASSERT_EQ(Writes.size(), 1u);
  EXPECT_EQ(cast<ConstantInt>(Writes.lookup(M->getNamedGlobal("state")))
                ->getSExtValue(),
            2);
  EXPECT_EQ(cast<ConstantInt>(M->getNamedGlobal("state")->getInitializer())
                ->getSExtValue(),
            7);
}

TEST_F(EvaluatorTest, RuntimeAllowsOnlyUnobservableStackStores) {
  Constant *Result = nullptr;
  ASSERT_TRUE(evaluate("local", {2}, Result));
  EXPECT_EQ(cast<ConstantInt>(Result)->getSExtValue(), 9);
  EXPECT_FALSE(evaluate("oversized_store", {2}, Result));
  EXPECT_FALSE(evaluate("atomic_store", {2}, Result));
  EXPECT_FALSE(evaluate("volatile_store", {2}, Result));
}

TEST_F(EvaluatorTest, OnlyExecutedEffectsMatter) {
  Constant *Result = nullptr;
  ASSERT_TRUE(evaluate("path", {2}, Result));
  EXPECT_EQ(cast<ConstantInt>(Result)->getSExtValue(), 2);
  EXPECT_FALSE(evaluate("path", {-1}, Result));
}

TEST_F(EvaluatorTest, PoisonFlagsAndImmediateUBAreRejected) {
  Constant *Result = nullptr;
  EXPECT_FALSE(evaluate("overflow", {127}, Result));
  EXPECT_FALSE(evaluate("negative", {-1}, Result));
  EXPECT_FALSE(evaluate("exact", {3}, Result));
  EXPECT_FALSE(evaluate("zero_divisor", {2}, Result));
  EXPECT_FALSE(evaluate("vector_overflow", {}, Result));
}

TEST_F(EvaluatorTest, PureIntrinsicUsesTheExistingConstantFolder) {
  Constant *Result = nullptr;
  ASSERT_TRUE(evaluate("reduce", {}, Result));
  EXPECT_EQ(cast<ConstantInt>(Result)->getSExtValue(), 3);
}

TEST_F(EvaluatorTest, WorkAndCallDepthAreBounded) {
  Constant *Result = nullptr;
  EXPECT_FALSE(evaluate("pure", {2}, Result, 1));
  EXPECT_FALSE(evaluate("nested", {2}, Result, 256, 1));
  ASSERT_TRUE(evaluate("nested", {2}, Result, 256, 2));
  EXPECT_EQ(cast<ConstantInt>(Result)->getSExtValue(), 6);
  EXPECT_FALSE(evaluate("recursive", {2}, Result));
  EXPECT_FALSE(evaluate("loop", {2}, Result));
  EXPECT_FALSE(evaluate("interposable", {2}, Result));
}

TEST_F(EvaluatorTest, CorrelatedExpressionArgumentsUseOneAssignment) {
  Function *F = M->getFunction("expression");
  Value *Root = cast<ReturnInst>(F->back().getTerminator())->getReturnValue();
  for (auto [Input, Expected] : {std::pair{3, 16}, std::pair{5, 36}}) {
    Evaluator Eval(M->getDataLayout(), nullptr, Evaluator::Mode::Runtime);
    Constant *Result =
        Eval.EvaluateExpression(Root, [&](Value *V) -> Constant * {
          return V == F->getArg(0)
                     ? ConstantInt::get(Type::getInt32Ty(Context), Input)
                     : nullptr;
        });
    ASSERT_TRUE(Result);
    EXPECT_EQ(cast<ConstantInt>(Result)->getSExtValue(), Expected);
  }
}

TEST_F(EvaluatorTest, UnboundPhiDoesNotVisitNonDominatingIncomingValues) {
  Function *F = M->getFunction("phi_expression");
  Value *Root = cast<ReturnInst>(F->back().getTerminator())->getReturnValue();
  bool VisitedIncoming = false;
  Evaluator Eval(M->getDataLayout(), nullptr, Evaluator::Mode::Runtime);
  Constant *Result = Eval.EvaluateExpression(Root, [&](Value *V) -> Constant * {
    VisitedIncoming |=
        V->getName() == "left.value" || V->getName() == "right.value";
    return V->getName() == "n" ? ConstantInt::get(Type::getInt32Ty(Context), 2)
                               : nullptr;
  });
  EXPECT_FALSE(Result);
  EXPECT_FALSE(VisitedIncoming);
}
TEST_F(EvaluatorTest, RuntimeRespectsCallAndFunctionRanges) {
  Constant *Result = nullptr;
  EXPECT_FALSE(evaluate("return_range", {2}, Result));
  EXPECT_TRUE(evaluate("return_range", {3}, Result));
  EXPECT_FALSE(evaluate("call_return_range", {2}, Result));
  EXPECT_TRUE(evaluate("call_return_range", {3}, Result));
  EXPECT_FALSE(evaluate("call_parameter_range", {4}, Result));
  EXPECT_TRUE(evaluate("call_parameter_range", {2}, Result));

  Function *F = M->getFunction("vector_parameter_range");
  auto *Ty = Type::getInt8Ty(Context);
  auto Check = [&](unsigned High) {
    SmallVector<Constant *, 2> Lanes{ConstantInt::get(Ty, 2),
                                     ConstantInt::get(Ty, High)};
    SmallVector<Constant *, 1> Args{ConstantVector::get(Lanes)};
    Evaluator Eval(M->getDataLayout(), nullptr, Evaluator::Mode::Runtime);
    return Eval.EvaluateFunction(F, Result, Args);
  };
  EXPECT_TRUE(Check(3));
  EXPECT_FALSE(Check(4));
}
TEST_F(EvaluatorTest, RuntimeValuesDoNotBecomeManifestConstants) {
  Constant *Result = nullptr;
  EXPECT_FALSE(evaluate("constant_query", {2}, Result));
}
} // namespace
