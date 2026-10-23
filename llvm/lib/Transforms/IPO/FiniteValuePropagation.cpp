//===- FiniteValuePropagation.cpp - Finite input folding ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Evaluate a computation for each member of a small input set, preserving the
// relationship between all uses of that input. In particular this can evaluate
// a helper's successful paths without inlining its unreachable error handling.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FiniteValuePropagation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "finite-value-propagation"
STATISTIC(NumFolded, "Computations folded over finite input sets");
STATISTIC(NumCallsFolded, "Calls folded over finite input sets");

namespace {

constexpr unsigned MaxValues = 4;
constexpr unsigned MaxExpressionSize = 32;
constexpr unsigned MaxEvaluationSteps = 256;

struct InputSet {
  Value *Input;
  SmallVector<ConstantInt *, MaxValues> Values;
  // For a select input, retain its original condition and arm order.
  Value *Condition = nullptr;
};

/// ConstantFoldInstOperands deliberately does not preserve poison-generating
/// flags. Check them before using its result, including each fixed vector lane.
static bool flagsHold(Instruction &I, ArrayRef<Constant *> Ops) {
  if (!I.hasPoisonGeneratingFlags())
    return true;
  unsigned Lanes = 1;
  if (auto *VT = dyn_cast<FixedVectorType>(Ops[0]->getType()))
    Lanes = VT->getNumElements();
  for (unsigned Lane = 0; Lane != Lanes; ++Lane) {
    auto GetInt = [&](unsigned Op) -> ConstantInt * {
      Constant *C = Ops[Op];
      if (C->getType()->isVectorTy())
        C = C->getAggregateElement(Lane);
      return dyn_cast_or_null<ConstantInt>(C);
    };
    auto *C0 = GetInt(0);
    if (!C0)
      return false;
    const APInt &A = C0->getValue();
    if (auto *Trunc = dyn_cast<TruncInst>(&I)) {
      APInt R = A.trunc(I.getType()->getScalarSizeInBits());
      if ((Trunc->hasNoUnsignedWrap() && R.zext(A.getBitWidth()) != A) ||
          (Trunc->hasNoSignedWrap() && R.sext(A.getBitWidth()) != A))
        return false;
      continue;
    }
    if (isa<ZExtInst>(I)) {
      if (I.hasNonNeg() && A.isNegative())
        return false;
      continue;
    }
    auto *C1 = Ops.size() > 1 ? GetInt(1) : nullptr;
    if (!C1)
      return false;
    const APInt &B = C1->getValue();
    if (auto *Cmp = dyn_cast<ICmpInst>(&I)) {
      if (Cmp->hasSameSign() && A.isNegative() != B.isNegative())
        return false;
      continue;
    }
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO)
      return false;
    bool SignedOverflow = false, UnsignedOverflow = false;
    switch (BO->getOpcode()) {
    case Instruction::Add:
      (void)A.sadd_ov(B, SignedOverflow);
      (void)A.uadd_ov(B, UnsignedOverflow);
      break;
    case Instruction::Sub:
      (void)A.ssub_ov(B, SignedOverflow);
      (void)A.usub_ov(B, UnsignedOverflow);
      break;
    case Instruction::Mul:
      (void)A.smul_ov(B, SignedOverflow);
      (void)A.umul_ov(B, UnsignedOverflow);
      break;
    case Instruction::Shl:
      if (B.uge(A.getBitWidth()))
        return false;
      (void)A.sshl_ov(B, SignedOverflow);
      (void)A.ushl_ov(B, UnsignedOverflow);
      break;
    case Instruction::LShr:
    case Instruction::AShr:
      if (B.uge(A.getBitWidth()) ||
          (BO->isExact() && A.countTrailingZeros() < B.getZExtValue()))
        return false;
      continue;
    case Instruction::UDiv:
    case Instruction::SDiv:
      if (B.isZero() || (BO->getOpcode() == Instruction::SDiv &&
                         A.isMinSignedValue() && B.isAllOnes()))
        return false;
      if (BO->isExact() &&
          !(BO->getOpcode() == Instruction::UDiv ? A.urem(B) : A.srem(B))
               .isZero())
        return false;
      continue;
    case Instruction::Or:
      if (!(A & B).isZero())
        return false;
      continue;
    default:
      return false;
    }
    if ((BO->hasNoSignedWrap() && SignedOverflow) ||
        (BO->hasNoUnsignedWrap() && UnsignedOverflow))
      return false;
  }
  return true;
}

/// A bounded, side-effect-free interpreter. Only the executed helper paths are
/// inspected. Unexecuted paths may contain throws, stores, and other effects.
class PartialEvaluator {
  const DataLayout &DL;
  const TargetLibraryInfo &TLI;
  unsigned Steps = MaxEvaluationSteps;
  SmallPtrSet<Function *, 4> ActiveCalls;
  DenseMap<Value *, Constant *> Cache;
  function_ref<Constant *(Value *)> ContextConstant;

  using Lookup = function_ref<Constant *(Value *)>;

  Constant *evaluateCall(CallInst &CI, Lookup Get) {
    Function *F = CI.getCalledFunction();
    if (!F || CI.isMustTailCall() || CI.hasOperandBundles() ||
        CI.isConvergent() || CI.hasFnAttr(Attribute::ReturnsTwice) ||
        CI.hasFnAttr("loader-replaceable") || F->isVarArg() ||
        F->getFunctionType() != CI.getFunctionType() ||
        F->getCallingConv() != CI.getCallingConv())
      return nullptr;
    SmallVector<Constant *, 8> Args;
    for (Value *Arg : CI.args()) {
      Constant *C = Get(Arg);
      if (!C)
        return nullptr;
      Args.push_back(C);
    }
    if (F->isIntrinsic()) {
      if (CI.mayHaveSideEffects() || !canConstantFoldCallTo(&CI, F, &TLI))
        return nullptr;
      return ConstantFoldCall(&CI, F, Args, &TLI,
                              /*AllowNonDeterministic=*/false);
    }
    if (any_of(Args, [](Constant *C) {
          return !C->getType()->isIntOrIntVectorTy() ||
                 !isGuaranteedNotToBeUndefOrPoison(C);
        }))
      return nullptr;
    if (!F->hasExactDefinition() || F->doesNotReturn() ||
        F->hasFnAttribute(Attribute::OptimizeNone) ||
        F->hasFnAttribute(Attribute::Naked) || ActiveCalls.size() == 4 ||
        ActiveCalls.contains(F))
      return nullptr;
    for (auto [Arg, C] : zip(F->args(), Args))
      if (auto R = Arg.getRange())
        if (auto *CI = dyn_cast<ConstantInt>(C))
          if (!R->contains(CI->getValue()))
            return nullptr;
    ActiveCalls.insert(F);

    DenseMap<Value *, Constant *> Frame;
    for (auto [Arg, C] : zip(F->args(), Args))
      Frame[&Arg] = C;
    auto GetLocal = [&](Value *V) -> Constant * {
      if (auto *C = dyn_cast<Constant>(V))
        return C;
      return Frame.lookup(V);
    };
    Constant *Result = nullptr;
    BasicBlock *BB = &F->getEntryBlock(), *Pred = nullptr;
    SmallPtrSet<BasicBlock *, 16> Visited;
    while (BB && Visited.insert(BB).second) {
      BasicBlock *Next = nullptr;
      bool Failed = false;
      for (Instruction &I : *BB) {
        if (!Steps) {
          Failed = true;
          break;
        }
        --Steps;
        if (auto *PN = dyn_cast<PHINode>(&I)) {
          if (!Pred ||
              !(Frame[PN] = GetLocal(PN->getIncomingValueForBlock(Pred)))) {
            Failed = true;
            break;
          }
        } else if (auto *RI = dyn_cast<ReturnInst>(&I)) {
          Result =
              RI->getReturnValue() ? GetLocal(RI->getReturnValue()) : nullptr;
          break;
        } else if (auto *BI = dyn_cast<UncondBrInst>(&I)) {
          Next = BI->getSuccessor(0);
          break;
        } else if (auto *BI = dyn_cast<CondBrInst>(&I)) {
          if (auto *C =
                  dyn_cast_or_null<ConstantInt>(GetLocal(BI->getCondition())))
            Next = BI->getSuccessor(C->isZero());
          else
            Failed = true;
          break;
        } else if (auto *SI = dyn_cast<SwitchInst>(&I)) {
          if (auto *C =
                  dyn_cast_or_null<ConstantInt>(GetLocal(SI->getCondition())))
            Next = SI->findCaseValue(C)->getCaseSuccessor();
          else
            Failed = true;
          break;
        } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          // An unused local stack slot has no observable effect. Its address
          // cannot be evaluated, so any executed escape or load still fails.
          if (!AI->isStaticAlloca() || AI->isUsedWithInAlloca() ||
              AI->isSwiftError()) {
            Failed = true;
            break;
          }
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          auto *AI = dyn_cast<AllocaInst>(SI->getPointerOperand());
          auto Size = AI ? AI->getAllocationSize(DL) : std::nullopt;
          TypeSize StoreSize =
              DL.getTypeStoreSize(SI->getValueOperand()->getType());
          if (SI->isVolatile() || SI->isAtomic() || !AI ||
              AI->getFunction() != F || !Size || Size->isScalable() ||
              StoreSize.isScalable() ||
              StoreSize.getFixedValue() > Size->getFixedValue() ||
              SI->getAlign() > AI->getAlign()) {
            Failed = true;
            break;
          }
        } else if (isa<DbgInfoIntrinsic>(I)) {
          continue;
        } else {
          Constant *C = fold(I, GetLocal);
          if (!C) {
            Failed = true;
            break;
          }
          Frame[&I] = C;
        }
      }
      if (Failed || Result || !Next)
        break;
      Pred = BB;
      BB = Next;
    }
    ActiveCalls.erase(F);
    return Result;
  }

  Constant *fold(Instruction &I, Lookup Get) {
    if (auto *CI = dyn_cast<CallInst>(&I))
      return evaluateCall(*CI, Get);
    if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects() ||
        I.isTerminator() || isa<PHINode, FreezeInst>(I) ||
        !I.getType()->isIntOrIntVectorTy())
      return nullptr;
    SmallVector<Constant *, 8> Ops;
    for (Value *V : I.operands()) {
      Constant *C = Get(V);
      if (!C)
        return nullptr;
      Ops.push_back(C);
    }
    if (Ops.empty() || !flagsHold(I, Ops))
      return nullptr;
    // Division by zero (and signed division overflow) is immediate UB, even
    // when the result would subsequently be masked by a select.
    if (I.getOpcode() == Instruction::UDiv ||
        I.getOpcode() == Instruction::SDiv ||
        I.getOpcode() == Instruction::URem ||
        I.getOpcode() == Instruction::SRem) {
      unsigned Lanes = 1;
      if (auto *VT = dyn_cast<FixedVectorType>(I.getType()))
        Lanes = VT->getNumElements();
      for (unsigned Lane = 0; Lane != Lanes; ++Lane) {
        auto Get = [&](unsigned Op) {
          return dyn_cast_or_null<ConstantInt>(
              I.getType()->isVectorTy() ? Ops[Op]->getAggregateElement(Lane)
                                        : Ops[Op]);
        };
        ConstantInt *A = Get(0), *B = Get(1);
        if (!A || !B || B->isZero() ||
            ((I.getOpcode() == Instruction::SDiv ||
              I.getOpcode() == Instruction::SRem) &&
             A->getValue().isMinSignedValue() && B->isMinusOne()))
          return nullptr;
      }
    }
    return ConstantFoldInstOperands(&I, Ops, DL, &TLI,
                                    /*AllowNonDeterministic=*/false);
  }

public:
  PartialEvaluator(const DataLayout &DL, const TargetLibraryInfo &TLI,
                   Value *Input, ConstantInt *C,
                   function_ref<Constant *(Value *)> ContextConstant)
      : DL(DL), TLI(TLI), ContextConstant(ContextConstant) {
    Cache[Input] = C;
  }

  Constant *evaluate(Value *V, unsigned Depth = 0) {
    if (auto *C = dyn_cast<Constant>(V))
      return C;
    if (Constant *C = Cache.lookup(V))
      return C;
    if (!Steps || Depth == MaxExpressionSize)
      return nullptr;
    --Steps;
    if (Constant *C = ContextConstant(V))
      return C;
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return nullptr;
    Constant *C = fold(*I, [&](Value *Op) { return evaluate(Op, Depth + 1); });
    if (C)
      Cache[V] = C;
    return C;
  }
};

static void findInputSets(Instruction &Root, LazyValueInfo &LVI,
                          DominatorTree &DT, AssumptionCache &AC,
                          SmallVectorImpl<InputSet> &Sets) {
  SmallVector<Value *, MaxExpressionSize> Work;
  SmallPtrSet<Value *, MaxExpressionSize> Seen;
  for (Value *Op : Root.operands())
    if (!isa<Constant>(Op) && Op->getType()->isIntegerTy())
      Work.push_back(Op);
  for (unsigned N = 0; N < Work.size() && N < MaxExpressionSize; ++N) {
    Value *V = Work[N];
    if (!Seen.insert(V).second)
      continue;
    if (auto *SI = dyn_cast<SelectInst>(V)) {
      auto *T = dyn_cast<ConstantInt>(SI->getTrueValue());
      auto *F = dyn_cast<ConstantInt>(SI->getFalseValue());
      if (T && F && T != F)
        Sets.push_back({V, {T, F}, SI->getCondition()});
    }
    if (isGuaranteedNotToBeUndefOrPoison(V, &AC, &Root, &DT)) {
      // A shared successor with no other predecessor admits exactly the
      // switch labels targeting it, even when several labels share that edge.
      auto *Node = DT.getNode(Root.getParent());
      if (Node)
        Node = Node->getIDom();
      for (unsigned D = 0; Node && D != 16; ++D, Node = Node->getIDom()) {
        auto *SI = dyn_cast<SwitchInst>(Node->getBlock()->getTerminator());
        if (!SI || SI->getCondition() != V || SI->getNumCases() > 32)
          continue;
        for (BasicBlock *Succ : successors(SI->getParent())) {
          if (Succ == SI->getDefaultDest() ||
              Succ->getUniquePredecessor() != SI->getParent() ||
              !DT.dominates(Succ, Root.getParent()))
            continue;
          InputSet S{V, {}};
          for (auto C : SI->cases())
            if (C.getCaseSuccessor() == Succ)
              S.Values.push_back(C.getCaseValue());
          if (!S.Values.empty() && S.Values.size() <= MaxValues)
            Sets.push_back(std::move(S));
          break;
        }
      }
      ConstantRange R = LVI.getConstantRange(V, &Root, /*UndefAllowed=*/false);
      if (R.isEmptySet())
        continue;
      APInt Min = R.getUnsignedMin(), Max = R.getUnsignedMax();
      if ((Max - Min).ult(MaxValues)) {
        InputSet S{V, {}};
        for (APInt C = Min;; ++C) {
          S.Values.push_back(ConstantInt::get(V->getContext(), C));
          if (C == Max)
            break;
        }
        Sets.push_back(std::move(S));
      }
    }
    if (auto *I = dyn_cast<Instruction>(V))
      if (!I->mayReadOrWriteMemory() && !isa<PHINode, FreezeInst>(I))
        for (Value *Op : I->operands())
          if (!isa<Constant>(Op) && Op->getType()->isIntegerTy())
            Work.push_back(Op);
  }
}

static InstructionCost instructionSize(Instruction &I,
                                       TargetTransformInfo &TTI) {
  InstructionCost Cost =
      isa<CastInst>(I)
          ? 0
          : TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);
  for (auto [Idx, Op] : enumerate(I.operands()))
    if (auto *C = dyn_cast<ConstantInt>(Op))
      Cost +=
          TTI.getIntImmCostInst(I.getOpcode(), Idx, C->getValue(), C->getType(),
                                TargetTransformInfo::TCK_CodeSize, &I);
  return Cost;
}

static InstructionCost
removableCost(Instruction &Root, TargetTransformInfo &TTI, Value *RetainedInput,
              TargetTransformInfo::TargetCostKind Kind,
              SmallPtrSetImpl<Instruction *> *Removed = nullptr) {
  InstructionCost Cost = 0;
  SmallVector<Instruction *, 16> Work{&Root};
  SmallPtrSet<Instruction *, 16> Seen;
  while (!Work.empty()) {
    Instruction *I = Work.pop_back_val();
    if (I == RetainedInput || !Seen.insert(I).second)
      continue;
    // Do not credit casts: lowering can fold an extension into its consumer,
    // which is not reflected by its standalone instruction cost.
    if (Kind == TargetTransformInfo::TCK_CodeSize)
      Cost += instructionSize(*I, TTI);
    else if (!isa<CastInst>(I))
      Cost += TTI.getInstructionCost(I, Kind);
    for (Value *V : I->operands())
      if (auto *Op = dyn_cast<Instruction>(V))
        if (Op->hasOneUse() && !Op->mayHaveSideEffects() &&
            !Op->mayReadOrWriteMemory() && !isa<PHINode, FreezeInst>(Op))
          Work.push_back(Op);
  }
  if (Removed)
    Removed->insert(Seen.begin(), Seen.end());
  return Cost;
}

/// Keep single-bit masks available to the target's bit-test/set/reset/toggle
/// combines. A standalone shift cost does not describe its cost in these users.
static bool hasOnlyBitwiseMaskUsers(Instruction &Root,
                                    ArrayRef<ConstantInt *> Results) {
  if (!Root.isShift() || !all_of(Results, [](ConstantInt *C) {
        return C->getValue().isPowerOf2();
      }))
    return false;
  SmallVector<Instruction *, 8> Work{&Root};
  SmallPtrSet<Instruction *, 8> Seen;
  unsigned Uses = 0;
  bool HasBitwiseUser = false;
  while (!Work.empty()) {
    Instruction *I = Work.pop_back_val();
    if (!Seen.insert(I).second)
      continue;
    for (User *U : I->users()) {
      // A large use graph is outside this local profitability check.
      if (++Uses > MaxExpressionSize)
        return true;
      auto *Use = dyn_cast<Instruction>(U);
      if (!Use)
        return false;
      if (Use->isBitwiseLogicOp()) {
        HasBitwiseUser = true;
      } else if (isa<CastInst>(Use) && Use->getType()->isIntegerTy()) {
        Work.push_back(Use);
      } else {
        // Another value consumer already requires materializing the mask, so
        // do not assume the bitwise uses can make this computation free.
        return false;
      }
    }
  }
  return HasBitwiseUser;
}

/// Share the offset of constant selects used as arithmetic offsets. For
/// example, select C, 14, 18 and select C, 15, 19 can share D = select C, 0, 4.
/// Then 1 << ((select C, 14, 18) - 1) becomes 8192 << D. Unlike materializing
/// both one-hot results, this also exposes the common offset to lowering.
static Value *shareShiftOffset(Instruction &Root, InputSet &S,
                               ArrayRef<ConstantInt *> Results,
                               TargetTransformInfo &TTI,
                               SmallVectorImpl<WeakTrackingVH> &Dead) {
  if (Root.getOpcode() != Instruction::Shl || !S.Condition ||
      Results.size() != 2)
    return nullptr;
  // ConstantData conditions (including true, undef and poison) have no
  // use-list.
  if (!S.Condition->hasUseList())
    return nullptr;
  // Minsize lowering can use different constant materializations (for example,
  // push/pop on x86) that are not represented by the instruction size costs.
  // Keep the ordinary finite-input folding policy, but do not use this
  // fallback.
  if (Root.getFunction()->hasMinSize())
    return nullptr;
  auto *Input = dyn_cast<SelectInst>(S.Input);
  auto *Count = dyn_cast<BinaryOperator>(Root.getOperand(1));
  auto *Seed = dyn_cast<ConstantInt>(Root.getOperand(0));
  if (!Input || !Count || !Seed || !Seed->getValue().isPowerOf2() ||
      Input->getType() != Root.getType() ||
      Count->getOpcode() != Instruction::Add || !Count->hasOneUse() ||
      Count->getOperand(0) != Input ||
      !isa<ConstantInt>(Count->getOperand(1)) ||
      Input->getParent() != Root.getParent() ||
      !Results[0]->getValue().isPowerOf2() ||
      !Results[1]->getValue().isPowerOf2())
    return nullptr;

  // Instruction costs do not account for legalization or combines with the
  // users of the shift. Restrict this size-only rewrite to a legal computation
  // used as an extended arithmetic offset, with a free extension. In
  // particular, retaining the original shift can be smaller for narrow types
  // and for bitwise consumers.
  constexpr auto Kind = TargetTransformInfo::TCK_CodeSize;
  if (!TTI.isTypeLegal(Root.getType()) || !Root.hasOneUse())
    return nullptr;
  auto *Wide = dyn_cast<ZExtInst>(*Root.user_begin());
  if (!Wide || !TTI.isTypeLegal(Wide->getType()) || Wide->use_empty() ||
      TTI.getCastInstrCost(Instruction::ZExt, Wide->getType(), Root.getType(),
                           TargetTransformInfo::CastContextHint::None,
                           Kind) != 0 ||
      any_of(Wide->users(), [](User *U) {
        auto *BO = dyn_cast<BinaryOperator>(U);
        return !BO || (BO->getOpcode() != Instruction::Add &&
                       BO->getOpcode() != Instruction::Sub);
      }))
    return nullptr;

  unsigned TrueShift = Results[0]->getValue().logBase2();
  unsigned FalseShift = Results[1]->getValue().logBase2();
  if (TrueShift == FalseShift)
    return nullptr;
  bool TrueIsLow = TrueShift < FalseShift;
  unsigned Delta = TrueIsLow ? FalseShift - TrueShift : TrueShift - FalseShift;
  // Consecutive arms already lower to an extended condition plus an offset;
  // there is no nontrivial common computation to amortize in that case.
  if (Delta == 1)
    return nullptr;
  auto *Ty = cast<IntegerType>(Root.getType());
  auto *Zero = ConstantInt::get(Ty, 0);
  auto *Offset = ConstantInt::get(Ty, Delta);
  ConstantInt *Base = Results[TrueIsLow ? 0 : 1];
  SmallVector<std::pair<SelectInst *, ConstantInt *>, 4> Selects;
  unsigned Scanned = 0;
  for (User *U : S.Condition->users()) {
    if (Scanned++ == MaxExpressionSize)
      break;
    auto *SI = dyn_cast<SelectInst>(U);
    if (!SI || SI->getParent() != Input->getParent() || SI->getType() != Ty ||
        SI->hasNUsesOrMore(MaxExpressionSize))
      continue;
    auto *T = dyn_cast<ConstantInt>(SI->getTrueValue());
    auto *F = dyn_cast<ConstantInt>(SI->getFalseValue());
    if (!T || !F ||
        (TrueIsLow ? F->getValue() - T->getValue()
                   : T->getValue() - F->getValue()) != Delta ||
        SI->use_empty() || any_of(SI->users(), [](User *U) {
          auto *BO = dyn_cast<BinaryOperator>(U);
          return !BO || (BO->getOpcode() != Instruction::Add &&
                         BO->getOpcode() != Instruction::Sub);
        }))
      continue;
    Selects.emplace_back(SI, TrueIsLow ? T : F);
    if (Selects.size() == 8)
      break;
  }
  // Amortize the common offset over two independently required selections.
  // Identical selections can already be CSE'd by lowering and must not be
  // credited as two instructions.
  if (Selects.size() != 2 || Selects[0].second == Selects[1].second ||
      none_of(Selects, [&](auto Entry) { return Entry.first == Input; }))
    return nullptr;

  // Only combine offsets that contribute with the same sign to one addition:
  //   Input + Other, or Input + (Other +/- X).
  // Otherwise arithmetic can cancel a shared offset, or separate consumers can
  // require extra copies that the per-instruction costs do not represent.
  auto *Other = Selects[0].first == Input ? Selects[1].first : Selects[0].first;
  BinaryOperator *Sum = nullptr;
  for (User *U : Input->users()) {
    if (U == Count)
      continue;
    if (Sum)
      return nullptr;
    Sum = cast<BinaryOperator>(U);
  }
  if (!Sum || Sum->getOpcode() != Instruction::Add || !Other->hasOneUse())
    return nullptr;
  auto *OtherUser = cast<BinaryOperator>(*Other->user_begin());
  if (OtherUser != Sum &&
      ((Sum->getOperand(0) != OtherUser && Sum->getOperand(1) != OtherUser) ||
       (OtherUser->getOpcode() != Instruction::Add &&
        (OtherUser->getOpcode() != Instruction::Sub ||
         OtherUser->getOperand(0) != Other))))
    return nullptr;

  auto *First = min_element(Selects, [](auto A, auto B) {
                  return A.first->comesBefore(B.first);
                })->first;
  // Sharing changes the value that stays live from the selections to the
  // shift. Do not rely on instruction costs to predict register preservation
  // across calls, or extend the lifetime past the shift's block.
  if (any_of(Selects, [&](auto Entry) {
        return !Entry.first->comesBefore(&Root) ||
               any_of(Entry.first->users(), [&](User *U) {
                 auto *I = cast<Instruction>(U);
                 return I->getParent() != Root.getParent() ||
                        !I->comesBefore(&Root);
               });
      }))
    return nullptr;
  unsigned Distance = 0;
  for (Instruction &I : make_range(First->getIterator(), Root.getIterator()))
    if (++Distance > MaxExpressionSize || isa<CallBase>(I))
      return nullptr;

  SmallPtrSet<Instruction *, 16> Removed;
  InstructionCost OldCost =
      removableCost(Root, TTI, S.Condition, Kind, &Removed);
  InstructionCost NewCost = TTI.getCmpSelInstrCost(
      Instruction::Select, Ty, S.Condition->getType(), CmpInst::ICMP_EQ, Kind);
  NewCost += TTI.getIntImmCostInst(Instruction::Select, TrueIsLow ? 2 : 1,
                                   Offset->getValue(), Ty, Kind);
  NewCost += TTI.getIntImmCostInst(Instruction::Select, TrueIsLow ? 1 : 2,
                                   Zero->getValue(), Ty, Kind);
  NewCost += TTI.getArithmeticInstrCost(Instruction::Shl, Ty, Kind);
  NewCost +=
      TTI.getIntImmCostInst(Instruction::Shl, 0, Base->getValue(), Ty, Kind);
  for (auto [SI, Low] : Selects) {
    if (Removed.insert(SI).second)
      OldCost += instructionSize(*SI, TTI);
    if (SI == Input && SI->hasOneUse())
      continue;
    NewCost += TTI.getArithmeticInstrCost(Instruction::Add, Ty, Kind);
    NewCost +=
        TTI.getIntImmCostInst(Instruction::Add, 1, Low->getValue(), Ty, Kind);
  }
  if (!OldCost.isValid() || !NewCost.isValid() || NewCost >= OldCost)
    return nullptr;

  IRBuilder<> B(First);
  Value *D = B.CreateSelect(S.Condition, TrueIsLow ? Zero : Offset,
                            TrueIsLow ? Offset : Zero, "finite.offset");
  for (auto [SI, Low] : Selects) {
    if (SI == Input && SI->hasOneUse())
      continue;
    B.SetInsertPoint(SI);
    Value *V = B.CreateAdd(D, Low, SI->getName() + ".offset");
    SI->replaceAllUsesWith(V);
    Dead.push_back(SI);
  }
  B.SetInsertPoint(&Root);
  return B.CreateShl(Base, D, Root.getName() + ".finite");
}

static bool foldFiniteInput(Instruction &Root, LazyValueInfo &LVI,
                            DominatorTree &DT, AssumptionCache &AC,
                            const TargetLibraryInfo &TLI,
                            TargetTransformInfo &TTI) {
  SmallVector<InputSet, 4> Sets;
  findInputSets(Root, LVI, DT, AC, Sets);
  for (InputSet &S : Sets) {
    if (isa<CallInst>(Root) &&
        !isGuaranteedNotToBeUndefOrPoison(S.Input, &AC, &Root, &DT))
      continue;
    SmallVector<ConstantInt *, MaxValues> Results;
    for (ConstantInt *Input : S.Values) {
      auto GetContextConstant = [&](Value *V) -> Constant * {
        if (V == &Root || !V->getType()->isIntegerTy() ||
            !isGuaranteedNotToBeUndefOrPoison(V, &AC, &Root, &DT))
          return nullptr;
        ConstantRange R = LVI.getConstantRange(V, &Root, false);
        if (auto *C = R.getSingleElement())
          return ConstantInt::get(V->getContext(), *C);
        return nullptr;
      };
      PartialEvaluator Eval(Root.getDataLayout(), TLI, S.Input, Input,
                            GetContextConstant);
      auto *C = dyn_cast_or_null<ConstantInt>(Eval.evaluate(&Root));
      if (!C)
        break;
      Results.push_back(C);
    }
    if (Results.size() != S.Values.size())
      continue;
    bool Same =
        all_of(Results, [&](ConstantInt *C) { return C == Results[0]; });
    if (Same && S.Condition &&
        !isGuaranteedNotToBePoison(S.Condition, &AC, &Root, &DT))
      continue;
    auto SelectionCost = [&](TargetTransformInfo::TargetCostKind Kind) {
      InstructionCost Constants = 0;
      for (ConstantInt *C : Results) {
        InstructionCost Imm =
            TTI.getIntImmCost(C->getValue(), C->getType(), Kind);
        // Cheap constants may fold into the selection's lowering. Expensive
        // materializations must also be paid for when optimizing for speed.
        if (Kind == TargetTransformInfo::TCK_CodeSize || !Imm.isValid() ||
            Imm > TargetTransformInfo::TCC_Basic)
          Constants += Imm;
      }
      if (Same)
        return Kind == TargetTransformInfo::TCK_CodeSize
                   ? TTI.getIntImmCost(Results[0]->getValue(), Root.getType(),
                                       Kind)
                   : InstructionCost(0);
      Type *CondTy = Type::getInt1Ty(Root.getContext());
      InstructionCost Cost = TTI.getCmpSelInstrCost(
          Instruction::Select, Root.getType(), CondTy, CmpInst::ICMP_EQ, Kind);
      if (!S.Condition) {
        Cost += TTI.getCmpSelInstrCost(Instruction::ICmp, S.Input->getType(),
                                       CondTy, CmpInst::ICMP_EQ, Kind);
        Cost *= Results.size() - 1;
        if (Kind == TargetTransformInfo::TCK_CodeSize)
          for (ConstantInt *C : drop_end(S.Values))
            Constants += TTI.getIntImmCostInst(
                Instruction::ICmp, 1, C->getValue(), C->getType(), Kind);
      }
      return Cost + Constants;
    };
    bool OptimizeForSize = Root.getFunction()->hasOptSize();
    auto CostKind = OptimizeForSize ? TargetTransformInfo::TCK_CodeSize
                                    : TargetTransformInfo::TCK_SizeAndLatency;
    InstructionCost NewCost = SelectionCost(CostKind);
    if (!NewCost.isValid())
      continue;
    Value *RetainedInput = Same ? nullptr : S.Condition ? S.Condition : S.Input;
    SmallPtrSet<Instruction *, 16> Removed;
    InstructionCost OldCost =
        removableCost(Root, TTI, RetainedInput, CostKind, &Removed);
    if (!Same && !isa<CallInst>(Root) && hasOnlyBitwiseMaskUsers(Root, Results))
      continue;
    Value *Replacement = nullptr;
    SmallVector<WeakTrackingVH, 8> Dead;
    if (OptimizeForSize) {
      // Compact arithmetic can be much smaller than loading several large
      // constants, even when a target counts each load as one instruction.
      if (any_of(Results, [&](ConstantInt *C) {
            return TTI.getIntImmCost(C->getValue(), C->getType(), CostKind) >
                   TargetTransformInfo::TCC_Basic;
          }))
        continue;
      // Require a strict code-size saving, including for calls. Other callers
      // can keep the callee alive, so do not credit its evaluated path here.
      if (!OldCost.isValid() || NewCost >= OldCost) {
        Replacement = shareShiftOffset(Root, S, Results, TTI, Dead);
        if (!Replacement)
          continue;
      }
    } else if (!isa<CallInst>(Root)) {
      // A folded call also removes the completely evaluated callee path. The
      // expression-only case must pay for all newly introduced instructions.
      if (!OldCost.isValid() || NewCost > OldCost)
        continue;
      if (!Same) {
        // Do not sum costs from different execution frequencies, or introduce
        // another selection of a retained value in a different type. Such
        // sharing and live-range changes are not represented by these costs.
        if (any_of(Removed, [&](Instruction *I) {
              return I->getParent() != Root.getParent();
            }))
          continue;
        if (S.Condition) {
          auto *Input = cast<SelectInst>(S.Input);
          if (Input->getParent() != Root.getParent() ||
              (!Removed.contains(Input) &&
               (Input->getType() != Root.getType() ||
                Input->hasNUsesOrMore(MaxExpressionSize) ||
                any_of(Input->users(),
                       [](User *U) { return isa<SelectInst, PHINode>(U); }))))
            continue;
          // Keep a retained selection coupled to the selects/PHIs forwarding
          // it through control flow. Splitting another value of its condition
          // can increase register pressure in successor blocks and loops even
          // when the local arithmetic becomes cheaper.
          unsigned Distance = 0;
          bool Local = true;
          for (Instruction &I :
               make_range(Input->getIterator(), Root.getIterator()))
            if (++Distance > MaxExpressionSize || isa<CallBase>(I)) {
              Local = false;
              break;
            }
          if (!Local)
            continue;
        }
        // Reusing a condition is not a free operation: require a strict
        // throughput benefit. In particular, replacing a shift of an already
        // shared count with an equally costly select can repeat its test and
        // obstruct combines.
        InstructionCost NewThroughput =
            SelectionCost(TargetTransformInfo::TCK_RecipThroughput);
        InstructionCost OldThroughput = removableCost(
            Root, TTI, RetainedInput, TargetTransformInfo::TCK_RecipThroughput);
        if (!NewThroughput.isValid() || !OldThroughput.isValid() ||
            NewThroughput >= OldThroughput)
          continue;
        InstructionCost RootThroughput = TTI.getInstructionCost(
            &Root, TargetTransformInfo::TCK_RecipThroughput);
        // Expensive operations such as division can pay for more code with
        // their strict throughput saving. A cheap root must also avoid size
        // growth: standalone costs do not describe all its lowering combines.
        if (!RootThroughput.isValid() ||
            RootThroughput <= TargetTransformInfo::TCC_Basic) {
          InstructionCost NewSize =
              SelectionCost(TargetTransformInfo::TCK_CodeSize);
          InstructionCost OldSize = removableCost(
              Root, TTI, RetainedInput, TargetTransformInfo::TCK_CodeSize);
          if (!NewSize.isValid() || !OldSize.isValid() || NewSize > OldSize)
            continue;
        }
      }
    }
    IRBuilder<> B(&Root);
    if (!Replacement && Same)
      Replacement = Results.back();
    if (!Replacement) {
      if (S.Condition)
        Replacement = B.CreateSelect(S.Condition, Results[0], Results[1],
                                     Root.getName() + ".finite");
      else {
        Replacement = Results.back();
        for (unsigned N = Results.size() - 1; N-- != 0;) {
          Value *Cond = B.CreateICmpEQ(S.Input, S.Values[N]);
          Replacement = B.CreateSelect(Cond, Results[N], Replacement,
                                       Root.getName() + ".finite");
        }
      }
    }
    if (isa<CallInst>(Root))
      ++NumCallsFolded;
    ++NumFolded;
    OptimizationRemarkEmitter ORE(Root.getFunction());
    ORE.emit([&]() {
      std::string RootName;
      raw_string_ostream OS(RootName);
      Root.printAsOperand(OS, false);
      return OptimizationRemark(DEBUG_TYPE, "FiniteInput", &Root)
             << "folded " << ore::NV("Root", RootName) << " over "
             << ore::NV("NumValues", S.Values.size()) << " values of "
             << ore::NV("Input", S.Input) << " ("
             << ore::NV("Operation", Root.getOpcodeName()) << ")";
    });
    Root.replaceAllUsesWith(Replacement);
    SmallVector<WeakTrackingVH, 8> Operands(Root.operands());
    append_range(Operands, Dead);
    Root.eraseFromParent();
    for (WeakTrackingVH &Op : Operands)
      if (auto *I = dyn_cast_or_null<Instruction>(Op))
        RecursivelyDeleteTriviallyDeadInstructions(I, &TLI);
    LVI.clear();
    return true;
  }
  return false;
}

} // namespace

PreservedAnalyses FiniteValuePropagationPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || F.hasFnAttribute(Attribute::OptimizeNone) ||
        F.hasFnAttribute(Attribute::StrictFP))
      continue;
    SmallVector<WeakTrackingVH, 32> Work;
    for (Instruction &I : instructions(F)) {
      if (I.use_empty() || !I.getType()->isIntegerTy())
        continue;
      // Cheap arithmetic and casts are best left to the ordinary
      // canonicalizers: replacing them with selections can obstruct target
      // combines.
      switch (I.getOpcode()) {
      case Instruction::Call:
        if (isa<IntrinsicInst>(I))
          break;
        [[fallthrough]];
      case Instruction::Mul:
      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr:
        Work.push_back(&I);
        break;
      default:
        break;
      }
    }
    if (Work.empty())
      continue;
    auto &LVI = FAM.getResult<LazyValueAnalysis>(F);
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    auto &AC = FAM.getResult<AssumptionAnalysis>(F);
    auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
    auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
    for (WeakTrackingVH &V : reverse(Work))
      if (auto *I = dyn_cast_or_null<Instruction>(V))
        Changed |= foldFiniteInput(*I, LVI, DT, AC, TLI, TTI);
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
