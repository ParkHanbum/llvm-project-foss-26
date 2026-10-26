//===- Evaluator.cpp - LLVM IR evaluator ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Function evaluator for LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Evaluator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "evaluator"

using namespace llvm;

static inline bool
isSimpleEnoughValueToCommit(Constant *C,
                            SmallPtrSetImpl<Constant *> &SimpleConstants,
                            const DataLayout &DL);

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

Constant *Evaluator::FoldRuntimeInstruction(Instruction &I,
                                            ArrayRef<Constant *> Ops) {
  if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects() || I.isTerminator() ||
      isa<PHINode, FreezeInst>(I) || !I.getType()->isIntOrIntVectorTy())
    return nullptr;
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
  return ConstantFoldInstOperands(&I, Ops, DL, TLI,
                                  /*AllowNonDeterministic=*/false);
}

/// An out-of-range call result is poison. In particular, do not turn that
/// into immediate UB by inferring noundef on a call whose result may be masked.
static bool constantIsInRange(Constant *C, const ConstantRange &Range) {
  if (auto *CI = dyn_cast<ConstantInt>(C))
    return Range.contains(CI->getValue());
  auto *VT = dyn_cast<FixedVectorType>(C->getType());
  if (!VT)
    return false;
  for (unsigned Lane = 0; Lane != VT->getNumElements(); ++Lane) {
    auto *CI = dyn_cast_or_null<ConstantInt>(C->getAggregateElement(Lane));
    if (!CI || !Range.contains(CI->getValue()))
      return false;
  }
  return true;
}

bool Evaluator::CanEvaluateRuntimeFunction(Function &F,
                                           ArrayRef<Constant *> Args) const {
  if (!F.hasExactDefinition() || F.isVarArg() || F.doesNotReturn() ||
      F.hasFnAttribute(Attribute::OptimizeNone) ||
      F.hasFnAttribute(Attribute::Naked) ||
      F.hasFnAttribute(Attribute::StrictFP) ||
      F.hasFnAttribute(Attribute::Convergent) ||
      F.hasFnAttribute(Attribute::ReturnsTwice) ||
      F.hasFnAttribute("loader-replaceable") ||
      !F.getReturnType()->isIntOrIntVectorTy() ||
      CallStack.size() >= MaxCallDepth)
    return false;
  for (auto [Arg, C] : zip(F.args(), Args)) {
    if (C->getType() != Arg.getType() || !C->getType()->isIntOrIntVectorTy() ||
        !isGuaranteedNotToBeUndefOrPoison(C))
      return false;
    if (auto R = Arg.getRange())
      if (!constantIsInRange(C, *R))
        return false;
  }
  return true;
}

Constant *Evaluator::EvaluateRuntimeCall(CallBase &CB,
                                         ArrayRef<Constant *> Args) {
  Function *F = CB.getCalledFunction();
  if (!isa<CallInst>(CB) || !F || CB.isMustTailCall() ||
      CB.hasOperandBundles() || CB.isConvergent() ||
      CB.hasFnAttr(Attribute::ReturnsTwice) ||
      CB.hasFnAttr("loader-replaceable") ||
      F->getFunctionType() != CB.getFunctionType() ||
      F->getCallingConv() != CB.getCallingConv())
    return nullptr;
  for (auto [N, C] : enumerate(Args)) {
    Attribute Range = CB.getParamAttr(N, Attribute::Range);
    if (Range.isValid() && !constantIsInRange(C, Range.getRange()))
      return nullptr;
  }
  Constant *Result = nullptr;
  if (F->isIntrinsic()) {
    // This observes manifest constants in the IR, not just runtime values.
    // As in simplifyWithOpReplaced(), assumptions must not make it true.
    if (F->getIntrinsicID() == Intrinsic::is_constant)
      return nullptr;
    if (CB.mayHaveSideEffects() || !canConstantFoldCallTo(&CB, F, TLI))
      return nullptr;
    Result = ConstantFoldCall(&CB, F, Args, TLI,
                              /*AllowNonDeterministic=*/false);
  } else {
    SmallVector<Constant *, 8> ActualArgs(Args);
    ValueStack.emplace_back();
    bool Success = EvaluateFunction(F, Result, ActualArgs);
    ValueStack.pop_back();
    if (!Success)
      return nullptr;
  }
  if (Result)
    if (auto Range = CB.getRange())
      if (!constantIsInRange(Result, *Range))
        return nullptr;
  return Result;
}

Constant *Evaluator::EvaluateExpression(Value *V, Lookup GetConstant) {
  assert(EvalMode == Mode::Runtime &&
         "Expression evaluation needs runtime mode");
  ExpressionCache.clear();
  return EvaluateExpressionImpl(V, GetConstant, 0);
}

Constant *Evaluator::EvaluateExpressionImpl(Value *V, Lookup GetConstant,
                                            unsigned Depth) {
  if (auto *C = dyn_cast<Constant>(V))
    return C;
  if (Constant *C = ExpressionCache.lookup(V))
    return C;
  if (Depth == 32 || !takeRuntimeStep())
    return nullptr;
  if (Constant *C = GetConstant(V)) {
    if (C->getType() != V->getType())
      return nullptr;
    return ExpressionCache[V] = C;
  }
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return nullptr;
  // In particular, do not visit PHI incoming values at the root's context:
  // those definitions need not dominate the expression being evaluated.
  if (!isa<CallInst>(I) &&
      (I->mayReadOrWriteMemory() || I->mayHaveSideEffects() ||
       I->isTerminator() || isa<PHINode, FreezeInst>(I) ||
       !I->getType()->isIntOrIntVectorTy()))
    return nullptr;
  SmallVector<Constant *, 8> Ops;
  if (auto *CI = dyn_cast<CallInst>(I)) {
    for (Value *Arg : CI->args()) {
      Constant *C = EvaluateExpressionImpl(Arg, GetConstant, Depth + 1);
      if (!C)
        return nullptr;
      Ops.push_back(C);
    }
    if (Constant *C = EvaluateRuntimeCall(*CI, Ops))
      return ExpressionCache[V] = C;
    return nullptr;
  }
  for (Value *Op : I->operands()) {
    Constant *C = EvaluateExpressionImpl(Op, GetConstant, Depth + 1);
    if (!C)
      return nullptr;
    Ops.push_back(C);
  }
  if (Constant *C = FoldRuntimeInstruction(*I, Ops))
    return ExpressionCache[V] = C;
  return nullptr;
}

/// Return true if the specified constant can be handled by the code generator.
/// We don't want to generate something like:
///   void *X = &X/42;
/// because the code generator doesn't have a relocation that can handle that.
///
/// This function should be called if C was not found (but just got inserted)
/// in SimpleConstants to avoid having to rescan the same constants all the
/// time.
static bool
isSimpleEnoughValueToCommitHelper(Constant *C,
                                  SmallPtrSetImpl<Constant *> &SimpleConstants,
                                  const DataLayout &DL) {
  // Simple global addresses are supported, do not allow dllimport or
  // thread-local globals.
  if (auto *GV = dyn_cast<GlobalValue>(C))
    return !GV->hasDLLImportStorageClass() && !GV->isThreadLocal();

  // Simple integer, undef, constant aggregate zero, etc are all supported.
  if (C->getNumOperands() == 0 || isa<BlockAddress>(C))
    return true;

  // Aggregate values are safe if all their elements are.
  if (isa<ConstantAggregate>(C)) {
    for (Value *Op : C->operands())
      if (!isSimpleEnoughValueToCommit(cast<Constant>(Op), SimpleConstants, DL))
        return false;
    return true;
  }

  // We don't know exactly what relocations are allowed in constant expressions,
  // so we allow &global+constantoffset, which is safe and uniformly supported
  // across targets.
  ConstantExpr *CE = dyn_cast<ConstantExpr>(C);
  if (!CE)
    return false;
  switch (CE->getOpcode()) {
  case Instruction::BitCast:
    // Bitcast is fine if the casted value is fine.
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
    // int <=> ptr is fine if the int type is the same size as the
    // pointer type.
    if (DL.getTypeSizeInBits(CE->getType()) !=
        DL.getTypeSizeInBits(CE->getOperand(0)->getType()))
      return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  // GEP is fine if it is simple + constant offset.
  case Instruction::GetElementPtr:
    for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
      if (!isa<ConstantInt>(CE->getOperand(i)))
        return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  case Instruction::Add:
    // We allow simple+cst.
    if (!isa<ConstantInt>(CE->getOperand(1)))
      return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);
  }
  return false;
}

static inline bool
isSimpleEnoughValueToCommit(Constant *C,
                            SmallPtrSetImpl<Constant *> &SimpleConstants,
                            const DataLayout &DL) {
  // If we already checked this constant, we win.
  if (!SimpleConstants.insert(C).second)
    return true;
  // Check the constant.
  return isSimpleEnoughValueToCommitHelper(C, SimpleConstants, DL);
}

void Evaluator::MutableValue::clear() {
  if (auto *Agg = dyn_cast_if_present<MutableAggregate *>(Val))
    delete Agg;
  Val = nullptr;
}

Constant *Evaluator::MutableValue::read(Type *Ty, APInt Offset,
                                        const DataLayout &DL) const {
  TypeSize TySize = DL.getTypeStoreSize(Ty);
  const MutableValue *V = this;
  while (const auto *Agg = dyn_cast_if_present<MutableAggregate *>(V->Val)) {
    Type *AggTy = Agg->Ty;
    std::optional<APInt> Index = DL.getGEPIndexForOffset(AggTy, Offset);
    if (!Index || Index->uge(Agg->Elements.size()) ||
        !TypeSize::isKnownLE(TySize, DL.getTypeStoreSize(AggTy)))
      return nullptr;

    V = &Agg->Elements[Index->getZExtValue()];
  }

  return ConstantFoldLoadFromConst(cast<Constant *>(V->Val), Ty, Offset, DL);
}

bool Evaluator::MutableValue::makeMutable() {
  Constant *C = cast<Constant *>(Val);
  Type *Ty = C->getType();
  unsigned NumElements;
  if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
    NumElements = VT->getNumElements();
  } else if (auto *AT = dyn_cast<ArrayType>(Ty))
    NumElements = AT->getNumElements();
  else if (auto *ST = dyn_cast<StructType>(Ty))
    NumElements = ST->getNumElements();
  else
    return false;

  MutableAggregate *MA = new MutableAggregate(Ty);
  MA->Elements.reserve(NumElements);
  for (unsigned I = 0; I < NumElements; ++I)
    MA->Elements.push_back(C->getAggregateElement(I));
  Val = MA;
  return true;
}

bool Evaluator::MutableValue::write(Constant *V, APInt Offset,
                                    const DataLayout &DL) {
  Type *Ty = V->getType();
  TypeSize TySize = DL.getTypeStoreSize(Ty);
  MutableValue *MV = this;
  while (Offset != 0 ||
         !CastInst::isBitOrNoopPointerCastable(Ty, MV->getType(), DL)) {
    if (isa<Constant *>(MV->Val) && !MV->makeMutable())
      return false;

    MutableAggregate *Agg = cast<MutableAggregate *>(MV->Val);
    Type *AggTy = Agg->Ty;
    std::optional<APInt> Index = DL.getGEPIndexForOffset(AggTy, Offset);
    if (!Index || Index->uge(Agg->Elements.size()) ||
        !TypeSize::isKnownLE(TySize, DL.getTypeStoreSize(AggTy)))
      return false;

    MV = &Agg->Elements[Index->getZExtValue()];
  }

  Type *MVType = MV->getType();
  MV->clear();
  if (Ty->isIntegerTy() && MVType->isPointerTy())
    MV->Val = ConstantExpr::getIntToPtr(V, MVType);
  else if (Ty->isPointerTy() && MVType->isIntegerTy())
    MV->Val = ConstantExpr::getPtrToInt(V, MVType);
  else if (Ty != MVType)
    MV->Val = ConstantExpr::getBitCast(V, MVType);
  else
    MV->Val = V;
  return true;
}

Constant *Evaluator::MutableAggregate::toConstant() const {
  SmallVector<Constant *, 32> Consts;
  for (const MutableValue &MV : Elements)
    Consts.push_back(MV.toConstant());

  if (auto *ST = dyn_cast<StructType>(Ty))
    return ConstantStruct::get(ST, Consts);
  if (auto *AT = dyn_cast<ArrayType>(Ty))
    return ConstantArray::get(AT, Consts);
  assert(isa<FixedVectorType>(Ty) && "Must be vector");
  return ConstantVector::get(Consts);
}

/// Return the value that would be computed by a load from P after the stores
/// reflected by 'memory' have been performed.  If we can't decide, return null.
Constant *Evaluator::ComputeLoadResult(Constant *P, Type *Ty) {
  APInt Offset(DL.getIndexTypeSizeInBits(P->getType()), 0);
  P = cast<Constant>(P->stripAndAccumulateConstantOffsets(
      DL, Offset, /* AllowNonInbounds */ true));
  Offset = Offset.sextOrTrunc(DL.getIndexTypeSizeInBits(P->getType()));
  if (auto *GV = dyn_cast<GlobalVariable>(P))
    return ComputeLoadResult(GV, Ty, Offset);
  return nullptr;
}

Constant *Evaluator::ComputeLoadResult(GlobalVariable *GV, Type *Ty,
                                       const APInt &Offset) {
  auto It = MutatedMemory.find(GV);
  if (It != MutatedMemory.end())
    return It->second.read(Ty, Offset, DL);

  if (!GV->hasDefinitiveInitializer())
    return nullptr;
  return ConstantFoldLoadFromConst(GV->getInitializer(), Ty, Offset, DL);
}

static Function *getFunction(Constant *C) {
  if (auto *Fn = dyn_cast<Function>(C))
    return Fn;

  if (auto *Alias = dyn_cast<GlobalAlias>(C))
    if (auto *Fn = dyn_cast<Function>(Alias->getAliasee()))
      return Fn;
  return nullptr;
}

Function *
Evaluator::getCalleeWithFormalArgs(CallBase &CB,
                                   SmallVectorImpl<Constant *> &Formals) {
  auto *V = CB.getCalledOperand()->stripPointerCasts();
  if (auto *Fn = getFunction(getVal(V)))
    return getFormalParams(CB, Fn, Formals) ? Fn : nullptr;
  return nullptr;
}

bool Evaluator::getFormalParams(CallBase &CB, Function *F,
                                SmallVectorImpl<Constant *> &Formals) {
  auto *FTy = F->getFunctionType();
  if (FTy != CB.getFunctionType()) {
    LLVM_DEBUG(dbgs() << "Signature mismatch.\n");
    return false;
  }

  for (Value *Arg : CB.args())
    Formals.push_back(getVal(Arg));
  return true;
}

/// Evaluate all instructions in block BB, returning true if successful, false
/// if we can't evaluate it.  NewBB returns the next BB that control flows into,
/// or null upon return. StrippedPointerCastsForAliasAnalysis is set to true if
/// we looked through pointer casts to evaluate something.
bool Evaluator::EvaluateBlock(BasicBlock::iterator CurInst, BasicBlock *&NextBB,
                              bool &StrippedPointerCastsForAliasAnalysis) {
  // This is the main evaluation loop.
  while (true) {
    Constant *InstResult = nullptr;

    if (EvalMode == Mode::Runtime) {
      if (!takeRuntimeStep())
        return false;
      Instruction &I = *CurInst;
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        if (!AI->isStaticAlloca() || AI->isUsedWithInAlloca() ||
            AI->isSwiftError())
          return false;
        // Runtime mode never evaluates a load or an escaping stack address.
        ++CurInst;
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        auto *AI = dyn_cast<AllocaInst>(SI->getPointerOperand());
        auto Size = AI ? AI->getAllocationSize(DL) : std::nullopt;
        TypeSize StoreSize =
            DL.getTypeStoreSize(SI->getValueOperand()->getType());
        if (SI->isVolatile() || SI->isAtomic() || !AI ||
            AI->getFunction() != I.getFunction() || !Size ||
            Size->isScalable() || StoreSize.isScalable() ||
            StoreSize.getFixedValue() > Size->getFixedValue() ||
            SI->getAlign() > AI->getAlign())
          return false;
        ++CurInst;
        continue;
      }
      if (isa<DbgInfoIntrinsic>(I)) {
        ++CurInst;
        continue;
      }
      if (!I.isTerminator()) {
        SmallVector<Constant *, 8> Ops;
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          for (Value *Arg : CI->args()) {
            Constant *C = getRuntimeVal(Arg);
            if (!C)
              return false;
            Ops.push_back(C);
          }
          InstResult = EvaluateRuntimeCall(*CI, Ops);
        } else {
          for (Value *Op : I.operands()) {
            Constant *C = getRuntimeVal(Op);
            if (!C)
              return false;
            Ops.push_back(C);
          }
          InstResult = FoldRuntimeInstruction(I, Ops);
        }
        if (!InstResult)
          return false;
        if (!I.use_empty())
          setVal(&I, InstResult);
        ++CurInst;
        continue;
      }
      if (!isa<UncondBrInst, CondBrInst, SwitchInst, ReturnInst>(I))
        return false;
    }

    LLVM_DEBUG(dbgs() << "Evaluating Instruction: " << *CurInst << "\n");

    if (StoreInst *SI = dyn_cast<StoreInst>(CurInst)) {
      if (SI->isVolatile()) {
        LLVM_DEBUG(dbgs() << "Store is volatile! Can not evaluate.\n");
        return false;  // no volatile accesses.
      }
      Constant *Ptr = getVal(SI->getOperand(1));
      Constant *FoldedPtr = ConstantFoldConstant(Ptr, DL, TLI);
      if (Ptr != FoldedPtr) {
        LLVM_DEBUG(dbgs() << "Folding constant ptr expression: " << *Ptr);
        Ptr = FoldedPtr;
        LLVM_DEBUG(dbgs() << "; To: " << *Ptr << "\n");
      }

      APInt Offset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
      Ptr = cast<Constant>(Ptr->stripAndAccumulateConstantOffsets(
          DL, Offset, /* AllowNonInbounds */ true));
      Offset = Offset.sextOrTrunc(DL.getIndexTypeSizeInBits(Ptr->getType()));
      auto *GV = dyn_cast<GlobalVariable>(Ptr);
      if (!GV || !GV->hasUniqueInitializer() || GV->hasSection()) {
        // hasUniqueInitializer() ensures that if we modify the initializer,
        // the modified initializer will be used.
        //
        // We can't modify global variables with an explicit section because
        // it might not be legal to emit the resulting initializer (for
        // example, emitting a non-zero value into a BSS section).
        LLVM_DEBUG(dbgs() << "Store is not to global with unique initializer: "
                          << *Ptr << "\n");
        return false;
      }

      // If this might be too difficult for the backend to handle (e.g. the addr
      // of one global variable divided by another) then we can't commit it.
      Constant *Val = getVal(SI->getOperand(0));
      if (!isSimpleEnoughValueToCommit(Val, SimpleConstants, DL)) {
        LLVM_DEBUG(dbgs() << "Store value is too complex to evaluate store. "
                          << *Val << "\n");
        return false;
      }

      auto Res = MutatedMemory.try_emplace(GV, GV->getInitializer());
      if (!Res.first->second.write(Val, Offset, DL))
        return false;
    } else if (LoadInst *LI = dyn_cast<LoadInst>(CurInst)) {
      if (LI->isVolatile()) {
        LLVM_DEBUG(
            dbgs() << "Found a Load! Volatile load, can not evaluate.\n");
        return false;  // no volatile accesses.
      }

      Constant *Ptr = getVal(LI->getOperand(0));
      Constant *FoldedPtr = ConstantFoldConstant(Ptr, DL, TLI);
      if (Ptr != FoldedPtr) {
        Ptr = FoldedPtr;
        LLVM_DEBUG(dbgs() << "Found a constant pointer expression, constant "
                             "folding: "
                          << *Ptr << "\n");
      }
      InstResult = ComputeLoadResult(Ptr, LI->getType());
      if (!InstResult) {
        LLVM_DEBUG(
            dbgs() << "Failed to compute load result. Can not evaluate load."
                      "\n");
        return false; // Could not evaluate load.
      }

      LLVM_DEBUG(dbgs() << "Evaluated load: " << *InstResult << "\n");
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(CurInst)) {
      if (AI->isArrayAllocation()) {
        LLVM_DEBUG(dbgs() << "Found an array alloca. Can not evaluate.\n");
        return false;  // Cannot handle array allocs.
      }
      Type *Ty = AI->getAllocatedType();
      AllocaTmps.push_back(std::make_unique<GlobalVariable>(
          Ty, false, GlobalValue::InternalLinkage, UndefValue::get(Ty),
          AI->getName(), /*TLMode=*/GlobalValue::NotThreadLocal,
          AI->getType()->getPointerAddressSpace()));
      InstResult = AllocaTmps.back().get();
      LLVM_DEBUG(dbgs() << "Found an alloca. Result: " << *InstResult << "\n");
    } else if (isa<CallInst>(CurInst) || isa<InvokeInst>(CurInst)) {
      CallBase &CB = *cast<CallBase>(&*CurInst);

      // Cannot handle inline asm.
      if (CB.isInlineAsm()) {
        LLVM_DEBUG(dbgs() << "Found inline asm, can not evaluate.\n");
        return false;
      }

      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&CB)) {
        if (MemSetInst *MSI = dyn_cast<MemSetInst>(II)) {
          if (MSI->isVolatile()) {
            LLVM_DEBUG(dbgs() << "Can not optimize a volatile memset "
                              << "intrinsic.\n");
            return false;
          }

          auto *LenC = dyn_cast<ConstantInt>(getVal(MSI->getLength()));
          if (!LenC) {
            LLVM_DEBUG(dbgs() << "Memset with unknown length.\n");
            return false;
          }

          Constant *Ptr = getVal(MSI->getDest());
          APInt Offset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
          Ptr = cast<Constant>(Ptr->stripAndAccumulateConstantOffsets(
              DL, Offset, /* AllowNonInbounds */ true));
          auto *GV = dyn_cast<GlobalVariable>(Ptr);
          if (!GV) {
            LLVM_DEBUG(dbgs() << "Memset with unknown base.\n");
            return false;
          }

          Constant *Val = getVal(MSI->getValue());
          // Avoid the byte-per-byte scan if we're memseting a zeroinitializer
          // to zero.
          if (!Val->isNullValue() || MutatedMemory.contains(GV) ||
              !GV->hasDefinitiveInitializer() ||
              !GV->getInitializer()->isNullValue()) {
            APInt Len = LenC->getValue();
            if (Len.ugt(64 * 1024)) {
              LLVM_DEBUG(dbgs() << "Not evaluating large memset of size "
                                << Len << "\n");
              return false;
            }

            while (Len != 0) {
              Constant *DestVal = ComputeLoadResult(GV, Val->getType(), Offset);
              if (DestVal != Val) {
                LLVM_DEBUG(dbgs() << "Memset is not a no-op at offset "
                                  << Offset << " of " << *GV << ".\n");
                return false;
              }
              ++Offset;
              --Len;
            }
          }

          LLVM_DEBUG(dbgs() << "Ignoring no-op memset.\n");
          ++CurInst;
          continue;
        }

        if (II->isLifetimeStartOrEnd()) {
          LLVM_DEBUG(dbgs() << "Ignoring lifetime intrinsic.\n");
          ++CurInst;
          continue;
        }

        if (II->getIntrinsicID() == Intrinsic::invariant_start) {
          // We don't insert an entry into Values, as it doesn't have a
          // meaningful return value.
          if (!II->use_empty()) {
            LLVM_DEBUG(dbgs()
                       << "Found unused invariant_start. Can't evaluate.\n");
            return false;
          }
          ConstantInt *Size = cast<ConstantInt>(II->getArgOperand(0));
          Value *PtrArg = getVal(II->getArgOperand(1));
          Value *Ptr = PtrArg->stripPointerCasts();
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
            uint64_t MinGVSize = GV->getGlobalSize(DL);
            if (!Size->isMinusOne() &&
                Size->getValue().getLimitedValue() >= MinGVSize) {
              Invariants.insert(GV);
              LLVM_DEBUG(dbgs() << "Found a global var that is an invariant: "
                                << *GV << "\n");
            } else {
              LLVM_DEBUG(dbgs()
                         << "Found a global var, but can not treat it as an "
                            "invariant.\n");
            }
          }
          // Continue even if we do nothing.
          ++CurInst;
          continue;
        } else if (II->getIntrinsicID() == Intrinsic::assume) {
          LLVM_DEBUG(dbgs() << "Skipping assume intrinsic.\n");
          ++CurInst;
          continue;
        } else if (II->getIntrinsicID() == Intrinsic::sideeffect) {
          LLVM_DEBUG(dbgs() << "Skipping sideeffect intrinsic.\n");
          ++CurInst;
          continue;
        } else if (II->getIntrinsicID() == Intrinsic::pseudoprobe) {
          LLVM_DEBUG(dbgs() << "Skipping pseudoprobe intrinsic.\n");
          ++CurInst;
          continue;
        } else {
          Value *Stripped = CurInst->stripPointerCastsForAliasAnalysis();
          // Only attempt to getVal() if we've actually managed to strip
          // anything away, or else we'll call getVal() on the current
          // instruction.
          if (Stripped != &*CurInst) {
            InstResult = getVal(Stripped);
          }
          if (InstResult) {
            LLVM_DEBUG(dbgs()
                       << "Stripped pointer casts for alias analysis for "
                          "intrinsic call.\n");
            StrippedPointerCastsForAliasAnalysis = true;
            InstResult = ConstantExpr::getBitCast(InstResult, II->getType());
          } else {
            LLVM_DEBUG(dbgs() << "Unknown intrinsic. Cannot evaluate.\n");
            return false;
          }
        }
      }

      if (!InstResult) {
        // Resolve function pointers.
        SmallVector<Constant *, 8> Formals;
        Function *Callee = getCalleeWithFormalArgs(CB, Formals);
        if (!Callee || Callee->isInterposable()) {
          LLVM_DEBUG(dbgs() << "Can not resolve function pointer.\n");
          return false; // Cannot resolve.
        }

        if (Callee->isDeclaration()) {
          // If this is a function we can constant fold, do it.
          if (Constant *C = ConstantFoldCall(&CB, Callee, Formals, TLI)) {
            InstResult = C;
            LLVM_DEBUG(dbgs() << "Constant folded function call. Result: "
                              << *InstResult << "\n");
          } else {
            LLVM_DEBUG(dbgs() << "Can not constant fold function call.\n");
            return false;
          }
        } else {
          if (Callee->getFunctionType()->isVarArg()) {
            LLVM_DEBUG(dbgs()
                       << "Can not constant fold vararg function call.\n");
            return false;
          }

          Constant *RetVal = nullptr;
          // Execute the call, if successful, use the return value.
          ValueStack.emplace_back();
          if (!EvaluateFunction(Callee, RetVal, Formals)) {
            LLVM_DEBUG(dbgs() << "Failed to evaluate function.\n");
            return false;
          }
          ValueStack.pop_back();
          InstResult = RetVal;
          if (InstResult) {
            LLVM_DEBUG(dbgs() << "Successfully evaluated function. Result: "
                              << *InstResult << "\n\n");
          } else {
            LLVM_DEBUG(dbgs()
                       << "Successfully evaluated function. Result: 0\n\n");
          }
        }
      }
    } else if (CurInst->isTerminator()) {
      LLVM_DEBUG(dbgs() << "Found a terminator instruction.\n");

      if (UncondBrInst *BI = dyn_cast<UncondBrInst>(CurInst)) {
        NextBB = BI->getSuccessor(0);
      } else if (CondBrInst *BI = dyn_cast<CondBrInst>(CurInst)) {
        ConstantInt *Cond = dyn_cast<ConstantInt>(getVal(BI->getCondition()));
        if (!Cond)
          return false; // Cannot determine.
        NextBB = BI->getSuccessor(!Cond->getZExtValue());
      } else if (SwitchInst *SI = dyn_cast<SwitchInst>(CurInst)) {
        ConstantInt *Val =
          dyn_cast<ConstantInt>(getVal(SI->getCondition()));
        if (!Val) return false;  // Cannot determine.
        NextBB = SI->findCaseValue(Val)->getCaseSuccessor();
      } else if (IndirectBrInst *IBI = dyn_cast<IndirectBrInst>(CurInst)) {
        Value *Val = getVal(IBI->getAddress())->stripPointerCasts();
        if (BlockAddress *BA = dyn_cast<BlockAddress>(Val))
          NextBB = BA->getBasicBlock();
        else
          return false;  // Cannot determine.
      } else if (isa<ReturnInst>(CurInst)) {
        NextBB = nullptr;
      } else {
        // invoke, unwind, resume, unreachable.
        LLVM_DEBUG(dbgs() << "Can not handle terminator.");
        return false;  // Cannot handle this terminator.
      }

      // We succeeded at evaluating this block!
      LLVM_DEBUG(dbgs() << "Successfully evaluated block.\n");
      return true;
    } else {
      SmallVector<Constant *> Ops;
      for (Value *Op : CurInst->operands())
        Ops.push_back(getVal(Op));
      InstResult = ConstantFoldInstOperands(&*CurInst, Ops, DL, TLI);
      if (!InstResult) {
        LLVM_DEBUG(dbgs() << "Cannot fold instruction: " << *CurInst << "\n");
        return false;
      }
      LLVM_DEBUG(dbgs() << "Folded instruction " << *CurInst << " to "
                        << *InstResult << "\n");
    }

    if (!CurInst->use_empty()) {
      InstResult = ConstantFoldConstant(InstResult, DL, TLI);
      setVal(&*CurInst, InstResult);
    }

    // If we just processed an invoke, we finished evaluating the block.
    if (InvokeInst *II = dyn_cast<InvokeInst>(CurInst)) {
      NextBB = II->getNormalDest();
      LLVM_DEBUG(dbgs() << "Found an invoke instruction. Finished Block.\n\n");
      return true;
    }

    // Advance program counter.
    ++CurInst;
  }
}

/// Evaluate a call to function F, returning true if successful, false if we
/// can't evaluate it.  ActualArgs contains the formal arguments for the
/// function.
bool Evaluator::EvaluateFunction(Function *F, Constant *&RetVal,
                                 const SmallVectorImpl<Constant*> &ActualArgs) {
  assert(ActualArgs.size() == F->arg_size() && "wrong number of arguments");

  if (EvalMode == Mode::Runtime && !CanEvaluateRuntimeFunction(*F, ActualArgs))
    return false;

  // Check to see if this function is already executing (recursion).  If so,
  // bail out.  TODO: we might want to accept limited recursion.
  if (is_contained(CallStack, F))
    return false;

  CallStack.push_back(F);

  // Initialize arguments to the incoming values specified.
  for (const auto &[ArgNo, Arg] : llvm::enumerate(F->args()))
    setVal(&Arg, ActualArgs[ArgNo]);

  // ExecutedBlocks - We only handle non-looping, non-recursive code.  As such,
  // we can only evaluate any one basic block at most once.  This set keeps
  // track of what we have executed so we can detect recursive cases etc.
  SmallPtrSet<BasicBlock*, 32> ExecutedBlocks;

  // CurBB - The current basic block we're evaluating.
  BasicBlock *CurBB = &F->front();

  BasicBlock::iterator CurInst = CurBB->begin();

  while (true) {
    BasicBlock *NextBB = nullptr; // Initialized to avoid compiler warnings.
    LLVM_DEBUG(dbgs() << "Trying to evaluate BB: " << *CurBB << "\n");

    bool StrippedPointerCastsForAliasAnalysis = false;

    if (!EvaluateBlock(CurInst, NextBB, StrippedPointerCastsForAliasAnalysis))
      return false;

    if (!NextBB) {
      // Successfully running until there's no next block means that we found
      // the return.  Fill it the return value and pop the call stack.
      ReturnInst *RI = cast<ReturnInst>(CurBB->getTerminator());
      if (RI->getNumOperands()) {
        // The Evaluator can look through pointer casts as long as alias
        // analysis holds because it's just a simple interpreter and doesn't
        // skip memory accesses due to invariant group metadata, but we can't
        // let users of Evaluator use a value that's been gleaned looking
        // through stripping pointer casts.
        if (StrippedPointerCastsForAliasAnalysis &&
            !RI->getReturnValue()->getType()->isVoidTy()) {
          return false;
        }
        RetVal = getVal(RI->getOperand(0));
        if (EvalMode == Mode::Runtime) {
          Attribute Range = F->getRetAttribute(Attribute::Range);
          if (Range.isValid() && !constantIsInRange(RetVal, Range.getRange()))
            return false;
        }
      }
      CallStack.pop_back();
      return true;
    }

    // Okay, we succeeded in evaluating this control flow.  See if we have
    // executed the new block before.  If so, we have a looping function,
    // which we cannot evaluate in reasonable time.
    if (!ExecutedBlocks.insert(NextBB).second)
      return false;  // looped!

    // Okay, we have never been in this block before.  Check to see if there
    // are any PHI nodes.  If so, evaluate them with information about where
    // we came from.
    PHINode *PN = nullptr;
    for (CurInst = NextBB->begin(); (PN = dyn_cast<PHINode>(CurInst));
         ++CurInst) {
      if (EvalMode == Mode::Runtime && !takeRuntimeStep())
        return false;
      Constant *C = getRuntimeVal(PN->getIncomingValueForBlock(CurBB));
      if (!C)
        return false;
      setVal(PN, C);
    }

    // Advance to the next block.
    CurBB = NextBB;
  }
}
