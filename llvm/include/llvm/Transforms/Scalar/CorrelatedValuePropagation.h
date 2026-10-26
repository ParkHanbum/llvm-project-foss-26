//===- CorrelatedValuePropagation.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CORRELATEDVALUEPROPAGATION_H
#define LLVM_TRANSFORMS_SCALAR_CORRELATEDVALUEPROPAGATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

struct CorrelatedValuePropagationPass
    : OptionalPassInfoMixin<CorrelatedValuePropagationPass> {
  /// Enable bounded evaluation over small, proven input sets. The default
  /// early pipeline keeps this off so that it does not perturb inlining.
  explicit CorrelatedValuePropagationPass(bool FoldFiniteValues = false)
      : FoldFiniteValues(FoldFiniteValues) {}

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);

private:
  bool FoldFiniteValues;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_CORRELATEDVALUEPROPAGATION_H
