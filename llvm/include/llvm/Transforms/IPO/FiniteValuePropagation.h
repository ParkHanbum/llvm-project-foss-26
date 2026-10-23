//===- FiniteValuePropagation.h - Finite inputs ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_FINITEVALUEPROPAGATION_H
#define LLVM_TRANSFORMS_IPO_FINITEVALUEPROPAGATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Partially evaluate computations, including direct calls, over small proven
/// input sets. This is a module pass because it inspects callee definitions.
class FiniteValuePropagationPass
    : public OptionalPassInfoMixin<FiniteValuePropagationPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
