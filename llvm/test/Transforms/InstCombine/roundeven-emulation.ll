; RUN: opt < %s -passes=instcombine -S | FileCheck %s

declare half @llvm.floor.f16(half)
declare float @llvm.floor.f32(float)
declare double @llvm.floor.f64(double)
declare <4 x float> @llvm.floor.v4f32(<4 x float>)

; Freeze a potentially undef input before reusing it in the zero fixup.
define float @scalar(float %x) {
; CHECK-LABEL: define float @scalar(
; CHECK-SAME: float [[X:%.*]])
; CHECK-NEXT: [[FROZEN:%.*]] = freeze float [[X]]
; CHECK-NEXT: [[ROUND:%.*]] = call float @llvm.roundeven.f32(float [[FROZEN]])
; CHECK-NEXT: [[POS:%.*]] = fadd float [[ROUND]], 0.000000e+00
; CHECK-NEXT: [[ISZERO:%.*]] = fcmp oeq float [[FROZEN]], 0.000000e+00
; CHECK-NEXT: [[RESULT:%.*]] = select i1 [[ISZERO]], float [[FROZEN]], float [[POS]]
; CHECK-NEXT: ret float [[RESULT]]
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

; An already defined input needs no freeze.
define float @scalar_noundef(float noundef %x) {
; CHECK-LABEL: define float @scalar_noundef(
; CHECK-SAME: float noundef [[X:%.*]])
; CHECK-NEXT: [[ROUND:%.*]] = call float @llvm.roundeven.f32(float [[X]])
; CHECK-NEXT: [[POS:%.*]] = fadd float [[ROUND]], 0.000000e+00
; CHECK-NEXT: [[ISZERO:%.*]] = fcmp oeq float [[X]], 0.000000e+00
; CHECK-NEXT: [[RESULT:%.*]] = select i1 [[ISZERO]], float [[X]], float [[POS]]
; CHECK-NEXT: ret float [[RESULT]]
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

define half @half(half %x) {
; CHECK-LABEL: define half @half(
; CHECK-SAME: half [[X:%.*]])
; CHECK-NEXT: [[FROZEN:%.*]] = freeze half [[X]]
; CHECK-NEXT: [[ROUND:%.*]] = call half @llvm.roundeven.f16(half [[FROZEN]])
; CHECK-NEXT: [[POS:%.*]] = fadd half [[ROUND]], 0xH0000
; CHECK-NEXT: [[ISZERO:%.*]] = fcmp oeq half [[FROZEN]], 0xH0000
; CHECK-NEXT: [[RESULT:%.*]] = select i1 [[ISZERO]], half [[FROZEN]], half [[POS]]
; CHECK-NEXT: ret half [[RESULT]]
;
  %floor = call half @llvm.floor.f16(half %x)
  %remainder = fsub half %x, %floor
  %greater = fcmp ogt half %remainder, 5.000000e-01
  %floor.plus.one = fadd half %floor, 1.000000e+00
  %non.tie = select i1 %greater, half %floor.plus.one, half %floor
  %half.floor = fmul half %floor, 5.000000e-01
  %half.floor.rounded = call half @llvm.floor.f16(half %half.floor)
  %twice.half.floor = fmul half %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub half %floor, %twice.half.floor
  %is.odd = fcmp oeq half %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, half %floor.plus.one, half %floor
  %is.half = fcmp oeq half %remainder, 5.000000e-01
  %rounded = select i1 %is.half, half %tie, half %non.tie
  ret half %rounded
}

define double @double(double %x) {
; CHECK-LABEL: define double @double(
; CHECK-SAME: double [[X:%.*]])
; CHECK-NEXT: [[FROZEN:%.*]] = freeze double [[X]]
; CHECK-NEXT: [[ROUND:%.*]] = call double @llvm.roundeven.f64(double [[FROZEN]])
; CHECK-NEXT: [[POS:%.*]] = fadd double [[ROUND]], 0.000000e+00
; CHECK-NEXT: [[ISZERO:%.*]] = fcmp oeq double [[FROZEN]], 0.000000e+00
; CHECK-NEXT: [[RESULT:%.*]] = select i1 [[ISZERO]], double [[FROZEN]], double [[POS]]
; CHECK-NEXT: ret double [[RESULT]]
;
  %floor = call double @llvm.floor.f64(double %x)
  %remainder = fsub double %x, %floor
  %greater = fcmp ogt double %remainder, 5.000000e-01
  %floor.plus.one = fadd double %floor, 1.000000e+00
  %non.tie = select i1 %greater, double %floor.plus.one, double %floor
  %half.floor = fmul double %floor, 5.000000e-01
  %half.floor.rounded = call double @llvm.floor.f64(double %half.floor)
  %twice.half.floor = fmul double %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub double %floor, %twice.half.floor
  %is.odd = fcmp oeq double %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, double %floor.plus.one, double %floor
  %is.half = fcmp oeq double %remainder, 5.000000e-01
  %rounded = select i1 %is.half, double %tie, double %non.tie
  ret double %rounded
}

define <4 x float> @vector(<4 x float> %x) {
; CHECK-LABEL: define <4 x float> @vector(
; CHECK-SAME: <4 x float> [[X:%.*]])
; CHECK-NEXT: [[FROZEN:%.*]] = freeze <4 x float> [[X]]
; CHECK-NEXT: [[ROUND:%.*]] = call <4 x float> @llvm.roundeven.v4f32(<4 x float> [[FROZEN]])
; CHECK-NEXT: [[POS:%.*]] = fadd <4 x float> [[ROUND]], zeroinitializer
; CHECK-NEXT: [[ISZERO:%.*]] = fcmp oeq <4 x float> [[FROZEN]], zeroinitializer
; CHECK-NEXT: [[RESULT:%.*]] = select <4 x i1> [[ISZERO]], <4 x float> [[FROZEN]], <4 x float> [[POS]]
; CHECK-NEXT: ret <4 x float> [[RESULT]]
;
  %floor = call <4 x float> @llvm.floor.v4f32(<4 x float> %x)
  %remainder = fsub <4 x float> %x, %floor
  %greater = fcmp ogt <4 x float> %remainder, splat (float 5.000000e-01)
  %floor.plus.one = fadd <4 x float> %floor, splat (float 1.000000e+00)
  %non.tie = select <4 x i1> %greater, <4 x float> %floor.plus.one, <4 x float> %floor
  %half.floor = fmul <4 x float> %floor, splat (float 5.000000e-01)
  %half.floor.rounded = call <4 x float> @llvm.floor.v4f32(<4 x float> %half.floor)
  %twice.half.floor = fmul <4 x float> %half.floor.rounded, splat (float 2.000000e+00)
  %odd.remainder = fsub <4 x float> %floor, %twice.half.floor
  %is.odd = fcmp oeq <4 x float> %odd.remainder, splat (float 1.000000e+00)
  %tie = select <4 x i1> %is.odd, <4 x float> %floor.plus.one, <4 x float> %floor
  %is.half = fcmp oeq <4 x float> %remainder, splat (float 5.000000e-01)
  %rounded = select <4 x i1> %is.half, <4 x float> %tie, <4 x float> %non.tie
  ret <4 x float> %rounded
}

define float @added_to_positive(float %x) {
; CHECK-LABEL: define float @added_to_positive(
; CHECK-SAME: float [[X:%.*]])
; CHECK-NEXT: [[ROUND:%.*]] = call float @llvm.roundeven.f32(float [[X]])
; CHECK-NEXT: [[SUM:%.*]] = fadd float [[ROUND]], 1.280000e+02
; CHECK-NEXT: ret float [[SUM]]
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  %sum = fadd float %rounded, 1.280000e+02
  ret float %sum
}

; Accept both orders of fadd/fmul constants and the consuming fadd.
define float @commuted(float %x) {
; CHECK-LABEL: define float @commuted(
; CHECK-SAME: float [[X:%.*]])
; CHECK-NEXT: [[ROUND:%.*]] = call float @llvm.roundeven.f32(float [[X]])
; CHECK-NEXT: [[SUM:%.*]] = fadd float [[ROUND]], 1.280000e+02
; CHECK-NEXT: ret float [[SUM]]
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float 1.000000e+00, %floor
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float 5.000000e-01, %floor
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float 2.000000e+00, %half.floor.rounded
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  %sum = fadd float 1.280000e+02, %rounded
  ret float %sum
}

; The other addend might be -0.0, so the zero fixup is required.
define float @added_to_unknown(float %x, float %y) {
; CHECK-LABEL: define float @added_to_unknown(
; CHECK-SAME: float [[X:%.*]], float [[Y:%.*]])
; CHECK-NEXT: [[FROZEN:%.*]] = freeze float [[X]]
; CHECK-NEXT: [[ROUND:%.*]] = call float @llvm.roundeven.f32(float [[FROZEN]])
; CHECK-NEXT: [[POS:%.*]] = fadd float [[ROUND]], 0.000000e+00
; CHECK-NEXT: [[ISZERO:%.*]] = fcmp oeq float [[FROZEN]], 0.000000e+00
; CHECK-NEXT: [[RESULT:%.*]] = select i1 [[ISZERO]], float [[FROZEN]], float [[POS]]
; CHECK-NEXT: [[SUM:%.*]] = fadd float [[RESULT]], [[Y]]
; CHECK-NEXT: ret float [[SUM]]
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  %sum = fadd float %rounded, %y
  ret float %sum
}

; A comparison cannot distinguish the sign of zero.
define i1 @compared_with_zero(float %x) {
; CHECK-LABEL: define i1 @compared_with_zero(
; CHECK-SAME: float [[X:%.*]])
; CHECK-NEXT: [[ROUND:%.*]] = call float @llvm.roundeven.f32(float [[X]])
; CHECK-NEXT: [[CMP:%.*]] = fcmp oeq float [[ROUND]], 0.000000e+00
; CHECK-NEXT: ret i1 [[CMP]]
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  %cmp = fcmp oeq float %rounded, 0.000000e+00
  ret i1 %cmp
}

; The entire expansion must become dead; keep a shared remainder.
define { float, float } @extra_use(float %x) {
; CHECK-LABEL: define { float, float } @extra_use(
; CHECK-NOT: @llvm.roundeven.
; CHECK: ret { float, float }
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  %result.0 = insertvalue { float, float } poison, float %rounded, 0
  %result.1 = insertvalue { float, float } %result.0, float %remainder, 1
  ret { float, float } %result.1
}

define float @root_nsz(float %x) {
; CHECK-LABEL: define float @root_nsz(
; CHECK-SAME: float [[X:%.*]])
; CHECK-NEXT: [[ROUND:%.*]] = call nsz float @llvm.roundeven.f32(float [[X]])
; CHECK-NEXT: ret float [[ROUND]]
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select nsz i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

define float @wrong_non_tie_threshold(float %x) {
; CHECK-LABEL: define float @wrong_non_tie_threshold(
; CHECK-NOT: @llvm.roundeven.
; CHECK: ret float
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 2.500000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

define float @wrong_odd_value(float %x) {
; CHECK-LABEL: define float @wrong_odd_value(
; CHECK-NOT: @llvm.roundeven.
; CHECK: ret float
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 0.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

; A flushed subnormal may compare equal to zero without being zero.
define float @flush_inputs(float %x) #0 {
; CHECK-LABEL: define float @flush_inputs(
; CHECK-NOT: @llvm.roundeven.
; CHECK: ret float
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

; The other addend can become -0.0 when inputs are flushed.
define float @flush_inputs_added_to_negative_subnormal(float %x) #0 {
; CHECK-LABEL: define float @flush_inputs_added_to_negative_subnormal(
; CHECK-NOT: @llvm.roundeven.
; CHECK: ret float
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  %sum = fadd float %rounded, 0xB6A0000000000000
  ret float %sum
}

define float @flush_outputs(float %x) #1 {
; CHECK-LABEL: define float @flush_outputs(
; CHECK-NOT: @llvm.roundeven.
; CHECK: ret float
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

define float @dynamic_denormals(float %x) #2 {
; CHECK-LABEL: define float @dynamic_denormals(
; CHECK-NOT: @llvm.roundeven.
; CHECK: ret float
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

; Use the scalar element type for the float override.
define <4 x float> @float_override_flush_vector(<4 x float> %x) #3 {
; CHECK-LABEL: define <4 x float> @float_override_flush_vector(
; CHECK-NOT: @llvm.roundeven.
; CHECK: ret <4 x float>
;
  %floor = call <4 x float> @llvm.floor.v4f32(<4 x float> %x)
  %remainder = fsub <4 x float> %x, %floor
  %greater = fcmp ogt <4 x float> %remainder, splat (float 5.000000e-01)
  %floor.plus.one = fadd <4 x float> %floor, splat (float 1.000000e+00)
  %non.tie = select <4 x i1> %greater, <4 x float> %floor.plus.one, <4 x float> %floor
  %half.floor = fmul <4 x float> %floor, splat (float 5.000000e-01)
  %half.floor.rounded = call <4 x float> @llvm.floor.v4f32(<4 x float> %half.floor)
  %twice.half.floor = fmul <4 x float> %half.floor.rounded, splat (float 2.000000e+00)
  %odd.remainder = fsub <4 x float> %floor, %twice.half.floor
  %is.odd = fcmp oeq <4 x float> %odd.remainder, splat (float 1.000000e+00)
  %tie = select <4 x i1> %is.odd, <4 x float> %floor.plus.one, <4 x float> %floor
  %is.half = fcmp oeq <4 x float> %remainder, splat (float 5.000000e-01)
  %rounded = select <4 x i1> %is.half, <4 x float> %tie, <4 x float> %non.tie
  ret <4 x float> %rounded
}

define float @float_override_ieee(float %x) #4 {
; CHECK-LABEL: define float @float_override_ieee(
; CHECK-SAME: float [[X:%.*]])
; CHECK-NEXT: [[ROUND:%.*]] = call nsz float @llvm.roundeven.f32(float [[X]])
; CHECK-NEXT: ret float [[ROUND]]
;
  %floor = call float @llvm.floor.f32(float %x)
  %remainder = fsub float %x, %floor
  %greater = fcmp ogt float %remainder, 5.000000e-01
  %floor.plus.one = fadd float %floor, 1.000000e+00
  %non.tie = select i1 %greater, float %floor.plus.one, float %floor
  %half.floor = fmul float %floor, 5.000000e-01
  %half.floor.rounded = call float @llvm.floor.f32(float %half.floor)
  %twice.half.floor = fmul float %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub float %floor, %twice.half.floor
  %is.odd = fcmp oeq float %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, float %floor.plus.one, float %floor
  %is.half = fcmp oeq float %remainder, 5.000000e-01
  %rounded = select nsz i1 %is.half, float %tie, float %non.tie
  ret float %rounded
}

define double @double_ignores_float_override(double noundef %x) #3 {
; CHECK-LABEL: define double @double_ignores_float_override(
; CHECK-SAME: double noundef [[X:%.*]])
; CHECK-NEXT: [[ROUND:%.*]] = call double @llvm.roundeven.f64(double [[X]])
; CHECK-NEXT: [[POS:%.*]] = fadd double [[ROUND]], 0.000000e+00
; CHECK-NEXT: [[ISZERO:%.*]] = fcmp oeq double [[X]], 0.000000e+00
; CHECK-NEXT: [[RESULT:%.*]] = select i1 [[ISZERO]], double [[X]], double [[POS]]
; CHECK-NEXT: ret double [[RESULT]]
;
  %floor = call double @llvm.floor.f64(double %x)
  %remainder = fsub double %x, %floor
  %greater = fcmp ogt double %remainder, 5.000000e-01
  %floor.plus.one = fadd double %floor, 1.000000e+00
  %non.tie = select i1 %greater, double %floor.plus.one, double %floor
  %half.floor = fmul double %floor, 5.000000e-01
  %half.floor.rounded = call double @llvm.floor.f64(double %half.floor)
  %twice.half.floor = fmul double %half.floor.rounded, 2.000000e+00
  %odd.remainder = fsub double %floor, %twice.half.floor
  %is.odd = fcmp oeq double %odd.remainder, 1.000000e+00
  %tie = select i1 %is.odd, double %floor.plus.one, double %floor
  %is.half = fcmp oeq double %remainder, 5.000000e-01
  %rounded = select i1 %is.half, double %tie, double %non.tie
  ret double %rounded
}

attributes #0 = { denormal_fpenv(ieee|preservesign) }
attributes #1 = { denormal_fpenv(preservesign|ieee) }
attributes #2 = { denormal_fpenv(dynamic) }
attributes #3 = { denormal_fpenv(ieee, float: preservesign) }
attributes #4 = { denormal_fpenv(preservesign, float: ieee) }
