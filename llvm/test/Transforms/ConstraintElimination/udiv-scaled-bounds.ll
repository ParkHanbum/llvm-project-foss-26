; RUN: opt -passes=constraint-elimination -S %s | FileCheck %s

declare void @llvm.assume(i1)
declare i64 @llvm.smax.i64(i64, i64)
declare i64 @llvm.smin.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare void @consume.i64(i64)

; Unsigned division has the implicit relation
; 5 * (length /u 5) <= length.  Together with position < quotient - 1, this
; proves that ten input elements remain for every output position.
define i64 @udiv_conv_bound(i64 noundef %length, i64 noundef %position) {
; CHECK-LABEL: define i64 @udiv_conv_bound(
; CHECK:       ret i64 10
;
entry:
  %length.nonnegative = icmp sge i64 %length, 0
  call void @llvm.assume(i1 %length.nonnegative)
  %position.nonnegative = icmp sge i64 %position, 0
  call void @llvm.assume(i1 %position.nonnegative)
  %quotient = udiv i64 %length, 5
  %output.length = add nsw i64 %quotient, -1
  %position.in.range = icmp slt i64 %position, %output.length
  call void @llvm.assume(i1 %position.in.range)
  %scaled.neg = mul nsw i64 %position, -5
  %remaining = add i64 %scaled.neg, %length
  %positive.remaining = call i64 @llvm.smax.i64(i64 %remaining, i64 1)
  %kernel.limit = call i64 @llvm.umin.i64(i64 %positive.remaining, i64 10)
  ret i64 %kernel.limit
}

; The upper and lower quotient relations connect independently canonicalized
; divisions.  In particular, 10*q10 <= length <= 5*q5+4 implies
; 2*q10 <= q5, which proves the kernel-3 bound of the next strided Conv.
define i64 @nested_udiv_conv_bound(i64 noundef %length,
                                   i64 noundef %position) {
; CHECK-LABEL: define i64 @nested_udiv_conv_bound(
; CHECK:       ret i64 3
;
entry:
  %length.nonnegative = icmp sge i64 %length, 0
  call void @llvm.assume(i1 %length.nonnegative)
  %position.nonnegative = icmp sge i64 %position, 0
  call void @llvm.assume(i1 %position.nonnegative)
  %q5 = udiv i64 %length, 5
  %input.length = add nsw i64 %q5, -1
  %q10 = udiv i64 %length, 10
  %output.length = add nsw i64 %q10, -1
  %position.in.range = icmp slt i64 %position, %output.length
  call void @llvm.assume(i1 %position.in.range)
  %scaled.neg = mul nsw i64 %position, -2
  %remaining = add i64 %scaled.neg, %input.length
  %positive.remaining = call i64 @llvm.smax.i64(i64 %remaining, i64 1)
  %kernel.limit = call i64 @llvm.umin.i64(i64 %positive.remaining, i64 3)
  ret i64 %kernel.limit
}

; Match the control-flow shape emitted for the next Conv in Distil-Wav2Vec2:
; the output-position bound is a loop-header branch and the kernel bound is
; computed inside a nested channel loop.  The true edge into position.body
; dominates the nested loop, so position < output.length must be available
; while simplifying the smin.
define void @nested_udiv_conv_bound_in_loop(i64 noundef %length) {
; CHECK-LABEL: define void @nested_udiv_conv_bound_in_loop(
; CHECK:       kernel.body:
; CHECK:       call void @consume.i64(i64 3)
;
entry:
  %length.nonnegative = icmp sge i64 %length, 0
  call void @llvm.assume(i1 %length.nonnegative)
  %q5 = udiv i64 %length, 5
  %input.length = add nsw i64 %q5, -1
  %q10 = udiv i64 %length, 10
  %output.length = add nsw i64 %q10, -1
  br label %position.header

position.header:
  %position = phi i64 [ 0, %entry ], [ %position.next, %position.latch ]
  %position.in.range = icmp slt i64 %position, %output.length
  br i1 %position.in.range, label %position.body, label %exit

position.body:
  br label %channel.header

channel.header:
  %channel = phi i64 [ 0, %position.body ], [ %channel.next, %kernel.body ]
  %channel.in.range = icmp slt i64 %channel, 512
  br i1 %channel.in.range, label %kernel.body, label %position.latch

kernel.body:
  %scaled = shl i64 %position, 1
  %negated = xor i64 %scaled, -1
  %remaining = add i64 %negated, %q5
  %kernel.limit = call i64 @llvm.smin.i64(i64 %remaining, i64 3)
  call void @consume.i64(i64 %kernel.limit)
  %channel.next = add nuw nsw i64 %channel, 1
  br label %channel.header

position.latch:
  %position.next = add i64 %position, 1
  br label %position.header

exit:
  ret void
}

; A non-negative dividend is at most signed-max, so division by ten has the
; signed upper bound floor(signed-max / 10).
define i1 @udiv_signed_upper_bound(i64 noundef %length) {
; CHECK-LABEL: define i1 @udiv_signed_upper_bound(
; CHECK:       ret i1 true
;
entry:
  %length.nonnegative = icmp sge i64 %length, 0
  call void @llvm.assume(i1 %length.nonnegative)
  %quotient = udiv i64 %length, 10
  %in.range = icmp sle i64 %quotient, 922337203685477580
  ret i1 %in.range
}

; The signed upper bound makes this unflagged shift equivalent to a
; non-wrapping multiplication by two.
define i1 @shl_with_proven_signed_range(i64 noundef %value) {
; CHECK-LABEL: define i1 @shl_with_proven_signed_range(
; CHECK:       ret i1 true
;
entry:
  %value.nonnegative = icmp sge i64 %value, 0
  call void @llvm.assume(i1 %value.nonnegative)
  %value.in.range = icmp sle i64 %value, 4611686018427387903
  call void @llvm.assume(i1 %value.in.range)
  %scaled = shl i64 %value, 1
  %scaled.nonnegative = icmp sge i64 %scaled, 0
  ret i1 %scaled.nonnegative
}

; Without the upper bound the shift may move a one into the sign bit, so it
; must remain opaque in the signed constraint system.
define i1 @shl_without_proven_signed_range(i64 noundef %value) {
; CHECK-LABEL: define i1 @shl_without_proven_signed_range(
; CHECK:       %[[SCALED:.*]] = shl{{( nuw)?}} i64 %value, 1
; CHECK-NEXT:  %[[SCALED_NONNEGATIVE:.*]] = icmp sge i64 %[[SCALED]], 0
; CHECK-NEXT:  ret i1 %[[SCALED_NONNEGATIVE]]
;
entry:
  %value.nonnegative = icmp sge i64 %value, 0
  call void @llvm.assume(i1 %value.nonnegative)
  %scaled = shl i64 %value, 1
  %scaled.nonnegative = icmp sge i64 %scaled, 0
  ret i1 %scaled.nonnegative
}

; Two non-negative operands can still signed-wrap, so an unflagged same-sign
; add must not be decomposed.
define i1 @same_sign_add_may_wrap(i64 noundef %lhs, i64 noundef %rhs) {
; CHECK-LABEL: define i1 @same_sign_add_may_wrap(
; CHECK:       %[[SUM:.*]] = add i64 %lhs, %rhs
; CHECK-NEXT:  %[[SUM_NONNEGATIVE:.*]] = icmp sge i64 %[[SUM]], 0
; CHECK-NEXT:  ret i1 %[[SUM_NONNEGATIVE]]
;
entry:
  %lhs.nonnegative = icmp sge i64 %lhs, 0
  call void @llvm.assume(i1 %lhs.nonnegative)
  %rhs.nonnegative = icmp sge i64 %rhs, 0
  call void @llvm.assume(i1 %rhs.nonnegative)
  %sum = add i64 %lhs, %rhs
  %sum.nonnegative = icmp sge i64 %sum, 0
  ret i1 %sum.nonnegative
}

; Adding a non-negative operand and a non-positive operand cannot signed-wrap.
; Decomposing this unflagged add is therefore valid in the signed system.
define i1 @opposite_sign_add_upper_bound(i64 noundef %nonnegative,
                                         i64 noundef %nonpositive) {
; CHECK-LABEL: define i1 @opposite_sign_add_upper_bound(
; CHECK:       ret i1 true
;
entry:
  %lhs.fact = icmp sge i64 %nonnegative, 0
  call void @llvm.assume(i1 %lhs.fact)
  %rhs.fact = icmp sle i64 %nonpositive, 0
  call void @llvm.assume(i1 %rhs.fact)
  %sum = add i64 %nonnegative, %nonpositive
  %upper = icmp sle i64 %sum, %nonnegative
  ret i1 %upper
}

define i1 @opposite_sign_add_lower_bound(i64 noundef %nonnegative,
                                         i64 noundef %nonpositive) {
; CHECK-LABEL: define i1 @opposite_sign_add_lower_bound(
; CHECK:       ret i1 true
;
entry:
  %lhs.fact = icmp sge i64 %nonnegative, 0
  call void @llvm.assume(i1 %lhs.fact)
  %rhs.fact = icmp sle i64 %nonpositive, 0
  call void @llvm.assume(i1 %rhs.fact)
  %sum = add i64 %nonnegative, %nonpositive
  %lower = icmp sge i64 %sum, %nonpositive
  ret i1 %lower
}

; The upper quotient relation also derives a lower bound on the quotient from
; a lower bound on the dividend, as required by the guarded Conv output loop.
define i64 @udiv_quotient_lower_bound(i64 noundef %length) {
; CHECK-LABEL: define i64 @udiv_quotient_lower_bound(
; CHECK:       ret i64 %output.length
;
entry:
  %length.nonnegative = icmp sge i64 %length, 0
  call void @llvm.assume(i1 %length.nonnegative)
  %quotient = udiv i64 %length, 10
  %length.large = icmp ugt i64 %length, 19
  call void @llvm.assume(i1 %length.large)
  %output.length = add nsw i64 %quotient, -1
  %bounded = call i64 @llvm.smax.i64(i64 %output.length, i64 1)
  ret i64 %bounded
}

; The unsigned facts do not require a signed non-negative dividend.
define i1 @unsigned_scaled_lower_bound(i8 noundef %x) {
; CHECK-LABEL: define i1 @unsigned_scaled_lower_bound(
; CHECK:       ret i1 true
  %quotient = udiv i8 %x, 5
  %scaled = mul nuw i8 %quotient, 5
  %in.range = icmp ule i8 %scaled, %x
  ret i1 %in.range
}

define i1 @unsigned_remainder_upper_bound(i8 noundef %x) {
; CHECK-LABEL: define i1 @unsigned_remainder_upper_bound(
; CHECK:       ret i1 true
  %quotient = udiv i8 %x, 6
  %scaled = mul nuw i8 %quotient, 6
  %remainder = sub nuw i8 %x, %scaled
  %in.range = icmp ult i8 %remainder, 6
  ret i1 %in.range
}

; Mathematical upper bounds must not be interpreted as wrapping IR arithmetic.
; For x = 255, q = 42 and the unflagged upper expression wraps from 257 to 1.
define i1 @unsigned_upper_expression_wraps(i8 noundef %x) {
; CHECK-LABEL: define i1 @unsigned_upper_expression_wraps(
; CHECK:       %in.range = icmp ule i8 %x, %upper
; CHECK-NEXT:  ret i1 %in.range
  %quotient = udiv i8 %x, 6
  %scaled = mul nuw i8 %quotient, 6
  %upper = add i8 %scaled, 5
  %in.range = icmp ule i8 %x, %upper
  ret i1 %in.range
}

; With x = 128, the unsigned product is 125 but x is negative as a signed i8.
define i1 @no_signed_bound_without_nonnegative_dividend(i8 noundef %x) {
; CHECK-LABEL: define i1 @no_signed_bound_without_nonnegative_dividend(
; CHECK:       %in.range = icmp sle i8 %scaled, %x
; CHECK-NEXT:  ret i1 %in.range
  %quotient = udiv i8 %x, 5
  %scaled = mul nuw i8 %quotient, 5
  %in.range = icmp sle i8 %scaled, %x
  ret i1 %in.range
}

; A fact from one predecessor must not enable signed bounds after the merge.
define i1 @nonnegative_fact_does_not_dominate(i8 noundef %x, i1 %condition) {
; CHECK-LABEL: define i1 @nonnegative_fact_does_not_dominate(
; CHECK:       merge:
; CHECK:       %in.range = icmp sle i8 %scaled, %x
; CHECK-NEXT:  ret i1 %in.range
entry:
  br i1 %condition, label %nonnegative, label %other
nonnegative:
  %nonnegative.fact = icmp sge i8 %x, 0
  call void @llvm.assume(i1 %nonnegative.fact)
  br label %merge
other:
  br label %merge
merge:
  %quotient = udiv i8 %x, 5
  %scaled = mul nuw i8 %quotient, 5
  %in.range = icmp sle i8 %scaled, %x
  ret i1 %in.range
}

; If exact division produces poison, freeze may pick a different quotient.
; In particular, x = 1 and frozen = 1 give a defined false comparison.
define i1 @exact_division_freeze(i8 noundef %x) {
; CHECK-LABEL: define i1 @exact_division_freeze(
; CHECK:       %frozen = freeze i8 %quotient
; CHECK:       %in.range = icmp ule i8 %scaled, %x
; CHECK-NEXT:  ret i1 %in.range
  %quotient = udiv exact i8 %x, 5
  %frozen = freeze i8 %quotient
  %scaled = mul nuw i8 %frozen, 5
  %in.range = icmp ule i8 %scaled, %x
  ret i1 %in.range
}

; Same-sign non-positive operands can also overflow (e.g. -128 + -1).
define i1 @negative_same_sign_add_may_wrap(i8 noundef %lhs, i8 noundef %rhs) {
; CHECK-LABEL: define i1 @negative_same_sign_add_may_wrap(
; CHECK:       %in.range = icmp sle i8 %sum, %lhs
; CHECK-NEXT:  ret i1 %in.range
  %lhs.fact = icmp sle i8 %lhs, 0
  call void @llvm.assume(i1 %lhs.fact)
  %rhs.fact = icmp sle i8 %rhs, 0
  call void @llvm.assume(i1 %rhs.fact)
  %sum = add i8 %lhs, %rhs
  %in.range = icmp sle i8 %sum, %lhs
  ret i1 %in.range
}

; Recursive sign queries must use a shared budget. Without one, proving either
; sign of a chain of unknown-sign additions can repeatedly visit each prefix.
define i1 @unknown_sign_add_chain(i64 noundef %x, i64 noundef %positive,
                                  i64 noundef %negative) {
; CHECK-LABEL: define i1 @unknown_sign_add_chain(
; CHECK:       %in.range = icmp sge i64 %a40, 0
; CHECK:       %result = and i1 %in.range, true
; CHECK-NEXT:  ret i1 %result
  %a1 = add i64 %x, 1
  %a2 = add i64 %a1, 1
  %a3 = add i64 %a2, 1
  %a4 = add i64 %a3, 1
  %a5 = add i64 %a4, 1
  %a6 = add i64 %a5, 1
  %a7 = add i64 %a6, 1
  %a8 = add i64 %a7, 1
  %a9 = add i64 %a8, 1
  %a10 = add i64 %a9, 1
  %a11 = add i64 %a10, 1
  %a12 = add i64 %a11, 1
  %a13 = add i64 %a12, 1
  %a14 = add i64 %a13, 1
  %a15 = add i64 %a14, 1
  %a16 = add i64 %a15, 1
  %a17 = add i64 %a16, 1
  %a18 = add i64 %a17, 1
  %a19 = add i64 %a18, 1
  %a20 = add i64 %a19, 1
  %a21 = add i64 %a20, 1
  %a22 = add i64 %a21, 1
  %a23 = add i64 %a22, 1
  %a24 = add i64 %a23, 1
  %a25 = add i64 %a24, 1
  %a26 = add i64 %a25, 1
  %a27 = add i64 %a26, 1
  %a28 = add i64 %a27, 1
  %a29 = add i64 %a28, 1
  %a30 = add i64 %a29, 1
  %a31 = add i64 %a30, 1
  %a32 = add i64 %a31, 1
  %a33 = add i64 %a32, 1
  %a34 = add i64 %a33, 1
  %a35 = add i64 %a34, 1
  %a36 = add i64 %a35, 1
  %a37 = add i64 %a36, 1
  %a38 = add i64 %a37, 1
  %a39 = add i64 %a38, 1
  %a40 = add i64 %a39, 1
  %in.range = icmp sge i64 %a40, 0
  ; A later, independent query must have a fresh budget.
  %positive.fact = icmp sge i64 %positive, 0
  call void @llvm.assume(i1 %positive.fact)
  %negative.fact = icmp sle i64 %negative, 0
  call void @llvm.assume(i1 %negative.fact)
  %sum = add i64 %positive, %negative
  %upper = icmp sle i64 %sum, %positive
  %result = and i1 %in.range, %upper
  ret i1 %result
}
