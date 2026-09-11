; RUN: opt -passes=constraint-elimination -constraint-elimination-dump-reproducers \
; RUN:   -pass-remarks=constraint-elimination -disable-output < %s 2>&1 | FileCheck %s

; A quotient used by implicit scaled facts must remain an instruction in the
; reproducer, even though it has an entry in the constraint variable mapping.
; Treating it as an independent argument would lose q = x /u 5.
;
; CHECK:      define i1 @{{.*}}udiv_scaled_reproducerrepro{{"?}}(i8 %x) {
; CHECK:      %quotient = udiv i8 %x, 5
; CHECK:      %scaled = mul nuw i8 %quotient, 5
; CHECK:      %in.range = icmp ule i8 %scaled, %x
; CHECK:      ret i1 %in.range
define i1 @udiv_scaled_reproducer(i8 noundef %x) {
  %quotient = udiv i8 %x, 5
  %scaled = mul nuw i8 %quotient, 5
  %in.range = icmp ule i8 %scaled, %x
  ret i1 %in.range
}
