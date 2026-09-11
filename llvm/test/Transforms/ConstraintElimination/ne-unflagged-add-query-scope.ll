; RUN: opt -passes=constraint-elimination -disable-output < %s
;
; Adding a NE fact also attempts to tighten bounds. Its direct decompose()
; call must be inside the same query scope as the preceding getConstraint()
; calls, including when the unflagged add needs recursive sign queries.

declare void @llvm.assume(i1)

define i1 @ne_unflagged_add(i32 %x, i32 %y) {
entry:
  %sum = add i32 %x, %y
  %cmp = icmp ne i32 %sum, 0
  br i1 %cmp, label %nonzero, label %zero

nonzero:
  ret i1 true

zero:
  ret i1 false
}

; The same path is reachable through llvm.assume. Exercise multiple facts and
; recursive sign queries on chained unflagged additions.
define i1 @ne_unflagged_add_assume(i32 %x, i32 %y, i32 %z) {
entry:
  %sum = add i32 %x, %y
  %cmp = icmp ne i32 %sum, 0
  call void @llvm.assume(i1 %cmp)
  %next = add i32 %sum, %z
  %next.cmp = icmp ne i32 %next, 0
  br i1 %next.cmp, label %nonzero, label %zero

nonzero:
  ret i1 true

zero:
  ret i1 false
}
