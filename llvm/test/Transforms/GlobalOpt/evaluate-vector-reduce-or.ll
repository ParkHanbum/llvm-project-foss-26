; RUN: opt -S -passes=globalopt %s | FileCheck %s

; The reduction operands are computed in the evaluated call frame, rather
; than being literal constants at the intrinsic call.

; CHECK: @byte = local_unnamed_addr global i8 0
; CHECK: @bit = local_unnamed_addr global i1 false
; CHECK: @wide = local_unnamed_addr global i64 0
; CHECK: @unsupported = local_unnamed_addr global i8 0
; CHECK: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }]

@byte = global i8 0
@bit = global i1 false
@wide = global i64 0
@unsupported = global i8 0
@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 0, ptr @init_or, ptr null },
  { i32, ptr, ptr } { i32 1, ptr @init_xor, ptr null }
]

define internal i8 @reduce_byte(<4 x i8> %v) {
  %r = call i8 @llvm.vector.reduce.or.v4i8(<4 x i8> %v)
  ret i8 %r
}

define internal void @init_or() {
  %byte = call i8 @reduce_byte(<4 x i8> <i8 2, i8 8, i8 32, i8 -128>)
  store i8 %byte, ptr @byte
  %bit = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> <i1 false, i1 false, i1 true, i1 false>)
  store i1 %bit, ptr @bit
  %wide = call i64 @llvm.vector.reduce.or.v2i64(<2 x i64> <i64 281474976710656, i64 3>)
  store i64 %wide, ptr @wide
  ret void
}

; Supporting OR must not enable other reductions implicitly.
define internal void @init_xor() {
  %r = call i8 @llvm.vector.reduce.xor.v4i8(<4 x i8> <i8 1, i8 2, i8 4, i8 8>)
  store i8 %r, ptr @unsupported
  ret void
}

declare i8 @llvm.vector.reduce.or.v4i8(<4 x i8>)
declare i1 @llvm.vector.reduce.or.v4i1(<4 x i1>)
declare i64 @llvm.vector.reduce.or.v2i64(<2 x i64>)
declare i8 @llvm.vector.reduce.xor.v4i8(<4 x i8>)
