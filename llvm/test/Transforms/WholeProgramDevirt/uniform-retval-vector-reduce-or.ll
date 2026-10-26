; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x ptr] [ptr @vf1], !type !0
@vt2 = constant [1 x ptr] [ptr @vf2], !type !0

define i8 @vf1(ptr %this, i8 %arg) memory(none) {
  %v0 = insertelement <2 x i8> poison, i8 %arg, i32 0
  %v1 = insertelement <2 x i8> %v0, i8 10, i32 1
  %r = call i8 @llvm.vector.reduce.or.v2i8(<2 x i8> %v1)
  ret i8 %r
}

define i8 @vf2(ptr %this, i8 %arg) memory(none) {
  %v = insertelement <2 x i8> <i8 0, i8 15>, i8 %arg, i32 0
  %r = call i8 @llvm.vector.reduce.or.v2i8(<2 x i8> %v)
  ret i8 %r
}

; Both targets return 15 for this argument, but not for arbitrary arguments.
; CHECK-LABEL: define i8 @constant_arg(
; CHECK: call i8 %fptr(ptr %obj, i8 5)
define i8 @constant_arg(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i8 %fptr(ptr %obj, i8 5)
  ret i8 %result
}

; A nonconstant argument must still use the generic virtual call.
; CHECK-LABEL: define i8 @nonconstant_arg(
; CHECK: call i8 %fptr(ptr %obj, i8 %arg)
define i8 @nonconstant_arg(ptr %obj, i8 %arg) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i8 %fptr(ptr %obj, i8 %arg)
  ret i8 %result
}

declare i8 @llvm.vector.reduce.or.v2i8(<2 x i8>)
declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
