; RUN: opt -passes='correlated-propagation<finite-values>' -print-pipeline-passes -disable-output %s | FileCheck %s --check-prefix=PIPELINE
; PIPELINE: function(correlated-propagation<finite-values>)
; RUN: opt -passes='correlated-propagation<finite-values>,verify' -S %s | FileCheck %s --check-prefix=FACTS
; RUN: opt -passes='correlated-propagation<finite-values>,instcombine,sccp,instcombine,verify' -S %s | FileCheck %s --check-prefix=CLEAN
; RUN: opt -passes='correlated-propagation,instcombine,verify' -S %s | FileCheck %s --check-prefix=DISABLED

@state = global i32 7
declare void @opaque()

define internal i32 @identity(i32 %x) {
  %ok = icmp sge i32 %x, 0
  br i1 %ok, label %pure, label %effect
pure:
  ret i32 %x
effect:
  call void @opaque()
  ret i32 14
}

; Facts belong to this call. Other callers may still execute the effect path.
define i32 @finite_identity(i32 noundef range(i32 2, 4) %x) {
; DISABLED-LABEL: define {{.*}} @finite_identity(
; DISABLED: call i32 @identity(i32 %x)
; FACTS-LABEL: define {{.*}} @finite_identity(
; FACTS: %r = call {{.*}} @identity(i32 {{.*}}returned %x)
; FACTS: ret i32 %r
; CLEAN-LABEL: define {{.*}} @finite_identity(
; CLEAN-NOT: call
; CLEAN: ret i32 %x
  %r = call i32 @identity(i32 %x)
  ret i32 %r
}

define i32 @unrestricted(i32 noundef %x) {
; FACTS-LABEL: define {{.*}} @unrestricted(
; FACTS: call i32 @identity(i32 %x)
; CLEAN-LABEL: define {{.*}} @unrestricted(
; CLEAN: call i32 @identity(i32 %x)
  %r = call i32 @identity(i32 %x)
  ret i32 %r
}

define internal i32 @uniform(i32 %x) {
  %ok = icmp sge i32 %x, 0
  br i1 %ok, label %pure, label %effect
pure:
  ret i32 9
effect:
  call void @opaque()
  ret i32 14
}

define i32 @finite_uniform(i32 noundef range(i32 2, 4) %x) {
; DISABLED-LABEL: define {{.*}} @finite_uniform(
; DISABLED: call i32 @uniform(i32 %x)
; FACTS-LABEL: define {{.*}} @finite_uniform(
; FACTS: %r = call noundef range(i32 9, 10) i32 @uniform(i32 %x) #[[PURE:[0-9]+]]
; FACTS: ret i32 %r
; CLEAN-LABEL: define {{.*}} @finite_uniform(
; CLEAN-NOT: call
; CLEAN: ret i32 9
  %r = call i32 @uniform(i32 %x)
  ret i32 %r
}

define internal i32 @read_mutable(i32 %x) {
  %r = load i32, ptr @state
  ret i32 %r
}

; A static initializer is not the memory state at this runtime call.
define i32 @after_write(i32 noundef range(i32 2, 4) %x) {
; FACTS-LABEL: define {{.*}} @after_write(
; FACTS: store i32 99, ptr @state
; FACTS: %r = call i32 @read_mutable(i32 %x)
; FACTS: ret i32 %r
; CLEAN-LABEL: define {{.*}} @after_write(
; CLEAN: store i32 99, ptr @state
; CLEAN: %r = call i32 @read_mutable(i32 %x)
; CLEAN: ret i32 %r
  store i32 99, ptr @state
  %r = call i32 @read_mutable(i32 %x)
  ret i32 %r
}

define internal range(i32 9, 10) i32 @poison_return(i32 %x) {
  ret i32 %x
}

; Out-of-range results are poison, which a select may mask. Inferring noundef
; here would turn that defined execution into immediate UB.
define i32 @masked_poison_return(i32 noundef range(i32 2, 4) %x, i1 %c) {
; FACTS-LABEL: define {{.*}} @masked_poison_return(
; FACTS: %r = call i32 @poison_return(i32 %x)
; FACTS: %s = select i1 %c, i32 %r, i32 0
; CLEAN-LABEL: define {{.*}} @masked_poison_return(
; CLEAN: %r = call i32 @poison_return(i32 %x)
; CLEAN: %s = select i1 %c, i32 9, i32 0
; CLEAN: ret i32 %s
  %r = call i32 @poison_return(i32 %x)
  %s = select i1 %c, i32 %r, i32 0
  ret i32 %s
}

define i32 @masked_callsite_range(i32 noundef range(i32 2, 4) %x, i1 %c) {
; FACTS-LABEL: define {{.*}} @masked_callsite_range(
; FACTS: %r = call range(i32 9, 10) i32 @identity(i32 %x)
; CLEAN-LABEL: define {{.*}} @masked_callsite_range(
  %r = call range(i32 9, 10) i32 @identity(i32 %x)
  %s = select i1 %c, i32 %r, i32 0
  ret i32 %s
}

define internal i32 @argument_identity(i32 %x) {
  ret i32 %x
}

define i32 @masked_parameter_range(i32 noundef range(i32 2, 4) %x, i1 %c) {
; FACTS-LABEL: define {{.*}} @masked_parameter_range(
; FACTS: %r = call i32 @argument_identity(i32 range(i32 9, 10) %x)
; CLEAN-LABEL: define {{.*}} @masked_parameter_range(
  %r = call i32 @argument_identity(i32 range(i32 9, 10) %x)
  %s = select i1 %c, i32 %r, i32 0
  ret i32 %s
}

declare i1 @llvm.is.constant.i32(i32)

define internal i1 @constant_query(i32 %x) {
  %r = call i1 @llvm.is.constant.i32(i32 %x)
  ret i1 %r
}

; Knowing a small runtime input set must not make these manifest constants.
define i1 @finite_is_constant(i1 noundef %c) {
; FACTS-LABEL: define {{.*}} @finite_is_constant(
; FACTS: %v = select i1 %c, i32 2, i32 3
; FACTS: %r = call i1 @llvm.is.constant.i32(i32 %v)
; FACTS: ret i1 %r
; CLEAN-LABEL: define {{.*}} @finite_is_constant(
  %v = select i1 %c, i32 2, i32 3
  %r = call i1 @llvm.is.constant.i32(i32 %v)
  ret i1 %r
}

define i1 @finite_helper_is_constant(i1 noundef %c) {
; FACTS-LABEL: define {{.*}} @finite_helper_is_constant(
; FACTS: %v = select i1 %c, i32 2, i32 3
; FACTS: %r = call i1 @constant_query(i32 %v)
; FACTS: ret i1 %r
; CLEAN-LABEL: define {{.*}} @finite_helper_is_constant(
  %v = select i1 %c, i32 2, i32 3
  %r = call i1 @constant_query(i32 %v)
  ret i1 %r
}

; FACTS: attributes #[[PURE]] = { nounwind willreturn memory(none) }
