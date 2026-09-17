; RUN: opt -S -passes=instcombine -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx70 < %s | FileCheck %s --check-prefix=SM80
; RUN: opt -S -passes=instcombine -mtriple=nvptx64-nvidia-cuda -mcpu=sm_75 -mattr=+ptx70 < %s | FileCheck %s --check-prefix=SM75

declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)

define i32 @full_warp_add(i32 %x) {
; SM80-LABEL: @full_warp_add(
; SM80:       call i32 @llvm.nvvm.redux.sync.add(i32 %x, i32 -1)
; SM80-NOT:   @llvm.nvvm.shfl.sync.bfly.i32
;
; SM75-LABEL: @full_warp_add(
; SM75-NOT:   @llvm.nvvm.redux.sync.add
; SM75:       call i32 @llvm.nvvm.shfl.sync.bfly.i32
  %s1 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %x, i32 1, i32 31)
  %a1 = add i32 %x, %s1
  %s2 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a1, i32 2, i32 31)
  %a2 = add i32 %a1, %s2
  %s4 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a2, i32 4, i32 31)
  %a4 = add i32 %a2, %s4
  %s8 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a4, i32 8, i32 31)
  %a8 = add i32 %a4, %s8
  %s16 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a8, i32 16, i32 31)
  %sum = add i32 %a8, %s16
  ret i32 %sum
}

define i32 @full_warp_add_sunk(i32 %x, i1 %take) {
; SM80-LABEL: @full_warp_add_sunk(
; SM80:       [[SUM:%.*]] = call i32 @llvm.nvvm.redux.sync.add(i32 %x, i32 -1)
; SM80-NEXT:  br i1 %take, label %use, label %exit
; SM80-NOT:   @llvm.nvvm.shfl.sync.bfly.i32
; SM80:       use:
; SM80-NEXT:  ret i32 [[SUM]]
; SM80:       exit:
; SM80-NEXT:  ret i32 0
  %s1 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %x, i32 1, i32 31)
  %a1 = add i32 %x, %s1
  %s2 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a1, i32 2, i32 31)
  %a2 = add i32 %a1, %s2
  %s4 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a2, i32 4, i32 31)
  %a4 = add i32 %a2, %s4
  %s8 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a4, i32 8, i32 31)
  %a8 = add i32 %a4, %s8
  %s16 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a8, i32 16, i32 31)
  br i1 %take, label %use, label %exit
use:
  %sum = add i32 %a8, %s16
  ret i32 %sum
exit:
  ret i32 0
}

define i32 @partial_mask(i32 %x) {
; SM80-LABEL: @partial_mask(
; SM80-NOT:   @llvm.nvvm.redux.sync.add
; SM80:       ret i32
  %s1 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %x, i32 1, i32 31)
  %a1 = add i32 %x, %s1
  %s2 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a1, i32 2, i32 31)
  %a2 = add i32 %a1, %s2
  %s4 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a2, i32 4, i32 31)
  %a4 = add i32 %a2, %s4
  %s8 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a4, i32 8, i32 31)
  %a8 = add i32 %a4, %s8
  %s16 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 65535, i32 %a8, i32 16, i32 31)
  %sum = add i32 %a8, %s16
  ret i32 %sum
}

define i32 @nowrap_add(i32 %x) {
; SM80-LABEL: @nowrap_add(
; SM80-NOT:   @llvm.nvvm.redux.sync.add
; SM80:       ret i32
  %s1 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %x, i32 1, i32 31)
  %a1 = add i32 %x, %s1
  %s2 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a1, i32 2, i32 31)
  %a2 = add i32 %a1, %s2
  %s4 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a2, i32 4, i32 31)
  %a4 = add i32 %a2, %s4
  %s8 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a4, i32 8, i32 31)
  %a8 = add i32 %a4, %s8
  %s16 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a8, i32 16, i32 31)
  %sum = add nsw i32 %a8, %s16
  ret i32 %sum
}

define i32 @partial_sum_extra_use(i32 %x) {
; SM80-LABEL: @partial_sum_extra_use(
; SM80-NOT:   @llvm.nvvm.redux.sync.add
; SM80:       ret i32
  %s1 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %x, i32 1, i32 31)
  %a1 = add i32 %x, %s1
  %s2 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a1, i32 2, i32 31)
  %a2 = add i32 %a1, %s2
  %s4 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a2, i32 4, i32 31)
  %a4 = add i32 %a2, %s4
  %s8 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a4, i32 8, i32 31)
  %a8 = add i32 %a4, %s8
  %s16 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a8, i32 16, i32 31)
  %sum = add i32 %a8, %s16
  %extra = xor i32 %a4, 1
  %ret = add i32 %sum, %extra
  ret i32 %ret
}
