// RUN: mlir-opt %s -convert-vector-to-llvm -convert-gpu-to-nvvm="has-redux=1" | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-nvvm="has-redux=1" | FileCheck %s
// Exercise both LLVM singleton packing and direct vector.broadcast/extract
// matching in the original corpus cases below. Do not pre-lower vectors in
// the second invocation.
gpu.module @singleton_wrapper {
  // CHECK-LABEL: llvm.func @singleton_vector_candidate(
  // CHECK-NOT: llvm.select
  // CHECK: nvvm.vote.sync ballot
  // CHECK-NOT: nvvm.shfl.sync idx
  // CHECK: llvm.return
  gpu.func @singleton_vector_candidate(%xs: vector<1xi32>, %out: !llvm.ptr) kernel {
    %min = arith.constant dense<-2147483648> : vector<1xi32>
    %zero = arith.constant dense<0> : vector<1xi32>
    %width = arith.constant 32 : i32
    %id = gpu.lane_id
    %lane = arith.index_castui %id : index to i32
    %lanes = vector.broadcast %lane : i32 to vector<1xi32>
    %nonmin = arith.cmpi sgt, %xs, %min : vector<1xi32>
    %candidates = arith.select %nonmin, %lanes, %zero : vector<1xi1>, vector<1xi32>
    %candidate = vector.extract %candidates[0] : i32 from vector<1xi32>
    %x = vector.extract %xs[0] : i32 from vector<1xi32>
    %max = gpu.subgroup_reduce maxsi %x uniform {} : (i32) -> (i32)
    %ismax = arith.cmpi eq, %x, %max : i32
    %mask = gpu.ballot %ismax : i32
    %winner = math.cttz %mask : i32
    %result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
    llvm.store %result, %out : i32, !llvm.ptr
    gpu.return
  }
}
gpu.module @cases {
// CHECK-LABEL: llvm.func @butterfly(
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @butterfly(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @signed_cast(
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @signed_cast(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_cast %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @lane_id(
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @lane_id(%x: i32, %y: i32, %out: !llvm.ptr) kernel {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.lane_id
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @multiwarp_y(
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @multiwarp_y(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 2, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @eq_sentinel(
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @eq_sentinel(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi eq, %x, %min : i32
%candidate = arith.select %p, %zero, %lane : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @ne_sentinel(
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @ne_sentinel(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi ne, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @commuted(
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @commuted(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %s0, %x : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %s1, %r0 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %s2, %r1 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %s3, %r2 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %s4, %r3 : i32
%eq = arith.cmpi eq, %r4, %x : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @redux(
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @redux(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%max = gpu.subgroup_reduce maxsi %x uniform {} : (i32) -> (i32)
%eq = arith.cmpi eq, %x, %max : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @bad_block16(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @bad_block16(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 16, 2, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @bad_block64(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @bad_block64(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @missing_block(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @missing_block(%x: i32, %y: i32, %out: !llvm.ptr) kernel {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @wrong_axis(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @wrong_axis(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id y
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @bad_lane(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @bad_lane(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%one = arith.constant 1 : i32
%bad_lane = arith.addi %lane, %one : i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %bad_lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @candidate_other(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @candidate_other(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %y, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @bad_sentinel(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @bad_sentinel(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -1 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @bad_predicate(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @bad_predicate(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sge, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @unsigned_cmp(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @unsigned_cmp(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi ugt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @tree_other(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @tree_other(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %y, %o0, %width : i32
%r0 = arith.maxsi %y, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @partial(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @partial(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%eq = arith.cmpi eq, %x, %r3 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @wrong_offset(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @wrong_offset(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 8 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @narrow_stage(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @narrow_stage(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %narrow : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @wrong_mode(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @wrong_mode(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle down %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @disconnected(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @disconnected(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %y, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @unsigned_max(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @unsigned_max(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxui %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxui %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxui %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxui %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxui %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @narrow_final(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @narrow_final(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %narrow : i32
llvm.store %result, %out : i32, !llvm.ptr
gpu.return
}

// CHECK-LABEL: llvm.func @valid_used(
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
gpu.func @valid_used(%x: i32, %y: i32, %out: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
%min = arith.constant -2147483648 : i32
%zero = arith.constant 0 : i32
%width = arith.constant 32 : i32
%narrow = arith.constant 16 : i32
%id = gpu.thread_id x
%lane = arith.index_castui %id : index to i32
%p = arith.cmpi sgt, %x, %min : i32
%candidate = arith.select %p, %lane, %zero : i32
%o0 = arith.constant 1 : i32
%s0, %v0 = gpu.shuffle xor %x, %o0, %width : i32
%r0 = arith.maxsi %x, %s0 : i32
%o1 = arith.constant 2 : i32
%s1, %v1 = gpu.shuffle xor %r0, %o1, %width : i32
%r1 = arith.maxsi %r0, %s1 : i32
%o2 = arith.constant 4 : i32
%s2, %v2 = gpu.shuffle xor %r1, %o2, %width : i32
%r2 = arith.maxsi %r1, %s2 : i32
%o3 = arith.constant 8 : i32
%s3, %v3 = gpu.shuffle xor %r2, %o3, %width : i32
%r3 = arith.maxsi %r2, %s3 : i32
%o4 = arith.constant 16 : i32
%s4, %v4 = gpu.shuffle xor %r3, %o4, %width : i32
%r4 = arith.maxsi %r3, %s4 : i32
%eq = arith.cmpi eq, %x, %r4 : i32
%mask = gpu.ballot %eq : i32
%winner = math.cttz %mask : i32
%result, %valid = gpu.shuffle idx %candidate, %winner, %width : i32
llvm.store %result, %out : i32, !llvm.ptr
llvm.store %valid, %out : i1, !llvm.ptr
gpu.return
}
}
// CHECK-LABEL: llvm.func @main_graph_async_dispatch_2_arg_compare_1x32xi32_dispatch_tensor_store(
// CHECK-COUNT-5: nvvm.shfl.sync bfly
// CHECK: nvvm.vote.sync ballot
// CHECK-NOT: nvvm.shfl.sync idx
// CHECK: llvm.return
// -----// IR Dump Before ConvertToNVVMPass: iree-convert-to-nvvm //----- //
gpu.module @actual_common {
  gpu.func @main_graph_async_dispatch_2_arg_compare_1x32xi32_dispatch_tensor_store(%0: memref<1x32xi32, 1>, %1: memref<1xi32, strided<[1], offset: ?>, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<-2147483648> : vector<1xi32>
    %cst_0 = arith.constant dense<0> : vector<1xi32>
    %thread_id_x = gpu.thread_id x upper_bound 32
    %assume_align = memref.assume_alignment %0, 64 : memref<1x32xi32, 1>
    %assume_align_1 = memref.assume_alignment %1, 64 : memref<1xi32, strided<[1], offset: ?>, 1>
    %2 = vector.load %assume_align[%c0, %thread_id_x] : memref<1x32xi32, 1>, vector<1xi32>
    %3 = arith.index_castui %thread_id_x : index to i32
    %4 = vector.broadcast %3 : i32 to vector<1xi32>
    %5 = arith.cmpi sgt, %2, %cst : vector<1xi32>
    %6 = arith.select %5, %4, %cst_0 : vector<1xi1>, vector<1xi32>
    %7 = vector.extract %2[0] : i32 from vector<1xi32>
    %8 = vector.extract %6[0] : i32 from vector<1xi32>
    %9 = arith.cmpi sgt, %7, %c-2147483648_i32 : i32
    %10 = arith.select %9, %8, %c0_i32 : i32
    %shuffleResult, %valid = gpu.shuffle xor %7, %c1_i32, %c32_i32 : i32
    %11 = arith.maxsi %7, %shuffleResult : i32
    %shuffleResult_2, %valid_3 = gpu.shuffle xor %11, %c2_i32, %c32_i32 : i32
    %12 = arith.maxsi %11, %shuffleResult_2 : i32
    %shuffleResult_4, %valid_5 = gpu.shuffle xor %12, %c4_i32, %c32_i32 : i32
    %13 = arith.maxsi %12, %shuffleResult_4 : i32
    %shuffleResult_6, %valid_7 = gpu.shuffle xor %13, %c8_i32, %c32_i32 : i32
    %14 = arith.maxsi %13, %shuffleResult_6 : i32
    %shuffleResult_8, %valid_9 = gpu.shuffle xor %14, %c16_i32, %c32_i32 : i32
    %15 = arith.maxsi %14, %shuffleResult_8 : i32
    %16 = arith.cmpi eq, %7, %15 : i32
    %17 = gpu.ballot %16 : i32
    %18 = math.cttz %17 : i32
    %shuffleResult_10, %valid_11 = gpu.shuffle idx %10, %18, %c32_i32 : i32
    %19 = vector.broadcast %shuffleResult_10 : i32 to vector<1xi32>
    %20 = arith.index_castui %thread_id_x : index to i32
    %21 = arith.cmpi eq, %20, %c0_i32 : i32
    cf.cond_br %21, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    vector.store %19, %assume_align_1[%c0] : memref<1xi32, strided<[1], offset: ?>, 1>, vector<1xi32>
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    gpu.return
  }
}
// CHECK-LABEL: llvm.func @do_not_fold_singleton_biased(
// CHECK-COUNT-5: nvvm.shfl.sync bfly
// CHECK: nvvm.vote.sync ballot
// CHECK: nvvm.shfl.sync idx
// CHECK: llvm.return
// -----// IR Dump Before ConvertToNVVMPass: iree-convert-to-nvvm //----- //
gpu.module @biased_common {
  gpu.func @do_not_fold_singleton_biased(%0: memref<1x32xi32, 1>, %1: memref<1xi32, strided<[1], offset: ?>, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<-2147483648> : vector<1xi32>
    %cst_0 = arith.constant dense<0> : vector<1xi32>
    %thread_id_x = gpu.thread_id x upper_bound 32
    %assume_align = memref.assume_alignment %0, 64 : memref<1x32xi32, 1>
    %assume_align_1 = memref.assume_alignment %1, 64 : memref<1xi32, strided<[1], offset: ?>, 1>
    %2 = vector.load %assume_align[%c0, %thread_id_x] : memref<1x32xi32, 1>, vector<1xi32>
    %3 = arith.index_castui %thread_id_x : index to i32
    %biased = arith.addi %3, %c1_i32 : i32
    %4 = vector.broadcast %biased : i32 to vector<1xi32>
    %5 = arith.cmpi sgt, %2, %cst : vector<1xi32>
    %6 = arith.select %5, %4, %cst_0 : vector<1xi1>, vector<1xi32>
    %7 = vector.extract %2[0] : i32 from vector<1xi32>
    %8 = vector.extract %6[0] : i32 from vector<1xi32>
    %9 = arith.cmpi sgt, %7, %c-2147483648_i32 : i32
    %10 = arith.select %9, %8, %c0_i32 : i32
    %shuffleResult, %valid = gpu.shuffle xor %7, %c1_i32, %c32_i32 : i32
    %11 = arith.maxsi %7, %shuffleResult : i32
    %shuffleResult_2, %valid_3 = gpu.shuffle xor %11, %c2_i32, %c32_i32 : i32
    %12 = arith.maxsi %11, %shuffleResult_2 : i32
    %shuffleResult_4, %valid_5 = gpu.shuffle xor %12, %c4_i32, %c32_i32 : i32
    %13 = arith.maxsi %12, %shuffleResult_4 : i32
    %shuffleResult_6, %valid_7 = gpu.shuffle xor %13, %c8_i32, %c32_i32 : i32
    %14 = arith.maxsi %13, %shuffleResult_6 : i32
    %shuffleResult_8, %valid_9 = gpu.shuffle xor %14, %c16_i32, %c32_i32 : i32
    %15 = arith.maxsi %14, %shuffleResult_8 : i32
    %16 = arith.cmpi eq, %7, %15 : i32
    %17 = gpu.ballot %16 : i32
    %18 = math.cttz %17 : i32
    %shuffleResult_10, %valid_11 = gpu.shuffle idx %10, %18, %c32_i32 : i32
    %19 = vector.broadcast %shuffleResult_10 : i32 to vector<1xi32>
    %20 = arith.index_castui %thread_id_x : index to i32
    %21 = arith.cmpi eq, %20, %c0_i32 : i32
    cf.cond_br %21, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    vector.store %19, %assume_align_1[%c0] : memref<1xi32, strided<[1], offset: ?>, 1>, vector<1xi32>
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    gpu.return
  }
}
