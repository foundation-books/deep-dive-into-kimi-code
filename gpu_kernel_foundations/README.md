# Chapter 32: From Basic CUDA to Kimi Kernels

The dependency-free Python reference records 32 lane-local integer sums,
five simultaneous XOR exchanges, the numerical matrix tile and worker-copy
addresses, the scalar XOR storage map, and the two-slot schedule. It checks ownership and exact arithmetic;
it does not simulate CUDA execution, memory transactions, or performance.

Run from the companion-code root:

```sh
python3 -B _maintenance/chapter_test_runner.py \
  gpu_kernel_foundations/test_gpu_kernel_foundations_reference.py
```

The tests cover all lengths 0–128 with all-one and nonuniform signed weights,
all 128 individual operand pairings, disjoint reduction groups, and invalid
domains. The manuscript fixtures yield 8256 (length 128), 5050 (100), 6 (3),
and −133 (128 with `w[i] = (i % 7) - 3`). Lane 0's full-length trace is
196, 396, 808, 1680, 3616, 8256.
Additional checks cover both matrix slices, the four CuTE-example worker rows,
relative offsets versus contiguous span, the compact-layout mismatch, all
1,024 scalar swizzle locations, row/column banks, the wrong-layout consumer,
and six-slice schedules with both load and compute as the slower resource.
The three-slice fixture is `(S0,0): [0,2) -> [2,5)`,
`(S1,0): [2,4) -> [5,8)`, `(S0,1): [5,7) -> [8,11)`.
Those generations are teaching identities rather than hardware barrier bits.

## Optional CUDA teaching program

`one_warp_dot.cu` contains a complete host driver and the kernel fragment used
in the chapter. Launch exactly one block with 32 threads and supply disjoint
FP32 buffers with a logical length from 0 through 128. Each lane accumulates
coordinates `lane + 32*r` for `r = 0, 1, 2, 3`, guarding both operand loads.
Every lane reaches all five shuffles with `0xffffffffu`; only lane 0 writes.
A short input changes data validity, not the collective's membership. This
program does not handle physically partial warps or arbitrary sparse masks.

The CUDA program checks all 258 length/weight cases against independent host
integer sums, uses tight input allocations, checks output sentinels, checks
allocation/copy/launch/completion errors, and releases device allocations.
The fixtures' small integer products and intermediate sums are exactly
representable in FP32. Arbitrary floating-point data would need a declared
tolerance and would not generally match a sequential reduction bit for bit.

On a machine with a supported NVIDIA GPU and CUDA Toolkit 12.9.1, from this
directory, build for that GPU's architecture and run, for example on SM80:

```sh
nvcc -std=c++17 -O2 -arch=sm_80 one_warp_dot.cu -o /tmp/kimi-one-warp-dot
/tmp/kimi-one-warp-dot
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/kimi-one-warp-dot
compute-sanitizer --tool synccheck --error-exitcode 1 /tmp/kimi-one-warp-dot
```

Use the installed toolkit's supported architecture option for a different
device. These optional commands are excluded from `make smoke` and introduce
no dependency into the Python companion.

**CUDA validation status:** source provided; CUDA compilation, GPU execution,
and Compute Sanitizer checks have not run. No CUDA compiler or NVIDIA runtime
was available in the implementation environment. A passing Python test does
not complete this separate validation stage. Record toolkit/compiler versions,
GPU model, target architecture, commands, and outcomes when it is performed.
There is no timing claim or production K3 kernel implementation here.

## Optional CuTE worker-partition program

`cute_tile_copy.cu` uses the C++ API documented in CUTLASS 4.2.1,
[CuTE Tensors: partitioning](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/cute/03_tensor.html#partitioning-a-tensor).
The documented `local_partition(Tensor, Layout, Idx)` inverts the static
worker layout to select each thread's values. `copy(src, dst)` performs the
data operation. Both worker-layout dimensions are 2, divide the tensor
dimensions exactly, and map `(u,v)` to `2*u+v`.

Launch exactly one block of four threads. The source is a valid 16-element
row-major FP32 allocation; its selected tile is based at element 8 with
shape `(2,2)` and stride `(4,1)`. The disjoint destination has four elements,
shape `(2,2)` and stride `(2,1)`. Worker 2 selects `(1,0)` and copies source
offset 12 to destination offset 2. The host driver initializes the source
from an independent matrix product, then checks all 16 impulse inputs,
two destination sentinels, the untouched source, and CUDA operation errors.
There is no matrix instruction, partial-warp collective, or async transfer.

The source-checkout registry has no CUTLASS entry. The source basis here is
the recorded 4.2.1 documentation release; no local research checkout or
immutable Git commit is asserted. On a CUDA machine with matching CUTLASS
4.2.1 headers already available, set `CUTLASS_421_INCLUDE` to that release's
include directory. With CUDA Toolkit 12.9.1 and an SM80 GPU, for example:

```sh
nvcc -std=c++17 -O2 -arch=sm_80 -I "$CUTLASS_421_INCLUDE" \
  cute_tile_copy.cu -o /tmp/kimi-cute-tile-copy
/tmp/kimi-cute-tile-copy
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/kimi-cute-tile-copy
compute-sanitizer --tool synccheck --error-exitcode 1 /tmp/kimi-cute-tile-copy
```

Use the installed toolkit's supported target for other devices. These commands
are optional and excluded from the dependency-free smoke suite.
**CuTE validation status:** API/documentation inspection and independent CPU
coordinate checks are complete; CUDA/CuTE compilation, the 17 device cases,
and sanitizer checks are unrun because `nvcc` and a CUDA runtime are unavailable.
Record the header revision, compiler/toolkit versions, target, device, commands
and outcomes when those gates run. CPU arithmetic cannot certify their outcome.
