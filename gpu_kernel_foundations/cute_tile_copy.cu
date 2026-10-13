// Original finite teaching program for the Chapter 32 worker-copy map.
// API basis: archived CUTLASS 4.2.1 CuTE Tensors, partitioning and copy.
// CUDA compilation, device execution and sanitizer checks are separate gates.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

__global__ void copy_selected_tile(const float* d_ptr, float* r_ptr) {
  using namespace cute;
  auto tile = make_layout(make_shape(2, 2), make_stride(4, 1));
  auto compact = make_layout(make_shape(2, 2), make_stride(2, 1));
  auto T = make_tensor(make_gmem_ptr(d_ptr + 8), tile);
  auto R = make_tensor(make_gmem_ptr(r_ptr), compact);
  auto workers = make_layout(make_shape(Int<2>{}, Int<2>{}),
                             make_stride(Int<2>{}, Int<1>{}));
  int worker = threadIdx.x;  // exactly four threads
  auto src = local_partition(T, workers, worker);
  auto dst = local_partition(R, workers, worker);
  copy(src, dst);
}

static std::array<float, 16> full_product() {
  const int A[4][4] = {{1, 0, 2, 0}, {0, 1, 0, 2},
                       {1, 2, 3, 4}, {2, 1, 0, 1}};
  const int B[4][4] = {{1, 2, 0, 0}, {3, 4, 0, 0},
                       {5, 6, 1, 0}, {7, 8, 0, 1}};
  std::array<float, 16> D{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k)
        D[4 * i + j] += A[i][k] * B[k][j];
  return D;
}

int main() {
  float *device_source = nullptr, *device_result = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&device_source), 16 * sizeof(float)));
  check(cudaMalloc(reinterpret_cast<void**>(&device_result), 6 * sizeof(float)));
  bool passed = true;
  for (int trial = 0; trial < 17; ++trial) {
    std::array<float, 16> source = full_product();
    if (trial > 0) {
      // All 16 impulse inputs expose unintended inclusion of gap elements.
      source.fill(0.0f);
      source[trial - 1] = static_cast<float>(trial);
    }
    std::array<float, 6> result;
    result.fill(-999.0f);
    check(cudaMemcpy(device_source, source.data(), 16 * sizeof(float),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_result, result.data(), 6 * sizeof(float),
                     cudaMemcpyHostToDevice));
    copy_selected_tile<<<1, 4>>>(device_source, device_result + 1);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(result.data(), device_result, 6 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    std::array<float, 16> source_after;
    check(cudaMemcpy(source_after.data(), device_source, 16 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    const std::array<float, 6> expected = {
        -999.0f, source[8], source[9], source[12], source[13], -999.0f};
    if (result != expected || source_after != source) {
      std::fprintf(stderr, "Worker-copy mapping failed on trial %d\n", trial);
      passed = false;
    }
  }
  check(cudaFree(device_result));
  check(cudaFree(device_source));
  if (!passed) return EXIT_FAILURE;
  std::puts("17 worker-copy cases passed; no timing measured.");
  return EXIT_SUCCESS;
}
