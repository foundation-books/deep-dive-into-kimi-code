// Optional CUDA 12.9.1 teaching program, separate from the Python smoke suite.
// Contract: exactly one block of 32 threads; 0 <= n <= 128; disjoint buffers.
// All lanes participate in every shuffle, including lanes without valid data.
// These bounded integer-valued FP32 fixtures have exact products and sums.
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

static void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t elements) {
    check(cudaMalloc(reinterpret_cast<void**>(&data), elements * sizeof(float)),
          "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  float* data = nullptr;
};

__global__ void one_warp_dot(const float* x, const float* w, float* out, int n) {
  int lane = threadIdx.x;
  float p = 0.0f;
  for (int r = 0; r < 4; ++r) {
    int i = lane + 32 * r;
    if (i < n) p += x[i] * w[i];
  }
  for (int offset = 1; offset < 32; offset *= 2)
    p += __shfl_xor_sync(0xffffffffu, p, offset);
  if (lane == 0) *out = p;
}

static void run_case(int n, bool signed_weights) {
  if (n < 0 || n > 128) throw std::runtime_error("unsupported length");
  std::array<float, 128> x{}, w{};
  int expected = 0;
  for (int i = 0; i < n; ++i) {
    int weight = signed_weights ? (i % 7) - 3 : 1;
    x[i] = static_cast<float>(i + 1);
    w[i] = static_cast<float>(weight);
    expected += (i + 1) * weight;
  }
  // Keep tail allocations tight so compute-sanitizer can detect bad loads.
  // The length-zero case has one allocated, unread element per operand.
  size_t count = static_cast<size_t>(n == 0 ? 1 : n);
  DeviceBuffer dx(count), dw(count), output(3);
  check(cudaMemcpy(dx.data, x.data(), count * sizeof(float), cudaMemcpyHostToDevice),
        "copy x");
  check(cudaMemcpy(dw.data, w.data(), count * sizeof(float), cudaMemcpyHostToDevice),
        "copy w");
  std::array<float, 3> result{-12345.0f, -12345.0f, -12345.0f};
  check(cudaMemcpy(output.data, result.data(), sizeof(result), cudaMemcpyHostToDevice),
        "initialize output sentinels");
  one_warp_dot<<<1, 32>>>(dx.data, dw.data, output.data + 1, n);
  check(cudaGetLastError(), "launch one_warp_dot");
  check(cudaDeviceSynchronize(), "complete one_warp_dot");
  check(cudaMemcpy(result.data(), output.data, sizeof(result), cudaMemcpyDeviceToHost),
        "read result");
  if (result[0] != -12345.0f || result[2] != -12345.0f ||
      result[1] != static_cast<float>(expected)) {
    throw std::runtime_error("wrong result or overwritten output sentinel at n=" +
                             std::to_string(n));
  }
  if ((n == 128 || n == 100 || n == 3) && !signed_weights)
    std::printf("n=%d, all-one weights: %.0f\n", n, result[1]);
  if (n == 128 && signed_weights)
    std::printf("n=128, signed weights: %.0f\n", result[1]);
}

int main() {
  try {
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) throw std::runtime_error("a CUDA device is required");
    check(cudaSetDevice(0), "cudaSetDevice");
    for (int n = 0; n <= 128; ++n) {
      run_case(n, false);
      run_case(n, true);
    }
    std::puts("PASS: 258 length/weight cases; output sentinels unchanged.");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
