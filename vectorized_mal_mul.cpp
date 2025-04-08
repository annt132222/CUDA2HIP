#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>
#include <tuple>
#include <string>

template<typename T>
__host__ void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T sum = 0;
      for (int k = 0; k < K; k++) {
        sum += h_a[i * K + k] * h_b[k * N + j];
      }
      assert(h_c[i * N + j] == sum);
    }
  }
  printf("Correct!\n");
}

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
  assert(numThreadsBlocktile == blockDim.x);
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    reinterpret_cast<float4 *>(&As[innerRowA * BK + innerColA * 4])[0] =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      tmp.x = threadResults[resIdxM * TN + resIdxN];
      tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
      tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
      tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

template<typename T>
__host__ void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N, size_t K) {
  size_t a_bytes = sizeof(T) * M * K;
  size_t b_bytes = sizeof(T) * K * N;
  hipError_t err = hipMemcpy(d_a, h_a, a_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy h_a to d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMemcpy(d_b, h_b, b_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy h_b to d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block((BM * BN) / (TM * TN), 1, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  gemm_kernel<T, BM, BN, BK, TM, TN><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  (void)hipDeviceSynchronize();
}

template<typename T>
__host__ void copyFromDeviceToHost(T* d_c, T* h_c, size_t M, size_t N) {
  size_t c_bytes = sizeof(T) * M * N;
  hipError_t err = hipMemcpy(h_c, d_c, c_bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy from d_c to h_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void deallocateMemory(T* d_a, T* d_b, T* d_c) {
  hipError_t err = hipFree(d_a);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipFree(d_b);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipFree(d_c);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void cleanUpDevice() {
  hipError_t err = hipDeviceReset();
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to clean up device (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024, N = 1024, K = 1024;
  for (int i = 1; i < argc; i++){
    std::string option(argv[i]);
    std::string value(argv[i+1]);
    i++;
    if (option.compare("-m") == 0) {
      M = std::stoi(value);
    } else if (option.compare("-n") == 0) {
      N = std::stoi(value);
    } else if (option.compare("-k") == 0) {
      K = std::stoi(value);
    }
  }
  return {M, N, K};
}

int main(int argc, char *argv[]) {
  std::tuple<int, int, int> parsedCmdLineArgsTuple = parseCmdLineArgs(argc, argv);
  int M = std::get<0>(parsedCmdLineArgsTuple);
  int N = std::get<1>(parsedCmdLineArgsTuple);
  int K = std::get<2>(parsedCmdLineArgsTuple);
  float* h_a = (float*)malloc(M * K * sizeof(float));
  float* h_b = (float*)malloc(K * N * sizeof(float));
  float* h_c = (float*)malloc(M * N * sizeof(float));
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      h_a[i * K + j] = rand() % 10;
    }
  }
  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
      h_b[i * N + j] = rand() % 10;
    }
  }
  float *d_a, *d_b, *d_c;
  hipError_t err = hipMalloc((void **)&d_a, M * K * sizeof(float));
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMalloc((void **)&d_b, K * N * sizeof(float));
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMalloc((void **)&d_c, M * N * sizeof(float));
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);
  executeKernel<float, 128, 128, 8, 8, 8>(d_a, d_b, d_c, M, N, K);
  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  verifyResult<float>(h_a, h_b, h_c, M, N, K);
  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();
  return 0;
}