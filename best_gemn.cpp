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
  __shared__ float As[2][BM * BK];
  __shared__ float Bs[2][BK * BN];
  const unsigned totalTileElems = BM * BK;
  const int threadTileIdx = threadIdx.x;
  const int threadsPerBlockTileRow = BN / TN;
  const int threadRow = threadTileIdx / threadsPerBlockTileRow;
  const int threadCol = threadTileIdx % threadsPerBlockTileRow;
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;
  float threadResults[TM * TN] = {0.0f};
  const int numThreads = blockDim.x;
  int bufIdx = 0;
  for (uint offset = threadTileIdx; offset < BM * BK; offset += numThreads) {
    uint row = offset / BK;
    uint col = offset % BK;
    As[bufIdx][row * BK + col] = A[row * K + col];
  }
  for (uint offset = threadTileIdx; offset < BK * BN; offset += numThreads) {
    uint row = offset / BN;
    uint col = offset % BN;
    Bs[bufIdx][row * BN + col] = B[row * N + col];
  }
  __syncthreads();
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    int nextBuf = 1 - bufIdx;
    if (bkIdx + BK < K) {
      A += BK;        
      B += BK * N;    
      for (uint offset = threadTileIdx; offset < BM * BK; offset += numThreads) {
        uint row = offset / BK;
        uint col = offset % BK;
        As[nextBuf][row * BK + col] = A[row * K + col];
      }
      for (uint offset = threadTileIdx; offset < BK * BN; offset += numThreads) {
        uint row = offset / BN;
        uint col = offset % BN;
        Bs[nextBuf][row * BN + col] = B[row * N + col];
      }
    }
    
    __syncthreads();
    
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float regA[TM];
      float regB[TN];
      for (uint i = 0; i < TM; ++i) {
        regA[i] = As[bufIdx][(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint j = 0; j < TN; ++j) {
        regB[j] = Bs[bufIdx][dotIdx * BN + threadCol * TN + j];
      }
      #pragma unroll
      for (uint i = 0; i < TM; ++i) {
        #pragma unroll
        for (uint j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regA[i] * regB[j];
        }
      }
    }
    
    __syncthreads();
    bufIdx = nextBuf;
  }
  for (uint i = 0; i < TM; ++i) {
    for (uint j = 0; j < TN; ++j) {
      C[(threadRow * TM + i) * N + threadCol * TN + j] = threadResults[i * TN + j];
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

template<typename T, const unsigned BM, const unsigned BN, const unsigned BK, const unsigned TM, const unsigned TN>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block((BM * BN) / (TM * TN), 1, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  hipLaunchKernelGGL(gemm_kernel<T, BM, BN, BK, TM, TN>, grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
  hipDeviceSynchronize();
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

__host__ std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024, N = 1024, K = 1024;
  for (int i = 1; i < argc; i++){
    std::string option(argv[i]);
    if(i+1 < argc) {
      std::string value(argv[i+1]);
      i++;
      if (option == "-m") {
        M = std::stoi(value);
      } else if (option == "-n") {
        N = std::stoi(value);
      } else if (option == "-k") {
        K = std::stoi(value);
      }
    }
  }
  return std::make_tuple(M, N, K);
}

int main(int argc, char *argv[]) {
  auto [M, N, K] = parseCmdLineArgs(argc, argv);

  float* h_a = (float*)malloc(sizeof(float) * M * K);
  float* h_b = (float*)malloc(sizeof(float) * K * N);
  float* h_c = (float*)malloc(sizeof(float) * M * N);

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
  hipError_t err = hipMalloc((void **)&d_a, sizeof(float) * M * K);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMalloc((void **)&d_b, sizeof(float) * K * N);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMalloc((void **)&d_c, sizeof(float) * M * N);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  executeKernel<float, 128, 128, 32, 8, 8>(d_a, d_b, d_c, M, N, K);

  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  float time_ms;
  hipEventElapsedTime(&time_ms, start, stop);
  std::cout << "Time taken for GEMM: " << time_ms << " ms, ";
  std::cout << "Performance: " << 2LL * M * N * K / (time_ms * 1e-3 * 1e9) << " GFLOPS" << std::endl;
  hipEventDestroy(start);
  hipEventDestroy(stop);

  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  verifyResult<float>(h_a, h_b, h_c, M, N, K);

  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
