#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>
#include <tuple>
#include <string>

// Hàm kiểm tra kết quả tính toán
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

// Kernel GEMM sử dụng double buffering cho shared memory
template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
  // Khai báo hai vùng shared memory để thực hiện double buffering
  __shared__ float As[2][BM * BK];
  __shared__ float Bs[2][BK * BN];

  // Tính số lượng thread tham gia trong block-tile:
  const unsigned totalTileElems = BM * BK;
  // Số thread được dùng để tải tile: ta giả sử mỗi thread sẽ tải một số phần tử theo bước nhảy
  // Các thread được sắp xếp theo một chiều đơn giản.
  const int threadTileIdx = threadIdx.x;
  
  // Xác định vị trí của thread trong tile kết quả: 
  // Giả sử block tile kết quả có kích thước BM x BN và được chia thành các block con TM x TN
  const int threadsPerBlockTileRow = BN / TN;
  const int threadRow = threadTileIdx / threadsPerBlockTileRow;
  const int threadCol = threadTileIdx % threadsPerBlockTileRow;

  // Điều chỉnh con trỏ tương ứng với block tile hiện hành
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Tích lũy kết quả trên registers cho mỗi thread
  float threadResults[TM * TN] = {0.0f};

  // Số lượng thread dùng cho tải dữ liệu (có thể dùng lại threadTileIdx)
  const int numThreads = blockDim.x;
  
  // Buffer hiện hành, bắt đầu load tile đầu tiên vào buffer 0
  int bufIdx = 0;

  // Tải tile đầu tiên của A vào shared memory (buffer bufIdx)
  for (uint offset = threadTileIdx; offset < BM * BK; offset += numThreads) {
    uint row = offset / BK;
    uint col = offset % BK;
    // Chỉ tải các phần tử trong phạm vi M nếu vượt ra ngoài thì vẫn load giá trị không ảnh hưởng
    As[bufIdx][row * BK + col] = A[row * K + col];
  }
  // Tải tile đầu tiên của B vào shared memory (buffer bufIdx)
  for (uint offset = threadTileIdx; offset < BK * BN; offset += numThreads) {
    uint row = offset / BN;
    uint col = offset % BN;
    Bs[bufIdx][row * BN + col] = B[row * N + col];
  }
  __syncthreads();

  // Vòng lặp qua chiều K theo bước BK
  // bkIdx đánh dấu vị trí bắt đầu của tile hiện hành trong chiều K
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    int nextBuf = 1 - bufIdx;
    
    // Nếu chưa đến tile cuối, tiến hành tải tile kế vào buffer nextBuf
    if (bkIdx + BK < K) {
      // Di chuyển con trỏ đến tile kế của A và B
      A += BK;        // Đi tới tile kế trong ma trận A (dọc theo chiều K)
      B += BK * N;    // Ma trận B: chuyển sang tile kế (theo chiều K, với mỗi tile có BN phần tử theo N)
      
      // Tải dữ liệu cho tile kế của A vào buffer nextBuf
      for (uint offset = threadTileIdx; offset < BM * BK; offset += numThreads) {
        uint row = offset / BK;
        uint col = offset % BK;
        As[nextBuf][row * BK + col] = A[row * K + col];
      }
      // Tải dữ liệu cho tile kế của B vào buffer nextBuf
      for (uint offset = threadTileIdx; offset < BK * BN; offset += numThreads) {
        uint row = offset / BN;
        uint col = offset % BN;
        Bs[nextBuf][row * BN + col] = B[row * N + col];
      }
    }
    
    __syncthreads();
    
    // Tính toán GEMM trên tile đã load sẵn trong buffer bufIdx
    // Lặp qua các phần tử theo chiều BK của tile
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float regA[TM];
      float regB[TN];
      // Tải dữ liệu từ shared memory cho A theo block con kích thước TM
      for (uint i = 0; i < TM; ++i) {
        regA[i] = As[bufIdx][(threadRow * TM + i) * BK + dotIdx];
      }
      // Tải dữ liệu từ shared memory cho B theo block con kích thước TN
      for (uint j = 0; j < TN; ++j) {
        regB[j] = Bs[bufIdx][dotIdx * BN + threadCol * TN + j];
      }
      // Tích lũy kết quả cho block con (TM x TN)
      #pragma unroll
      for (uint i = 0; i < TM; ++i) {
        #pragma unroll
        for (uint j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regA[i] * regB[j];
        }
      }
    }
    
    __syncthreads();
    // Hoán đổi buffer để tile kế được sử dụng cho lần tính toán tiếp theo
    bufIdx = nextBuf;
  }
  
  // Ghi kết quả từ registers về global memory
  for (uint i = 0; i < TM; ++i) {
    for (uint j = 0; j < TN; ++j) {
      C[(threadRow * TM + i) * N + threadCol * TN + j] = threadResults[i * TN + j];
    }
  }
}

// Hàm copy dữ liệu từ host sang device cho A và B
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

// Hàm khởi chạy kernel với cấu hình block và grid phù hợp
template<typename T, const unsigned BM, const unsigned BN, const unsigned BK, const unsigned TM, const unsigned TN>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  // Số thread mỗi block = (BM*BN) / (TM*TN)
  dim3 block((BM * BN) / (TM * TN), 1, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  hipLaunchKernelGGL(gemm_kernel<T, BM, BN, BK, TM, TN>, grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
  hipDeviceSynchronize();
}

// Hàm copy dữ liệu từ device sang host cho C
template<typename T>
__host__ void copyFromDeviceToHost(T* d_c, T* h_c, size_t M, size_t N) {
  size_t c_bytes = sizeof(T) * M * N;
  hipError_t err = hipMemcpy(h_c, d_c, c_bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy from d_c to h_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Hàm giải phóng bộ nhớ thiết bị
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

// Hàm reset device
__host__ void cleanUpDevice() {
  hipError_t err = hipDeviceReset();
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to clean up device (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Hàm parse tham số dòng lệnh: -m, -n, -k
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
  // Parse các tham số kích thước ma trận M, N, K
  auto [M, N, K] = parseCmdLineArgs(argc, argv);

  // Cấp phát bộ nhớ host cho các ma trận A, B, C
  float* h_a = (float*)malloc(sizeof(float) * M * K);
  float* h_b = (float*)malloc(sizeof(float) * K * N);
  float* h_c = (float*)malloc(sizeof(float) * M * N);

  // Khởi tạo ma trận A, B với các giá trị ngẫu nhiên
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

  // Cấp phát bộ nhớ device cho ma trận A, B, C
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

  // Copy dữ liệu từ host sang device cho A và B
  copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);

  // Dùng hipEvent để đo thời gian thực hiện kernel
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  // Thực hiện kernel GEMM với các tham số đã chọn: BM=128, BN=128, BK=32, TM=8, TN=8
  executeKernel<float, 128, 128, 32, 8, 8>(d_a, d_b, d_c, M, N, K);

  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  float time_ms;
  hipEventElapsedTime(&time_ms, start, stop);
  std::cout << "Time taken for GEMM: " << time_ms << " ms, ";
  std::cout << "Performance: " << 2LL * M * N * K / (time_ms * 1e-3 * 1e9) << " GFLOPS" << std::endl;
  hipEventDestroy(start);
  hipEventDestroy(stop);

  // Copy kết quả từ device sang host và kiểm tra tính đúng đắn
  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  verifyResult<float>(h_a, h_b, h_c, M, N, K);

  // Giải phóng bộ nhớ device
  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();

  // Giải phóng bộ nhớ host
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
