#include <iostream>
using namespace std;

__constant__ float f[] = { 0.f, 1.f, 2.f, 3.f };

#define BLOCK_SIZE 256
// Adjacent Difference application:
// compute result[i] = input[i] – input[i-1]
__global__ void adj_diff_naive(int *result, int *input) { // compute this thread’s global index
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
 if(i > 0) {
   // each thread loads two elements from device memory
   int x_i = input[i];
   int x_i_minus_one = input[i-1];
   result[i] = f[i%4] * x_i - x_i_minus_one;
 }
}

__global__ void adj_diff(int *result, int *input) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ int s_data[BLOCK_SIZE]; // shared, 1 elt / thread
  // each thread reads 1 device memory elt, stores it in s_data
  s_data[threadIdx.x] = input[i];
  // avoid race condition: ensure all loads are complete
  __syncthreads();
  if(threadIdx.x > 0) {
    result[i] = s_data[threadIdx.x] - s_data[threadIdx.x - 1];
  } else if(i > 0) {
    // I am thread 0 in this block: handle thread block boundary
    result[i] = s_data[threadIdx.x] - input[i-1];
  }
}

#define N (1 << 22)
int main() {
  cudaSetDevice(0);
  int* h_arr = new int[N];
  for(int i = 0; i < N; i++) {
    h_arr[i] = i;
  }
  int* d_arr;
  cudaMalloc((void **)&d_arr, sizeof(int) * N);
  cudaMemcpy(d_arr, h_arr, sizeof(int) * N, cudaMemcpyHostToDevice);

  int* d_out;
  cudaMalloc((void **)&d_out, sizeof(int) * N);

  cout << "start naive" << endl;
  for(int i = 0; i < 5; i++) {
    adj_diff_naive<<<N / BLOCK_SIZE, BLOCK_SIZE >>>(d_arr, d_out);
    cudaDeviceSynchronize();
  }

  cout << "start shared" << endl;
  for(int i = 0; i < 5; i++) {
    adj_diff<<<N / BLOCK_SIZE, BLOCK_SIZE >>>(d_arr, d_out);
    cudaDeviceSynchronize();
  }
  return 0;
}
