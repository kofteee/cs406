#include <iostream>
#include <omp.h>
using namespace std;

#define SZ ((1 << 25) + 10)

__global__ void add(int* d_a, int* d_b, int* d_c) {
  int tid = ((blockDim.x * blockDim.y * blockDim.z) * blockIdx.x) + threadIdx.x;
  if(tid < SZ) {
    d_c[tid] = d_a[tid] + d_b[tid];
  }
}


int main(void) {
  int size = sizeof(int) * SZ;
  int *a = new int[SZ];
  int *b = new int[SZ];
  int *c = new int[SZ];

  int *d_a, *d_b, *d_c;

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Setup input values
  for(int  i = 0; i < SZ; i++) {
    a[i] = b[i] = 7;
  }

  double start = omp_get_wtime();
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU
  add<<<(SZ + 1023) / 1024, 1024>>>(d_a, d_b, d_c);
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  double end = omp_get_wtime();
  cout << "GPU: " << end - start << endl;

  long long sum = 0;
  for(int  i = 0; i < SZ; i++) {
    sum += c[i];
  }
  cout << sum << " " << (7 + 7)  * SZ << endl;

  start = omp_get_wtime();
#pragma omp parallel for
  for(int i = 0; i < SZ; i++) {
    c[i] = a[i] + b[i];
  }
  end = omp_get_wtime();
  cout << "CPU: " << end - start << endl;

  
  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
