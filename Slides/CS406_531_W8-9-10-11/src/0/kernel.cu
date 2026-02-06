#include <stdio.h>
#include <iostream>
using namespace std;
__global__ void vecadd(int* v1, int* v2, int* v3, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < N) {
    v3[i] = v2[i] * v1[i];
  }
}

#define N 1024
int main(int argc, char** argv) {
  int device = atoi(argv[1]);
  cudaSetDevice(device);

  int* v1 = new int[N];
  int* v2 = new int[N];
  int* v3 = new int[N];
  for(int i = 0; i < N; i++) v1[i] = v2[i] = 1;

  int* v1_d;
  cudaMalloc((void **)&v1_d, N * sizeof(int));
  cudaMemcpy(v1_d, v1, N * sizeof(int), cudaMemcpyHostToDevice);

  int* v2_d;
  cudaMalloc((void **)&v2_d, N * sizeof(int));
  cudaMemcpy(v2_d, v2, N * sizeof(int), cudaMemcpyHostToDevice);

  int* v3_d;
  cudaMalloc((void **)&v3_d, N * sizeof(int));

  vecadd<<< (N+31)/32, 32>>>(v1_d, v2_d, v3_d, N);
  cudaDeviceSynchronize();

  cudaMemcpy(v3, v3_d,  N * sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < N; i++) {
    if(v3[i] != 2) {
          cout << "you did bad " << endl;
    }
  }
  
  return 0;
}
