#include <stdio.h>
//const char STR_LENGTH = 13;
//__device__ const char *STR = "HELLO WORLD! ";

const char STR_LENGTH = 52;
__device__ const char *STR = "HELLO WORLD! HELLO WORLD! HELLO WORLD! HELLO WORLD! ";

__global__ void hello() {
  if(threadIdx.x < 52) {
    printf("%c", STR[threadIdx.x]);
  }
}

__global__ void simple(int a) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  printf("hi %d %d %d\n", a, bid, tid);
}

int main(int argc, char** argv) {
  /*simple<<<2, 4>>>(3);
  cudaDeviceSynchronize();
  */
  int device = atoi(argv[1]);
  cudaSetDevice(device);
  
  hello<<<1, 5000>>>();
  printf("\n");
  cudaDeviceSynchronize();		

  return 0;
}
