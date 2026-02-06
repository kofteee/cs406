#include <iostream>
#include "common.h"

using namespace std;

struct f8 {
  float a, b, c, d, e, f, g, h, dummy;
};		

__global__ void shared_user(int st) {
  __shared__ f8 sa[1024];
  
  for(int i = 0; i < 20000; i++) {
    sa[threadIdx.x + 2].a = sa[threadIdx.x % st].a + i;
  }  
}

int main(int argc, char** argv) {
  int devId = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devId);
  cout << "Device : " << prop.name << endl;

  int st = atoi(argv[1]);
  for(int i = 0; i < 1; i++) {
    //measure the time with nvprof
    shared_user<<<1024, 1024>>>(st);
    cudaDeviceSynchronize(); //what happens when you don't use this. check nvprof output
  }
  return 0;
} /* end main */
