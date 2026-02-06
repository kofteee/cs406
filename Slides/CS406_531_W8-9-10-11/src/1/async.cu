//see https://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/

/*
All device operations (kernels and data transfers) in CUDA run in a
stream. When no stream is specified, the default stream (also called
the “null stream”) is used. The default stream is different from other
streams because it is a synchronizing stream with respect to
operations on the device: no operation in the default stream will
begin until all previously issued operations in any stream on the
device have completed, and an operation in the default stream must
complete before any other operation (in any stream on the device) will
begin.
*/

#include <stdio.h>

#define DEBUG
//Convenience function for checking CUDA runtime API results
//can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    if(result != cudaSuccess) {
      exit(1);
    }
  }
#endif
  return result;
}

__global__ void kernel(float *a, int offset) {
  int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  float x = (float)i;
  float s = sinf(x); 
  float c = cosf(x);
  a[i] = a[i] + sqrtf((s * s) + (c * c));
}

float maxError(float *a, int n) {
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i] - 1.0f);
    if (error > maxE) {
      maxE = error;
    }
  }
  return maxE;
}

int main(int argc, char **argv) {
  const int blockSize = 256, nStreams = 4;
  const int n = 4096 * blockSize * nStreams;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 1) {
    devId = atoi(argv[1]);
  }

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  printf("Device overlap: %d\n", prop.deviceOverlap);	
  checkCuda( cudaSetDevice(devId) );
  
  // allocate pinned host memory and device memory
  float *a, *d_a;
  checkCuda( cudaMallocHost((void**)&a, bytes) ); // host pinned

  /*
    Allocates size bytes of host memory that is page-locked and
    accessible to the device. The driver tracks the virtual memory
    ranges allocated with this function and automatically accelerates
    calls to functions such as cudaMemcpy*(). Since the memory can be
    accessed directly by the device, it can be read or written with much
    higher bandwidth than pageable memory obtained with functions such
    as malloc().
    
    It is more convenient to use page-locked / pinned memory which enables
    a DMA on the GPU to request transfers to and from the host memory
    without the involvement of the CPU. In other words, locked memory is
    stored in the physical memory (RAM), so the device can fetch it
    without the help of the host (synchronous copy). 
  */

  checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device

  float ms; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  for (int i = 0; i < nStreams; ++i) {
    checkCuda( cudaStreamCreate(&stream[i]) );
  }
  
  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  checkCuda( cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice) );
  kernel<<<n/blockSize, blockSize>>>(d_a, 0);
  checkCuda( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice, 
                               stream[i]) );
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 2: 
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  for (int i = 0; i < nStreams; ++i) {
    checkCuda( cudaStreamDestroy(stream[i]) );
  }
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}
