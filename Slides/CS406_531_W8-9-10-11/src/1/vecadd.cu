#include <stdio.h>
#include <omp.h>
#include "common.h"

#define N (256 * 1024 * 1024)
#define THREADS_PER_BLOCK 256

__global__ void vector_add(float *a, float *b, float *c) {
  int blockID =  gridDim.y * blockIdx.x +  blockIdx.y;
  int index = (THREADS_PER_BLOCK * blockID) + threadIdx.x;
  c[index] = a[index] + b[index];
}

int main() {
  cudaSetDevice(0);
  
  /******************************************************************************/
  /* Getting the device properties here */
  int device;
  cudaGetDevice(&device);
  
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("  Available memory in MBs: %f\n", double(prop.totalGlobalMem) / 1e+6);
  /******************************************************************************/
  
  /******************************************************************************/
  //Preparing the memory
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  size_t size = N * sizeof(float);
  printf("%f MBbytes will be allocated for a b c (size variable is %ld)\n", ((double)N / 1e+6) * sizeof(float) * 3, size );
  /* allocate space for device copies of a, b, c */

  cudaCheck(cudaMalloc( (void **) &d_a, size ));
  cudaCheck(cudaMalloc( (void **) &d_b, size ));
  cudaCheck(cudaMalloc( (void **) &d_c, size ));

  cudaCheck(cudaPeekAtLastError());
  
  /* allocate space for host copies of a, b, c and setup input values */

#ifdef PAGED
  a = (float *)malloc( size );
  b = (float *)malloc( size );
  c = (float *)malloc( size );
#else
  cudaCheck(cudaMallocHost( (void **) &a, size ));
  cudaCheck(cudaMallocHost( (void **) &b, size ));
  cudaCheck(cudaMallocHost( (void **) &c, size ));
#endif
  
  for(int i = 0; i < N; i++ ) {
    a[i] = b[i] = 1; c[i] = 0;
  }
  
  /******************************************************************************/
  //Timing
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 block( 1024, 1024 );
  
  /******************************************************************************/

  cudaCheck(cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice ));
  cudaCheck(cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice ));
  cudaCheck(cudaMemcpy( d_c, c, size, cudaMemcpyHostToDevice ));
  cudaEventRecord(start,0);
  vector_add<<<block, THREADS_PER_BLOCK >>>(d_a, d_b, d_c);
  cudaCheck(cudaPeekAtLastError());
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaCheck(cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost ));
  cudaEventElapsedTime(&elapsedTime,start,stop);

  for( int i = 0; i < N; i++) {
    if(c[i] != 2) {
      printf("GPU Error: value c[%d] = %f\n", i, c[i]);
      break;
    }
  }
  printf("GPU time is: %lf seconds\n", elapsedTime / 1000);
  /******************************************************************************/
  
  for( int i = 0; i < N; i++) c[i] = 0;
  double start_time = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp single
    printf("number of threads is: %d\n", omp_get_num_threads());
#pragma omp for schedule(static)
    for( int i = 0; i < N; i++ )
      c[i] = a[i] + b[i];
  }
  double end_time = omp_get_wtime();
  for( int i = 0; i < N; i++) {
    if(c[i] != 2) {
      printf("CPU Error: value c[%d] = %f\n", i, c[i]);
      break;
    }
  }
  printf("CPU time is: %lf seconds\n", end_time - start_time);
  
  /* clean up */
#ifdef PAGED
  free(a);
  free(b);
  free(c);
#else
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
#endif
  
  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );
  
  return 0;
} /* end main */
