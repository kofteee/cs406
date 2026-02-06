//declare constant memory
__constant__ float cangle[360];

__global__ void test_kernel_constant(float* darray) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  for(int loop = 0; loop < 360; loop++) {
    darray[index] += cangle[loop];
  }   
}	

__global__ void test_kernel_shared(float* darray, float* g_cangle) {
  int local_index = threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ float s_cangle[360];
  if(local_index < 360) {
    s_cangle[local_index] = g_cangle[local_index];
  }
  __syncthreads();
  
  for(int loop = 0; loop < 360; loop++) {
    darray[index] += s_cangle[loop];
  }
}	

__global__ void test_kernel_global(float* darray, float* g_cangle) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
 
  for(int loop = 0; loop < 360; loop++) { 
    darray[index] += g_cangle[loop];
  }
}


int main(int argc,char** argv) {
  cudaSetDevice(1);      

  int size = 64000;

  float *darray, *g_cangle;
  float hangle[360];

  //allocate device memory
  cudaMalloc((void**)&darray, sizeof(float) * size);

  //allocate device memory for g_cangle
  cudaMalloc((void**)&g_cangle,sizeof(float)*360);
													         
  //initialize angle array on host
  for(int loop = 0; loop < 360; loop++){ 
    hangle[loop] = acos(-1.0f) * (loop/180.0f);
  }

  //copy host angle data to constant memory
  cudaMemcpyToSymbol(cangle, hangle, sizeof(float) * 360);
  //copy host angle data to global memory
  cudaMemcpy(g_cangle, hangle, sizeof(float) * 360, cudaMemcpyHostToDevice);

  //test constant
  cudaMemset (darray, 0, sizeof(float) * size);
  test_kernel_constant<<<size/64, 64>>>  (darray);
  cudaDeviceSynchronize();

  //test global
  cudaMemset (darray, 0, sizeof(float) * size);
  test_kernel_global<<<size/64, 64>>>  (darray, g_cangle);
  cudaDeviceSynchronize();

  //test shared
  cudaMemset (darray, 0, sizeof(float) * size);
  test_kernel_shared<<<size/64, 64>>>  (darray, g_cangle);
  cudaDeviceSynchronize();
  																				     
  //free device memory
  cudaFree(darray);
  cudaFree(g_cangle);
  return 0;
}

