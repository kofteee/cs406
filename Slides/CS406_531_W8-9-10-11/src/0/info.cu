#include <stdio.h>

int main() {
  int nDevices;
  
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1000);
    printf("  Clock rate (MHz): %d\n", prop.clockRate / 1000);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Global memory size (GB): %.2f\n", (prop.totalGlobalMem + .0f) / (1000000000));
    printf("  Shared mem per block: %ld\n", prop.sharedMemPerBlock);
    printf("  Major-minor: %d.%d\n", prop.major, prop.minor);
    printf("  Device overlap: %d\n", prop.deviceOverlap); //Device can concurrently copy memory and execute a kernel
    printf("  Compute mode: %d\n", prop.computeMode); //0 is cudaComputeModeDefault (Multiple threads can use cudaSetDevice() with this device)
    puts("");
  }
}
