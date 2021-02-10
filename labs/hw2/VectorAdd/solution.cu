#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  
  //@@ Insert code to implement vector addition here
  //   and launch your kernel from the main function
  int i = threadIdx.x + blockIdx.x * blockDim.x; // thread index

  if (i < len) // check boundary condition -- index must be less than vector length
    out[i] = in1[i] + in2[i];  // perform single pair-wise vector addition and store in output vector
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");

  //@@ Allocate GPU memory here
  // allocate device input and output objects in global memory
  // parameters:  address of pointer to allocated object,
  //              size of allocated object (in bytes)
  // need to cast address of pointer to (void **) to fit generic pointer expected from function
  cudaMalloc((void **) &deviceInput1, inputLength * sizeof(float));
  cudaMalloc((void **) &deviceInput2, inputLength * sizeof(float));
  cudaMalloc((void **) &deviceOutput, inputLength * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");

  //@@ Copy memory to the GPU here
  // parameters:  pointer to destination (device input),
  //              pointer to source (host input),
  //              size in bytes,
  //              direction of transfer (host to device)  
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // defining grid size (num blocks) and block size (num threads per block)
  // block size is 256 threads
  // ceiling function ensures enough thread blocks in grid to cover input vector length
  dim3 myGrid(ceil(inputLength / 256.0));
  dim3 myBlock(256);

  wbTime_start(Compute, "Performing CUDA computation");

  //@@ Launch the GPU Kernel here
  // perform vector addition of device inputs, store results in device output
  vecAdd<<<myGrid, myBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");

  //@@ Copy the GPU memory back to the CPU here
  // parameters:  pointer to destination (host output),
  //              pointer to source (device output),
  //              size in bytes,
  //              direction of transfer (device to host)
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
