#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT DEVICE CODE HERE
__global__ void colorConvert(float *grayImage, float *rgbImage, int width, int height, int numChannels) {
  
  int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
  int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

  if (col < width && row < height) {  // check boundary condition
    int idx = row * width + col; // mapping 2D to 1D coordinate

    float r = rgbImage[numChannels * idx];      // red component
    float g = rgbImage[numChannels * idx + 1];  // green component
    float b = rgbImage[numChannels * idx + 2];  // blue component

    // rescale pixel using rgb values and floating point constants
    // store new pixel value in grayscale image
    grayImage[idx] = (0.21 * r) + (0.71 * g) + (0.07 * b); 
  }
}

// Also modify the main function to launch thekernel. 
int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  
  //@@ INSERT CODE HERE

  // defining grid size (num blocks) and block size (num threads per block)
  // block size is 16x16
  dim3 myGrid(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
  dim3 myBlock(16, 16, 1);

  // kernel launch, perform color conversion on device input image data and store in device output image data
  colorConvert<<<myGrid, myBlock>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight, imageChannels);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
