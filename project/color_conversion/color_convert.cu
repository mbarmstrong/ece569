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

__global__ void convert_rgb_grayscale(float *grayImage, float *rgbImage, int width, int height, int numChannels) {
  
  int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
  int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

  if (col < width && row < height) {  // check boundary condition
    int idx = row * width + col;      // mapping 2D to 1D coordinate

    float r = rgbImage[numChannels * idx];      // red component
    float g = rgbImage[numChannels * idx + 1];  // green component
    float b = rgbImage[numChannels * idx + 2];  // blue component

    // rescale pixel using rgb values and floating point constants
    // store new pixel value in grayscale image
    grayImage[idx] = (0.21 * r) + (0.71 * g) + (0.07 * b); 
  }
}

__global__ void convert_rgb_yuv(float *yuvImage, float *rgbImage, int width, int height, int numChannels) {

  int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
  int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

  if (col < width && row < height) {  // check boundary condition
    int idx = row * width + col;      // mapping 2D to 1D coordinate

    // FIXME -- don't need to multiply by num channels since both have 3 channels?
    float r = rgbImage[numChannels * idx];      // red component
    float g = rgbImage[numChannels * idx + 1];  // green component
    float b = rgbImage[numChannels * idx + 2];  // blue component

    // Y  = R *  0.29900 + G *  0.58700 + B *  0.11400
    // Cb = R * -0.16874 + G * -0.33126 + B *  0.50000 + 128
    // Cr = R *  0.50000 + G * -0.41869 + B * -0.08131 + 128

    float y = (r * 0.299) + (g * 0.587) + (b * 0.114);          // luminance component
    float u = (r * -0.169) + (g * -0.331) + (b * 0.500) + 128;  // blue chrominance component
    float v = (r * 0.500) + (g * -0.419) + (b * -0.081) + 128;  // red chrominance component

    yuvImage[numChannels * idx]     = y;
    yuvImage[numChannels * idx + 1] = u;
    yuvImage[numChannels * idx + 2] = v; 
  }
}

int main(int argc, char *argv[]) {
  
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;

  char *inputImageFile;

  wbImage_t inputImage_RGB;
  wbImage_t outputImage_Gray;
  wbImage_t outputImage_YUV;

  float *hostInputImageData_RGB;
  float *hostOutputImageData_Gray;
  float *hostOutputImageData_YUV;

  float *deviceInputImageData_RGB;
  float *deviceOutputImageData_Gray;
  float *deviceOutputImageData_YUV;

  args = wbArg_read(argc, argv); // parse the input arguments

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage_RGB = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage_RGB);
  imageHeight = wbImage_getHeight(inputImage_RGB);
  imageChannels = wbImage_getChannels(inputImage_RGB);

  outputImage_Gray = wbImage_new(imageWidth, imageHeight, 3);
  outputImage_YUV = wbImage_new(imageWidth, imageHeight, 3);

  hostInputImageData_RGB = wbImage_getData(inputImage_RGB);
  hostOutputImageData_Gray = wbImage_getData(outputImage_Gray);
  hostOutputImageData_YUV = wbImage_getData(outputImage_YUV);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData_RGB,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData_Gray,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData_YUV,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData_RGB, hostInputImageData_RGB,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////

  wbTime_start(Compute, "Doing the computation on the GPU");

  // defining grid size (num blocks) and block size (num threads per block)
  dim3 myGrid(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
  dim3 myBlock(16, 16, 1);

  // kernel launch, perform color conversion on device input image data and store in device output image data
  convert_rgb_grayscale<<<myGrid, myBlock>>>(deviceOutputImageData_Gray, deviceInputImageData_RGB, imageWidth, imageHeight, imageChannels);
  convert_rgb_yuv<<<myGrid, myBlock>>>(deviceOutputImageData_YUV, deviceInputImageData_RGB, imageWidth, imageHeight, imageChannels);
  
  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData_Gray, deviceOutputImageData_Gray,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutputImageData_YUV, deviceOutputImageData_YUV,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage_Gray, outputImage_YUV);

  cudaFree(deviceInputImageData_RGB);
  cudaFree(deviceOutputImageData_Gray);
  cudaFree(deviceOutputImageData_YUV);

  wbImage_delete(outputImage_Gray);
  wbImage_delete(outputImage_YUV);
  wbImage_delete(inputImage_RGB);

  return 0;
}
