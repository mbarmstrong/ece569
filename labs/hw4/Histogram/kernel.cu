/* 
Histogram generation algorithm for an input array of integers within a given range. 
Each integer will map into a single bin, so the values will range from 0 to (NUM_BINS - 1). 
The histogram bins will use unsigned 32-bit counters that must be saturated at 127, 
meaning all bins with values larger than 127 need to be clipped to 127.  
This clipping operation is a key step during histogram equalization process. 
The input length can be assumed to be less than 2^32. NUM_BINS is fixed at 4096.
*/

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

// version 0
// global memory only interleaved version
// include comments describing your approach
__global__ void histogram_global_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

	// insert your code here
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index
	int stride = blockDim.x * gridDim.x; // total number of threads

	while (i < num_elements) {
		int pos = input[i]; // bin position
		if (pos >= 0 && pos < num_bins) // boundary condition check
			atomicAdd(&bins[pos], 1); // atomically increment appropriate bin
		i += stride;
	}
}


// version 1
// shared memory privatized version
// include comments describing your approach
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

	// insert your code here
	__shared__ unsigned int bins_private[4096]; // privatized bins
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index
	int stride = blockDim.x * gridDim.x; // total number of threads

	// initialize privatized bins to 0
	if (threadIdx.x < 4096) bins_private[threadIdx.x] = 0;
	__syncthreads();

	// build local histogram
	while (i < num_elements) {
		int pos = input[i]; // bin position
		if (pos >= 0 && pos < 4096) // boundary condition check
			atomicAdd(&bins_private[pos], 1); // atomically increment appropriate privatized bin
		i += stride;
	}
	__syncthreads();

	// build global histogram
	// number of bins > block size -- need multiple bins per thread
	for (int j = 0; j < num_bins; j += blockDim.x) {
		atomicAdd(&bins[threadIdx.x + j], bins_private[threadIdx.x + j]);
	}
}


// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
__global__ void histogram_shared_accumulate_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

	// insert your code here
	__shared__ unsigned int bins_private[4096]; // privatized bins
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index
	int stride = blockDim.x * gridDim.x; // total number of threads

	// initialize privatized bins to 0
	if (threadIdx.x < 4096) bins_private[threadIdx.x] = 0;
	__syncthreads();

	// build local histogram
	while (i < num_elements) {
		int pos = input[i]; // bin position
		int j = 0;
		int count = 0;

		if (pos >= 0 && pos < 4096) // boundary condition check
			j = i + 1;
			count = 1;
			if (j < num_elements) {
				while (input[j] == input[i]) {
				count++;
				j++;
				}
			}
			atomicAdd(&bins_private[pos], count); // atomically increment appropriate privatized bin
		
		i += stride;
	}
	__syncthreads();

	// build global histogram
	// number of bins > block size -- need multiple bins per thread
	for (int j = 0; j < num_bins; j += blockDim.x) {
		atomicAdd(&bins[threadIdx.x + j], bins_private[threadIdx.x + j]);
	}

	// counter to track if multiple bins are being updated sequentially, only one atomic add needed

	// sorting based approach
	// reduce by key
	// compression before reduction

}

// clipping function
// resets bins that have value larger than 127 to 127. 
// that is if bin[i]>127 then bin[i]=127
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

	// insert your code here
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index
	if (bins[i] > 127) bins[i] = 127;
}
