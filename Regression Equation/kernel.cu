
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		c[i] = 5*a[i] +3* b[i];
	}
}

int main(int argc, char *argv[])
{
	int numElements = 5e+4;
	// Allocate vectors a, b and c in host memory.
	size_t numBytes = sizeof(float)* numElements;


	float *h_a = (float *)malloc(numBytes);
	float *h_b = (float *)malloc(numBytes);
	float *h_c = (float *)malloc(numBytes);


	// Initialize vectors a and b.
	for (int i = 0; i < numElements; ++i)
	{
		h_a[i] = rand() / (float)RAND_MAX;
		h_b[i] = rand() / (float)RAND_MAX;
	}

	printf("dumping some arrays elements values\n");
	for (int i = 0; i < 10; ++i)
	{
		printf("%d \n", h_a[i]);
		printf("%d \n", h_b[i]);

	}


	printf("End of dumping\n");

	// Allocate vectors a, b and c in device memory.
	float *d_a;
	float *d_b;
	float *d_c;

	printf("Parallelizing a big task for GPU card\n");


	cudaMalloc((void **)&d_a, numBytes);
	cudaMalloc((void **)&d_b, numBytes);
	cudaMalloc((void **)&d_c, numBytes);
	// Copy vectors a and b from host memory to device memory synchronously.
	cudaMemcpy(d_a, h_a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, numBytes, cudaMemcpyHostToDevice);


	// Determine the number of threads per block and the number of blocks per grid.
	int numThreadsPerBlock = 256;
	int numBlocksPerGrid = (numElements + numThreadsPerBlock - 1) / numThreadsPerBlock;
	// Invoke the kernel on device asynchronously.


	vectorAdd << <numBlocksPerGrid, numThreadsPerBlock >> >(d_a, d_b, d_c, numElements);
	// Copy vector c from device memory to host memory synchronously.
	cudaMemcpy(h_c, d_c, numBytes, cudaMemcpyDeviceToHost);
	// Validate the result.

	for (int i = 0; i < numElements; ++i)
	{
		float actual = h_c[i];
		float expected = h_a[i] + h_b[i];

		printf("h_a[%d]=%f  \n",i,h_a[i]);
		printf("h_b[%d]=%f \n",  i, h_b[i]);

		printf("h_c[%d]=%f\n",i, h_c[i]);


		/*if (fabs(actual - expected) > 1e-7)
		{
			printf("h_c[%d] = %f, expected = %f\n", i, actual, expected);
			break;
		}*/
	}
	// Cleanup.
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	cudaDeviceReset();
	/*free(h_c);
	free(h_b);
	free(h_a);*/

}