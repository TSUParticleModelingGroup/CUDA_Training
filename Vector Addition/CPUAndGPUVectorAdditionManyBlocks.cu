// To compile: nvcc CPUAndGPUVectorAdditionManyBlocks.cu -o temp
// To run: ./temp
#include <sys/time.h>
#include <stdio.h>

// 65536 upper bound for 16 bit integer
// 2147483647 upper bound for 32 bit signed integer
// worked with 214748364 but not 2147483640 worked at (3*214748364) but not at (4*214748364)

#define BLOCK_SIZE 256
#define N 100000


//This is the CUDA kernel that will add the two vectors. 
__global__ void Addition(float *A, float *B, float *C, long n)
{

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(id < n) //This if keeps rogue threads from digging ditches in your nieghbors yard.
	{
		C[id] = A[id] + B[id];
	}
}

int main()
{
	long id;
	float time;
	float *A_CPU, *B_CPU, *C_CPU; //Pointers for memory on the Host
	float *A_GPU, *B_GPU, *C_GPU; //Pointers for memory on the Device
	dim3 dimBlock; //This variable will hold the Dimensions of your block
	dim3 dimGrid; //This variable will hold the Dimensions of your grid
	timeval start, end;
	double sum;
	long n = N;
	
	printf("\n\n size of dim3 %d", sizeof(dim3));
	printf("\n size of int %d", sizeof(int));
	printf("\n size of long %d \n", sizeof(long));
	
	printf("\n N = %d\n\n",n);
	
	//Threads in a block
	dimBlock.x = BLOCK_SIZE;
	dimBlock.y = 1;
	dimBlock.z = 1;
	
	//Blocks in a grid
	dimGrid.x = (n-1)/dimBlock.x + 1;
	dimGrid.y = 1;
	dimGrid.z = 1;
	
	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(n*sizeof(float));
	B_CPU = (float*)malloc(n*sizeof(float));
	C_CPU = (float*)malloc(n*sizeof(float));
	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,n*sizeof(float));
	cudaMalloc(&B_GPU,n*sizeof(float));
	cudaMalloc(&C_GPU,n*sizeof(float));
	
	//Loads values into vectors that we will add.
	for(id = 0; id < n; id++)
	{		
		A_CPU[id] = 1.0;  //(float)id*2;	
		B_CPU[id] = 3.0;  //(float)id;
	}
	
	//********************** CPU addicudaGetErrorString(cudaGetLastError()) tion start ****************************************
	//Starting a timer	
	gettimeofday(&start, NULL);

	//Add the two vectors
	for(id = 0; id < n; id++)
	{ 
		C_CPU[id] = A_CPU[id] + B_CPU[id];
	}
	
	//Stopping the timer
	gettimeofday(&end, NULL);
	//********************** CPU addition finish ****************************************
	
	//Calculating the total time used in the addition on the CPU and converting it to milliseconds and printing it to the screen.
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("------ CPU Results ------\n");
	printf(" CPU Time in milliseconds= %.15f\n", (time/1000.0));
	
	//Summing up the vector C and printing it so we can have a spot check for the correctness of the CPU addition.
	sum = 0.0;
	for(id = 0; id < n; id++)
	{ 
		sum += C_CPU[id];
	}
	printf(" Sum of C_CPU from CPU addition= %f\n", sum);
	printf(" C_CPU[%d]= %f\n", n-1, C_CPU[n-1]);
	
	
	//Resetting C_CPU to 0 to make sure the GPU really did an addition.
	for(id = 0; id < n; id++)
	{ 
		C_CPU[id] = 0;
	}

	//********************** GPU addition start ****************************************
	//Starting a timer	
	gettimeofday(&start, NULL);
	
	//Copying vectors A_CPU and B_CPU that were loaded on the CPU up to the GPU
	cudaMemcpyAsync(A_GPU, A_CPU, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, n*sizeof(float), cudaMemcpyHostToDevice);
	
	Addition<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, C_GPU, n);
	//Copy C_GPU that was calculated on the GPU down to the CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);

	//Stopping the timer
	gettimeofday(&end, NULL);
	//********************** GPU addition finish ****************************************
	
	//Calculating the total time used in the addition on the GPU and converting it to milliseconds and printing it to the screen.
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\n------ GPU Results ------\n");
	printf(" GPU Time in milliseconds= %.15f\n", (time/1000.0));
	
	//Summing up the vector C and printing it so we can have a spot check for the correctness of the GPU addition.
	sum = 0.0;
	for(id = 0; id < n; id++)
	{ 
		sum += C_CPU[id];
	}
	printf(" Sum of C_CPU from GPU addition= %f\n", sum);
	printf(" C_GPU[%d]= %f\n\n", n-1, C_CPU[n-1]);

	//Your done so cleanup your mess.	
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
	
	return(0);
}
