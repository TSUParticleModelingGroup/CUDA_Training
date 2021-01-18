// To compile: nvcc CPUAndGPUVectorAddition1Block.cu -o temp2
// To run: ./temp2
#include <sys/time.h>
#include <stdio.h>

#define N 1030
#define SIZEOFBLOCKS 5

//This is the CUDA kernel that will add the two vectors. 
__global__ void Addition(float *A, float *B, float *C, int n)
{

	int id = threadIdx.x;
	
	if(id < N) //This if keeps rogue threads from digging ditches in your nieghbors yard.
	{
		C[id] = A[id] + B[id];
	}
}

int main()
{
	int id;
	float sum, time;
	float *A_CPU, *B_CPU, *C_CPU; //Pointers for memory on the Host
	float *A_GPU, *B_GPU, *C_GPU; //Pointers for memory on the Device
	dim3 dimBlock; //This variable will hold the Dimensions of your block
	dim3 dimGrid; //This variable will hold the Dimensions of your grid
	timeval start, end;
	
	//Threads in a block
	dimBlock.x = N;
	dimBlock.y = 1;
	dimBlock.z = 1;
	
	//Blocks in a grid
	dimGrid.x = 1;
	dimGrid.y = 1;
	dimGrid.z = 1;
	
	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));
	
	//Loads values into vectors that we will add.
	for(id = 0; id < N; id++)
	{		
		A_CPU[id] = (float)id*2;	
		B_CPU[id] = (float)id;
	}
	
	printf("\nNumber of threads in block:  %d\n\n", N);
	//********************** CPU addition start ****************************************
	//Starting a timer	
	gettimeofday(&start, NULL);

	//Add the two vectors
	for(id = 0; id < N; id++)
	{ 
		C_CPU[id] = A_CPU[id] + B_CPU[id];
	}
	
	//Stopping the timer
	gettimeofday(&end, NULL);
	//********************** CPU addition finish ****************************************
	
	//Calculating the total time used in the addition on the CPU and converting it to milliseconds and printing it to the screen.
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("------ CPU Results ------\n");
	printf("CPU Time in milliseconds= %.15f\n", (time/1000.0));
	
	//Summing up the vector C and printing it so we can have a spot check for the correctness of the CPU addition.
	sum = 0.0;
	for(id = 0; id < N; id++)
	{ 
		sum += C_CPU[id];
	}
	printf("Sum of C_CPU from CPU addition= %.15f\n", sum);
	printf("C_CPU[%d]= %f\n", N-1, C_CPU[N-1]);
	for(id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0;
	}

	//********************** GPU addition start ****************************************
	//Starting a timer	
	gettimeofday(&start, NULL);
	
	//Copying vectors A_CPU and B_CPU that were loaded on the CPU up to the GPU
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	
	Addition<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, C_GPU, N);
	
	//Copy C_GPU that was calculated on the GPU down to the CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);

	//Stopping the timer
	gettimeofday(&end, NULL);
	//********************** GPU addition finish ****************************************
	
	//Calculating the total time used in the addition on the GPU and converting it to milliseconds and printing it to the screen.
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\n------ GPU Results ------\n");
	printf("GPU Time in milliseconds= %.15f\n", (time/1000.0));
	
	//Summing up the vector C and printing it so we can have a spot check for the correctness of the GPU addition.
	sum = 0.0;
	for(id = 0; id < N; id++)
	{ 
		sum += C_CPU[id];
	}
	printf("Sum of C_CPU from GPU addition= %.15f\n", sum);
	printf("C_CPU[%d]= %f\n", N-1, C_CPU[N-1]);

	//Your done so cleanup your mess.	
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
	
	return(0);
}
