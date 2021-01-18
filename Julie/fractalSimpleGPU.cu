//nvcc fractalSimpleGPU.cu -o fractalGPU -lglut -lGL -lm

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


#define A  -0.8
#define B  0.0

unsigned int window_width = 1024;
unsigned int window_height = 1024;

__global__ void fractal(float *pixels, float xMin, float yMin, float dx, float dy )
{	
	float x,y;
	float mag,maxMag,t1;
	
	int id = 3*(blockDim.x*blockIdx.x + threadIdx.x);
	x = xMin + dx*threadIdx.x;
	y = yMin + dy*blockIdx.x;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
	
	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;	
		x = x*x - y*y + A;
		y = (2.0 * t1 * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) pixels[id] = 0.0;
	else pixels[id] = 1.0;
	pixels[id+1] = 0.0; 
	pixels[id+2] = 0.0;
}

void display(void) 
{ 
	float *pixelsCPU, *pixelsGPU; 
	dim3 dimBlock; 
	dim3 dimGrid;
	
	float xMin = -2.0;
	float xMax =  2.0;
	float yMin = -2.0;
	float yMax =  2.0;

	float stepSizeX = (xMax - xMin)/((float)window_width);
	float stepSizeY = (yMax - yMin)/((float)window_height);
	
	//Threads in a block
	if(window_width > 1024) printf("The window width is too large to run with this program\n");
	dimBlock.x = window_width;
	dimBlock.y = 1;
	dimBlock.z = 1;
	
	//Blocks in a grid
	dimGrid.x = window_height;
	dimGrid.y = 1;
	dimGrid.z = 1;

	pixelsCPU = (float *)malloc(window_width*window_height*3*sizeof(float));
	cudaMalloc(&pixelsGPU,window_width*window_height*3*sizeof(float));
	
	fractal<<<dimGrid, dimBlock>>>(pixelsGPU, xMin, yMin, stepSizeX, stepSizeY);
	
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, window_width*window_height*3*sizeof(float), cudaMemcpyDeviceToHost);

	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixelsCPU); 
	glFlush(); 
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

