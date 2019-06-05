// Array multiplication: C = A * B:

// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE					// 16, 32, and 64
#define BLOCKSIZE		16			// number of threads per block 
#endif

#ifndef SIZE						// 16K, 32K, 64K, 128K, 256K, and 512K
#define SIZE			16000		// array size
#endif

#ifndef NUMTRIALS
#define NUMTRIALS		100			// to make the timing more accurate
#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

// ranges for the random numbers:
const float XCMIN = 0.0;
const float XCMAX = 2.0;
const float YCMIN = 0.0;
const float YCMAX = 2.0;
const float RMIN = 0.5;
const float RMAX = 2.0;

// function prototypes:
float		Ranf(float, float);
int			Ranf(int, int);
void		TimeOfDaySeed();


// Monte Carlo simulation (CUDA Kernel) on the device

__global__  void MonteCarlo( float *xcs, float *ycs, float *rs, int *hits )
{
	// get thread info
	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	// accumulate numHits per thread
	__shared__ float numHits[BLOCKSIZE];
	numHits[tnum] = 0;					

	// randomize the location and radius of the circle:
	float xc = xcs[gid];
	float yc = ycs[gid];
	float  r = rs[gid];

	// solve for the intersection using the quadratic formula:
	float a = 2.;
	float b = -2. * (xc + yc);
	float c = xc * xc + yc * yc - r * r;
	float d = b * b - 4. * a * c;

	// case A: circle completely missed (d < 0.)
	if (d < 0.)
	{
		numHits[tnum] = 0;
		goto finish_analysis;
	}
		
	// if not case A, hits the circle:
	// get the first intersection:
	d = sqrtf(d);
	float t1 = (-b + d) / (2. * a);						 // time to intersect the circle
	float t2 = (-b - d) / (2. * a);						 // time to intersect the circle
	float tmin = t1 < t2 ? t1 : t2;					     // only care about the first intersection

	// case B: circle engulfs line (tmin < 0.)
	if (tmin < 0.)
	{
		numHits[tnum] = 0;
		goto finish_analysis;
	}
		
	// if not case A or case B, where does it intersect the circle?
	float xcir = tmin;
	float ycir = tmin;

	// get the unitized normal vector at the point of intersection:
	float nx = xcir - xc;
	float ny = ycir - yc;
	float n = sqrtf(nx * nx + ny * ny);
	nx /= n;											// unit vector
	ny /= n;											// unit vector
							
	// get the unitized incoming vector:
	float inx = xcir - 0.;
	float iny = ycir - 0.;
	float in = sqrtf(inx * inx + iny * iny);
	inx /= in;											// unit vector
	iny /= in;											// unit vector

	// get the outgoing (bounced) vector:
	float dot = inx * nx + iny * ny;
	float outx = inx - 2. * nx * dot;					// angle of reflection = angle of incidence`
	float outy = iny - 2. * ny * dot;					// angle of reflection = angle of incidence`

	// find out if it hits the infinite plate:
	float t = (0. - ycir) / outy;

	// case C: line bounced back up (t < 0.)
	if (t < 0.)
	{
		numHits[tnum] = 0;
		goto finish_analysis;
	}

	// case D: if not case A, B, or C, line hit the plate
	numHits[tnum] = 1;

	// do the reduction (all threads execute simultaneously)
	finish_analysis:									
	for (int offset = 1; offset < numItems; offset *= 2)
	{
		int mask = 2 * offset - 1;
		__syncthreads();
		if ((tnum & mask) == 0)
		{
			numHits[tnum] += numHits[tnum + offset];
		}
	}

	// add to results array after all threads have finished
	__syncthreads();						
	if (tnum == 0)
		hits[wgNum] = numHits[0];
}


// main program:

int
main( int argc, char* argv[ ] )
{
	int dev = findCudaDevice(argc, (const char **)argv);

	TimeOfDaySeed();

	// allocate host memory:

	float *hXCS    = new float[ SIZE ];			// x centers
	float *hYCS    = new float[ SIZE ];			// y centers
	float *hRS     = new float[ SIZE ];			// radius
	int   *hHits   = new int  [ SIZE ];			// return results from work groups to add on CPU

	for (int n = 0; n < SIZE; n++)
	{
		hXCS[n]    = Ranf(XCMIN, XCMAX);
		hYCS[n]    = Ranf(YCMIN, YCMAX);
		hRS[n]     = Ranf(RMIN, RMAX);
		hHits[n]   = 0;
	}

	// allocate device memory:

	float *dXCS, *dYCS, *dRS;
	int   *dHits;

	dim3 dimsXCS(SIZE, 1, 1);
	dim3 dimsYCS(SIZE, 1, 1);
	dim3 dimsRS(SIZE, 1, 1);
	dim3 dimsHits(SIZE / BLOCKSIZE, 1, 1);

	//__shared__ float prods[SIZE/BLOCKSIZE];

	cudaError_t status;			
	status = cudaMalloc(reinterpret_cast<void **>(&dXCS), SIZE * sizeof(float));
		checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void **>(&dYCS), SIZE * sizeof(float));
		checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void **>(&dRS),  SIZE * sizeof(float));
		checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void **>(&dHits), (SIZE / BLOCKSIZE) * sizeof(int));
		checkCudaErrors(status);


	// copy host memory to the device:

	status = cudaMemcpy(dXCS, hXCS, SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaErrors(status);
	status = cudaMemcpy(dYCS, hYCS, SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaErrors(status);
	status = cudaMemcpy(dRS,  hRS,  SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaErrors(status);

	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid( SIZE / threads.x, 1, 1 );

	// Create and start timer

	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );

	// execute the kernel:

	for (int t = 0; t < NUMTRIALS; t++)
	{
			MonteCarlo << < grid, threads >> > (dXCS, dYCS, dRS, dHits);
	}

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
		checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );
		
	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double trialsPerSecond = (float)SIZE * (float)NUMTRIALS / secondsTotal;
	double megaTrialsPerSecond = trialsPerSecond / 1000000.;
	fprintf( stderr, "Block Size = %10d, Array Size = %10d, MegaTrials/Second = %10.2lf\n", BLOCKSIZE, SIZE, megaTrialsPerSecond );
	
	// copy result from the device to the host:

	status = cudaMemcpy( hHits, dHits, (SIZE/BLOCKSIZE)*sizeof(float), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

	// check the probability -- should be about 42%? 

	float totalHits = 0.;
	for (int i = 0; i < SIZE/BLOCKSIZE; i++)						
	{
		totalHits += hHits[i];
	}
	float probability = (totalHits / SIZE) * 100;
	printf("Probability: %10.2lf\n", probability);

	// read performance and probability to file

	FILE* outputResults = fopen("monteCarlo_results.csv", "a");
	if (outputResults == NULL)
	{
		printf("Error: no output file\n");
		exit(1);
	}

	fprintf(outputResults, "%d, %d, %.2lf, %.2lf\n", BLOCKSIZE, SIZE, megaTrialsPerSecond, probability);

	fclose(outputResults);

	// clean up memory:

	delete [ ] hXCS;
	delete [ ] hYCS;
	delete [ ] hRS;
	delete [ ] hHits;

	status = cudaFree(dXCS);
		checkCudaErrors(status);
	status = cudaFree(dYCS);
		checkCudaErrors(status);
	status = cudaFree(dRS);
		checkCudaErrors(status);


	return 0;
}

// functions for random numbers
float
Ranf(float low, float high)
{
	float r = (float)rand();               // 0 - RAND_MAX
	float t = r / (float)RAND_MAX;       // 0. - 1.

	return   low + t * (high - low);
}

int
Ranf(int ilow, int ihigh)
{
	float low = (float)ilow;
	float high = ceil((float)ihigh);

	return (int)Ranf(low, high);
}

// seed time of day
void
TimeOfDaySeed()
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time(&timer);
	double seconds = difftime(timer, mktime(&y2k));
	unsigned int seed = (unsigned int)(1000.*seconds);    // milliseconds
	srand(seed);
}