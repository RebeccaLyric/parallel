/******************************************************************************
** Program name: Vectorized Array Multiplication and Reduction Using SSE
** Author: Rebecca L. Taylor
** Date: 13 May 2019
** Description: This main file implements a program to test SIMD vs. Non-SIMD
	array-multiplication and array-multiplication-reduction. 
******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "Rand.hpp"
#include "simd.p4.h"

#ifndef ARRAYSIZE
#define ARRAYSIZE   1000
#endif

#ifndef FUNCTION
#define FUNCTION    1                           ; function 1 is non-SIMD array multiply
#endif

#ifndef NUMTRIES
#define NUMTRIES    10
#endif

// ranges for the random numbers
const float MIN = -10.;
const float MAX = 10.;

// non-SIMD array multiplication and multiplication-reduction functions
void  NonSimdMul( float *a, float *b, float *c, int len)
{
    for (int i=0; i < len; i++)
    {
	c[i] = a[i] * b[i];
    }
}

float NonSimdMulSum( float *a, float *b, int len )
{
    float sum = 0.;
    for( int i = 0; i < len; i++ )
    {
    	sum += a[i] * b[i];
    }
    return sum;
}

int main()
{
#ifndef _OPENMP
    fprintf(stderr, "OpenMP is not supported here -- sorry.\n");
    return 1;
#endif
    // seed the random number generator
    TimeOfDaySeed();                            

    // define arrays
    float *A = new float [ARRAYSIZE];
    float *B = new float [ARRAYSIZE];
    float *C = new float [ARRAYSIZE];

    // fill arrays 
    for (int i=0; i < ARRAYSIZE; i++)
    {
        A[i] = Ranf( MIN, MAX );
        B[i] = Ranf( MIN, MAX );
        C[i] = 0.;
    }

    // variables for performance measurement
    double maxPerformance = 0.;                   
    double sumPerformance = 0.;

    // run experiment multiple tries to get best performance
    for (int t = 0; t < NUMTRIES; t++)
    {
        // start timing
        double time0 = omp_get_wtime();         

        // run current matrix operation
        float result = 0.;                                  
        switch (FUNCTION)
        {
            case 1: SimdMul(A, B, C, ARRAYSIZE);         
                    break;
            case 2: NonSimdMul(A, B, C, ARRAYSIZE);         
                    break;
            case 3: result = SimdMulSum(A, B, ARRAYSIZE); 
                    break;
            case 4: result = NonSimdMulSum(A, B, ARRAYSIZE);
                    break;
            default: printf("Error: no function\n");
                    break;
        }

        // stop timing
        double time1 = omp_get_wtime();         
        
        // calculate performance and save maximum performance result
        double performance = (double)ARRAYSIZE/(time1-time0)/1000000.;
        sumPerformance += performance;
        if (performance > maxPerformance)
        {
            maxPerformance = performance;
        }
    }                                           // end num tries loop

    // print performance results
    printf("\nArray size       = %8d elements\n", ARRAYSIZE);
    printf("Function number  = %8d\n", FUNCTION);
    printf("Avg. performance = %8.2lf MegaMults/Sec\n", sumPerformance/(double)NUMTRIES);
    printf("Peak performance = %8.2lf MegaMults/Sec\n", maxPerformance);
    printf("\t%8.2lf\n", maxPerformance);

    // free array memory
    delete [] A;
    delete [] B;
    delete [] C;

    return 0;
}

