/******************************************************************************
** Program name: OpenMP: Numeric Integration with OpenMP
** Author: Rebecca L. Taylor
** Date: 28 April 2019
** Description: This main file calculates the volume between two Bezier surfaces.
   It uses OpenMP to create multiple threads to calculate volume for the given
   number of nodes.
******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "Proj2.hpp"

// function prototypes:
float Height( int, int );

// main program:
int main( int argc, char *argv[ ] )
{
#ifndef _OPENMP
	fprintf( stderr, "No OpenMP support!\n" );
	return 1;
#endif
    omp_set_num_threads( NUMT );	

    // the area of a single full-sized tile:
    float fullTileArea = (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) )  *
                            ( ( YMAX - YMIN )/(float)(NUMNODES-1) )  );

    // get ready to record the maximum performance and the volume
    float maxPerformance = 0.;              // must be declared outside the NUMTRIES loop
    float sumPerformance = 0.;
    double volume;

    // sum up the weighted heights into the variable "volume"
    // using an OpenMP for loop and a reduction:
    for( int t = 0; t < NUMTRIES; t++ )
    {
        double time0 = omp_get_wtime( );    // start timing
        volume = 0.;                        // accumulate volume 

        #pragma omp parallel for default(none) shared(fullTileArea) reduction(+:volume)
        for( int i = 0; i < NUMNODES*NUMNODES; i++ )
	{
            int iu = i % NUMNODES;
            int iv = i / NUMNODES;
            float height = Height(iu, iv);
            float currTileArea = fullTileArea;

            // half area of edge tiles; corner tiles will be halfed twice
            if (iv == 0 || iv == NUMNODES-1) 
            {
                currTileArea *= 0.5;
            }
            if (iu == 0 || iu == NUMNODES-1)
            {
                currTileArea *= 0.5;
            }

            volume += ( currTileArea * height ); 
        }                                   // end (num nodes * num nodes) loop

        // get end time and calculate difference in microseconds
        double time1 = omp_get_wtime( );    
        double megaHeightsPerSecond = double(NUMNODES * NUMNODES) / (time1 - time0) / 1000000.;
        sumPerformance += megaHeightsPerSecond;

        if (megaHeightsPerSecond > maxPerformance)
        {
            maxPerformance = megaHeightsPerSecond;
        }

    }                                       // end num tries loop (find maximum performance)

         printf("Num threads: %i\n", NUMT);
         printf("Num nodes:   %i\n", NUMNODES);
         printf("Volume:      %.2lf\n", volume);
         printf("Peak perf:   %.2lf\n", maxPerformance); 
         printf("Avg. perf:   %.2lf\n", sumPerformance / (double)NUMTRIES); 
         printf("\t%.2lf\n", maxPerformance);
        
        return 0;
}                                               // end main

