/******************************************************************************
** Program name: OpenMP: Monte Carlo Simulation
** Author: Rebecca L. Taylor
** Date: 17 April 2019
** Description: This main file implements a Monte Carlo simulation and uses
    OpenMP to test speedup with various numbers of parallel threads.
******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "Proj1.hpp"

// setting the number of threads:
#ifndef NUMT
#define NUMT		1
#endif

// setting the number of trials in the monte carlo simulation:
#ifndef NUMTRIALS
#define NUMTRIALS	1000000
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

// ranges for the random numbers:
const float XCMIN =	-1.0;
const float XCMAX =	 1.0;
const float YCMIN =	 0.0;
const float YCMAX =	 2.0;
const float RMIN  =	 0.5;
const float RMAX  =	 2.0;

// function prototypes:
float		Ranf( float, float );
int		Ranf( int, int );
void		TimeOfDaySeed( );

// main program:
int
main( int argc, char *argv[ ] )
{
#ifndef _OPENMP
	fprintf( stderr, "No OpenMP support!\n" );
	return 1;
#endif

    TimeOfDaySeed( );		            // seed the random number generator

    omp_set_num_threads( NUMT );	    // set the number of threads to use in the for-loop:`

    // better to define these here so that the rand() calls don't get into the thread timing:
    float *xcs = new float [NUMTRIALS];
    float *ycs = new float [NUMTRIALS];
    float * rs = new float [NUMTRIALS];

    // fill the random-value arrays:
    for( int n = 0; n < NUMTRIALS; n++ )
    {       
        xcs[n] = Ranf( XCMIN, XCMAX );
        ycs[n] = Ranf( YCMIN, YCMAX );
        rs[n] = Ranf(  RMIN,  RMAX ); 
    }       

    // get ready to record the maximum performance and the probability:
    float maxPerformance = 0.;      // must be declared outside the NUMTRIES loop
    float currentProb;              // must be declared outside the NUMTRIES loop
    double sumMegaTrials = 0.;
    double bestExecutionTime = 99999.;

    // looking for the maximum performance:
    for( int t = 0; t < NUMTRIES; t++ )
    {
        double time0 = omp_get_wtime( );

        int numHits = 0;
        #pragma omp parallel for default(none) shared(xcs,ycs,rs) reduction(+:numHits)
        for( int n = 0; n < NUMTRIALS; n++ )
	{
            // randomize the location and radius of the circle:
	    float xc = xcs[n];
	    float yc = ycs[n];
	    float  r =  rs[n];

	    // solve for the intersection using the quadratic formula:
	    float a = 2.;
	    float b = -2.*( xc + yc );
	    float c = xc*xc + yc*yc - r*r;
	    float d = b*b - 4.*a*c;
    
            // case A: circle completely missed (d < 0.)
            if (d < 0.) 
                continue;
            
            // if not case A, hits the circle:
            // get the first intersection:
	    d = sqrt( d );
	    float t1 = (-b + d ) / ( 2.*a );	// time to intersect the circle
	    float t2 = (-b - d ) / ( 2.*a );	// time to intersect the circle
	    float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection

            // case B: circle engulfs line (tmin < 0.)
            if (tmin < 0.)
                continue; 

            // if not case A or case B, where does it intersect the circle?
            float xcir = tmin;
	    float ycir = tmin;

	    // get the unitized normal vector at the point of intersection:
	    float nx = xcir - xc;
	    float ny = ycir - yc;
	    float n = sqrt( nx*nx + ny*ny );
	    nx /= n;	// unit vector
	    ny /= n;	// unit vector

	    // get the unitized incoming vector:
	    float inx = xcir - 0.;
	    float iny = ycir - 0.;
	    float in = sqrt( inx*inx + iny*iny );
	    inx /= in;	// unit vector
	    iny /= in;	// unit vector

	    // get the outgoing (bounced) vector:
	    float dot = inx*nx + iny*ny;
	    float outx = inx - 2.*nx*dot;	// angle of reflection = angle of incidence`
	    float outy = iny - 2.*ny*dot;	// angle of reflection = angle of incidence`

	    // find out if it hits the infinite plate:
	     float t = ( 0. - ycir ) / outy;
                
            // case C: line bounced back up (t < 0.)
            if (t < 0.)
                continue; 

            // if not case A, B, or C, line hit the plate
            numHits += 1;
        }                                       // end num trials loop

        double time1 = omp_get_wtime( );
        double executionTime = ( time1 - time0 ) * 1000000.;  

	double megaTrialsPerSecond = (double)NUMTRIALS / ( time1 - time0 ) / 1000000.;
        sumMegaTrials += megaTrialsPerSecond;
	if( megaTrialsPerSecond > maxPerformance )
        {
		maxPerformance = megaTrialsPerSecond;
                bestExecutionTime = executionTime;
        }        
	currentProb = (float)numHits/(float)NUMTRIALS;

    }                                           // end num tries loop (find maximum performance)
        
        // Print out: (1) the number of threads, (2) the number of trials, 
        // (3) the probability of hitting the plate, and (4) the MegaTrialsPerSecond. 
        // Printing this as a single line with tabs between the numbers is nice so that you can import these lines right into Excel. 
        printf("Best execution time: %8.2lf\n", bestExecutionTime);
        printf("Num threads: %8i\nNum trials: %8i\nHit probability: %8.2lf\nMegaTrials/Sec: %8.2lf\n", NUMT, NUMTRIALS, currentProb, maxPerformance); 
        printf("%i\t%i\t%8.2lf\t%8.2lf\n", NUMT, NUMTRIALS, currentProb, maxPerformance);

        delete [] xcs;
        delete [] ycs;
        delete [] rs;

        return 0;
}                                               // end main

