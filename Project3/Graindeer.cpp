/******************************************************************************
** Program name: Functional Decomposition: Graindeer Simulation
** Author: Rebecca L. Taylor
** Date: 6 May 2019
** Description: This main file executes a simulation of 6 years (72) months
    of a grain-growing operation. It uses functional decomposition to assign
    each simulation function to a different thread. The Graindeer, Grain, and
    GrainDietPopularity functions use current data to calculate the next
    month of the simulation. A Watcher thread prints current data and updates
    environmental variables.
******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "Rand.hpp"

#define END_YEAR 2025

// global variables to define the system state
int     monthCount;             // use to print consecutive month data
int	NowYear;		// 2019 - 2024
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	NowNumDeer;		// number of deer in the current population
float   NowDietPopularity;      // 10000 units sold of "Grain of Thrones" book

// constant parameters
const float GRAIN_GROWS_PER_MONTH =		8.0;    // inches
const float ONE_DEER_EATS_PER_MONTH =		0.5;

const float AVG_PRECIP_PER_MONTH =		6.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				50.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;   // degrees Fahrenheit
const float MIDPRECIP =				10.0;   // inches

const float RANDOM_SOCIAL =                     5.0;    // social media influence
const float GRAIN_DIET_DEPLETION_PERCENT =      0.15; 

unsigned int seed = 0;

// function prototypes for each program section
void	Watcher();
void	Graindeer();
void	Grain();
void 	DietPopularity();

// main program
int main( int argc, char *argv[ ] )
{
#ifndef _OPENMP
	fprintf( stderr, "No OpenMP support!\n" );
	return 1;
#endif

    // starting date and time:
    NowMonth =    0;
    NowYear  = 2019;
    monthCount = 1;

    // starting state (feel free to change this if you want):
    NowNumDeer = 1;
    NowHeight =  1.;
    NowDietPopularity = 20.;

    // calculate starting environmental parameters 
    float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

    float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
    if( NowPrecip < 0. )
     	NowPrecip = 0.;

    // start the threads with a parallel sections directive
    omp_set_num_threads( 4 );               // same as # of sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Graindeer();
        }

        #pragma omp section
        {
            Grain();
        }

        #pragma omp section
        {
            Watcher();
        }

        #pragma omp section
        {
            DietPopularity();
        }
    }                                       // implied barrier (end sections)

   return 0;
}                                           // end main

// simulation functions
void Watcher()
{
    while ( NowYear < END_YEAR)
    {
        // DoneComputing barrier: wait for threads to calculate next values
	#pragma omp barrier

        // DoneAssigning barrier: wait for threads to assign vals to variables
	#pragma omp barrier

        // print results 
        float NowTempCelsius = (5./9.)*(NowTemp-32);
        float NowPrecipCM = NowPrecip*2.54;
        float NowHeightCM = NowHeight*2.54;

        printf("NowYear: %8d\t NowMonth: %8d\n", NowYear, NowMonth+1);
        printf("NowDeer: %8d\t NowDiet: %8.2lf\n", NowNumDeer, NowDietPopularity);
        printf("NowTemp: %8.2lf F / %.2lf C\n", NowTemp, NowTempCelsius); 
        printf("NowPrec: %8.2lfin / %.2lfcm\n", NowPrecip, NowPrecipCM);
        printf("GrainHt: %8.2lfin / %.2lfcm\n", NowHeight, NowHeightCM);
        printf("%d\t%.2lf\t%.2lf\t%.2lf\t%d\t%.2lf\n", monthCount, NowTempCelsius, NowPrecipCM, NowHeightCM, NowNumDeer, NowDietPopularity);

        // increment time
        NowMonth++;
        monthCount++;
        if (NowMonth > 11)
        {
            NowMonth = 0;
            NowYear++;
        }

        // calculate new environmental parameters 
        float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

        float temp = AVG_TEMP - AMP_TEMP * cos( ang );
        NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
        NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
        if( NowPrecip < 0. )
        	NowPrecip = 0.;

        // DonePrinting barrier: other threads wait for printing to complete
	#pragma omp barrier
    }    
}

void Graindeer()
{
    while( NowYear < END_YEAR)
    {
        // compute a temporary next-value for Graindeer quantity
        // based on the current state of the simulation:
        int nextNumDeer = NowNumDeer;

        if ((float)NowNumDeer > NowHeight)      // if deer exceed grain
        {
            nextNumDeer--;
            if (nextNumDeer < 0)
                nextNumDeer = 0;
        }

        else if ((float)NowNumDeer < NowHeight) // if deer less than grain
        {
            nextNumDeer++;
        }

        // DoneComputing barrier: wait for other threads to finish calculating
	#pragma omp barrier
        NowNumDeer = nextNumDeer;

        // DoneAssigning barrier: wait for other threads to update values
	#pragma omp barrier

        // DonePrinting barrier: wait for Watcher thread to print and update environment
	#pragma omp barrier
    }    
}

void Grain()
{
    while( NowYear < END_YEAR)
    {
        // compute a temporary next-value for Grain quantity
        // based on the current state of the simulation:
        float nextHeight = NowHeight;
        float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
        float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );

        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        nextHeight -= (NowDietPopularity * GRAIN_DIET_DEPLETION_PERCENT);
        if (nextHeight < 0.)
            nextHeight = 0.;

        // DoneComputing barrier: wait for other threads to finish calculating
	#pragma omp barrier
        NowHeight = nextHeight;

        // DoneAssigning barrier: wait for other threads to finish updating values
	#pragma omp barrier

        // DonePrinting barrier: wait for Watcher thread to print and update environment
	#pragma omp barrier
    }    
}

void DietPopularity()
{
    while( NowYear < END_YEAR)
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        float nextDietPopularity = NowDietPopularity;  

        // account for random social media influence
        nextDietPopularity += Ranf( &seed, -RANDOM_SOCIAL, RANDOM_SOCIAL);

        // popularity peaks in December due to holidays 
        if (NowMonth == 11)                     
            nextDietPopularity *= 1.5;

        // popularity dips from May to July due to swimsuit pressure
        if (NowMonth >= 4 || NowMonth <= 6)
            nextDietPopularity -= (nextDietPopularity * 0.025);

        // DoneComputing barrier:
	#pragma omp barrier
        NowDietPopularity = nextDietPopularity;

        // DoneAssigning barrier:
	#pragma omp barrier

        // DonePrinting barrier:
	#pragma omp barrier
    }    
}


